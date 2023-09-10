import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.optim import AdamW
from pyro.infer import SVI, Trace_ELBO

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, hidden_dim2, Md):
        super().__init__()
        # setup the three linear transformations used
        # self.cv1 = nn.Conv1d(132,hidden_dim,4)
        self.fc1 = nn.Linear(Md, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.Md = Md

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        # then compute the hidden units
        x = x.reshape(-1, self.Md)
        hidden2 = self.relu(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        hidden = self.relu(self.fc2(hidden2))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, hidden_dim2, Md):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, Md)
        self.fc22 = nn.Linear(hidden_dim, Md)
        # setup the non-linearities
        # self.softplus = nn.Softplus()
        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden2 = self.relu(self.fc1(z))
        hidden = self.relu(self.fc2(hidden2))
        loc_y = self.fc21(hidden)
        scale_y = torch.exp(self.fc22(hidden))
        return loc_y, scale_y
# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=15, hidden_dim=67, hidden_dim2=32, Md=132):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, hidden_dim2, Md)
        self.decoder = Decoder(z_dim, hidden_dim, hidden_dim2, Md)
        self.z_dim = z_dim
        self.Md = Md

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_y, scale_y = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Normal(loc_y, scale_y, validate_args=False).to_event(1),
                obs=x.reshape(-1, self.Md),
            )
            # return the loc so we can visualize it later
            return loc_y, scale_y
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    # define a helper functiyon for reconstructing images
    def reconstruct_y(self, x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        # z = dist.Normal(z_loc, z_scale).sample()+z_plus
        z = z_loc
        loc, _ = self.decoder(z)
        return loc
def setup_data_loaders(y_train, batch_size=128, use_cuda=False):
    train_set = y_train
    # test_set = y_test
    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset=test_set,
    #     batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader 
def train(vae, y_train, load_model=0, dir=None, its=5000):
    if load_model:
        pyro.clear_param_store()
        vae = torch.load(dir)
    else:
        optimizer = AdamW({"lr": 1.0e-3, "betas":(0.9, 0.999), "weight_decay":0.01, "amsgrad":False})
        elbo = Trace_ELBO()
        svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
        train_elbo = []
        TEST_FREQUENCY = 10
        normalizer_train = len(y_train)
        for epoch in range(its):
            epoch_loss = 0.0
            train_loader = setup_data_loaders(y_train, batch_size=200)
            for x in train_loader:
                epoch_loss += svi.step(x)
            if epoch % TEST_FREQUENCY == 0:
                total_epoch_loss_train = epoch_loss / normalizer_train
                print(
                    "[epoch %03d]  average training loss: %.4f"
                    % (epoch, total_epoch_loss_train)
                    )
            train_elbo.append(total_epoch_loss_train)
        if dir is not None:
            torch.save(vae, dir)
    return vae
                
        