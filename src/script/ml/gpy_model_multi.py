import gpytorch
import torch
import torch.utils.data as Data
import numpy as np
import os
from gpytorch.kernels import AdditiveKernel
path = os.getcwd()

def choose_kernel(k_id, ts):
    defaul_k = gpytorch.kernels.RBFKernel(ard_num_dims=ts, active_dims=tuple(np.arange(ts))) 
    m_dict = {
        0: defaul_k,
        1: gpytorch.kernels.SpectralMixtureKernel(ard_num_dims=ts, num_mixtures=6),
        2: gpytorch.kernels.MaternKernel(ard_num_dims=3, active_dims=(0,1,2), nu=0.5),
        3: gpytorch.kernels.PiecewisePolynomialKernel(q=3, ard_num_dims=6, active_dims=(0,1,2,3,4,5)),
        4: gpytorch.kernels.PolynomialKernel(power=3),
        5: gpytorch.kernels.RQKernel(ard_num_dims=6, active_dims=(0,1,2,3,4,5,)),
        6: gpytorch.kernels.PolynomialKernel(num_dimensions=6, active_dims=(0,1,2,3,4,5), power=3),
    }
    fun = m_dict.get(k_id, defaul_k)
    return fun

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, k_id, ts):
        ker_custom = choose_kernel(k_id, train_x.shape[1])
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=train_x.shape[1]), num_tasks=ts
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(ker_custom, num_tasks=ts, rank=3, batch_size=[])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def setup_data_loaders(x, y, batch_size=128):
    torch_data = Data.TensorDataset(x, y)
    kwargs = {'num_workers': 0, 'pin_memory': False}
    loader = Data.DataLoader(
        dataset=torch_data, 
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    return loader       
       
def train(gpr, likelihood, x_train, y_train, load_model=0, dir=None):
    if load_model:
        # state_dict = torch.load(dir)
        # gpr.load_state_dict(state_dict)
        gpr = torch.load(dir)
    else:
        optimizer = torch.optim.AdamW([
        {'params': gpr.covar_module.parameters()},
        {'params': gpr.mean_module.parameters()},
        {'params': gpr.likelihood.parameters()},
        ], lr=0.001, betas=(0.9, 0.999))
        # optimizer = torch.optim.AdamW(gpr.parameters(), lr = 0.01, betas=(0.9, 0.999), weight_decay=0.01, amsgrad=True)
        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)
        train_loss = []
        TEST_FREQUENCY = 10
        normalizer_train = len(y_train)
        for epoch in range(100):
            epoch_loss = 0.0
            loader = setup_data_loaders(x_train, y_train, batch_size=500)
            for batch_x, batch_y in loader:
                gpr.set_train_data(batch_x, batch_y, strict=False)
                optimizer.zero_grad()
                output = gpr(batch_x)
                loss = -loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss+=loss
                # print(loss.item())
            if epoch % TEST_FREQUENCY == 0:
                total_epoch_loss_train = epoch_loss / normalizer_train
                print(
                    "[epoch %03d]  average training loss: %.4f"
                    % (epoch, total_epoch_loss_train)
                    )
                train_loss.append(total_epoch_loss_train)
        # torch.save(gpr.state_dict(), dir) 
        torch.save(gpr, dir)     
    return gpr, likelihood
