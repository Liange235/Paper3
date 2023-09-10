import torch
from torch import nn
import torch.utils.data as Data
class NeuralNetwork(nn.Module):
    def __init__(self, input, output, d1, d2, d3):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input, d1),
            nn.Tanh(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, d3),
            nn.ReLU(),
            nn.Linear(d3, output),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
def setup_data_loaders(x, y, batch_size=128):
    torch_data = Data.TensorDataset(x, y)
    kwargs = {'num_workers': 0, 'pin_memory': False}
    loader = Data.DataLoader(
        dataset=torch_data, 
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    return loader 

def train(nn_model, x_train, y_train, load_model=0, dir=None, its=50):
    if load_model:
        nn_model = torch.load(dir)
    else:
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01)
        train_loss = []
        TEST_FREQUENCY = 10
        normalizer_train = len(y_train)
        for epoch in range(its):
            epoch_loss = 0.0
            loader = setup_data_loaders(x_train, y_train, batch_size=500)
            for batch_x, batch_y in loader:
                pred = nn_model(batch_x)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss+=loss
            if epoch % TEST_FREQUENCY == 0:
                total_epoch_loss_train = epoch_loss / normalizer_train
                print(
                    "[epoch %03d]  average training loss: %.4f"
                    % (epoch, total_epoch_loss_train)
                    )
                train_loss.append(total_epoch_loss_train)
        if dir is not None:
            # torch.save(gpr.state_dict(), dir)
            torch.save(nn_model, dir)
    return nn_model
