import gpytorch
import torch
import torch.utils.data as Data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from gpytorch.kernels import AdditiveKernel
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior, NormalPrior, UniformPrior
import os
path = os.getcwd()

def choose_kernel(k_id, ts):
    defaul_k = gpytorch.kernels.RBFKernel(ard_num_dims=ts, active_dims=tuple(np.arange(ts)),
                                        lengthscale_prior=GammaPrior(3, 8)) 
    m_dict = {
        0: defaul_k,
        1: gpytorch.kernels.SpectralMixtureKernel(ard_num_dims=ts, num_mixtures=2,),
                                                # mixture_means_prior=LogNormalPrior(0.1, 0.1),
                                                # mixture_scales_prior=LogNormalPrior(0.1, 0.1),
                                                # mixture_weights_prior=LogNormalPrior(0.1, 0.1)),
        2: gpytorch.kernels.MaternKernel(ard_num_dims=3, active_dims=(0,1,2), nu=0.5),
        3: gpytorch.kernels.PiecewisePolynomialKernel(q=3, ard_num_dims=6, active_dims=(0,1,2,3,4,5)),
        4: gpytorch.kernels.PolynomialKernel(power=3),
        5: gpytorch.kernels.RQKernel(ard_num_dims=6, active_dims=(0,1,2,3,4,5,)),
    }
    fun = m_dict.get(k_id, defaul_k)
    return fun

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ts, feature_augument):
        projected_x = feature_augument(train_x)
        skm = choose_kernel(1, ts)
        skm.initialize_from_data(projected_x, train_y)
        rbf = choose_kernel(0, ts)
        poly = choose_kernel(4, ts)
        ker_custom = AdditiveKernel(
            skm,
            rbf,
            poly
        )
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=ts), num_tasks=ts
        )
        # for p in vae.parameters():
        #     p.requires_grad = False
        self.covar_module = gpytorch.kernels.MultitaskKernel(ker_custom, num_tasks=ts, rank=1)
        self.feature_augument = feature_augument
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        projected_x = self.feature_augument(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
class Feature_Augument(torch.nn.Sequential):
    def __init__(self, data_dim=6, aug_feature1=10):
        super(Feature_Augument, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, aug_feature1))
        self.add_module('acfun1', torch.nn.Sigmoid())

def setup_data_loaders(x, y, batch_size=128):
    torch_data = Data.TensorDataset(x, y)
    kwargs = {'num_workers': 0, 'pin_memory': False}
    loader = Data.DataLoader(
        dataset=torch_data, 
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    return loader       
       
def train(gpr, likelihood, x_train, y_train, load_model=0, dir=None, its=50):
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gpr.parameters()), lr=0.01, \
    # betas=(0.9, 0.999))
    if load_model:
        state_dict = torch.load(dir)
        gpr.load_state_dict(state_dict)
        # gpr = torch.load(dir)
    else:
        optimizer = torch.optim.AdamW([
        {'params': gpr.feature_augument.parameters()},
        {'params': gpr.covar_module.parameters()},
        {'params': gpr.mean_module.parameters()},
        {'params': gpr.likelihood.parameters()},
        ], lr=0.001, betas=(0.9, 0.999))
        # optimizer = torch.optim.AdamW(gpr.parameters(), lr = 0.001, betas=(0.9, 0.999), weight_decay=0.01, amsgrad=True)
        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)
        train_loss = []
        TEST_FREQUENCY = 10
        normalizer_train = len(y_train)
        for epoch in range(its):
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
        if dir is not None:
            # torch.save(gpr.state_dict(), dir)
            torch.save(gpr, dir)   
    return gpr, likelihood


# def train(gpr, likelihood, X_train, y_train):
#     optimizer = torch.optim.AdamW(gpr.parameters(), lr=.1, betas=(0.9, 0.95))
#     # optimizer = torch.optim.SGD(gpr.parameters(), lr=1)
#     loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)
#     batch_size = round(X_train.shape[0]*0.5)
#     # iterations = 20*round(X_train.shape[0]/batch_size)
#     iterations = 1
#     epochs_t = 20
#     lambda1 = lambda epo_ite: 1/(epochs_t*iterations) * epo_ite
#     scheduler_t = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
#     losses_opt = []
#     lrs_opt = []
#     for _ in range(epochs_t):
#         # for  param in gpr.parameters():
#             # param.grad = None
#         for _ in range(iterations):
#             # flag = torch.randperm(batch_size)
#             # x_train_lp = X_train[flag]
#             x_train_lp = X_train
#             y_train_lp = y_train + (1.0e-4)*torch.randn(x_train_lp.shape[0], y_train.shape[1])
#             gpr.set_train_data(x_train_lp, y_train_lp, strict=False)
#             optimizer.zero_grad()
#             output = gpr(x_train_lp)
#             loss = -loss_fn(output, y_train_lp)
#             loss.backward()
#             optimizer.step()
#             lrs_opt.append(optimizer.param_groups[0]["lr"])
#             losses_opt.append(loss.item())
#             scheduler_t.step()
#     loss_filt = signal.savgol_filter(losses_opt, 3, 2)
#     df_l = np.diff(loss_filt)
#     lr_p = np.zeros(df_l.shape[0])
#     ###############strategy 1############################
#     # lr_down = np.abs(df_l[df_l<0])
#     # fl1 = np.min(np.argwhere(lr_down>np.quantile(np.sort(lr_down),0.5)))
#     # min_lr = lrs_opt[fl1]
#     # if np.max(df_l)>0:
#     #     lr_up = df_l[df_l>0]
#     #     upv = np.quantile(np.sort(lr_up), 0.5)
#     #     fl2 = np.min(np.argwhere(df_l>=upv))+1
#     #     max_lr = lrs_opt[fl2]
#     # else:
#     #     max_lr = lrs_opt[-1]
#     ###############strategy 2############################
#     lr_p[df_l<=0] = -1
#     lr_p[df_l>0] = 1
#     lr_p2 = np.diff(lr_p)
#     loc1 = (np.argwhere(lr_p2==-2))+1
#     loc2 = (np.argwhere(lr_p2==2))+1
#     if (loc1.size>0)&(loc2.size>0):
#         if loc2.size>1:
#             lr_lowest = loc2[np.argmin([losses_opt[_] for _ in loc2.squeeze()])]
#         else:
#             lr_lowest = loc2[np.argmin(losses_opt[loc2.squeeze()])]
#         if losses_opt[lr_lowest[0]]<losses_opt[0]:
#             if loc1[loc1<lr_lowest].size>0:
#                 ds = loc1[np.argmin(np.abs(loc1[loc1<lr_lowest]-lr_lowest))]
#                 min_lr = lrs_opt[ds[0]]
#                 max_lr = lrs_opt[lr_lowest[0]]
#             else:
#                 min_lr = lrs_opt[0]
#                 max_lr = lrs_opt[lr_lowest[0]]
#         else:
#             min_lr = lrs_opt[0]
#             max_lr = lrs_opt[2]
#     elif loc1.size+loc2.size==0:
#         min_lr = lrs_opt[0]
#         max_lr = lrs_opt[-1]
#     elif loc1.size==0:
#         min_lr = lrs_opt[0]
#         max_lr = lrs_opt[loc2.squeeze()]
#     elif loc2.size==0:
#         min_lr = lrs_opt[0]
#         max_lr = lrs_opt[2]
#         if losses_opt[-1]<losses_opt[0]:
#             min_lr = lrs_opt[loc1[0].item()]
#             max_lr = lrs_opt[-1]
#     # min_lr = 0.01
#     # max_lr = 0.8
#     # batch_size = round(X_train.shape[0]*0.9)
#     # iterations = 5*round(X_train.shape[0]/batch_size)
#     iterations = 1
#     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr,
#                                         step_size_up = 1*iterations, step_size_down = 1*iterations,
#                                         cycle_momentum=False)
#     epochs = 10
#     losses = []
#     lrs = []
#     for _ in range(epochs):
#     # for  param in gpr.parameters():
#             # param.grad = None
#         for _ in range(iterations):
#             # flag = torch.randperm(batch_size)
#             # x_train_lp = X_train[flag]
#             x_train_lp = X_train
#             y_train_lp = y_train + (1.0e-4)*torch.randn(x_train_lp.shape[0], y_train.shape[1])
#             gpr.set_train_data(x_train_lp, y_train_lp, strict=False)
#             optimizer.zero_grad()
#             output = gpr(x_train_lp)
#             loss = -loss_fn(output, y_train_lp)
#             loss.backward()
#             optimizer.step()
#             lrs.append(optimizer.param_groups[0]["lr"])
#             scheduler.step()
#         losses.append(loss.item())
#     # loss.backward(retain_graph=True)
#     # optimizer.zero_grad()
#     # for param_name, param in gpr.named_parameters():
#         # print(f'Parameter name: {param_name:42} value = {param.detach().numpy()}')
#     # gpr.set_train_data(X_train, y_train, strict=False)
#     tx1 = np.arange(len(losses))
#     tx2 = np.arange(len(lrs))
#     fig1 = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=[f'Learning Rate Test- {epochs_t:d} epochs'
#                         , f'Learning Rate Test Filter- {epochs_t:d} epochs'],
#     )
#     fig1.add_trace(
#                 go.Scatter(x=lrs_opt, y=losses_opt, line_shape='linear'),
#                 row=1, col=1
#     )
#     fig1.add_trace(
#                 go.Scatter(x=lrs_opt, y=loss_filt, line_shape='linear'),
#                 row=1, col=2
#     )
#     fig1.update_layout(title=f'Final Training Loss: {losses[-1]:.2f}',
#                     xaxis1_title='Learning Rate',
#                     xaxis2_title='Learning Rate',
#                     yaxis1_title='Loss',
#                     yaxis2_title='Loss',)
#     # fig1.show()

#     fig2 = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=[f'Final Training Loss: {losses[-1]:.2f}'
#                         , f'CLR Policy'],
#     )
#     fig2.add_trace(
#                 go.Scatter(x=tx1, y=losses, line_shape='linear'),
#                 row=1, col=1
#     )
#     fig2.add_trace(
#                 go.Scatter(x=tx2, y=lrs, line_shape='linear'),
#                 row=1, col=2
#     )
#     fig2.update_layout(title=f'Final Training Loss: {losses[-1]:.2f}',
#                     xaxis1_title='Function Evaluations',
#                     xaxis2_title='Epochs X Iterations',
#                     yaxis1_title='Loss',
#                     yaxis2_title='Learning Rate',)
#     # fig2.show()
#     fig1.write_image(path+'/Data/result_set/lr_loss1.png', scale=2)
#     fig2.write_image(path+'/Data/result_set/lr_loss2.png', scale=2)
#     return losses[-1]
        

