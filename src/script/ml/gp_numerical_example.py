from distutils.command.sdist import sdist
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import os
import r_pca as ra
import Model_preparation as mp
import Model_preparation_np as mpn
import gpy_model_multi as gp_struct
import pyro.distributions as dist
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior, NormalPrior, UniformPrior
import datetime

path = os.getcwd()
plt.ion()
plt.show()
starttime = datetime.datetime.now()
def f(x):
    return (.6 * x - .1*x**2) + torch.sin(.9 * x)*5
path_c = r"C:\Users\t1714\Desktop\Academic\Coding_Files\GP building polymers\Data\result_set\numerical example"
# x = torch.linspace(-10,10,100)
# y = torch.Tensor(0)
x = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
y = torch.Tensor([1,2,1,1,2,3,2,1,1,2,3,4,3,2,1,1,2,3,4,5,4,3,2,1])
# for _ in x:
    # y = torch.hstack([y, f(_)])
# trainx = x[::10]
# trainy = y[::10]
# trainx = trainx[1:]
# trainy = trainy[1:]
trainx = x[:16]
trainy = y[:16]
ss = {'min': torch.Tensor([1]), 'max': torch.Tensor([4])}
x_sc = mp.scale_f_zo(trainx, ss)['val']
sy = mp.scale_f(trainy)
y_sc = sy['val']
fig, ax = plt.subplots(1,1, figsize=(9.02,4.8))
ax.plot(x, y)
ax.scatter(trainx, trainy)
dir_c = path_c+'\\fun.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(ard_num_dims=0, num_mixtures=10))
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=5)+\
        gpytorch.kernels.RBFKernel()*gpytorch.kernels.PeriodicKernel()+\
                                                    gpytorch.kernels.PiecewisePolynomialKernel(power=3)
        # self.covar_module.initialize_from_data(train_x, train_y)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_sc, y_sc, likelihood)
# for param_name, param in model.named_parameters():
    # print(f'Parameter name: {param_name:42} value = {param.item()}')
# hypers = {
#     'likelihood.noise_covar.noise': torch.tensor(5.1),
#     'covar_module.base_kernel.lengthscale': torch.tensor(.5),
#     'covar_module.outputscale': torch.tensor(1.),
# }
# model.initialize(**hypers)
# print(f'Actual outputscale after setting: {model.likelihood.noise_covar.noise}')
# print(f'Actual outputscale after setting: {model.covar_module.base_kernel.lengthscale}')
# print(f'Actual outputscale after setting: {model.covar_module.outputscale}')
model.train(), likelihood.train()
model.eval(), likelihood.eval()
testx1 = torch.linspace(-10,10,10)
x_te1 = mp.scale_f_zo(testx1, ss)['val']
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_prior = likelihood(model(x_te1))
#################################################################
model.train(), likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005, betas=(0.9, 0.999))
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 3000
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(x_sc)
    loss = -mll(output, trainy)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()    
model.eval(), likelihood.eval()
testx2 = torch.linspace(-10,10,100)
x_te2 = mp.scale_f_zo(testx2, ss)['val']
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_post = (model(x_te2))
    l, u = observed_pred_post.confidence_region()

fig, ax = plt.subplots(1,2, figsize=(9.02,4.8))
for _ in np.arange(20):
    z = dist.Normal(mp.rescale_f(observed_pred_prior.mean, sy), observed_pred_prior.variance).sample()
    ax[0].plot(testx1, z)
    ax[0].scatter(trainx, trainy)
ax[0].set(title='Prior', ylabel='y')
for _ in np.arange(1000):
    z = dist.Normal(mp.rescale_f(observed_pred_post.mean, sy), observed_pred_post.variance).sample()
    ax[1].plot(testx2, z)
ax[1].scatter(trainx, trainy)
ax[1].plot(testx2, mp.rescale_f(observed_pred_post.mean, sy))
ax[1].plot(testx2, mp.rescale_f(l, sy), ls='--', color=(0,0,0))
ax[1].plot(testx2, mp.rescale_f(u, sy), ls='--', color=(0,0,0))
# ax[1].scatter(trainx, trainy)
ax[1].set(title='Posterior')
fig.supxlabel('x')
fig.suptitle('Gaussian Process')
dir_c = path_c+'\\gp.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500)

class ExactGPModel1(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel1, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()*gpytorch.kernels.PeriodicKernel()
                                                        +gpytorch.kernels.PiecewisePolynomialKernel(power=3))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
model1 = ExactGPModel1(x_sc, y_sc, likelihood1)
model1.train(), likelihood1.train()
optimizer = torch.optim.AdamW(model1.parameters(), lr = 0.01, betas=(0.9, 0.999), weight_decay=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, model1)
training_iter = 3000
for i in range(training_iter):
    optimizer.zero_grad()
    output = model1(x_sc)
    loss = -mll(output, trainy)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()    
model1.eval(), likelihood1.eval()
############################################################
class ExactGPModel2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_augument):
        super(ExactGPModel2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()
                                                        +gpytorch.kernels.PeriodicKernel())
        self.feature_augument = feature_augument
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0., 1.)
        # self.covar_module.initialize_from_data(train_x, train_y)
    def forward(self, x):
        projected_x = self.feature_augument(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class Feature_Augument(torch.nn.Sequential):
    def __init__(self, data_dim=1, aug_feature1=100, aug_feature2=100):
        super(Feature_Augument, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, aug_feature1))
        self.add_module('acfun1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(aug_feature1, aug_feature2))
        self.add_module('acfun2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(aug_feature2, 1))
        self.add_module('acfun3', torch.nn.Tanh())
feature_expand = Feature_Augument()
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
model2 = ExactGPModel2(x_sc, y_sc, likelihood2, feature_expand)

model2.train(), likelihood2.train()
optimizer = torch.optim.AdamW(model2.parameters(), lr = 0.01, betas=(0.9, 0.999), weight_decay=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, model2)
training_iter = 3000
for i in range(training_iter):
    optimizer.zero_grad()
    output = model2(x_sc)
    loss = -mll(output, trainy)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()    
model2.eval(), likelihood2.eval()
###############################################################
# x = torch.linspace(-10,20,150)
# y = torch.Tensor(0)
# for _ in x:
    # y = torch.hstack([y, f(_)])
testx3 = torch.linspace(-1,30,150)
x_te3 = mp.scale_f_zo(testx3, ss)['val']
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    yp1 = (model(x_te3))
    l1, u1 = yp1.confidence_region()
    yp2 = (model1(x_te3))
    l2, u2 = yp2.confidence_region()
    yp3 = (model2(x_te3))
    l3, u3 = yp3.confidence_region()
fig, ax = plt.subplots(1,3, figsize=(9.02,4.8))
ax[0].plot(x, y, label='obs')
ax[0].scatter(trainx, trainy, label='trns')
ax[0].plot(testx3, mp.rescale_f(yp1.mean, sy), label='mean')
ax[0].plot(testx3, mp.rescale_f(l1, sy), ls='--', color=(0,0,0), label='u1')
ax[0].plot(testx3, mp.rescale_f(u1, sy), ls='--', color=(0,0,0), label='u2')
ax[0].set(title='RBF')
ax[0].legend()

ax[1].plot(x, y)
ax[1].scatter(trainx, trainy)
ax[1].plot(testx3, mp.rescale_f(yp2.mean, sy))
ax[1].plot(testx3, mp.rescale_f(l2, sy), ls='--', color=(0,0,0))
ax[1].plot(testx3, mp.rescale_f(u2, sy), ls='--', color=(0,0,0))
ax[1].set(title='RBF+Per')

ax[2].plot(x, y)
ax[2].scatter(trainx, trainy)
ax[2].plot(testx3, mp.rescale_f(yp3.mean, sy))
ax[2].plot(testx3, mp.rescale_f(l3, sy), ls='--', color=(0,0,0))
ax[2].plot(testx3, mp.rescale_f(u3, sy), ls='--', color=(0,0,0))
ax[2].set(title='Deep(RBF+Per)')

endtime = datetime.datetime.now()
print (f"Total training time: {(endtime - starttime).seconds:d}")
plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')