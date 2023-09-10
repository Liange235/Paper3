from distutils.command.sdist import sdist
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import os
import r_pca as ra
import Model_preparation as mp
import Model_preparation_np as mpn
import mwd_data_preparation as mdp
import gpvae_model_multi_large as gp_vae
import gpy_model_multi as gp_struct
import vae_model
import visualization_of_mwd as vom
from torch.distributions import normal
import pyro.distributions as dist
import datetime
import pybobyqa

path = os.getcwd()
plt.ion()
plt.show()
starttime = datetime.datetime.now()
str_n = 'MWD'
# str_n = 'CSD'
ind = 'passive'
apx = os.path.join(path,'Data','result_set','model_'+str_n)
path_c = r"C:\Users\t1714\Desktop\Academic\Monthly Reports\paper1\pic"
apc = os.path.join(path_c, str_n)
load_model = 1
data_dict = mdp.generate_scenario1(str_n)
def case1():
    X_Sim = data_dict['x_sim']
    Y_Sim = data_dict['y_sim']
    chain = data_dict['chain']
    hidden = 6
    pcom = 7
    h1 = 60
    h2 = 32
    str_x = 'Chain Length'
    return X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x
def case2():
    X_Sim = data_dict['x_sim']['input']
    Y_Sim = data_dict['y_sim']['output']
    chain = np.linspace(0, 4, 400)[:,None]
    hidden = 10
    pcom = 4
    h1 = 100
    h2 = 50
    str_x = 'Length(mm)'
    return X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x
switch = {'MWD': case1,
          'CSD': case2}
X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x = switch[str_n]()
Md = Y_Sim.shape[1]
rpca = ra.R_pca(Y_Sim)
LowY, SparseY = rpca.fit(0.005, max_iter=10000, iter_print=100)
abn_p = np.count_nonzero(SparseY, axis=1)
lower_q = np.quantile(np.sort(abn_p), 0.25, method='lower')
higher_q = np.quantile(np.sort(abn_p), 0.75, method='higher')
iqr = higher_q-lower_q
thre = higher_q+1.5*iqr
X_Sim_Valid = torch.Tensor(X_Sim[abn_p<=thre, :])[:1500]
Y_Sim_Valid = torch.Tensor(Y_Sim[abn_p<=thre, :])[:1500]
# X_Sim_Valid = torch.Tensor(X_Sim)[:1500]
# Y_Sim_Valid = torch.Tensor(Y_Sim)[:1500]
#########################delete abnormal data using rpca############################
np.random.seed(0)
tf = round(0.8*X_Sim_Valid.shape[0])
# tf2 = round(0.2*X_Sim_Valid.shape[0])
ts = np.random.randint(1,X_Sim_Valid.shape[0], size=X_Sim_Valid.shape[0])
TraX = X_Sim_Valid[ts[:tf],:]
TraY = Y_Sim_Valid[ts[:tf],:]
# fig, ax = plt.subplots(1,1)
# ax.plot(chain, TraY[2])
# ax.set(xlabel=str_x, ylabel=str_n, title='Molecular weight distribution')
# dir_c = apc+'\\exp.png'
# plt.tight_layout()
# plt.savefig(dir_c, dpi=500) 
###########################################
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# for _ in np.arange(10):
#     ax.scatter(X_Sim[_,0], X_Sim[_,1], X_Sim[_,2], s=100)
# ax.set(xlabel='C_in', ylabel='Q', zlabel='R', title='Process variables')
# dir_c = apc+'\\csd_x.png'
# plt.tight_layout()
# plt.savefig(dir_c, dpi=800) 
# fig, ax = plt.subplots(1,1)
# for _ in np.arange(10):
#     ax.plot(chain, Y_Sim[_])
# ax.set(xlabel='Particle size(mm)', ylabel='Volumn pdf(mm^(-1))', title='Crystal size distribution')
# dir_c = apc+'\\csd_y.png'
# plt.tight_layout()
# plt.savefig(dir_c, dpi=800) 
###########################################
# fig, ax = plt.subplots(1,1, figsize=(4.8, 2.76))
# ax.plot(chain, Y_Sim[0])
# ax.set(xlabel='Particle size(mm)', ylabel='Volumn pdf(mm^(-1))', title='Crystal size distribution')
# dir_c = apc+'\\csd_example.png'
# plt.tight_layout()
# plt.savefig(dir_c, dpi=800)
fig, ax = plt.subplots(1,1, figsize=(4.8, 2.76))
ax.plot(chain, Y_Sim[0])
ax.set(xlabel='Chain length', ylabel='MWD', title='Molecular weight distribution')
dir_c = apc+'\\mwd_example.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=800)  
###########################################
sy_zo = mp.scale_f_zo(TraY)
TraY_Zo = sy_zo['val']
# ValX = X_Sim_Valid[ts[tf:tf+tf2],:]
# ValY = Y_Sim_Valid[ts[tf:tf+tf2],:]
ValX = X_Sim_Valid[ts[tf:],:]
ValY = Y_Sim_Valid[ts[tf:],:]
# TeX = X_Sim_Valid[ts[tf+tf2:],:]
# TeY = Y_Sim_Valid[ts[tf+tf2:],:]
ValY_Zo = mp.scale_f_zo(ValY, sy_zo)['val']
# TeY_Zo = mp.scale_f_zo(TeY, sy_zo)['val']
# ValY_Zo2 = mp.scale_f_zo(ValY2, sy_zo)['val']
vae = vae_model.VAE(z_dim=hidden, Md=Md, hidden_dim=h1, hidden_dim2=h2)
dir = apx+'_vae.pth'
vae = vae_model.train(vae, TraY_Zo, dir=dir, load_model=1)
reco_tra = vae.reconstruct_y(TraY_Zo)
reco_tes = vae.reconstruct_y(ValY_Zo)
reco_tra_vae = mpn.rescale_f_zo(reco_tra, sy_zo).detach()
reco_tra_vae[reco_tra_vae<0] = 0
reco_tes_vae = mpn.rescale_f_zo(reco_tes, sy_zo).detach()
reco_tes_vae[reco_tes_vae<0] = 0
#########################find and plot the biggest reconstruction error wmd using vae############################
err = []
for _, value in enumerate(reco_tra_vae):
    m = sum((TraY[_]-value)**2).item()
    err.append(m)
err = np.asarray(err)
sse_vae = sum(err)
max_ind = np.argmax(err)
# fig, ax = plt.subplots(1,2, figsize=(9.02,4.8))
fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(np.arange(len(err)), err)
ax.set(xlabel='observation', ylabel='error', title='sum square error for vae: %.4f (train)' % (sse_vae))
ax = fig.add_subplot(232)
ax.plot(chain, TraY[max_ind], label='y')
ax.plot(chain, reco_tra_vae[max_ind], label='y_hat')
ax.set(xlabel=str_x, ylabel=str_n, title='sample1')
ax.legend()
ax = fig.add_subplot(233)
ax.plot(chain, TraY[max_ind], label='y')
ax.plot(chain, reco_tra_vae[max_ind], label='y_hat')
ax.set(xlabel=str_x, ylabel=str_n, title='sample2')
ax.legend()
ax = fig.add_subplot(235)
ax.plot(chain, TraY[max_ind], label='y')
ax.plot(chain, reco_tra_vae[max_ind], label='y_hat')
ax.set(xlabel=str_x, ylabel=str_n, title='sample1')
ax.legend()
ax = fig.add_subplot(236)
ax.plot(chain, TraY[max_ind], label='y')
ax.plot(chain, reco_tra_vae[max_ind], label='y_hat')
ax.set(xlabel=str_x, ylabel=str_n, title='sample1')
ax.legend()


dir_c = apc+'\\train_vae_case1.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 
err = []
for _, value in enumerate(reco_tes_vae):
    m = sum((ValY[_]-value)**2).item()
    err.append(m)
err = np.asarray(err)
sse_vae = sum(err)
max_ind = np.argmax(err)
fig, ax = plt.subplots(1,2, figsize=(9.02,4.8))
ax[0].scatter(np.arange(len(err)), err)
ax[0].set(xlabel='observation', ylabel='error', title='sum square error for vae: %.4f (validate)' % (sse_vae))
ax[1].plot(chain, ValY[max_ind], label='original')
ax[1].plot(chain, reco_tes_vae[max_ind], label='reconstructed')
ax[1].set(xlabel=str_x, ylabel=str_n, title='Largest reconstruction error case')
ax[1].legend()
dir_c = apc+'\\valid_vae_case1.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 
#########################find and plot the biggest reconstruction error wmd using pca############################
sy_std = mpn.scale_f(TraY.numpy())
TraY_Std = sy_std['val']
ValY_Std = mpn.scale_f(ValY.numpy(), sy_std)['val']
U_Y, S_Y, V_Y = np.linalg.svd(TraY_Std , full_matrices=False)
pro = V_Y[0:pcom, :].T
TraY_p = np.dot(TraY_Std, pro)
ValY_p = np.dot(ValY_Std, pro)
p_inv = np.linalg.lstsq(pro, np.eye(Md), rcond=None)[0]
reco_tra_pca = torch.Tensor(mpn.rescale_f(np.dot(TraY_p, p_inv), sy_std))
reco_tes_pca = torch.Tensor(mpn.rescale_f(np.dot(ValY_p, p_inv), sy_std))
err = []
for _, value in enumerate(reco_tra_pca):
    m = sum((TraY[_]-value)**2).item()
    err.append(m)
err = np.asarray(err)
sse_pca = sum(err)
max_ind = np.argmax(err)
fig, ax = plt.subplots(1,2, figsize=(9.02,4.8))
ax[0].scatter(np.arange(len(err)), err)
ax[0].set(xlabel='observation', ylabel='error', title='sum square error for pca: %.4f (train)' % (sse_pca))
ax[1].plot(chain, TraY[max_ind], label='original')
ax[1].plot(chain, reco_tra_pca[max_ind], label='reconstructed')
ax[1].set(xlabel=str_x, ylabel=str_n, title='Largest reconstruction error case')
ax[1].legend()
dir_c = apc+'\\train_pca_case1.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 
err = []
for _, value in enumerate(reco_tes_pca):
    m = sum((ValY[_]-value)**2).item()
    err.append(m)
err = np.asarray(err)
sse_pca = sum(err)
max_ind = np.argmax(err)
fig, ax = plt.subplots(1,2, figsize=(9.02,4.8))
ax[0].scatter(np.arange(len(err)), err)
ax[0].set(xlabel='observation', ylabel='error', title='sum square error for pca: %.4f (validate)' % (sse_pca))
ax[1].plot(chain, ValY[max_ind], label='original')
ax[1].plot(chain, reco_tes_pca[max_ind], label='reconstructed')
ax[1].set(xlabel=str_x, ylabel=str_n, title='Largest reconstruction error case')
ax[1].legend()
dir_c = apc+'\\valid_pca_case1.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 
#########################gp_pca modeling############################
Sx = mp.scale_f_zo(TraX)
train_x = Sx['val']
train_y = torch.Tensor(TraY_p)
test_x = mp.scale_f_zo(ValX, Sx)['val']
k_id = 0
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=pcom, rank=1)
gpr = gp_struct.MultitaskGPModel(train_x, train_y, likelihood, k_id, pcom)
dir = apx+'_gpca.pth'
gpr.train(), likelihood.train()
gpr, likelihood = gp_struct.train(gpr, likelihood, train_x, train_y, dir=dir, load_model=load_model)
gpr.eval(), likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = gpr(train_x)
    lower, upper = observed_pred.confidence_region()
# with torch.no_grad():
#     observed_pred = gpr(train_x)
#     sd = observed_pred.variance.sqrt()
#     lower = observed_pred.mean-2*sd
#     upper = observed_pred.mean+2*sd
Gp_Pca = mpn.rescale_f(np.dot(observed_pred.mean.numpy(), p_inv), sy_std)
quan05_value = np.dot(lower.numpy(), p_inv)
quan95_value = np.dot(upper.numpy(), p_inv)
Y_Hat_F05 = mpn.rescale_f(quan05_value, sy_std)
Y_Hat_F95 = mpn.rescale_f(quan95_value, sy_std)
vom.plot_mwd_animation(np.squeeze(chain), Gp_Pca, TraY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gp_pca_tr')

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = gpr(test_x)
    lower, upper = observed_pred.confidence_region()
# with torch.no_grad():
#     observed_pred = gpr(test_x)
#     sd = observed_pred.variance.sqrt()
#     lower = observed_pred.mean-2*sd
#     upper = observed_pred.mean+2*sd
Gp_Pca = mpn.rescale_f(np.dot(observed_pred.mean.numpy(), p_inv), sy_std)
quan05_value = np.dot(lower.numpy(), p_inv)
quan95_value = np.dot(upper.numpy(), p_inv)
Y_Hat_F05 = mpn.rescale_f(quan05_value, sy_std)
Y_Hat_F95 = mpn.rescale_f(quan95_value, sy_std)
vom.plot_mwd_animation(np.squeeze(chain), Gp_Pca, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gp_pca_te')
#########################gp_vae modeling############################
h_feature = vae.encoder(TraY_Zo)[0].detach()
h_feature_te = vae.encoder(ValY_Zo)[0].detach()
th = np.arange(h_feature.shape[1])
fig, ax = plt.subplots(1,2, figsize=(15.36,7.65))
ax[0].plot(th, h_feature.T)
ax[1].plot(th, h_feature_te.T)
ax[0].set(xlabel='x', ylabel='hidden feature', title='Feature extraction(train)')
ax[1].set(xlabel='x', ylabel='hidden feature', title='Feature extraction(validate)')
dir_c = apc+'\\overall.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 
sf = mp.scale_f(h_feature)
train_y = sf['val']
feature_expand = gp_vae.Feature_Augument(data_dim=TraX.shape[1], aug_feature1=hidden)
likelihood_vae = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=hidden, rank=1)
gprvae = gp_vae.MultitaskGPModel(train_x, train_y, likelihood_vae, hidden, feature_expand)
dir = apx+'_gpvae.pth'
gprvae.train(), likelihood_vae.train()
gprvae, likelihood_vae = gp_vae.train(gprvae, likelihood_vae, train_x, train_y, load_model=1, dir=dir, its=2000)
gprvae.eval(), likelihood_vae.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = gprvae(train_x)
    lower, upper = observed_pred.confidence_region()
h_pred = mp.rescale_f(observed_pred.mean, sf)
y_pred = vae.decoder(h_pred)[0]
h_pred_lo = mp.rescale_f(lower, sf)
y_pred_lo = vae.decoder(h_pred_lo)[0]
h_pred_up = mp.rescale_f(upper, sf)
y_pred_up = vae.decoder(h_pred_up)[0]
Gp_Vae = mp.rescale_f_zo(y_pred, sy_zo).detach().numpy()
Y_Hat_F05 = mp.rescale_f_zo(y_pred_lo, sy_zo).detach().numpy()
Y_Hat_F95 = mp.rescale_f_zo(y_pred_up, sy_zo).detach().numpy()
Gp_Vae[Gp_Vae<0] = 0
Y_Hat_F05[Y_Hat_F05<0] = 0
Y_Hat_F95[Y_Hat_F95<0] = 0
vom.plot_mwd_animation(np.squeeze(chain), Gp_Vae, TraY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gp_vae_tr')

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = gprvae(test_x)
    lower, upper = observed_pred.confidence_region()
h_pred = mp.rescale_f(observed_pred.mean, sf)
y_pred = vae.decoder(h_pred)[0]
h_pred_lo = mp.rescale_f(lower, sf)
y_pred_lo = vae.decoder(h_pred_lo)[0]
h_pred_up = mp.rescale_f(upper, sf)
y_pred_up = vae.decoder(h_pred_up)[0]
Gp_Vae = mp.rescale_f_zo(y_pred, sy_zo).detach().numpy()
Y_Hat_F05 = mp.rescale_f_zo(y_pred_lo, sy_zo).detach().numpy()
Y_Hat_F95 = mp.rescale_f_zo(y_pred_up, sy_zo).detach().numpy()
Gp_Vae[Gp_Vae<0] = 0
Y_Hat_F05[Y_Hat_F05<0] = 0
Y_Hat_F95[Y_Hat_F95<0] = 0
cmap = plt.get_cmap('Oranges')
colors = [cmap(i) for i in np.linspace(0.3, 0.8, 1500)]
fig, ax = plt.subplots(1,1, figsize=(15.36,7.65))
for _ in np.arange(500):
    z = dist.Normal(observed_pred.mean[0], observed_pred.variance[0]).sample()
    z_r = mp.rescale_f(z, sf)
    loc, scale = vae.decoder(z_r)
    for k in np.arange(3):
        y_samp = dist.Normal(loc, scale).sample()
        ys = mp.rescale_f_zo(y_samp, sy_zo).detach().numpy()
        ax.plot(chain, ys.T, color=colors[_*k], alpha=1)
loc, scale = vae.decoder(h_pred_up[0])
y1 = mp.rescale_f_zo(loc+2*scale, sy_zo).detach().numpy()
y2 = mp.rescale_f_zo(loc-2*scale, sy_zo).detach().numpy()
loc, scale = vae.decoder(h_pred_lo[0])
y3 = mp.rescale_f_zo(loc+2*scale, sy_zo).detach().numpy()
y4 = mp.rescale_f_zo(loc-2*scale, sy_zo).detach().numpy()
yb = np.vstack([y1,y2,y3,y4])
Y_upper = np.min(yb,axis=0)
Y_lower = np.max(yb,axis=0)
ax.plot(chain, Gp_Vae[0].T, color=(1,0,0), label='mean value')
ax.plot(chain, ValY.numpy()[0].T, color=(0,1,0), label='true ob')
ax.plot(chain, Y_lower.T, color=(0,0,0), ls='--', label='5%')
ax.plot(chain, Y_upper.T, color=(0,0,0), ls='--', label='95%')
ax.set(xlabel='x', ylabel='y', title='VAE-DGP')
ax.legend()
dir_c = apc+'\\pp.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 

vom.plot_mwd_animation(np.squeeze(chain), Gp_Vae, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, '')

sse = np.linalg.norm(Gp_Vae-ValY.numpy(), axis=1)
fig, ax = plt.subplots(1,1, figsize=(9.02,4.8))
ax.scatter(np.arange(sse.shape[0]), sse)
# ax.scatter(max_ind, sse[max_ind], c='r')
ax.set(xlabel='obs', ylabel='sse', title='Distribution of SSE')
dir_c = apc+'\\var_result.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 


endtime = datetime.datetime.now()
print (f"Total training time: {(endtime - starttime).seconds:d}")

plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')