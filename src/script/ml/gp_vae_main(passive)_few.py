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
import gpvae_model_multi_few as gp_vae
import gpy_model_multi as gp_struct
import vae_model
# import vae_model_csd
import visualization_of_mwd as vom
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import datetime

path = os.getcwd()
plt.ion()
plt.show()
starttime = datetime.datetime.now()

def del_tensor_ele_n(arr, index, n):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始, 需要删除的行数
    """
    arr1 = arr[0:index]
    arr2 = arr[index+n:]
    return torch.cat((arr1,arr2),dim=0)
str_n = 'MWD'
# str_n = 'CSD'
ind = 'passive'
path_c = r"C:\Users\t1714\Desktop\Academic\Coding_Files\GP building polymers\Data\result_set"
apc = os.path.join(path_c, str_n)
load_model = 1
plot_fig = 0
data_dict = mdp.generate_scenario1(str_n)
def case1():
    X_Sim = data_dict['x_sim']
    Y_Sim = data_dict['y_sim']
    chain = data_dict['chain']
    hidden = 6
    pcom = 7
    h1 = 60
    h2 = 35
    str_x = 'Chain Length'
    str_y = 'MWD'
    return X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x, str_y
def case2():
    X_Sim = data_dict['x_sim']['input']
    Y_Sim = data_dict['y_sim']['output']
    chain = np.linspace(0, 4, 400)[:,None]
    hidden = 7
    pcom = 4
    h1 = 50
    h2 = 100
    str_x = 'Particle size(mm)'
    str_y = 'Volumn pdf(mm^-1)'
    return X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x, str_y
switch = {'MWD': case1,
          'CSD': case2}
X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x, str_y = switch[str_n]()
Md = Y_Sim.shape[1]
rpca = ra.R_pca(Y_Sim)
LowY, SparseY = rpca.fit(0.005, max_iter=10000, iter_print=100)
abn_p = np.count_nonzero(SparseY, axis=1)
lower_q = np.quantile(np.sort(abn_p), 0.25)
higher_q = np.quantile(np.sort(abn_p), 0.75)
iqr = higher_q-lower_q
thre = higher_q+1.5*iqr
X_Sim_Valid = torch.Tensor(X_Sim[abn_p<=thre, :])[:1500]
Y_Sim_Valid = torch.Tensor(Y_Sim[abn_p<=thre, :])[:1500]
#########################delete abnormal data using rpca############################
np.random.seed(0)
tf = round(0.05*X_Sim_Valid.shape[0])
# tf2 = round(0.02*X_Sim_Valid.shape[0])
ts = np.random.randint(1,X_Sim_Valid.shape[0], size=X_Sim_Valid.shape[0])
TraX = X_Sim_Valid[ts[0:tf],:]
TraY = Y_Sim_Valid[ts[0:tf],:]
sy_zo = mp.scale_f_zo(TraY)
TraY_Zo = sy_zo['val']
# ValX = X_Sim_Valid[ts[tf:tf+tf2],:]
# ValY = Y_Sim_Valid[ts[tf:tf+tf2],:]
ValX = X_Sim_Valid[ts[tf:],:]
ValY = Y_Sim_Valid[ts[tf:],:]
ValY_Zo = mp.scale_f_zo(ValY, sy_zo)['val']
# TeY_Zo = mp.scale_f_zo(TeY, sy_zo)['val']
vae = vae_model.VAE(z_dim=hidden, Md=Md, hidden_dim=h1, hidden_dim2=h2)
# vae = vae_model_csd.VAE(z_dim=hidden, Md=Md, hidden_dim=h1)
dir = apc+'\\vae_5.pth'
vae = vae_model.train(vae, TraY_Zo, dir=dir, load_model=load_model)
# vae = vae_model_csd.train(vae, TraY_Zo, dir=dir, load_model=load_model)
reco_tra = vae.reconstruct_y(TraY_Zo)
reco_tra_vae = mpn.rescale_f_zo(reco_tra, sy_zo).detach()
reco_tra_vae[reco_tra_vae<0] = 0

reco_tva = vae.reconstruct_y(ValY_Zo)
reco_tva_vae = mpn.rescale_f_zo(reco_tva, sy_zo).detach()
reco_tva_vae[reco_tva_vae<0] = 0
err = []
for _, value in enumerate(reco_tva_vae):
    m = torch.norm(ValY[_]-value).item()
    err.append(m)
err = np.asarray(err)
rmse_vae = np.sqrt(np.mean(err))
print(f"rmse validation vae is:{rmse_vae:.4f}")
#########################find and plot the biggest reconstruction error wmd using vae############################
err1 = []
for _, value in enumerate(reco_tra_vae):
    m = torch.norm(TraY[_]-value).item()
    err1.append(m)
err1 = np.asarray(err1)
rmse_vae = np.sqrt(np.mean(err1))
seq = np.argsort(err1)
ind = np.floor(np.linspace(0,tf,4)).astype(int)
ind[-1] = ind[-1]-1
fig = plt.figure(figsize=(6,3.45))
plt.rcParams['font.sans-serif']=['Times New Roman']
ax = fig.add_subplot(131)
ax.scatter(np.arange(len(err1)), err1)
ax.scatter(seq[ind], err1[seq[ind]], c='r')
ax.set(xlabel='observation', ylabel='value', title='RMSE for vae: %.4f (Ytr1)' % (rmse_vae))
ax.text(seq[ind[0]], err1[seq[ind[0]]], r'1')
ax.text(seq[ind[1]], err1[seq[ind[1]]], r'2')
ax.text(seq[ind[2]], err1[seq[ind[2]]], r'3')
ax.text(seq[ind[3]], err1[seq[ind[3]]], r'4')
ax = fig.add_subplot(232)
ax.plot(chain, TraY[seq[ind[0]]], label='y')
ax.plot(chain, reco_tra_vae[seq[ind[0]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob1')
ax.legend()
ax = fig.add_subplot(233)
ax.plot(chain, TraY[seq[ind[1]]], label='y')
ax.plot(chain, reco_tra_vae[seq[ind[1]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob2')
ax.legend()
ax = fig.add_subplot(235)
ax.plot(chain, TraY[seq[ind[2]]], label='y')
ax.plot(chain, reco_tra_vae[seq[ind[2]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob3')
ax.legend()
ax = fig.add_subplot(236)
ax.plot(chain, TraY[seq[ind[3]]], label='y')
ax.plot(chain, reco_tra_vae[seq[ind[3]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob4')
ax.legend()
dir_c = apc+'\\f_vae.svg'
plt.tight_layout()
plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 
# ax.plot(chain, TraY[1], label='y2', c='g')
# ax[1].plot(chain, reco_tra_vae[1], label='y_hat2', marker='*', c='g', linestyle=':')
# ax[1].set(xlabel=str_x, ylabel=str_n, title='2 reconstruction case')
# ax[1].legend()

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
err2 = []
for _, value in enumerate(reco_tra_pca):
    m = torch.norm(TraY[_]-value).item()
    err2.append(m)
err2 = np.asarray(err2)
rmse_pca = np.sqrt(np.mean(err2))
seq = np.argsort(err2)
ind = np.floor(np.linspace(0,tf,4)).astype(int)
ind[-1] = ind[-1]-1
fig = plt.figure(figsize=(6,3.45))
plt.rcParams['font.sans-serif']=['Times New Roman']
ax = fig.add_subplot(131)
ax.scatter(np.arange(len(err2)), err2)
ax.scatter(seq[ind], err2[seq[ind]], c='r')
ax.set(xlabel='observation', ylabel='value', title='RMSE for pca: %.4f (Ytr1)' % (rmse_pca))
ax.text(seq[ind[0]], err2[seq[ind[0]]], r'1')
ax.text(seq[ind[1]], err2[seq[ind[1]]], r'2')
ax.text(seq[ind[2]], err2[seq[ind[2]]], r'3')
ax.text(seq[ind[3]], err2[seq[ind[3]]], r'4')
ax = fig.add_subplot(232)
ax.plot(chain, TraY[seq[ind[0]]], label='y')
ax.plot(chain, reco_tra_pca[seq[ind[0]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob1')
ax.legend()
ax = fig.add_subplot(233)
ax.plot(chain, TraY[seq[ind[1]]], label='y')
ax.plot(chain, reco_tra_pca[seq[ind[1]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob2')
ax.legend()
ax = fig.add_subplot(235)
ax.plot(chain, TraY[seq[ind[2]]], label='y')
ax.plot(chain, reco_tra_pca[seq[ind[2]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob3')
ax.legend()
ax = fig.add_subplot(236)
ax.plot(chain, TraY[seq[ind[3]]], label='y')
ax.plot(chain, reco_tra_pca[seq[ind[3]]], label=r'$\^y$')
ax.set(xlabel=str_x, ylabel=str_y, title='ob4')
ax.legend()
dir_c = apc+'\\f_pca.svg'
plt.tight_layout()
plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 

fig, ax = plt.subplots(1,1, figsize=(6,3.45))
ax.scatter(np.arange(len(err2)), err2, label='PCA: %.4f' % rmse_pca, c='none',marker='o',edgecolors='#1f77b4')
ax.scatter(np.arange(len(err1)), err1, label='VAE: %.4f' % rmse_vae, c='#ff7f0e')
ax.legend(fontsize=12)
ax.set_xlabel('Observation', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('RMSE of Ytr1', fontsize=12)
dir_c = apc+'\\f_pca_vae_cp.svg'
plt.tight_layout()
plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 

reco_tva_pca = torch.Tensor(mpn.rescale_f(np.dot(ValY_p, p_inv), sy_std))
reco_tva_pca[reco_tva_pca<0] = 0
err = []
for _, value in enumerate(reco_tva_pca):
    m = torch.norm(ValY[_]-value).item()
    err.append(m)
err = np.asarray(err)
rmse_pca = np.sqrt(np.mean(err))
print(f"rmse validation pca is:{rmse_pca:.4f}")

# err = []
# for _, value in enumerate(reco_tes_pca):
#     m = sum((ValY[_]-value)**2).item()
#     err.append(m)
# err = np.asarray(err)
# sse_pca = sum(err)
# max_ind = np.argmax(err)
# fig, ax = plt.subplots(1,2, figsize=(9.02,4.8))
# ax[0].scatter(np.arange(len(err)), err)
# ax[0].set(xlabel='observation', ylabel='error', title='sum square error for pca: %.4f (validate)' % (sse_pca))
# ax[1].plot(chain, ValY[max_ind], label='original')
# ax[1].plot(chain, reco_tes_pca[max_ind], label='reconstructed')
# ax[1].set(xlabel=str_x, ylabel=str_n, title='Largest reconstruction error case')
# ax[1].legend()
# dir_c = apc+'\\valid_pca_case2.png'
# plt.tight_layout()
# plt.savefig(dir_c, dpi=500) 
#########################pca_mgp modeling############################
ValY_p1 = np.delete(ValY_p, 5, axis=0)
ValY_p1 = np.delete(ValY_p1, 28, axis=0)
th = np.arange(TraY_p.shape[1]).astype(dtype=np.str)
fig, ax = plt.subplots(1,3, figsize=(7.2, 4.14))
ax[0].plot(th, TraY_p.T, linestyle=':', marker='*')
ax[1].plot(th, ValY_p.T, linestyle=':', marker='*')
ax[2].plot(th, ValY_p1.T, linestyle=':', marker='*')
ax[0].set_xlabel('Number of PCs\n(a) Dataset Ytr1', fontsize=12)
ax[0].set_ylabel('Feature value', fontsize=12)
ax[1].set_xlabel('Number of PCs\n(b) Dataset Ytr2', fontsize=12)
ax[1].set_ylabel('Feature value', fontsize=12)
ax[2].set_xlabel('Number of PCs\n(c) Dataset Ytr2 without outliers', fontsize=12)
ax[2].set_ylabel('Feature value', fontsize=12)
dir_c = apc+'\\f_all_pca.svg'
plt.tight_layout()
# plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 

Sx = mp.scale_f_zo(TraX)
train_x = Sx['val']
train_y = torch.Tensor(TraY_p)
test_x = mp.scale_f_zo(ValX, Sx)['val']
k_id = 0
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=pcom, rank=1)
gpr = gp_struct.MultitaskGPModel(train_x, train_y, likelihood, k_id, pcom)
dir = apc+'\\pca_mgp5.pth'
gpr.train(), likelihood.train()
gpr, likelihood = gp_struct.train(gpr, likelihood, train_x, train_y, dir=dir, load_model=1)
gpr.eval(), likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = gpr(train_x)
    lower, upper = observed_pred.confidence_region()
# with torch.no_grad():
#     observed_pred = gpr(train_x)
#     sd = observed_pred.variance.sqrt()
#     lower = observed_pred.mean-2*sd
#     upper = observed_pred.mean+2*sd
pca_mgp = mpn.rescale_f(np.dot(observed_pred.mean.numpy(), p_inv), sy_std)
quan05_value = np.dot(lower.numpy(), p_inv)
quan95_value = np.dot(upper.numpy(), p_inv)
Y_Hat_F05 = mpn.rescale_f(quan05_value, sy_std)
Y_Hat_F95 = mpn.rescale_f(quan95_value, sy_std)
# vom.plot_mwd_animation(np.squeeze(chain), Gp_Pca, TraY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gp_pca_tr')

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = (gpr(test_x))
    lower, upper = observed_pred.confidence_region()
# with torch.no_grad():
#     observed_pred = gpr(test_x)
#     sd = observed_pred.variance.sqrt()
#     lower = observed_pred.mean-2*sd
#     upper = observed_pred.mean+2*sd
pca_mgp = mpn.rescale_f(np.dot(observed_pred.mean.numpy(), p_inv), sy_std)
quan05_value = np.dot(lower.numpy(), p_inv)
quan95_value = np.dot(upper.numpy(), p_inv)
Y_Hat_F05 = mpn.rescale_f(quan05_value, sy_std)
Y_Hat_F95 = mpn.rescale_f(quan95_value, sy_std)
rmse_pcamgp = np.sqrt(np.mean(np.linalg.norm(pca_mgp-ValY.numpy(), axis=1)))
mae_pcamgp = np.mean(np.linalg.norm(pca_mgp-ValY.numpy(), axis=1)/
                    np.linalg.norm(pca_mgp+ValY.numpy(), axis=1))
mre_pcamgp = np.mean(np.linalg.norm(pca_mgp-ValY.numpy(), axis=1)/
                    np.linalg.norm(ValY.numpy(), axis=1))
if plot_fig:
    vom.plot_mwd_animation(np.squeeze(chain), pca_mgp, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, 'PCA-MGP')
print(f"PCA-MGP rmse is: {rmse_pcamgp:.4f}")
print(f"PCA-MGP mae is: {mae_pcamgp:.4f}")
print(f"PCA-MGP mre is: {mre_pcamgp:.4f}")

#########################vae-mgp modeling############################
# h_feature = vae.encoder(TraY_Zo)[0].detach()
# h_feature_te = vae.encoder(ValY_Zo)[0].detach()
# sf = mp.scale_f(h_feature)
# train_y = sf['val']
# likelihood1 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=hidden, rank=3)
# gpr1 = gp_struct.MultitaskGPModel(train_x, train_y, likelihood1, k_id, hidden)
# dir = apc+'\\vae_mgp5.pth'
# gpr1.train(), likelihood1.train()
# gpr1, likelihood1 = gp_struct.train(gpr1, likelihood1, train_x, train_y, load_model=1, dir=dir)
# gpr1.eval(), likelihood1.eval()
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = (gpr1(test_x))
#     lower, upper = observed_pred.confidence_region()
# h_pred = mp.rescale_f(observed_pred.mean, sf)
# y_pred = vae.decoder(h_pred)[0]
# h_pred_lo = mp.rescale_f(lower, sf)
# y_pred_lo = vae.decoder(h_pred_lo)[0]
# h_pred_up = mp.rescale_f(upper, sf)
# y_pred_up = vae.decoder(h_pred_up)[0]
# vae_mgp = mp.rescale_f_zo(y_pred, sy_zo).detach().numpy()
# Y_Hat_F05 = mp.rescale_f_zo(y_pred_lo, sy_zo).detach().numpy()
# Y_Hat_F95 = mp.rescale_f_zo(y_pred_up, sy_zo).detach().numpy()
# vae_mgp[vae_mgp<0] = 0
# Y_Hat_F05[Y_Hat_F05<0] = 0
# Y_Hat_F95[Y_Hat_F95<0] = 0
# rmse_vaemgp = np.sqrt(np.mean(np.linalg.norm(vae_mgp-ValY.numpy(), axis=1)))
# mae_vaemgp = np.mean(np.linalg.norm(vae_mgp-ValY.numpy(), axis=1)/
#                     np.linalg.norm(vae_mgp+ValY.numpy(), axis=1))
# mre_vaemgp = np.mean(np.linalg.norm(vae_mgp-ValY.numpy(), axis=1)/
#                     np.linalg.norm(ValY.numpy(), axis=1))
# if plot_fig:
#     vom.plot_mwd_animation(np.squeeze(chain), vae_mgp, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, 'VAE-MGP')
# print(f"VAE-MGP rmse is: {rmse_vaemgp:.4f}")
# print(f"VAE-MGP mae is: {mae_vaemgp:.4f}")
# print(f"VAE-MGP mre is: {mre_vaemgp:.4f}")

#########################vae-dcgp modeling############################
h_feature = vae.encoder(TraY_Zo)[0].detach()
h_feature_te = vae.encoder(ValY_Zo)[0].detach()
# rr = vae.decoder(h_feature_te)[0]
# err = []
# for _, value in enumerate(ValY_Zo):
#     m = torch.norm(rr[_]-value).item()
    # err.append(m)
h_feature_te1 = del_tensor_ele_n(h_feature_te, 5, 1)
h_feature_te1 = del_tensor_ele_n(h_feature_te1, 28, 1)
th = np.arange(h_feature.shape[1])
fig, ax = plt.subplots(1,3, figsize=(7.2, 4.14))
ax[0].plot(th, h_feature.T, linestyle=':', marker='*')
ax[1].plot(th, h_feature_te.T, linestyle=':', marker='*')
ax[2].plot(th, h_feature_te1.T, linestyle=':', marker='*')
ax[0].set_xlabel('Number of features\n(a) Dataset Ytr1', fontsize=12)
ax[0].set_ylabel('Feature value', fontsize=12)
ax[1].set_xlabel('Number of features\n(b) Dataset Ytr2', fontsize=12)
ax[1].set_ylabel('Feature value', fontsize=12)
ax[2].set_xlabel('Number of features\n(c) Dataset Ytr2 without outliers', fontsize=12)
ax[2].set_ylabel('Feature value', fontsize=12)
dir_c = apc+'\\f_all_vae.svg'
plt.tight_layout()
# plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 

sf = mp.scale_f(h_feature)
train_y = sf['val']
feature_expand = gp_vae.Feature_Augument(data_dim=TraX.shape[1], aug_feature1=15)##15
likelihood_vae = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=hidden, rank=3)##3
gprvae = gp_vae.MultitaskGPModel(train_x, train_y, likelihood_vae, hidden, feature_expand)
dir = apc+'\\vaedcgp5.pth'
gprvae.train(), likelihood_vae.train()
gprvae, likelihood_vae = gp_vae.train(gprvae, likelihood_vae, train_x, train_y, load_model=load_model, dir=dir, its=2000)
gprvae.eval(), likelihood_vae.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # observed_pred = gprvae.likelihood(gprvae(train_x))
    observed_pred = gprvae(train_x)
    lower, upper = observed_pred.confidence_region()
h_pred = mp.rescale_f(observed_pred.mean, sf)
y_pred = vae.decoder(h_pred)[0]
h_pred_lo = mp.rescale_f(lower, sf)
y_pred_lo = vae.decoder(h_pred_lo)[0]
h_pred_up = mp.rescale_f(upper, sf)
y_pred_up = vae.decoder(h_pred_up)[0]
vae_dcgp = mp.rescale_f_zo(y_pred, sy_zo).detach().numpy()
Y_Hat_F05 = mp.rescale_f_zo(y_pred_lo, sy_zo).detach().numpy()
Y_Hat_F95 = mp.rescale_f_zo(y_pred_up, sy_zo).detach().numpy()
vae_dcgp[vae_dcgp<0] = 0
Y_Hat_F05[Y_Hat_F05<0] = 0
Y_Hat_F95[Y_Hat_F95<0] = 0

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # observed_pred = gprvae.likelihood(gprvae(test_x))
    observed_pred = gprvae(test_x)
    lower, upper = observed_pred.confidence_region()
h_pred = mp.rescale_f(observed_pred.mean, sf)
y_pred = vae.decoder(h_pred)[0]
h_pred_lo = mp.rescale_f(lower, sf)
y_pred_lo = vae.decoder(h_pred_lo)[0]
h_pred_up = mp.rescale_f(upper, sf)
y_pred_up = vae.decoder(h_pred_up)[0]
vae_dcgp = mp.rescale_f_zo(y_pred, sy_zo).detach().numpy()
Y_Hat_F05 = mp.rescale_f_zo(y_pred_lo, sy_zo).detach().numpy()
Y_Hat_F95 = mp.rescale_f_zo(y_pred_up, sy_zo).detach().numpy()
vae_dcgp[vae_dcgp<0] = 0
Y_Hat_F05[Y_Hat_F05<0] = 0
Y_Hat_F95[Y_Hat_F95<0] = 0
rmse_vaedcgp = np.sqrt(np.mean(np.linalg.norm(vae_dcgp-ValY.numpy(), axis=1)))
mae_vaedcgp = np.mean(np.linalg.norm(vae_dcgp-ValY.numpy(), axis=1)/
                    np.linalg.norm(vae_dcgp+ValY.numpy(), axis=1))
mre_vaedcgp = np.mean(np.linalg.norm(vae_dcgp-ValY.numpy(), axis=1)/
                    np.linalg.norm(ValY.numpy(), axis=1))
if plot_fig:
    vom.plot_mwd_animation(np.squeeze(chain), vae_dcgp, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, 'VAE-DCGP')
sse = sum(np.linalg.norm(vae_dcgp-ValY.numpy(), axis=1)**2)
print(f"sse is: {sse:.4f}")
print(f"VAE-DCGP rmse is: {rmse_vaedcgp:.4f}")
print(f"VAE-DCGP mae is: {mae_vaedcgp:.4f}")
print(f"VAE-DCGP mre is: {mre_vaedcgp:.4f}")


rmse_each = np.linalg.norm(vae_dcgp-ValY.numpy(), axis=1, keepdims=True)
# kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(rmse_each)
params = {"bandwidth": np.logspace(-3, 0.2, 50)}
kde = GridSearchCV(KernelDensity(kernel="gaussian"), params)
kde.fit(rmse_each)
X_plot = np.linspace(0, 1.1*max(rmse_each), 2000)
log_dens = kde.score_samples(X_plot)
fig, ax = plt.subplots(1,2, figsize=(6, 3.45))
ax[0].plot(X_plot, np.exp(log_dens))
ax[0].set_xlabel('RMSE\n(a)', fontsize=12)
ax[0].set_ylabel('Distribution', fontsize=12)
ax[0].set_title('RMSE distribution', fontsize=12)
qq = [0.25, 0.5, 0.75, 0.9]
cc = ['r', 'c', 'm', 'y']
for _ in np.arange(4):
    quant = np.quantile(np.sort(rmse_each), qq[_])
    ind = np.argsort(abs(rmse_each-quant), axis=0)[0]
    y_ind = kde.score_samples(rmse_each[ind[0]][:,None])
    ax[0].scatter(rmse_each[ind[0]], np.exp(y_ind), c='r')
    ax[0].text(rmse_each[ind[0]]+0.01, np.exp(y_ind), f"{_+1:1d}")
    ax[1].plot(chain, vae_dcgp[ind[0]], label=f"m{_+1:1d}", color=cc[_], linestyle='--')
    ax[1].plot(chain, ValY[ind[0]], label=f"ob{_+1:1d}", color=cc[_], )
ax[1].set_xlabel(str_x+"\n(b)", fontsize=12)
ax[1].set_ylabel(str_y, fontsize=12)
ax[1].set_title('4 different results', fontsize=12)
ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
dir_c = apc+'\\error_pdf.svg'
plt.tight_layout()
plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 

# max_ind = np.argmax(rmse_each.squeeze())
max_ind = 221  ## csd 45 mwd 175
loc, scale = vae.decoder(h_pred_lo[max_ind])
y1_05 = mp.rescale_f_zo(loc+2*scale, sy_zo).detach().numpy()
y2_05 = mp.rescale_f_zo(loc-2*scale, sy_zo).detach().numpy()
loc, scale = vae.decoder(h_pred_up[max_ind])
y1_95 = mp.rescale_f_zo(loc+2*scale, sy_zo).detach().numpy()
y2_95 = mp.rescale_f_zo(loc-2*scale, sy_zo).detach().numpy()
yb = np.vstack([y1_05,y2_05,y1_95,y2_95])
Y_upper = np.min(yb,axis=0)
Y_upper[Y_upper<0] = 0
Y_lower = np.max(yb,axis=0)
Y_lower[Y_lower<0] = 0
fig, ax = plt.subplots(1,1, figsize=(6,3.45))
ax.plot(chain, vae_dcgp[max_ind], label='model_mean')
# ax.plot(chain, Y_Hat_F05[max_ind], label=r'5% confidence', color='black', linestyle=':')
ax.plot(chain, Y_lower, label=r'5% confidence', color='black', linestyle=':')
# ax.plot(chain, Y_Hat_F95[max_ind], label=r'95% confidence', color='black', linestyle=':')
ax.plot(chain, Y_upper, label=r'95% confidence', color='black', linestyle=':')
ax.plot(chain, ValY[max_ind], label='observation')
# ax.fill_between(chain[:,0], Y_Hat_F05[max_ind], Y_Hat_F95[max_ind], alpha=.25)
ax.fill_between(chain[:,0], Y_lower, Y_upper, alpha=.25)
# ax.set(xlabel='Particle size(mm)', ylabel='Volumn pdf(mm^(-1))', title='Crystal Size Distribution')
# ax.set(xlabel='Chain Length', ylabel='MWD', title=f'Worst case No.{max_ind:d} test result')
ax.set_xlabel(str_x, fontsize=12)
ax.set_ylabel(str_y, fontsize=12)
ax.set_title(f'The {max_ind:d}th validation result', fontsize=12)
ax.legend(fontsize=12)
dir_c = apc+'\\result of worse case.svg'
plt.tight_layout()
plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 

# fig, ax = plt.subplots(1,1, figsize=(9.02,4.8))
# ax.scatter(np.arange(sse.shape[0]), sse)
# # ax.scatter(max_ind, sse[max_ind], c='r')
# ax.set(xlabel='obs', ylabel='sse', title='Distribution of SSE')
# dir_c = apc+'\\var_result.png'
# plt.tight_layout()
# plt.savefig(dir_c, dpi=500) 

endtime = datetime.datetime.now()
print (f"Total training time: {(endtime - starttime).seconds:d}")

plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')