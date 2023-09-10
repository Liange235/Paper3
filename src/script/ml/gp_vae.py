from cProfile import label
from pickle import FALSE
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

path = os.getcwd()
plt.ion()
plt.show()
str_n = 'MWD'
dir = os.path.join(path,'Data','result_set','vae_'+str_n+'.pth')
load_model = 0
str_n = "MWD"
data_dict = mdp.generate_scenario1(str_n)
X_Sim = data_dict['x_sim']
Y_Sim = data_dict['y_sim']
chain = data_dict['chain']
# Md = Y_Sim.shape[1]
Md = Y_Sim.shape[1]
# h = X_Sim.shape[1]
h = 6
sy = mpn.scale_f(Y_Sim)
load_model = 1
Y_Sim_std = sy['val']
rpca = ra.R_pca(Y_Sim)
LowY, SparseY = rpca.fit(0.005, max_iter=10000, iter_print=100)
abn_p = np.count_nonzero(SparseY, axis=1)
lower_q = np.quantile(np.sort(abn_p), 0.25, method='lower')
higher_q = np.quantile(np.sort(abn_p), 0.75, method='higher')
iqr = higher_q-lower_q
thre = higher_q+1.5*iqr
X_Sim_Valid = torch.Tensor(X_Sim[abn_p<=thre, :])
Y_Sim_Valid = torch.Tensor(Y_Sim[abn_p<=thre, :])

np.random.seed(0)
tf = round(0.8*X_Sim_Valid.shape[0])
ts = np.random.randint(1,X_Sim_Valid.shape[0], size=X_Sim_Valid.shape[0])
TraX = X_Sim_Valid[ts[0:tf],:]
TraY = Y_Sim_Valid[ts[0:tf],:]
sy = mp.scale_f_zo(TraY)
TraY_Zo = sy['val']
ValX = X_Sim_Valid[ts[tf:],:]
ValY = Y_Sim_Valid[ts[tf:],:]
ValY_Zo = mp.scale_f_zo(ValY, sy)['val']
vae = vae_model.VAE(z_dim=h, Md=Md, hidden_dim=60, hidden_dim2=32)
vae = vae_model.train(vae, TraY_Zo, ValY_Zo, dir=dir, load_model=load_model)
reco_tra = vae.reconstruct_y(TraY_Zo)
reco_tes = vae.reconstruct_y(ValY_Zo)
reco_tra1 = mpn.rescale_f_zo(reco_tra, sy).detach()
reco_tra1[reco_tra1<0] = 0
reco_tes1 = mpn.rescale_f_zo(reco_tes, sy).detach()
reco_tes1[reco_tes1<0] = 0
err = []
for _, value in enumerate(reco_tes1):
    m = sum((ValY[_]-value)**2).item()
    err.append(m)
err = np.asarray(err)
max_ind = np.argmax(err)
# max_ind = 2
se = torch.norm(TraY-reco_tra1)**2
print(
    "average training loss for vae: %.4f"
    % (se)
)
se = torch.norm(reco_tes1-ValY)**2
print(
    "average testing loss for vae: %.4f"
    % (se)
)
# fig, ax = plt.subplots(1,1)
# ax.plot(err)
# fig, ax = plt.subplots(1,1)
# ax.plot(chain, ValY[max_ind], label='orig')
# ax.plot(chain, reco_tes1[max_ind], label='rec')
# ax.legend()
# fig, ax = plt.subplots(1,2)
# ax[0].plot(chain, ValY.T)
# ax[1].plot(chain, reco_tes1.T)
###################################generative model for generating data#######################
# N = 50
# gen_data = torch.zeros([N,132])
# c = torch.linspace(-0.1,0.1,N)
# for _ in np.arange(N):
#     z_plus = torch.ones([1, h])*c[_]
#     gen_data[_] = vae.reconstruct_img(ValY_Zo[max_ind], z_plus)
# gen_data = mp.rescale_f_zo(gen_data, sy).detach()
# fig, ax = plt.subplots(1,1)
# ax.plot(chain, ValY[max_ind], 'k')
# ax.plot(chain, gen_data.T)
#######################using pca for reconstruction##########################################
sy2 = mpn.scale_f(TraY.numpy())
TraY_Std = sy2['val']
ValY_Std = mpn.scale_f(ValY.numpy(), sy2)['val']
U_Y, S_Y, V_Y = np.linalg.svd(TraY_Std , full_matrices=False)
pcom = 6
pro = V_Y[0:pcom, :].T
TraY_p = np.dot(TraY_Std, pro)
ValY_p = np.dot(ValY_Std, pro)
p_inv = np.linalg.lstsq(pro, np.eye(Md), rcond=None)[0]
Tra_tilde = torch.Tensor(mpn.rescale_f(np.dot(TraY_p, p_inv), sy2))
Val_tilde = torch.Tensor(mpn.rescale_f(np.dot(ValY_p, p_inv), sy2))
se = torch.norm(TraY-Tra_tilde)**2
print(
    "average training loss for pca: %.4f"
    % (se)
)
se = torch.norm(ValY-Val_tilde)**2
print(
    "average testing loss for pca: %.4f"
    % (se)
)
###########################################GP_VAE##########################################
Sx = mp.scale_f_zo(TraX)
h_feature = vae.encoder(TraY_Zo)[0].detach()
h_feature_te = vae.encoder(ValY_Zo)[0].detach()
th = np.arange(h_feature.shape[1])
fig, ax = plt.subplots(1,2)
ax[0].plot(th, h_feature.T)
ax[1].plot(th, h_feature_te.T)
train_x = Sx['val']
test_x = mp.scale_f_zo(ValX, Sx)['val']
sf = mp.scale_f(h_feature)
train_y = sf['val']
k_id = 0   ##kernel label
feature_expand = gp_vae.Feature_Augument(data_dim=TraX.shape[1], aug_feature1=h)
likelihood_vae = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=h, rank=1)
gprvae = gp_vae.MultitaskGPModel(train_x, train_y, likelihood_vae, h, feature_expand)
gprvae.train(), likelihood_vae.train()
loss = gp_vae.train(gprvae, likelihood_vae, train_x, train_y)
gprvae.eval(), likelihood_vae.eval()
Gp_vae = mp.rescale_f(gprvae(train_x).mean, sf)
Gp_vae_r = vae.decoder(Gp_vae)[0]
Gp_vae_r1 = mp.rescale_f_zo(Gp_vae_r, sy)
se_tr = torch.norm(TraY-Gp_vae_r1)**2
Gp_vae = mp.rescale_f(gprvae(test_x).mean, sf)
Gp_vae_r = vae.decoder(Gp_vae)[0]
Gp_vae_r2 = mp.rescale_f_zo(Gp_vae_r, sy)
se_te = torch.norm(ValY-Gp_vae_r2)**2
print("Training loss for gp_vae : %.2f, Testing loss for gp_vae : %.2f" % (se_tr, se_te))
fig, ax = plt.subplots(1,2)
ax[0].plot(chain, ValY.detach().T)
ax[1].plot(chain, Gp_vae_r2.detach().T)
###########################################GP##########################################
train_y = torch.Tensor(TraY_p)
k_id = 0
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=pcom, rank=1)
gpr = gp_struct.MultitaskGPModel(train_x, train_y, likelihood, k_id, pcom)
gpr.train(), likelihood.train()
loss = gp_struct.train(gpr, likelihood, train_x, train_y)
gpr.eval(), likelihood.eval()
Gp_r1 = torch.Tensor(mpn.rescale_f(np.dot(gpr(train_x).mean.detach().numpy(), p_inv), sy2))
Gp_r2 = torch.Tensor(mpn.rescale_f(np.dot(gpr(test_x).mean.detach().numpy(), p_inv), sy2))
se_tr = torch.norm(TraY-Gp_r1)**2
se_te = torch.norm(ValY-Gp_r2)**2
print("Training loss for gp_pca : %.2f, Testing loss for gp_pca : %.2f" % (se_tr, se_te))

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood_vae(gprvae(test_x))
lower, upper = observed_pred.confidence_region()
hh = mp.rescale_f(lower, sf)
lower = vae.decoder(hh)[0]
hh = mp.rescale_f(upper, sf)
upper = vae.decoder(hh)[0]
quan05_value = mp.rescale_f_zo(lower.detach(), sy).numpy()
quan95_value = mp.rescale_f_zo(upper.detach(), sy).numpy()

Y_Hat_F = Gp_vae_r2.detach().numpy()
Y_Hat_F05 = quan05_value
Y_Hat_F95 = quan95_value
vom.plot_mwd_animation(np.squeeze(chain), Y_Hat_F, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gp_vae')

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gpr(test_x))
lower, upper = observed_pred.confidence_region()
quan05_value = lower.detach().numpy()
quan95_value = upper.detach().numpy()

Y_Hat_F = Gp_r2.detach().numpy()
Y_Hat_F05 = mpn.rescale_f(np.dot(quan05_value, p_inv), sy2)
Y_Hat_F95 = mpn.rescale_f(np.dot(quan95_value, p_inv), sy2)
vom.plot_mwd_animation(np.squeeze(chain), Y_Hat_F, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gp_pca')

plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')
     
        