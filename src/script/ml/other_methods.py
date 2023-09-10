from distutils.command.sdist import sdist
import numpy as np
import torch
from joblib import dump, load
import matplotlib.pyplot as plt
import os
import r_pca as ra
import Model_preparation as mp
import Model_preparation_np as mpn
import mwd_data_preparation as mdp
import nn_model as Neural
import gpytorch
import MSVR 
import visualization_of_mwd as vom
from smt.surrogate_models import KPLS
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from gpytorch.priors.torch_priors import GammaPrior
from sklearn.multioutput import MultiOutputRegressor
import pybobyqa

import datetime

path = os.getcwd()
plt.ion()
plt.show()
starttime = datetime.datetime.now()
plt.rcParams['font.sans-serif']=['Times New Roman']
str_n = 'MWD'
# str_n = 'CSD'
ind = 'passive'
path_c = r"C:\Users\t1714\Desktop\Academic\Coding_Files\GP building polymers\Data\result_set"
apc = os.path.join(path_c, str_n)
load_model = 1
plot_fig = 0
mod = 'Deep-NN'
data_dict = mdp.generate_scenario1(str_n)
def case1():
    X_Sim = data_dict['x_sim']
    Y_Sim = data_dict['y_sim']
    chain = data_dict['chain']
    pcom = 7
    p1 = [60, 32, 6]   ### h1, h2, z
    p2 = [40, 90, 110]
    str_x = 'Chain Length'
    str_y = 'MWD'
    return X_Sim, Y_Sim, chain, pcom, str_x, p1, p2, str_y
def case2():
    X_Sim = data_dict['x_sim']['input']
    Y_Sim = data_dict['y_sim']['output']
    chain = np.linspace(0, 4, 400)[:,None]
    pcom = 4
    p1 = [100, 50, 10]   ### h1, h2, z
    p2 = [100, 200, 150]
    p3 = [60, 100, 8]  ###VAE h1, h2, z
    str_x = 'Particle size(mm)'
    str_y = 'Volumn pdf(mm^-1)'
    return X_Sim, Y_Sim, chain, pcom, str_x, p1, p2, str_y
switch = {'MWD': case1,
          'CSD': case2}
X_Sim, Y_Sim, chain, pcom, str_x, p1, p2, str_y = switch[str_n]()
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
#########################delete abnormal data using rpca############################
np.random.seed(0)
tf = round(0.05*X_Sim_Valid.shape[0])
ts = np.random.randint(1,X_Sim_Valid.shape[0], size=X_Sim_Valid.shape[0])
TraX = X_Sim_Valid[ts[0:tf],:]
TraY = Y_Sim_Valid[ts[0:tf],:]
ValX = X_Sim_Valid[ts[tf:],:]
ValY = Y_Sim_Valid[ts[tf:],:]
sx = mpn.scale_f(TraX.numpy())
sy = mpn.scale_f(TraY.numpy())
dy_std = sy['val']
U_Y, S_Y, V_Y = np.linalg.svd(dy_std , full_matrices=False)
pro = V_Y[0:pcom, :].T
p_inv = np.linalg.lstsq(pro, np.eye(Md), rcond=None)[0]
TraY_p = np.dot(dy_std, pro)
######################################modelling using different methods########################################
def PCA_KRR():
    train_y = TraY_p
    train_x = sx['val']
    test_x = mpn.scale_f(ValX.numpy(), sx)['val']
    yp_pred = np.zeros([test_x.shape[0], train_y.shape[1]])
    krr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        param_grid={"alpha": [1e0, 1e-1, 1e-2], "gamma": np.logspace(-2, 2, 5)},
    )
    if load_model:
        for _ in np.arange(train_y.shape[1]):
            dir = apc+'\\krr'+str(_+1)+'.joblib'
            krr = load(dir) 
            yp_pred[:,_] = krr.predict(test_x)
            y_krr = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
    else:
        for _ in np.arange(train_y.shape[1]):
            dir = apc+'\\krr'+str(_+1)+'.joblib'
            krr.fit(train_x, train_y[:,_])
            yp_pred[:,_] = krr.predict(test_x)
            y_krr = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
            dump(krr, dir)
    y_krr[y_krr<0] = 0
    return y_krr
def PCA_KPLS():
    train_x = sx['val']
    train_y = TraY_p
    test_x = mpn.scale_f(ValX.numpy(), sx)['val']
    yp_pred = np.zeros([test_x.shape[0], train_y.shape[1]])
    if load_model:
        for _ in np.arange(train_y.shape[1]):
            dir = apc+'\\kpls'+str(_+1)+'.joblib'
            kpls = load(dir) 
            yp_pred[:,_] = kpls.predict_values(test_x).squeeze()
            y_kpls = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
    else:
        kpls = KPLS(eval_n_comp=True, corr='squar_exp')
        for _ in np.arange(train_y.shape[1]):
            dir = apc+'\\kpls'+str(_+1)+'.joblib'
            kpls.set_training_values(train_x, train_y[:,_])
            kpls.train()
            yp_pred[:,_] = kpls.predict_values(test_x).squeeze()
            y_kpls = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
            dump(kpls, dir)
    y_kpls[y_kpls<0] = 0
    return y_kpls
def PCA_SVR():
    train_y = TraY_p
    train_x = sx['val']
    test_x = mpn.scale_f(ValX.numpy(), sx)['val']
    yp_pred = np.zeros([test_x.shape[0], train_y.shape[1]])
    svr = GridSearchCV(
        SVR(kernel="rbf", gamma=0.1),
        param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
    )
    if load_model:
        for _ in np.arange(train_y.shape[1]):
            dir = apc+'\\svr'+str(_+1)+'.joblib'
            svr = load(dir) 
            yp_pred[:,_] = svr.predict(test_x)
            y_svr = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
    else:
        for _ in np.arange(train_y.shape[1]):
            dir = apc+'\\svr'+str(_+1)+'.joblib'
            svr.fit(train_x, train_y[:,_])
            yp_pred[:,_] = svr.predict(test_x)
            y_svr = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
            dump(svr, dir)
    y_svr[y_svr<0] = 0
    return y_svr
def PCA_GP():
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            ts = train_x.shape[1]
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ts, active_dims=tuple(np.arange(ts)),
                                                                                    lengthscale_prior=GammaPrior(3, 8)))
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    train_x = torch.Tensor(sx['val'])
    train_y = torch.Tensor(TraY_p)
    test_x = torch.Tensor(mpn.scale_f(ValX.numpy(), sx)['val'])
    yp_pred = torch.zeros([test_x.shape[0], train_y.shape[1]])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if load_model:
        for _ in np.arange(train_y.shape[1]):
            gpr = ExactGPModel(train_x, train_y[:,_], likelihood)
            dir = apc+'\\pca_gp'+str(_+1)+'.pth'
            # gpr = torch.load(dir)
            state_dict = torch.load(dir)
            gpr.load_state_dict(state_dict)
            gpr.eval(), likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                yp_pred[:,_] = (gpr(test_x)).mean
    else:
        for _ in np.arange(train_y.shape[1]):
            dir = apc+'\\pca_gp'+str(_+1)+'.pth'
            gpr = ExactGPModel(train_x, train_y[:,_], likelihood)
            gpr.train(), likelihood.train()
            optimizer = torch.optim.AdamW(gpr.parameters(), lr = 0.05, betas=(0.9, 0.999), weight_decay=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)
            training_iter = 5000
            for i in range(training_iter):
                optimizer.zero_grad()
                output = gpr(train_x)
                loss = -mll(output, train_y[:,_])
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
                optimizer.step()    
            gpr.eval(), likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                yp_pred[:,_] = (gpr(test_x)).mean
            torch.save(gpr.state_dict(), dir)
    observed_pred = mpn.rescale_f(np.dot(yp_pred.numpy(), p_inv), sy)
    return observed_pred
# def err(c, x, y):
#     model = MSVR.MSVR(gamma=c)
#     model.fit(x, y)
#     yp = model.predict(x)
#     rmse = np.sqrt(np.mean(np.linalg.norm(yp-y, axis=1)))
#     return rmse
# def PCA_MSVR():
#     train_x = sx['val']
#     train_y = TraY_p
#     test_x = mpn.scale_f(ValX.numpy(), sx)['val']
#     # yp_pred = np.zeros([test_x.shape[0], train_y.shape[1]])
#     if load_model:
#         dir = apc+'\\msvr.joblib'
#         msvr = load(dir) 
#         yp_pred = msvr.predict(test_x)
#         y_msvr = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
#     else:
#         # obj = lambda c: err(c, train_x, train_y)
#         # soln = pybobyqa.solve(obj, np.array([0.001]), bounds=(np.array([0.]), np.array([.05])), maxfun=500, scaling_within_bounds=True,
#         #                       seek_global_minimum=True)
#         # print(soln)
#         # msvr = MSVR.MSVR(gamma=soln.x[0])
#         dir = apc+'\\msvr.joblib'
#         # msvr.fit(train_x, train_y)
#         # yp_pred = msvr.predict(test_x)
#         # y_msvr = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
#         svr = GridSearchCV(
#             SVR(kernel="rbf", gamma=0.1),
#             param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
#         )
#         msvr = MultiOutputRegressor(svr)
#         msvr = msvr.fit(train_x, train_y)
#         yp_pred = msvr.predict(test_x)
#         y_msvr = mpn.rescale_f(np.dot(yp_pred, p_inv), sy)
#         dump(msvr, dir)
#     y_msvr[y_msvr<0] = 0
#     return y_msvr
def DNN():
    sx = mp.scale_f_zo(TraX)
    sy = mp.scale_f_zo(TraY)
    train_x = sx['val']
    train_y = sy['val']
    test_x = mp.scale_f_zo(ValX, sx)['val']
    model = Neural.NeuralNetwork(train_x.shape[1], train_y.shape[1], d1=p2[0], d2=p2[1], d3=p2[2])
    dir = apc+'\\dnn.pth'
    model = Neural.train(model, train_x, train_y, load_model=load_model, its=1500, dir=dir)
    nn_pred_te = mp.rescale_f_zo(model(test_x), sy).detach().numpy()
    nn_pred_te[nn_pred_te<0] = 0
    return nn_pred_te
def choose_model(mod):
    switch = {'PCA-KRR': PCA_KRR,
              'Deep-NN': DNN,
              'PCA-SVR': PCA_SVR,
              'PCA-GP': PCA_GP,
              'PCA-KPLS': PCA_KPLS,
            #   'PCA-MSVR': PCA_MSVR,
            }
    method = switch.get(mod, DNN)
    return method
pred_te = choose_model(mod)()
#########################calculate rmse############################
rmse = np.sqrt(np.mean(np.linalg.norm(pred_te-ValY.numpy(), axis=1)))
mae = np.mean(np.linalg.norm(pred_te-ValY.numpy(), axis=1)/
                    np.linalg.norm(pred_te+ValY.numpy(), axis=1))
mre = np.mean(np.linalg.norm(pred_te-ValY.numpy(), axis=1)/
                    np.linalg.norm(ValY.numpy(), axis=1))
# vom.plot_mwd_animation(np.squeeze(chain), nn_pred_tra, TraY.numpy(), str_n = 'nn_tr')
if plot_fig:
    vom.plot_mwd_animation(np.squeeze(chain), pred_te, ValY.numpy(), str_n = mod)
print(f"rmse is: {rmse:.4f}")
print(f"mae is: {mae:.4f}")
print(f"mre is: {mre:.4f}")
rmse_each = np.linalg.norm(pred_te-ValY.numpy(), axis=1)
max_ind = 221#### csd   mwd 163
# max_ind = np.argmax(rmse_each)
fig, ax = plt.subplots(1,1, figsize=(6, 3.45))
ax.plot(chain, pred_te[max_ind], label='model')
ax.plot(chain, ValY.numpy()[max_ind], label='observation')
ax.set_xlabel(str_x, fontsize=12)
ax.set_ylabel(str_y, fontsize=12)
ax.set_title(f'The {max_ind:d}th validation result', fontsize=12)
ax.legend(fontsize=12)
dir_c = apc+'\\result of worse case other methods.svg'
plt.tight_layout()
plt.savefig(dir_c, dpi=600, bbox_inches='tight', format='svg') 
####modelling using nn feature extracted by vae############################
# vae = vae_model.VAE(z_dim=hidden, Md=Md, hidden_dim=h1, hidden_dim2=h2)
# dir = apx+'_vae.pth'
# vae = vae_model.train(vae, TraY_Zo, dir=dir, load_model=1)
# h_feature = vae.encoder(TraY_Zo)[0].detach()
# h_feature_te = vae.encoder(ValY_Zo)[0].detach()
# sf_zo = mp.scale_f_zo(h_feature)
# train_y = sf_zo['val']
# dir = apx+'_nnvae.pth'
# d1=8
# d2=12
# # d1=6
# # d2=15
# model_vae = Neural.NeuralNetwork(TraX.shape[1], train_y.shape[1], d1=d1, d2=d2)
# model_vae = Neural.train(model_vae, train_x, train_y, load_model=load_model, its=500, dir=dir)
# #########################prediction############################
# Nn_f = mp.rescale_f_zo(model_vae(train_x), sf_zo)
# Nnvae_pred_tra = mp.rescale_f_zo(vae.decoder(Nn_f)[0], sy_zo).detach().numpy()
# test_x = mp.scale_f_zo(ValX, Sx)['val']
# Nn_f = mp.rescale_f_zo(model_vae(test_x), sf_zo)
# Nnvae_pred_te = mp.rescale_f_zo(vae.decoder(Nn_f)[0], sy_zo).detach().numpy()
# vom.plot_mwd_animation(np.squeeze(chain), Nnvae_pred_tra, TraY.numpy(), str_n = 'nn_vae_tr')
# vom.plot_mwd_animation(np.squeeze(chain), Nnvae_pred_te, ValY.numpy(), str_n = 'nn_vae_te')

#########################svr kr vae############################
# train_x = train_x.numpy()
# train_y = train_y.numpy()
# test_x = test_x.numpy()
# 
# y_svr = np.zeros([test_x.shape[0], train_y.shape[1]])
# for _ in np.arange(train_y.shape[1]):
#     kr.fit(train_x, train_y[:,_])
#     y_kr[:,_] = kr.predict(test_x)
# for _ in np.arange(train_y.shape[1]):
#     svr.fit(train_x, train_y[:,_])
#     y_svr[:,_] = svr.predict(test_x)
# Nn_f = mp.rescale_f_zo(torch.Tensor(y_kr), sf_zo)
# Krvae_pred_te = mp.rescale_f_zo(vae.decoder(Nn_f)[0], sy_zo).detach().numpy()
# vom.plot_mwd_animation(np.squeeze(chain), Krvae_pred_te, ValY.numpy(), str_n = 'kr_vae_te')
# Nn_f = mp.rescale_f_zo(torch.Tensor(y_svr), sf_zo)
# Svrvae_pred_te = mp.rescale_f_zo(vae.decoder(Nn_f)[0], sy_zo).detach().numpy()
# vom.plot_mwd_animation(np.squeeze(chain), Svrvae_pred_te, ValY.numpy(), str_n = 'svr_vae_te')
#########################svr kr pca############################
# sy_std = mpn.scale_f(TraY.numpy())
# TraY_Std = sy_std['val']
# ValY_Std = mpn.scale_f(ValY.numpy(), sy_std)['val']
# U_Y, S_Y, V_Y = np.linalg.svd(TraY_Std , full_matrices=False)
# pro = V_Y[0:pcom, :].T
# TraY_p = np.dot(TraY_Std, pro)
# ValY_p = np.dot(ValY_Std, pro)
# 
# train_x = train_x.numpy()
# train_y = TraY_p
# test_x = test_x.numpy()
# y_kr = np.zeros([test_x.shape[0], train_y.shape[1]])
# y_svr = np.zeros([test_x.shape[0], train_y.shape[1]])

# for _ in np.arange(train_y.shape[1]):
#     svr.fit(train_x, train_y[:,_])
#     y_svr[:,_] = svr.predict(test_x)
# Nn_f = np.dot(y_kr, p_inv)
# Krpca_pred_te = mpn.rescale_f(Nn_f, sy_std)
# vom.plot_mwd_animation(np.squeeze(chain), Krpca_pred_te, ValY.numpy(), str_n = 'kr_pca_te')
# Nn_f = np.dot(y_svr, p_inv)
# Svrpca_pred_te = mpn.rescale_f(Nn_f, sy_std)
# vom.plot_mwd_animation(np.squeeze(chain), Svrpca_pred_te, ValY.numpy(), str_n = 'svr_pca_te')



endtime = datetime.datetime.now()
print (f"Total training time: {(endtime - starttime).seconds:d}")

plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')