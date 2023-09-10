from email.policy import default
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.io as io_mat
import os
import r_pca as ra
import mwd_data_preparation as mdp
import Model_preparation_np as mpn
import Model_preparation as mp
import scipy.fftpack as spfft
import gpytorch
import gpy_model_multi as gp_struc
# import pyro.contrib.gp as gp
# import pyro.distributions as dist
# from torch.distributions import constraints
# from pyro.infer import SVI, Trace_ELBO
import visualization_of_mwd as vom
import positive_search as ps
# import pyro.optim
# import cvxpy as cvx
import torch
import scipy.optimize as spopt
import datetime
starttime = datetime.datetime.now()
plt.style.use('ggplot')

path = os.getcwd()
plt.ion()
plt.show()

# str_n = 'MWD'
str_n = 'CSD'
ind = 'passive'
# ind = 'positive'
dir = os.path.join(path,'Data','result_set','model_'+str_n+'.pth')
data_dict = mdp.generate_scenario1(str_n)
def case1():
    X_Sim = data_dict['x_sim']
    Y_Sim = data_dict['y_sim']
    chain = data_dict['chain']
    pcom1 = 9 
    pcom2 = 6
    THr = 700
    return X_Sim, Y_Sim, chain, pcom1, pcom2, THr
def case2():
    X_Sim = data_dict['x_sim']['input']
    Y_Sim = data_dict['y_sim']['output']
    chain = np.linspace(0, 4, 400)[:,None]
    pcom1 = 4
    pcom2 = 3
    THr = 600
    return X_Sim, Y_Sim, chain, pcom1, pcom2, THr
switch = {'MWD': case1,
          'CSD': case2}

X_Sim, Y_Sim, chain, pcom1, pcom2, THr = switch[str_n]()
Nd = Y_Sim.shape[0]
Md = Y_Sim.shape[1]
Y_Sim_std = mdp.scale(Y_Sim)
Y_Sim_sca = (Y_Sim-np.min(Y_Sim,axis=0))/(np.max(Y_Sim, axis=0)-np.min(Y_Sim, axis=0) + 1.0e-6)
U_Y, S_Y, V_Y = np.linalg.svd(Y_Sim_std['norm'], full_matrices=False)

Y_tilde = np.dot(U_Y[:, :pcom1] * S_Y[:pcom1], V_Y[:pcom1, :])
VarPcomY = np.cumsum(S_Y)/np.sum(S_Y)
SeY = np.linalg.norm(Y_Sim-mdp.rescale(Y_tilde, Y_Sim_std['mu'], Y_Sim_std['sigma']), ord='fro')**2
labels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
width = 0.35
rpca = ra.R_pca(Y_Sim_sca)
LowY, SparseY = rpca.fit(0.005, max_iter=10000, iter_print=100)
_, S_L, _ = np.linalg.svd(LowY, full_matrices=False)
VarPcomL = np.cumsum(S_L)/np.sum(S_L)
_, S_S, _ = np.linalg.svd(SparseY, full_matrices=False)
VarPcomS = np.cumsum(S_S)/np.sum(S_S)
rankL = np.linalg.matrix_rank(LowY)
rankS = np.linalg.matrix_rank(SparseY)
condL = np.linalg.cond(LowY)
condS = np.linalg.cond(SparseY)
# 1 figure  ###################
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(chain, Y_Sim.T)
ax.set(xlabel='Chain Length', ylabel=str_n, title=str(Nd)+' '+str_n+' Observations')
ax.grid()
ax = fig.add_subplot(222)
ax.plot(chain, LowY.T)
ax.set(xlabel='Chain Length', ylabel=str_n, title='Low-rank part of '+str_n)
ax.grid()
ax = fig.add_subplot(224)
ax.plot(chain, SparseY.T)
ax.set(xlabel='Chain Length', ylabel=str_n, title='Sparse part of '+str_n)
ax.grid()
# 2 figure  ###################
fig, ax = plt.subplots(2,1)
ax[0].plot(chain, mdp.rescale(Y_tilde, Y_Sim_std['mu'], Y_Sim_std['sigma']).T)
ax[0].set(xlabel='chain', ylabel=str_n+'_Tilde', title='Squared_Error= '+str(round(SeY, 2))+' with '+str(pcom1)+' PCs')
ax[1].bar(labels[:pcom1], VarPcomY[:pcom1], width)
ax[1].set(ylabel='scores', title='each_component_proportion')
abn_p = np.count_nonzero(SparseY, axis=1)
######### plotly plot in browser
# fig1 = px.bar(x=np.arange(0,Nd,1), y=abn_p)
# fig1.update_traces(marker_color='blue')
# fig2 = px.box(y=abn_p, points='all')
# fig1.show()
# fig2.show()
##########################################
lower_q = np.quantile(np.sort(abn_p), 0.25)
higher_q = np.quantile(np.sort(abn_p), 0.75)
iqr = higher_q-lower_q
thre = higher_q+1.5*iqr
X_Sim_Valid = X_Sim[abn_p<=thre, :]
Y_Sim_Valid = Y_Sim[abn_p<=thre, :]
Y_Sim_Valid[Y_Sim_Valid<=1.0e-5] = 0
# X_Sim_Valid = X_Sim_Valid[:1200]  ###固定训练量在1200
# Y_Sim_Valid = Y_Sim_Valid[:1200]  ###固定训练量在1200
N_Valid = abn_p[abn_p<=thre].shape[0]
Y_Sim_Valid_sca = (Y_Sim_Valid-np.min(Y_Sim_Valid,axis=0))/(np.max(Y_Sim_Valid, axis=0)-np.min(Y_Sim_Valid, axis=0)+1.0e-6)
rpca_two_stage = ra.R_pca(Y_Sim_Valid_sca)
LowY_V, SparseY_V = rpca_two_stage.fit(0.001, max_iter=10000, iter_print=100)
Y_Sim_Valid_std = mdp.scale(Y_Sim_Valid)
_, S_f_Valid, V_f_Valid = np.linalg.svd(Y_Sim_Valid_std['norm'], full_matrices=False)
VarPcomSfV = np.cumsum(S_f_Valid)/np.sum(S_f_Valid)
per_var = S_f_Valid/np.sum(S_f_Valid)

pro = V_f_Valid[0:pcom2, :].T
Y_Sim_Valid_p = np.dot(Y_Sim_Valid_std['norm'], pro)
p_inv = np.linalg.lstsq(pro, np.eye(Md), rcond=None)[0]
Y_Valid_tilde = np.dot(Y_Sim_Valid_p, p_inv)
YV_hat_ori = mdp.rescale(Y_Valid_tilde, Y_Sim_Valid_std['mu'], Y_Sim_Valid_std['sigma'])
SqErSum = np.linalg.norm(Y_Sim_Valid-YV_hat_ori, ord='fro')**2
# 3 figure  ###################
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(chain, Y_Sim_Valid.T)
ax.set(xlabel='Chain Length', ylabel=str_n, title=str(N_Valid)+' valid '+str_n+' observations')
ax.grid()
ax = fig.add_subplot(222)
ax.plot(chain, LowY_V.T)
ax.set(xlabel='Chain Length', ylabel=str_n, title='Low-rank part of Valid '+str_n)
ax.grid()
ax = fig.add_subplot(224)
ax.plot(chain, SparseY_V.T)
ax.set(xlabel='Chain Length', ylabel=str_n, title='Sparse part of Valid '+str_n)
ax.grid()
# 4 figure  ###################
fig, ax = plt.subplots(2,1)
ax[0].plot(chain, YV_hat_ori.T, '.-')
ax[0].set(xlabel='Chain Length', ylabel='Valid_'+str_n+'_Tilde', title='Squared_Error= '+str(round(SqErSum, 2))+' with '+str(pcom2)+' PCs')
ax[1].bar(labels[:pcom2], VarPcomSfV[:pcom2], width)
ax[1].set(ylabel='scores', title='each_component_proportion')
_, S_LYV, _ = np.linalg.svd(LowY_V, full_matrices=False)
_, S_SYV, _ = np.linalg.svd(SparseY_V, full_matrices=False)
VarPcomSFV_Low = np.cumsum(S_LYV)/np.sum(S_LYV)
VarPcomSFV_Spar = np.cumsum(S_SYV)/np.sum(S_SYV)
# 5 figure  ###################
fig, ax = plt.subplots(2,1)
xp = np.arange(len(labels))
rects1 = ax[0].bar(xp - width/2, VarPcomL[0:10], width, label='Original')
rects2 = ax[0].bar(xp + width/2, VarPcomSFV_Low[0:10], width, label='Valid')
ax[0].set_ylabel('Scores')
ax[0].set_title('each_component_proportion of low_rank part')
ax[0].set_xticks(xp)
ax[0].set_xticklabels(labels)
ax[0].legend()
# ax[0].bar_label(rects1, padding=3)
# ax[0].bar_label(rects2, padding=3)
rects3 = ax[1].bar(xp - width/2, VarPcomS[0:10], width, label='Original')
rects4 = ax[1].bar(xp + width/2, VarPcomSFV_Spar[0:10], width, label='Valid')
ax[1].set_ylabel('Scores')
ax[1].set_title('each_component_proportion of sparse part')
ax[1].set_xticks(xp)
ax[1].set_xticklabels(labels)
ax[1].legend()
# ax[1].bar_label(rects3, padding=3)
# ax[1].bar_label(rects4, padding=3)
##########fast Fourier transformation######################
# Y_Sim_Ft = spfft.dct(Y_Sim_Valid, type=3, norm='ortho')
# # zengjia figure  ###################
# fig, ax = plt.subplots()
# ax.plot(Y_Sim_Ft.T)

############  figure  ###################
# fig, ax = plt.subplots()
# ax = sns.set_theme(style="white")
# kcom =40
# Y_Sim_Ft1 = mdp.scale(Y_Sim_Ft[:,:kcom])
# _, S_f_Valid, V_f_Valid = np.linalg.svd(Y_Sim_Ft1['norm'], full_matrices=False)
# VarPcomSFV_F = np.cumsum(S_f_Valid)/np.sum(S_f_Valid)
# pcom = 5
# pro = V_f_Valid[:pcom, :].T
# Y_Sim_Ft_p = np.dot(Y_Sim_Ft1['norm'], pro)
# per_r = np.corrcoef(x=X_Sim_Valid.T, y=Y_Sim_Ft_p.T)
# yxc = np.sum(np.abs(per_r[6:, :6]))
# mask = np.triu(np.ones_like(per_r, dtype=bool))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# ax = sns.heatmap(per_r, mask=mask, cmap=cmap, center=0,
#             square=True, linewidths=0.5)
# ax.set(title='Total_R2= '+str(round(yxc, 2)))
# p_inv = np.linalg.lstsq(pro, np.eye(kcom), rcond=None)[0]
# Y_Sim_Ft_til = np.dot(Y_Sim_Ft_p, p_inv)
# Y_Sim_Ft_til = mdp.rescale(Y_Sim_Ft_til, Y_Sim_Ft1['mu'], Y_Sim_Ft1['sigma'])
# Y_Sim_Ft_filte = np.concatenate([Y_Sim_Ft_til, 0.1*np.ones((N_Valid,Md-kcom))], axis=1)
# YVF_hat_ori = spfft.idct(Y_Sim_Ft_filte, type=3, norm='ortho')
# SqErSum_Ft = np.linalg.norm(Y_Sim_Valid-YVF_hat_ori, ord='fro')**2
# plt.style.use('ggplot')
# # 8 figure  ###################
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(chain, YVF_hat_ori.T, '.-')
# ax[0].set(xlabel='chain', ylabel='MWD_Tilde', title='Squared_Error= '+str(round(SqErSum_Ft, 2))+' with '+str(kcom)
#           +' fre with '+ str(pcom)+' pcs')
# ax[1].bar(labels, VarPcomSFV_F[0:10], width)
# ax[1].set(ylabel='scores', title='each_component_proportion')

##########################feature extraction##############################
# d1 = np.diff(Y_Sim_Valid)
# d1[d1<=0] = -1
# d1[d1>0] = 1
# d2 = np.diff(d1)
# fea1 = []
# fea2 = np.argmax(Y_Sim_Valid, axis=1)
# fea3 = []
# fea4 = []
# fea5 = []
# fea1y = []
# fea2y = []
# fea3y = []
# fea4y = []
# fea5y = []
# for i, j in enumerate(d2):
#     m1 = np.min(np.argwhere(Y_Sim_Valid[i]>1.0e-3))
#     fea1.append(m1)
#     fea1y.append(Y_Sim_Valid[i][m1])
#     fea2y.append(Y_Sim_Valid[i][fea2[i]])
#     f1 = np.argwhere(j==2)
#     ind1 = np.min(f1[f1>fea2[i]])
#     fea3.append(ind1+2)
#     fea3y.append(Y_Sim_Valid[i][ind1+2])
#     f2 = np.argwhere(j==-2)
#     ind2 = np.min(f2[f2>ind1])
#     fea4.append(ind2+2)
#     fea4y.append(Y_Sim_Valid[i][ind2+2])
#     f3 = np.argwhere(Y_Sim_Valid[i]<1.0e-3)
#     m2 = f3[f3>ind2+2]
#     if m2.size==0:
#         m2 = 131
#     fea5.append(np.min(m2))
#     fea5y.append(Y_Sim_Valid[i][np.min(m2)])
# fea1x = chain[fea1]; fea1y = np.array(fea1y)
# fea2x = chain[fea2]; fea2y = np.array(fea2y)
# fea3x = chain[fea3]; fea3y = np.array(fea3y)
# fea4x = chain[fea4]; fea4y = np.array(fea4y)
# fea5x = chain[fea5]; fea5y = np.array(fea5y)
# fea_x = np.hstack([fea1x, fea2x, fea3x, fea4x, fea5x])
# fea_y = np.vstack([fea1y, fea2y, fea3y, fea4y, fea5y]).T
# fig, ax = plt.subplots(1,1)
# for _ in range(fea_x.shape[0]):
#     ax.plot(fea_x[_], fea_y[_], '*-')
# ax.set(xlabel='X', ylabel='Y', title='Feature')

# # 6 figure  ###################
# fig, ax = plt.subplots()
# ax = sns.set_theme(style="white")
# per_r = np.corrcoef(x=X_Sim_Valid.T, y=fea_x.T)
# yxc = np.sum(np.abs(per_r[6:, :6]))
# mask = np.triu(np.ones_like(per_r, dtype=bool))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# ax = sns.heatmap(per_r, mask=mask, cmap=cmap, center=0,
#             square=True, linewidths=0.5)
# ax.set(title='Total_R2= '+str(round(yxc, 2)))


####################separate the data into training and validation part####################
np.random.seed(0)
tf = round(0.8*X_Sim_Valid.shape[0])
ts = np.random.randint(1,X_Sim_Valid.shape[0],size=X_Sim_Valid.shape[0])
X_Sim_Valid_std = mpn.scale_f(X_Sim_Valid)['val']
# TraX = X_Sim_Valid_std['norm'][ts[0:tf],:]
TraX = X_Sim_Valid_std[ts[0:tf],:]
TraX_Ori = X_Sim_Valid[ts[0:tf],:]
TraY_P = Y_Sim_Valid_p[ts[0:tf],:]
TraY_Ori = Y_Sim_Valid[ts[0:tf],:]
# ValX = X_Sim_Valid_std['norm'][ts[tf:],:]
ValX = X_Sim_Valid_std[ts[tf:],:]
ValY_P = Y_Sim_Valid_p[ts[tf:],:]
ValY_Ori = Y_Sim_Valid[ts[tf:],:]
#################### 7 figure  ###################
fig, ax = plt.subplots()
ax = sns.set_theme(style="white")
per_r = np.corrcoef(x=TraX.T, y=TraY_P.T)
yxc = np.sum(np.abs(per_r[X_Sim_Valid.shape[0]:, :X_Sim_Valid.shape[0]]))
mask = np.triu(np.ones_like(per_r, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
ax = sns.heatmap(per_r, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=0.5)
ax.set(title='Total_R2= '+str(round(yxc, 2)))
    


# # Gaussian Process on Pyro  ###################
# # TX = torch.tensor(TraX)
# # TY = torch.tensor(TraY_P.T)
# # kernel = gp.kernels.RBF(
# #                         input_dim=TraX.shape[1], 
# #                         variance=torch.tensor(.5),
# #                         lengthscale=torch.tensor(.2))
# # kernel.lengthscale = pyro.nn.PyroSample(dist.Gamma(1, 0.05))
# # kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0., 2.0))
# # gpr = gp.models.GPRegression(TX, TY, kernel)
# # mean, cov = gpr(TX, full_cov=False)
# # optimizer = torch.optim.Adam(gpr.parameters(), lr=0.001, betas=[0.8, 0.99])
# # loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
# # losses=[]
# # num_steps = 150
# # fig, ax = plt.subplots()
# # for _ in range(num_steps):
# #     # svi.step()
# #     optimizer.zero_grad()
# #     loss = loss_fn(gpr.model, gpr.guide)
# #     loss.backward()
# #     optimizer.step()
# #     losses.append(loss.item())
# # ax.plot(losses)
# # ax.set(xlabel='iteration', ylabel='gpr_obj', title='loss= '+str(losses[-1]))

# # TeX = torch.tensor(ValX)
# # Gp_out_mean, Gp_std = gpr(TeX, full_cov=False)
# # Gp_out_mean = Gp_out_mean.detach().numpy().T
# # Gp_std = Gp_std.detach().numpy().T
# # quan95_value = Gp_out_mean + 2*Gp_std
# # quan05_value = Gp_out_mean - 2*Gp_std

###############passive Gaussian process#######################
ts = pcom2
k_id = 0   ##kernel label
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=ts, rank=1)
###############positive Gaussian process#######################
if ind=='passive':
    TraX_t = torch.Tensor(TraX)
    TraY_P_t = torch.Tensor(TraY_P)
    gpr = gp_struc.MultitaskGPModel(TraX_t, TraY_P_t, likelihood, k_id, ts).float()
    gpr.train(), likelihood.train()
    # loss = gp_struc.train(gpr, likelihood, TraX_t, TraY_P_t)
    state_dict = torch.load(dir)
    gpr.load_state_dict(state_dict)
elif ind=='positive':
    s = 4*ts+TraX.shape[1]+1+5
    # TraX_ta = torch.Tensor(TraX_Ori)
    # TraY_ta = torch.Tensor(TraY_Ori)
    TraX_ta = torch.Tensor(TraX)
    TraY_ta = torch.Tensor(TraY_P)
    TraX_t = TraX_ta[:s]
    TraY_t = TraY_ta[:s]
    
    # sub = mdp.scale(TraY_Ori[:s])
    # _, _, V_f_Valid = np.linalg.svd(sub['norm'], full_matrices=False)
    # pro = V_f_Valid[0:pcom2, :].T
    # TraY_p = torch.Tensor(np.dot(sub['norm'], pro))
    gpr = gp_struc.MultitaskGPModel(TraX_t, TraY_t, likelihood, k_id, ts).float()
    gpr.train(), likelihood.train()
    gp_struc.train(gpr, likelihood, TraX_t, TraY_t)
    # min_x, min_y, gpr = ps.search(TraX_Ori, TraY_Ori, gpr, likelihood, loss, per_var[:pcom2], s)
    min_x, min_y, gpr, losses = ps.search(TraX, TraY_P, gpr, likelihood, per_var[:pcom2], s, ValX, ValY_P, THr)
    print(f"Required points: {min_x.shape[0]:d}")
    np.savetxt(path+r"\Data\XSet_min.txt", min_x)
    np.savetxt(path+r"\Data\YSet_min.txt", min_y)
    np.savetxt(path+r"\Data\LossSet.txt", losses)
    # 7 figure  ###################
    fig, ax = plt.subplots()
    dx = np.arange(min_x.shape[0]-s)
    ax.plot(dx, losses)
    ax.set(xlabel='Evaluations', ylabel='loss', title='Convergence')
ValX_t = torch.Tensor(ValX)
gpr.eval(), likelihood.eval()
Gp_out_mean = likelihood(gpr(ValX_t.float())).mean.detach().numpy()
# observed_val = likelihood(ValX_t.float().T).detach()
# y_std = gpr(ValX_t.float()).variance.detach().numpy()**(0.5)
# quan05_value = Gp_out_mean-2*y_std
# quan95_value = Gp_out_mean+2*y_std
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gpr(ValX_t.float()))
lower, upper = observed_pred.confidence_region()
quan05_value = lower.detach().numpy()
quan95_value = upper.detach().numpy()

################Gaussian process on sepia with mcmc#######################
# x_ind = np.arange(1,TraY_P.shape[1]+1)
# data = SepiaData(y_sim=TraY_P, y_ind_sim=x_ind, x_sim=TraX)
# data.transform_xt()
# data.standardize_y(scale='columnwise')
# data.create_K_basis(TraY_P.shape[1])
# data.create_D_basis(D_type='linear')
# print(data)
# model = SepiaModel(data)
# # model.tune_step_sizes(50, 20, update_vals=True)
# # model.do_mcmc(1000)
# # model.save_model_info('saved_mwd_model_5features',overwrite=True)
# model.restore_model_info('saved_mwd_model_5features')
# samples_dict = model.get_samples()
# mcmc_trace = SepiaPlot.mcmc_trace(samples_dict)
# # fig = plt.figure()
# model.print_mcmc_info()
# pred_samples=model.get_samples(numsamples=24)
# pred=SepiaEmulatorPrediction(samples=pred_samples, model=model, x_pred=ValX)
# predy=pred.get_y()
# Gp_out_mean = np.zeros(shape=(ValY_P.shape))
# quan05_value = np.zeros(shape=(ValY_P.shape))
# quan95_value = np.zeros(shape=(ValY_P.shape))
# for _ in range(ValY_Ori.shape[0]):
#     Gp_out_mean[_, :] = np.mean(predy[:,_,:], 0)
#     quan05_value[_, :] = np.quantile(predy[:,_,:],0.05,0)
#     quan95_value[_, :] = np.quantile(predy[:,_,:],0.95,0)



Y_Hat_F = mdp.rescale(np.dot(Gp_out_mean, p_inv), Y_Sim_Valid_std['mu'], Y_Sim_Valid_std['sigma'])
Y_Hat_F05 = mdp.rescale(np.dot(quan05_value, p_inv), Y_Sim_Valid_std['mu'], Y_Sim_Valid_std['sigma'])
Y_Hat_F95 = mdp.rescale(np.dot(quan95_value, p_inv), Y_Sim_Valid_std['mu'], Y_Sim_Valid_std['sigma'])

# Y_Hat_Fi = np.concatenate([Y_Hat_F, 0.1*np.ones((ValY_Ori.shape[0],Md-kcom))], axis=1)
# Y_Hat_F05i = np.concatenate([Y_Hat_F05, 0.1*np.ones((ValY_Ori.shape[0],Md-kcom))], axis=1)
# Y_Hat_F95i = np.concatenate([Y_Hat_F95, 0.1*np.ones((ValY_Ori.shape[0],Md-kcom))], axis=1)
# Y_Hat = spfft.idct(Y_Hat_Fi, type=3, norm='ortho')
# Y_Hat_05 = spfft.idct(Y_Hat_F05i, type=3, norm='ortho')
# Y_Hat_95 = spfft.idct(Y_Hat_F95i, type=3, norm='ortho')

# # nearest, cubic, slinear or quadratic interpolate
# x_new = np.linspace(np.min(chain), np.max(chain), 150)
# Y_Hat_In = np.zeros(shape=(Y_Hat.shape))
# Y_Hat_05In = np.zeros(shape=(Y_Hat.shape))
# Y_Hat_95In = np.zeros(shape=(Y_Hat.shape))
# chain1 = chain.squeeze() # for interpolate.interp1d , chain shape is(132,1), while Y_hatp[_, :] is (132,)
# for _ in range(ValY_Ori.shape[0]):
#     fun_interp = interpolate.interp1d(chain1, Y_Hat[_,:], kind='cubic')
#     Y_Hat_In[_, :] = fun_interp(chain1)
#     fun_interp = interpolate.interp1d(chain1, Y_Hat_05[_,:], kind='cubic')
#     Y_Hat_05In[_, :] = fun_interp(chain1)
#     fun_interp = interpolate.interp1d(chain1, Y_Hat_95[_,:], kind='cubic')
#     Y_Hat_95In[_, :] = fun_interp(chain1)
# np.savetxt(path+r"\Data\chain_inter.txt", x_new)
# torch.save(gpr.state_dict(), dir)

vom.plot_mwd_animation(np.squeeze(chain), Y_Hat_F, ValY_Ori, Y_Hat_F05, Y_Hat_F95, 'test')
endtime = datetime.datetime.now()
print (f"Total training time: {(endtime - starttime).seconds:d}")
# print("error"+str(SS))



plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')