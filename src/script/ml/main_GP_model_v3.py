import numpy as np
import matplotlib.pyplot as plt
import os
import r_pca as ra
import mwd_data_preparation as mdp
import Model_preparation_np as mpn
import Model_preparation as mp
import scipy.fftpack as spfft
import scipy.signal as signal
import gpytorch
import gpvae_model_multi_large as gp_struc
import visualization_of_mwd as vom
import torch
import datetime
import pybobyqa
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.interpolate import interp1d
starttime = datetime.datetime.now()

path = os.getcwd()
plt.ion()
plt.show()

str_n = 'MWD'
# str_n = 'CSD'
ind = 'passive'
# ind = 'positive'
# dir = os.path.join(path,'Data','result_set','model_'+str_n+'.pth')
data_dict = mdp.generate_scenario1(str_n)
def case1():
    X_Sim = data_dict['x_sim']
    Y_Sim = data_dict['y_sim']
    chain = data_dict['chain']
    pcom1 = 8
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
apc = r"C:\Users\t1714\Desktop\Academic\Monthly Reports\Submission to pse 2022\pic"
X_Sim, Y_Sim, chain, pcom1, pcom2, THr = switch[str_n]()
labels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
width = 0.35

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
def curve_filter(y, yp ,yt):
    y[y<=0.1]=0
    df_l = np.diff(y)
    loc1 = min((np.argwhere(df_l<0))+1)[0]
    df_l[:loc1]=0
    loc2 = min((np.argwhere(df_l>0))+1)[0]
    i = 0
    while (loc2-loc1)<5:
        loc2 = np.sort((np.argwhere(df_l>0))+1)[i][0]
        i+=1
    df_l[:loc2]=0
    loc3 = min((np.argwhere(df_l<0))+1)[0]
    i = 0
    while (loc3-loc2)<5:
        loc3 = np.sort((np.argwhere(df_l>0))+1)[i][0]
        i+=1
    s = min(np.argwhere(yp>=0.03))[0]
    n = min(np.argwhere(yp[loc3:]<0.03))[0]+loc3
    # y0 = np.zeros(s)
    # y1 = signal.savgol_filter(yp[s:loc1], int(np.floor((loc1-s)/2)), 2)
    # y2 = signal.savgol_filter(yp[loc1:loc2], int(np.floor((loc2-loc1)/2)), 2)
    # y3 = signal.savgol_filter(yp[loc2:loc3], int(np.floor((loc3-loc2)/2)), 2)
    # y4 = signal.savgol_filter(yp[loc3:n], int(np.floor((n-loc3)/2)), 2)
    # y5 = np.zeros(Md-n)
    # y_filter = np.concatenate([y0,y1,y2,y3,y4,y5])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(chain, yp)
    # ax.plot(chain, y_filter)
    # ax.plot(chain, aa[1])
    yp[:s] = 0
    yp[n:] = 0
    yp[loc1] = 0.82
    yp[loc3] = 0.44
    y1 = signal.savgol_filter(yp[:loc1], 5, 3)
    y2 = signal.savgol_filter(yp[loc1:loc3], 5, 3)
    y3 = signal.savgol_filter(yp[loc3:], 5, 3)
    y_filter = np.concatenate([y1,y2,y3])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(chain, yp)
    # ax.plot(chain, y_filter)
    # ax.plot(chain, yt)
    return y_filter
    # return loc1,loc2,loc3

rpca = ra.R_pca(Y_Sim)
LowY, SparseY = rpca.fit(max_iter=1000, iter_print=500, tol=0.005)
abn_p = np.count_nonzero(SparseY, axis=1)
lower_q = np.quantile(np.sort(abn_p), 0.25)
higher_q = np.quantile(np.sort(abn_p), 0.75)
iqr = higher_q-lower_q
thre = higher_q+1.5*iqr
X_Sim_Valid = X_Sim[abn_p<=thre, :]
Y_Sim_Valid = Y_Sim[abn_p<=thre, :]
Y_Sim = Y_Sim_Valid
Nd = Y_Sim.shape[0]
Md = Y_Sim.shape[1]
mu = np.ones((Nd, Md))*0.
sigma = 0.05*np.tile(np.max(Y_Sim, axis=1, keepdims=True), Md)*np.ones((Nd, Md))
z = np.random.normal(mu, sigma)
Y_Sim_ob = Y_Sim+z
Y_Sim_ob[Y_Sim_ob<0]=0

lp = moving_average(Y_Sim_ob[0], 6)
# lp = curve_filter(Y_Sim_ob[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(chain, Y_Sim[0].T, label='y', marker='s')
ax.plot(chain[:-5], lp.T, label='y_filter')
ax.plot(chain, Y_Sim_ob[0].T, label='y_ob', linestyle=':', marker='*')
ax.set(xlabel='k', ylabel='yt', title='FFT')
ax.legend()
########################fig1########################
fig, ax = plt.subplots(1,2)
ax[0].plot(chain, Y_Sim.T)
ax[0].set(xlabel='Chain Length', ylabel=str_n, title=str(Nd)+' Noiseless values')
ax[1].plot(chain, Y_Sim_ob.T)
ax[1].set(xlabel='Chain Length', ylabel=str_n, title=str(Nd)+' Observations')
########################fig2########################
ns = 60
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(chain, Y_Sim[ns].T, label='y')
ax.set(xlabel='Chain Length', ylabel=str_n, title=str(1)+' Observation')
ax.plot(chain, Y_Sim_ob[ns].T, label='y_ob')
ax.legend()
Y_Sim_Ft_p = spfft.dct(Y_Sim[ns], norm='ortho', type=1) 
Y_Sim_Ft1 = spfft.dct(Y_Sim_ob[ns], norm='ortho', type=1)
mod = 20
Y_Sim_Ft2 = Y_Sim_Ft1.copy()
Y_Sim_Ft_fl = np.concatenate([Y_Sim_Ft2[:mod], 0.0001*np.ones(Md-mod)])
X = spfft.idct(Y_Sim_Ft_fl, norm='ortho',  axis=0, type=1)
e_single = np.linalg.norm(Y_Sim[ns]-X)
########################fig3########################
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(Y_Sim_Ft_p.T, label='y')
ax.plot(Y_Sim_Ft1.T, label='y_ob')
ax.set(xlabel='k', ylabel='yt', title='FFT')
ax.legend()
ax = fig.add_subplot(122)
ax.plot(chain, Y_Sim[ns], label='Y')
ax.plot(chain, X, label='FFT', marker='x')
ax.set(xlabel='Chain Length', ylabel='MWD', title='IFFT')
ax.legend()
########################FFT########################
Yf = np.zeros((Nd,Md))
mod = np.zeros(Nd)
for _ in np.arange(Nd):
    YY = Y_Sim_ob[_]
    Y_Sim_Ft = spfft.dct(YY, norm='ortho', type=1) 
    sup = np.min(Y_Sim_Ft)
    ind = min(np.argwhere(Y_Sim_Ft-sup)<np.median(Y_Sim_Ft-sup))[0]
    mod[_] = ind
    Yf[_] = Y_Sim_Ft
rpca = ra.R_pca(Yf)
LowY, SparseY = rpca.fit(max_iter=1000, iter_print=500)
ll = signal.savgol_filter(LowY[ns][17:], 5, 2)
y_r = spfft.idct(LowY[ns], norm='ortho',  axis=0, type=1)
low_f = np.concatenate([LowY[ns][:17],ll])
y_r1 = spfft.idct(low_f, norm='ortho',  axis=0, type=1)
er1 = np.linalg.norm(X-Y_Sim[ns])
er2 = np.linalg.norm(y_r-Y_Sim[ns])
er3 = np.linalg.norm(y_r1-Y_Sim[ns])
########################fig4########################
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(Y_Sim_Ft_p, label='y')
ax.plot(Y_Sim_Ft_fl, label='yob_fft')
ax.plot(LowY[ns], label='yob_low1')
ax.plot(low_f, label='yob_low2')
ax.set(xlabel='k', ylabel='yt',title='FFT')
ax.legend()
ax = fig.add_subplot(122)
ax.plot(chain, Y_Sim[ns], label='y')
ax.plot(chain, X, label='yob_fft')
ax.plot(chain, y_r, label='yob_treated1')
ax.plot(chain, y_r1, label='yob_treated2')
ax.set(xlabel='Chain Length', ylabel='MWD',title=f'{er1:.2f}vs{er2:.2f}vs{er3:.2f}')
ax.legend()

Low_f = []
for _ in LowY:
    sup = min(_)
    ab_low = _-sup
    a = moving_average(ab_low,3)
    fl = np.min(np.argwhere(abs(a-np.median(a))<0.013))
    ll = signal.savgol_filter(_[fl:], 5, 2)
    Low_f.append(np.concatenate([_[:fl], ll]))
Low_f = np.asarray(Low_f)   
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(LowY[:2].T)
ax.plot(Low_f[:2].T, linestyle='-.')
Low_t = []
for _ in Y_Sim:
    Low_t.append(spfft.dct(_, norm='ortho', type=1))
Low_t = np.asarray(Low_t)
e1 = np.linalg.norm(Low_t-LowY)**2
e2 = np.linalg.norm(Low_t-Low_f)**2

sy = mpn.scale_f(Low_f)
_, S_f, V_f = np.linalg.svd(sy['val'], full_matrices=False)
VarPcomSfV = np.cumsum(S_f)/np.sum(S_f)
pro = V_f[0:2, :].T
TraY_P = np.dot(sy['val'], pro)
########################fig5########################
Y_Sim_NFt = []
p_ls = [0,5,60]
for _ in p_ls:
    Y_Sim_Ft = spfft.dct(Y_Sim[_], norm='ortho', type=1) 
    Y_Sim_NFt.append(Y_Sim_Ft)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(Y_Sim_NFt[0].T, label='yt1')
ax.plot(Y_Sim_NFt[1].T, label='yt2')
ax.plot(Y_Sim_NFt[2].T, label='yt3')
ax.set(xlabel='k', ylabel='yt', title='Y-FFT')
ax.legend()
ax = fig.add_subplot(122)
ax.plot(Yf[p_ls[0]].T, label='yt1')
ax.plot(Yf[p_ls[1]].T, label='yt2')
ax.plot(Yf[p_ls[2]].T, label='yt3')
ax.set(xlabel='k', ylabel='yt', title='Y_OB-FFT')
ax.legend()
########################KNN########################
kc = 3
k_means = KMeans(init="k-means++", n_clusters=kc, n_init=10)
k_means.fit(sy['val'])
k_means_cluster_centers = k_means.cluster_centers_
kp = np.dot(k_means_cluster_centers, pro)
k_means_labels = pairwise_distances_argmin(sy['val'], k_means_cluster_centers)
########################fig6########################
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", '#331a36', '#a19825', '#ba0b0b', '#1b4a06', '#293308', '#082233', '#78431a']
for k, col in zip(range(kc), colors):
    my_members = k_means_labels == k
    # cluster_center = k_means_cluster_centers[k]
    cluster_center = kp[k]
    ax.plot(TraY_P[my_members, 0], TraY_P[my_members, 1], "w", markerfacecolor=col, marker=".", markersize=13)
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
########################fig7########################
sn = []
aa, bb, cc = [], [], []
l_rf = []
for _ in LowY:
    l_rf.append(spfft.idct(_, norm='ortho', type=1))
l_rf = np.asarray(l_rf)
########################fft########################
# for k in np.arange(kc):
#     my_members = k_means_labels == k
#     rpca = ra.R_pca(Yf[my_members])
#     LowY2, SparseY = rpca.fit(max_iter=1000, iter_print=500)
#     rf = []
#     for _ in LowY2:
#         sup = min(_)
#         ab_low = _-sup
#         a = moving_average(ab_low,3)
#         fl = np.min(np.argwhere(abs(a-np.median(a))<0.013))
#         ll = signal.savgol_filter(_[fl:], 5, 2)
#         L_f = np.concatenate([_[:fl], ll])
#         rf.append(spfft.idct(L_f, norm='ortho', type=1))
#     rf = np.asarray(rf)
#     o_idx = np.argmax(np.linalg.norm(rf-Y_Sim[my_members], axis=1))
#     aa.append(Y_Sim[my_members][o_idx])
#     bb.append(l_rf[my_members][o_idx])
#     cc.append(rf[o_idx])
#     error_o = np.linalg.norm(rf-Y_Sim[my_members])**2
#     sn.append(error_o)
#     print(f'num is:{rf.shape[0]:d}')
# sn_l = np.linalg.norm(l_rf-Y_Sim)**2
# print(sn_l)
# print(sum(sn))
# aa = np.asarray(aa)
# bb = np.asarray(bb)
# cc = np.asarray(cc)
# cc[cc<=0]=0
# cc_f = signal.savgol_filter(cc, 10, 3)
# cc_f[cc_f<=0]=0
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(chain, aa.T, label='true')
# ax.plot(chain, bb.T, label='m1')
# ax.plot(chain, cc_f.T, label='m2')
# ax.legend()
########################y_inverse########################
low_m = []
for k in np.arange(kc):
    my_members = k_means_labels == k
    rpca = ra.R_pca(Y_Sim_ob[my_members])
    LowY2, SparseY = rpca.fit(max_iter=1000, iter_print=500)
    rf = LowY2
    o_idx = np.argmax(np.linalg.norm(rf-Y_Sim[my_members], axis=1))
    aa.append(Y_Sim[my_members][o_idx])
    bb.append(l_rf[my_members][o_idx])
    cc.append(rf[o_idx])
    error_o = np.linalg.norm(rf-Y_Sim[my_members])**2
    sn.append(error_o)
    print(f'num is:{rf.shape[0]:d}')
    low_m.append(rf)
sn_l = np.linalg.norm(l_rf-Y_Sim)**2
print(sn_l)
print(sum(sn))
# aa = np.asarray(aa)
# bb = np.asarray(bb)
# cc = np.asarray(cc)
# cc[cc<=0.03]=0
# cc_f = signal.savgol_filter(cc, 10, 3)
# cc_f[cc_f<=0.03]=0
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(chain, aa.T, label='true')
# ax.plot(chain, bb.T, label='m1')
# ax.plot(chain, cc_f.T, label='m2')
# ax.legend()
aa, bb, cc, dd = [], [], [], []
sn = []
for k, m in zip(low_m, np.arange(kc)):
    my_members = k_means_labels == m
    low_ad = []
    for _ in np.arange(k.shape[0]):
        cc_f = moving_average(k[_], 3)
        tp = curve_filter(cc_f, Y_Sim_ob[my_members][_], Y_Sim[my_members][_])
        # if abs(_[tp[0]]-0.82)>=0.1:
        #     _[tp[0]] = 0.82
        # if abs(_[tp[1]]-0.27)>=0.1:
        #     _[tp[1]] = 0.27
        # if abs(_[tp[2]]-0.44)>=0.1:
        #     _[tp[2]] = 0.44
        low_ad.append(tp)
    low_ad = np.asarray(low_ad)
    rpca = ra.R_pca(low_ad)
    LowY3, SparseY = rpca.fit(max_iter=1000, iter_print=500)
    o_idx = np.argmax(np.linalg.norm(LowY3-Y_Sim[my_members], axis=1))
    aa.append(Y_Sim[my_members][o_idx])
    bb.append(l_rf[my_members][o_idx])
    cc.append(LowY3[o_idx])
    dd.append(Y_Sim_ob[my_members][o_idx])
    error_o = np.linalg.norm(LowY3-Y_Sim[my_members])**2
    sn.append(error_o)
    print(f'num is:{LowY3.shape[0]:d}')
print(sum(sn))
aa = np.asarray(aa)
bb = np.asarray(bb)
cc = np.asarray(cc)
dd = np.asarray(dd)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(chain, aa.T, label='true')
# ax.plot(chain, bb.T, label='m1')
ax.plot(chain, cc.T, label='m2')
ax.legend()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(chain, aa.T, label='true')
# ax.plot(chain, bb.T, label='m1')
ax.plot(chain, dd.T, label='ob')
ax.legend()
####################################################rest#############################
y_filt = signal.savgol_filter(Y_Sim_ob, 15, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(chain, y_filt[1].T)
ax.plot(chain, Y_Sim_ob[1].T, linestyle='--')
error = np.linalg.norm(y_filt-Y_Sim)**2
# ax.plot(chain, bb.T, label='true')

e = []
for _ in np.arange(Md):
    Y_Sim_Ft_filte = np.concatenate([Y_Sim_Ft[:,:_+1], 0.00001*np.ones((Nd,Md-_-1))], axis=1)
    Y_h = spfft.idct(Y_Sim_Ft_filte, type=3, norm='ortho')
    SqErSum = np.linalg.norm(Y_Sim-Y_h, ord='fro')**2
    e.append(SqErSum)
idx = np.argmin(e)
Y_Sim_Ft_filte = np.concatenate([Y_Sim_Ft[:,:idx], 0.01*np.ones((Nd,Md-idx))], axis=1)
Y_h = spfft.idct(Y_Sim_Ft_filte, type=3, norm='ortho')
def opt_fun(x):
    rpca = ra.R_pca(Y_h, mu=x[0], lmbda=x[1])
    LowY, SparseY = rpca.fit( max_iter=500, iter_print=10)
    error = np.linalg.norm(Y_Sim-LowY, ord='fro')**2
    print(f'e is:{error:.2f}')
    return error
xop = [30, 0.01]
Lbb_s = [1, 0.001]
Ubb_s = [150, 0.1]
# vom.plot_mwd_animation(np.squeeze(chain),LowY, Y_Sim, str_n='rpca')
# soln = pybobyqa.solve(opt_fun, xop, bounds=(Lbb_s, Ubb_s), maxfun=500,
#                             scaling_within_bounds=True)
# rpca = ra.R_pca(Y_h, mu=soln.x[0], lmbda=soln.x[1])
rpca = ra.R_pca(Y_h, mu=82, lmbda=0.005)
LowY, SparseY = rpca.fit( max_iter=500, iter_print=10, tol=1.0e-6)
# 1 figure  ###################
# ip = np.argmax(np.sum(abs(SparseY), axis=1))
# Y_y = mpn.rescale_f_zo(LowY, sft)
Y_Sim_Ft = spfft.dct(LowY, type=3, norm='ortho')
e = []
for _ in np.arange(Md):
    Y_Sim_Ft_filte = np.concatenate([Y_Sim_Ft[:,:_+1], 0.01*np.ones((Nd,Md-_-1))], axis=1)
    Y_h = spfft.idct(Y_Sim_Ft_filte, type=3, norm='ortho')
    SqErSum = np.linalg.norm(Y_Sim-Y_h, ord='fro')**2
    e.append(SqErSum)
idx = np.argmin(e)
Y_Sim_Ft_filte = np.concatenate([Y_Sim_Ft[:,:idx], 0.01*np.ones((Nd,Md-idx))], axis=1)
Y_h = spfft.idct(Y_Sim_Ft_filte, type=3, norm='ortho')
fig = plt.figure()
idx=np.argmax(np.linalg.norm(Y_Sim-Y_h, axis=1))
ax = fig.add_subplot(111)
ax.plot(chain, Y_Sim_ob[idx].T)
ax.plot(chain, Y_Sim[idx].T)
ax.plot(chain, Y_h[idx].T)
ax.set(xlabel='Chain Length', ylabel=str_n, title=str(Nd)+' '+str_n+' Observations')
ax.grid()
# ax = fig.add_subplot(222)
# ax.plot(chain, LowY[idx].T)
# ax.set(xlabel='Chain Length', ylabel=str_n, title='Low-rank part of '+str_n)
# ax.grid()
# ax = fig.add_subplot(224)
# ax.plot(chain, SparseY[idx].T)
# ax.set(xlabel='Chain Length', ylabel=str_n, title='Sparse part of '+str_n)
# ax.grid()

idx = np.argsort(np.linalg.norm(Y_Sim-Y_h, axis=1))[:555]
fig, ax = plt.subplots()
ax.plot(chain, Y_Sim_ob[idx].T)
fig, ax = plt.subplots()
ax.plot(chain, Y_h[idx].T)
ax.plot(chain, Y_Sim[idx].T,'r--', label='or')
# ax.legend()
ipp = np.argsort(np.linalg.norm(Y_Sim-Y_h, axis=1))[-140:]
worst = Y_h[ipp]
yy = np.vstack([Y_h[idx], worst])
rpca = ra.R_pca(yy)
LowY, SparseY = rpca.fit( max_iter=100, iter_print=10)
fig, ax = plt.subplots()
ax.plot(chain, Y_Sim_ob[ipp[-1]].T)
ax.plot(chain, LowY[-1].T, label='a')
ax.plot(chain, Y_h[ipp[-1]].T, label='p')
ax.plot(chain, Y_Sim[ipp[-1]].T, label='o')
ax.legend()


pc=40
syf = mpn.scale_f(Y_h[idx])
vf = spfft.dct(Y_h[idx], type=3, norm='ortho')
syff = mpn.scale_f(Y_Sim_ob[ipp])
vff = spfft.dct(Y_Sim_ob[ipp], type=3, norm='ortho')
yy = np.vstack([vf[:,:pc], vff[:,:pc]])
rpca = ra.R_pca(yy)
Ndd = yy.shape[0]
LowY, SparseY = rpca.fit( max_iter=100, iter_print=10)
Y_Sim_Ft_filte = np.concatenate([LowY, 0.01*np.ones((Ndd,Md-pc))], axis=1)
hh = spfft.idct(Y_Sim_Ft_filte, type=3, norm='ortho')
# hh = mpn.rescale_f(hh, syf)
fig, ax = plt.subplots()
ax.plot(chain, Y_Sim_ob[ipp[-1]].T)
ax.plot(chain, hh[-1].T, label='a')
ax.plot(chain, Y_h[ipp[-1]].T, label='p')
ax.plot(chain, Y_Sim[ipp[-1]].T, label='o')
ax.legend()

pc=18
vf = spfft.dct(Y_Sim_ob, type=3, norm='ortho')
rpca = ra.R_pca(vf[:,:pc])
LowY, SparseY = rpca.fit( max_iter=2000, iter_print=10)
Y_Sim_Ft_filte = np.concatenate([LowY, 0.00001*np.ones((Nd,Md-pc))], axis=1)
hh = spfft.idct(Y_Sim_Ft_filte, type=3, norm='ortho')
# hh = mpn.rescale_f(hh, syf)
fig, ax = plt.subplots()
ax.plot(chain, Y_Sim_ob[ipp[-1]].T)
ax.plot(chain, hh[-1].T, label='a')
ax.plot(chain, Y_h[ipp[-1]].T, label='p')
ax.plot(chain, Y_Sim[ipp[-1]].T, label='o')
ax.legend()

idx = np.argmax(np.linalg.norm(Y_Sim-hh, axis=1))
fig, ax = plt.subplots()
ax.plot(chain, Y_Sim_ob[idx].T)
ax.plot(chain, hh[idx].T, label='a')
ax.plot(chain, Y_h[idx].T, label='p')
ax.plot(chain, Y_Sim[idx].T, label='o')
ax.legend()
err = np.linalg.norm(Y_Sim-hh)**2

fig, ax = plt.subplots()
ax.plot(chain, hh[:10].T)

dir_c = apc+'\\Data_y.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 

abn_p = np.count_nonzero(SparseY, axis=1)
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
rpca_two_stage = ra.R_pca(Y_Sim_Valid)
LowY_V, SparseY_V = rpca_two_stage.fit(0.005, max_iter=10000, iter_print=500)
# 2 figure  ###################
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
dir_c = apc+'\\Data_y_valid.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 

####################separate the data into training and validation part####################
np.random.seed(0)
tf = round(0.8*X_Sim_Valid.shape[0])
ts = np.random.randint(1,X_Sim_Valid.shape[0],size=X_Sim_Valid.shape[0])
TraX = X_Sim_Valid[ts[0:tf],:]
TraY = LowY_V[ts[0:tf],:]
TraY_Ori = Y_Sim_Valid[ts[0:tf],:]
sx = mpn.scale_f(TraX)
TraX = sx['val']
ValX = X_Sim_Valid[ts[tf:],:]
ValX = mpn.scale_f(ValX, sx)['val']
ValY_Ori = Y_Sim_Valid[ts[tf:],:]
#########svd###############
sy = mpn.scale_f(TraY)
_, S_f_Valid, V_f_Valid = np.linalg.svd(sy['val'], full_matrices=False)
VarPcomSfV = np.cumsum(S_f_Valid)/np.sum(S_f_Valid)
per_var = S_f_Valid/np.sum(S_f_Valid)
pro = V_f_Valid[0:pcom1, :].T
TraY_P = np.dot(sy['val'], pro)
p_inv = np.linalg.lstsq(pro, np.eye(Md), rcond=None)[0]
Y_tilde = np.dot(TraY_P, p_inv)
YV_hat_ori = mpn.rescale_f(Y_tilde, sy)
SqErSum = np.linalg.norm(TraY_Ori-YV_hat_ori, ord='fro')**2
#####figure 3####################################
fig, ax = plt.subplots(2,1)
ax[0].plot(chain, YV_hat_ori.T)
ax[0].set(xlabel='chain', ylabel=str_n+'_Tilde', title='Squared_Error= '+str(round(SqErSum, 2))+' with '+str(pcom1)+' PCs')
ax[1].bar(labels[:pcom1], VarPcomSfV[:pcom1], width)
ax[1].set(ylabel='scores', title='each_component_proportion')
dir_c = apc+'\\pca.png'
plt.tight_layout()
plt.savefig(dir_c, dpi=500) 
###############passive Gaussian process#######################
ts = pcom1
k_id = 0   ##kernel label
###############positive Gaussian process#######################
if ind=='passive':
    TraX_t = torch.Tensor(TraX)
    TraY_P_t = torch.Tensor(TraY_P)
    feature_expand = gp_struc.Feature_Augument(data_dim=TraX_t.shape[1], aug_feature1=ts)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=ts, rank=1)
    gpr = gp_struc.MultitaskGPModel(TraX_t, TraY_P_t, likelihood, ts, feature_expand).float()
    gpr.train(), likelihood.train()
    dir_c = apc+'\\gp.pth'
    gpr, likelihood = gp_struc.train(gpr, likelihood, TraX_t, TraY_P_t, dir=dir_c, load_model=0, its=800)
elif ind=='positive':
    s = 4*ts+TraX.shape[1]+1+5
    # TraX_ta = torch.Tensor(TraX_Ori)
    # TraY_ta = torch.Tensor(TraY_Ori)
    TraX_ta = torch.Tensor(TraX)
    TraY_ta = torch.Tensor(TraY_P)
    TraX_t = TraX_ta[:s]
    TraY_t = TraY_ta[:s]
ValX_t = torch.Tensor(ValX)
gpr.eval(), likelihood.eval()
Gp_out_mean = likelihood(gpr(TraX_t.float())).mean.detach().numpy()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gpr(TraX_t.float()))
lower, upper = observed_pred.confidence_region()
quan05_value = lower.detach().numpy()
quan95_value = upper.detach().numpy()
Y_Hat_F = mpn.rescale_f(np.dot(Gp_out_mean, p_inv), sy)
Y_Hat_F05 = mpn.rescale_f(np.dot(quan05_value, p_inv), sy)
Y_Hat_F95 = mpn.rescale_f(np.dot(quan95_value, p_inv), sy)
vom.plot_mwd_animation(np.squeeze(chain), Y_Hat_F, TraY_Ori, Y_Hat_F05, Y_Hat_F95, 'train')

Gp_out_mean = likelihood(gpr(ValX_t.float())).mean.detach().numpy()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gpr(ValX_t.float()))
lower, upper = observed_pred.confidence_region()
quan05_value = lower.detach().numpy()
quan95_value = upper.detach().numpy()
Y_Hat_F = mpn.rescale_f(np.dot(Gp_out_mean, p_inv), sy)
Y_Hat_F05 = mpn.rescale_f(np.dot(quan05_value, p_inv), sy)
Y_Hat_F95 = mpn.rescale_f(np.dot(quan95_value, p_inv), sy)
vom.plot_mwd_animation(np.squeeze(chain), Y_Hat_F, ValY_Ori, Y_Hat_F05, Y_Hat_F95, 'test')
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

endtime = datetime.datetime.now()
print (f"Total training time: {(endtime - starttime).seconds:d}")
# print("error"+str(SS))



plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')