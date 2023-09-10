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
import vae_model
import visualization_of_mwd as vom
import similarity_search as ss
import positive_search_v2 as ps
from skopt.space import Space
from skopt.sampler import Grid

import datetime

path = os.getcwd()
plt.ion()
plt.show()
starttime = datetime.datetime.now()
# str_n = 'MWD'
str_n = 'CSD'
apx = os.path.join(path,'Data','result_set','model_'+str_n)
load_model = 1
data_dict = mdp.generate_scenario1(str_n)
def case1():
    X_Sim = data_dict['x_sim']
    Y_Sim = data_dict['y_sim']
    chain = data_dict['chain']
    Lbb = np.array([0.1, 35., 103., 0.000015, 0.01, 328.])
    Ubb = np.array([1., 150., 300., 0.000055, 0.05, 378.])
    hidden = 5
    pcom = 7
    h1 = 60
    h2 = 32
    str_x = 'Chain Length'
    threshold = 2
    ns = 200
    return X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x, Lbb, Ubb, threshold, ns
def case2():
    X_Sim = data_dict['x_sim']['input']
    Y_Sim = data_dict['y_sim']['output']
    chain = np.linspace(0, 4, 400)[:,None]
    Lbb = np.array([4.1, 0.05, 1.0])
    Ubb = np.array([5.0, 0.09, 3.0])
    hidden = 2
    pcom = 4
    h1 = 150
    h2 = 55
    str_x = 'Length(mm)'
    threshold = 1
    ns = 50
    return X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x, Lbb, Ubb, threshold, ns
switch = {'MWD': case1,
          'CSD': case2}
X_Sim, Y_Sim, chain, hidden, pcom, h1, h2, str_x, Lbb, Ubb, threshold, ns = switch[str_n]()
Md = Y_Sim.shape[1]
sx = {'min': torch.Tensor(Lbb), 'max': torch.Tensor(Ubb)}
rpca = ra.R_pca(Y_Sim)
LowY, SparseY = rpca.fit(0.005, max_iter=10000, iter_print=100)
abn_p = np.count_nonzero(SparseY, axis=1)
lower_q = np.quantile(np.sort(abn_p), 0.25,)
higher_q = np.quantile(np.sort(abn_p), 0.75,)
iqr = higher_q-lower_q
thre = higher_q+1.5*iqr
X_Sim_Valid = torch.Tensor(X_Sim[abn_p<=thre, :])
Y_Sim_Valid = torch.Tensor(Y_Sim[abn_p<=thre, :])
#########################delete abnormal data using rpca############################
np.random.seed(0)
zipped = zip(Lbb, Ubb)
space = Space(list(zipped))
grid = Grid(border="include", use_full_layout=False)
n_samples = 12*len(Lbb)
tf = round(0.8*X_Sim_Valid.shape[0])
ts = np.random.randint(1,X_Sim_Valid.shape[0], size=X_Sim_Valid.shape[0])
TraX = X_Sim_Valid[ts[0:tf],:]
TraY = Y_Sim_Valid[ts[0:tf],:]
sample_x = torch.Tensor(np.asarray(grid.generate(space.dimensions, n_samples)))
seq_all = []
Smpx, Smpy, seq_all = ss.search(TraX, TraY, sample_x, seq_all)
ValX = Smpx[2*len(Lbb):]
ValY = Smpy[2*len(Lbb):]
TesX = X_Sim_Valid[ts[tf:],:]
TesY = Y_Sim_Valid[ts[tf:],:]
vae = vae_model.VAE(z_dim=hidden, Md=Md, hidden_dim=h1, hidden_dim2=h2)
feature_expand = gp_vae.Feature_Augument(data_dim=TraX.shape[1], aug_feature1=hidden)
likelihood_vae = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=hidden, rank=1)
loss = []
# dir = apx+'_vae.pth'
# vae = vae_model.train(vae, TraY_Zo, ValY_Zo, dir=dir, load_model=load_model)
#########################gp_vae positive modeling############################
if not load_model:
    seq = seq_all[:2*len(Lbb)]
    Dx = Smpx[:2*len(Lbb)]
    Dy = Smpy[:2*len(Lbb)]
    sy_zo = mp.scale_f_zo(Dy)
    Dy_Zo = sy_zo['val']
    vae = vae_model.train(vae, Dy_Zo, load_model=0, its=ns)
    train_x = mp.scale_f_zo(Dx, sx)['val']
    h_feature = vae.encoder(Dy_Zo)[0].detach()
    sf = mp.scale_f(h_feature)
    train_y = sf['val']
    gprvae = gp_vae.MultitaskGPModel(train_x, train_y, likelihood_vae, hidden, feature_expand)
    gprvae.train(), likelihood_vae.train()
    gprvae, likelihood_vae = gp_vae.train(gprvae, likelihood_vae, train_x, train_y, load_model=0, its=ns)
    gprvae.eval(), likelihood_vae.eval()
    epoch = 0
    sse = 30
    while sse>threshold:
        opt_x = ps.search(gprvae, vae, sy_zo, sf, sx)
        opt_x = opt_x.unsqueeze(0)
        Dx, Dy, seq = ss.search(TraX, TraY, opt_x, seq)
        sy_zo = mp.scale_f_zo(Dy)
        Dy_Zo = sy_zo['val']
        vae = vae_model.train(vae, Dy_Zo, load_model=0, its=ns)
        train_x = mp.scale_f_zo(Dx, sx)['val']
        h_feature = vae.encoder(Dy_Zo)[0].detach()
        sf = mp.scale_f(h_feature)
        train_y = sf['val']
        gprvae.set_train_data(train_x, train_y, strict=False)
        gprvae.train(), likelihood_vae.train()
        gprvae, likelihood_vae = gp_vae.train(gprvae, likelihood_vae, train_x, train_y, load_model=0, its=ns)
        gprvae.eval(), likelihood_vae.eval()
        epoch+=1
        val_x = mp.scale_f_zo(ValX ,sx)['val']
        observed_pred = gprvae(val_x)
        h_pred = mp.rescale_f(observed_pred.mean, sf)
        y_pred = vae.decoder(h_pred)[0]
        Gp_Vae = mp.rescale_f_zo(y_pred, sy_zo).detach()
        sse = (torch.norm(Gp_Vae-ValY)**2).item()
        print("Current epoch is: %03d" % (epoch))
        print('error is :%02.2f' % (sse))
        loss.append(sse)
        if sse<10:
            ns = 100
    Dx = torch.vstack([Dx, ValX])
    Dy = torch.vstack([Dy, ValY])
    torch.save(Dx, path+r'\Data\result_set\Data_x.pt')
    torch.save(Dy, path+r'\Data\result_set\Data_y.pt')
    np.savetxt(path+r'\Data\result_set\loss.txt', loss)
    sy_zo = mp.scale_f_zo(Dy)
    Dy_Zo = sy_zo['val']
    dir = apx+'_vae_po.pth'
    vae = vae_model.train(vae, Dy_Zo, load_model=0, dir=dir, its=50)
    train_x = mp.scale_f_zo(Dx, sx)['val']
    h_feature = vae.encoder(Dy_Zo)[0].detach()
    sf = mp.scale_f(h_feature)
    train_y = sf['val']
    dir = apx+'_gpvae_po.pth'
    gprvae.set_train_data(train_x, train_y, strict=False)
    gprvae.train(), likelihood_vae.train()
    gprvae, likelihood_vae = gp_vae.train(gprvae, likelihood_vae, train_x, train_y, load_model=0, dir=dir, its=50)
    gprvae.eval(), likelihood_vae.eval()
else:
    loss = np.loadtxt(path+r'\Data\result_set\loss.txt')
    Dx = torch.load(path+r'\Data\result_set\Data_x.pt')
    Dy = torch.load(path+r'\Data\result_set\Data_y.pt')
    train_x = mp.scale_f_zo(Dx, sx)['val']
    sy_zo = mp.scale_f_zo(Dy)
    Dy_Zo = sy_zo['val']
    dir = apx+'_vae_po.pth'
    vae = vae_model.train(vae, Dy_Zo, load_model=load_model, dir=dir)
    h_feature = vae.encoder(Dy_Zo)[0].detach()
    sf = mp.scale_f(h_feature)
    train_y = sf['val']
    gprvae = gp_vae.MultitaskGPModel(train_x, train_y, likelihood_vae, hidden, feature_expand)
    dir = apx+'_gpvae_po.pth'
    gprvae.train(), likelihood_vae.train()
    gprvae, likelihood_vae = gp_vae.train(gprvae, likelihood_vae, train_x, train_y, load_model=load_model, dir=dir)
    gprvae.eval(), likelihood_vae.eval()
fig, ax = plt.subplots(1,1)
ax.plot(loss)
ax.set(xlabel='sample', ylabel='sse', title='sum square error on validation set per sample')

test_x = mp.scale_f_zo(ValX, sx)['val']
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

vom.plot_mwd_animation(np.squeeze(chain), Gp_Vae, ValY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gpvae_positive_val')

test_x = mp.scale_f_zo(TesX, sx)['val']
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



vom.plot_mwd_animation(np.squeeze(chain), Gp_Vae, TesY.numpy(), Y_Hat_F05, Y_Hat_F95, 'gpvae_positive_te')
endtime = datetime.datetime.now()
print (f"Total training time: {(endtime - starttime).seconds:d}")

plt.draw()
plt.pause(0.01)
input("Press [enter] to close all the figure windows.")
plt.close('all')