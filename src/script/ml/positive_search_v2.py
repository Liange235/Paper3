import numpy as np
import torch
import Model_preparation as mp
from botorch.utils.sampling import draw_sobol_samples
from scipy.optimize import minimize
from scipy.optimize import Bounds
def objective(gpr, vae, x, sy, sf):
    if torch.is_tensor(x):
        xt = x
    else:
        xt = torch.tensor(x, requires_grad=True).unsqueeze(0).float()
    ob_pred = gpr(xt)
    ob_std = ob_pred.variance.sqrt()
    h_std = mp.rescale_f(ob_std, sf)
    y_std = vae.decoder(h_std)[0]
    obj = torch.sum(mp.rescale_f_zo(y_std, sy), dim=1)
    return obj
def gradient(gpr, vae, x, sy, sf):
    xt = torch.tensor(x, requires_grad=True).unsqueeze(0).float()
    fi = objective(gpr, vae, xt, sy, sf)
    gt = torch.autograd.grad(fi, xt, retain_graph=True)[0]
    return gt.squeeze().numpy()
def hessian(gpr, vae, x, sy, sf):
    xt = torch.tensor(x, requires_grad=True).unsqueeze(0).float()
    fi = objective(gpr, vae, xt, sy, sf)
    grad_b = torch.autograd.grad(fi, xt, create_graph=True)[0][0]
    ht = torch.zeros(xt.shape[1], xt.shape[1])
    for i, _ in enumerate(grad_b):
        gi = torch.autograd.grad(grad_b[i], xt, retain_graph=True)[0]
        ht[i] = gi
    return ht.numpy()
def search(gpr, vae, sy, sf, sx):
    # torch.manual_seed(0)
    obj_mwd = lambda x: -objective(gpr, vae, x, sy, sf).detach().numpy()
    grad = lambda x: gradient(gpr, vae, x, sy, sf)
    hess = lambda x: hessian(gpr, vae, x, sy, sf)
    m = gpr.train_inputs[0].shape[1]
    bounds = torch.Tensor(2,m)
    bounds[0] = torch.zeros(m)
    bounds[1] = torch.ones(m)
    bounds_sci = Bounds(np.zeros(m), np.ones(m))
    nsamples = 1
    ops = {"disp": True}
    opx = []
    opjac = []
    for _ in np.arange(nsamples):
        x_rnd = draw_sobol_samples(bounds=bounds, n=100, q=50).view(-1,m)
        fv = obj_mwd(x_rnd)
        min_ids = np.argmin(fv)
        x0 = x_rnd[min_ids].numpy()
        res = minimize(obj_mwd, x0, method='SLSQP', jac=grad, bounds=bounds_sci, options=ops)
        opx.append(res.x)
        n_gra = np.linalg.norm(res.jac)
        opjac.append(n_gra)
    idx = np.argmin(opjac)
    mx = opx[idx]
    gi = grad(mx)
    # hi = hess(mx)
    n_gra = np.linalg.norm(gi)
    # e_vals = np.linalg.eig(hi)[0]
    # pv = np.sum(e_vals>=1.0e-6)
    print(f"Gradient is: {n_gra:.4f}")
    # print(f"Positive egv is: {pv:d}")
    opx_t = mp.rescale_f_zo(torch.Tensor(mx), sx).detach()
    return opx_t
        