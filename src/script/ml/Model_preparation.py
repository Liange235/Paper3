import torch
import numpy as np
# BRANIN 2-DIMENSIONAL FUNCTION
def f1(x):
    x1 = x[...,0]
    x2 = x[...,1]
    a,b,c,r,s,t = 1,5.1/(4*torch.pi**2),5/torch.pi,6,10,1/(8*torch.pi)
    fun = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*torch.cos(x1)+s
    return fun
# HARTMANN 3-DIMENSIONAL FUNCTION
def f2(x):
    alpha = torch.Tensor([1., 1.2, 3., 3.2]).double()
    A = torch.Tensor([[3., 10., 30.],[0.1, 10., 35.],[3., 10., 30.],[0.1, 10., 35.]]).double()
    P = 1.0e-4*torch.Tensor([[3689., 1170., 2673.],[4699., 4387., 7470.],[1091., 8732., 5547.],[381., 5743., 8828.]]).double()
    t1 = torch.zeros(4).double()
    exp = lambda x:x.squeeze(0) if x.ndim>1 else x
    x = exp(x)
    for _ in np.arange(len(t1)):
        t1[_] = -torch.matmul(A[_], (x-P[_])**2)
    fun = -torch.matmul(alpha, torch.exp(t1))
    return fun
# HARTMANN 6-DIMENSIONAL FUNCTION
def f3(x):
    alpha = torch.Tensor([1., 1.2, 3., 3.2]).double()
    A = torch.Tensor([[10, 3, 17, 3.5, 1.7, 8],[0.05, 10, 17, 0.1, 8, 14],[3, 3.5, 1.7, 10, 17, 8],[17, 8, 0.05, 10, 0.1 ,14]]).double()
    P = 1.0e-4*torch.Tensor([[1312, 1696, 5569, 124, 8283, 5886],[2329, 4135, 8307, 3736, 1004, 9991],[2348, 1451, 3522, 2883, 3047, 6650],[4047, 8828, 8732, 5743, 1091, 381]]).double()
    t1 = torch.zeros(4).double()
    exp = lambda x:x.squeeze(0) if x.ndim>1 else x
    x = exp(x)
    for _ in np.arange(len(t1)):
        t1[_] = -torch.matmul(A[_], (x-P[_])**2)
    fun = -torch.matmul(alpha, torch.exp(t1))
    return fun
Lbb1 = torch.Tensor([-5., 0.])
Ubb1 = torch.Tensor([10., 15.])
Lbb2 = torch.Tensor([0., 0., 0.])
Ubb2 = torch.Tensor([1., 1., 1.])
Lbb3 = torch.Tensor([0., 0., 0., 0., 0., 0.])
Ubb3 = torch.Tensor([1., 1., 1., 1., 1., 1.])
def choose_model(id):
    m_dict = {
        1: f1,
        2: f2,
        3: f3,
    }
    method = m_dict.get(id, f1)
    return method
def choose_bound(id):
    b_dict = {
        1: [Lbb1, Ubb1],
        2: [Lbb2, Ubb2],
        3: [Lbb3, Ubb3],
    }
    bound = b_dict.get(id, [Lbb1, Ubb1])
    return bound
#zero mean_unit variance normalize and renormalize
def scale_f(x, input=None):
    if input:
        mu = input['mu']
        sigma = input['sigma']
        x_scale = (x-mu)/(sigma + 1.0e-9)
    else:
        mu = torch.mean(x, axis=0)
        sigma = torch.std(x, axis=0)
        x_scale = (x-mu)/(sigma + 1.0e-9)
    return {'val': x_scale, 'mu': mu, 'sigma': sigma}
def rescale_f(x, input):
    mu = input['mu']
    sigma = input['sigma']
    x_rescaled = x*sigma+mu
    return x_rescaled
#min_max normalize and renormalize    
def scale_f_zo(x, input=None):
    if input:
        minv = input['min']
        maxv = input['max']
        x_scale = (x-minv)/(maxv-minv + 1.0e-9)
    else:
        minv = torch.min(x, axis=0).values
        maxv = torch.max(x, axis=0).values
        x_scale = (x-minv)/(maxv-minv + 1.0e-9)
    return {'val': x_scale, 'min': minv, 'max': maxv}
def rescale_f_zo(x, input):
    minv = input['min']
    maxv = input['max']
    x_rescaled = x*(maxv-minv)+minv
    return x_rescaled