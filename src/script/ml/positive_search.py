import numpy as np
import torch
import gpy_model_multi as gp_struc
import mwd_data_preparation as mdp

# def del_tensor_ele(arr,index):
#     arr1 = arr[0:index]
#     arr2 = arr[index+1:]
#     return torch.cat((arr1,arr2),dim=0)

def search(X, Y, gpr, likelihood, weight, s, ValX, ValY, thre):
    TraX = X[:s]
    TraY = Y[:s]
    pcom2 = weight.shape[0]
    # x_s = torch.Tensor(mdp.scale(X)['norm'])
    weight = torch.Tensor(weight[:,None])
    Xt = torch.Tensor(X)
    losses = []
    loss = 800
    ValX_t = torch.Tensor(ValX)
    n = ValX.shape[0]*4
    while loss >= thre:
        gpr.eval(), likelihood.eval()
        y_std = likelihood(gpr(Xt.float())).variance**(0.5)
        gpr.eval(), likelihood.eval()
        Gp_out_mean = likelihood(gpr(ValX_t.float())).mean.detach().numpy()
        loss = np.linalg.norm(ValY-Gp_out_mean, ord='fro')**2
        obj = torch.matmul(y_std,weight)
        sorted, s_ind = torch.sort(obj, axis=0, descending=True)
        ind = s_ind[0]
        nx = X[ind.item()]
        ny = Y[ind.item()]
        if TraX.shape[0]>=n-1:
            break
        TraX = np.vstack([TraX, nx])
        TraY = np.vstack([TraY, ny])
        X = np.delete(X, ind, axis=0)
        Y = np.delete(Y, ind, axis=0)
        Xt = torch.Tensor(X)
        # TraX_std = torch.Tensor(mdp.scale(TraX)['norm'])
        # TraY_std = mdp.scale(TraY)['norm']
        TraX_std = torch.Tensor(TraX)
        # _, _, V_f_Valid = np.linalg.svd(TraY_std, full_matrices=False)
        # pro = V_f_Valid[0:pcom2, :].T
        # TraY_p = torch.Tensor(np.dot(TraY_std, pro))
        TraY_p = torch.Tensor(TraY)
        gpr.set_train_data(TraX_std, TraY_p, strict=False)
        gpr.train(), likelihood.train()
        gp_struc.train(gpr, likelihood, TraX_std, TraY_p)
        losses.append(loss)
        print(f"Current iteration: {TraX.shape[0]-s:d}")
    return TraX, TraY, gpr, losses
        
    
    
