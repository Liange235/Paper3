import numpy as np
import scipy.io as io_mat
def generate_scenario1(str_n):
    dir1 = 'Data/'+str_n+'_X'
    dir2 = 'Data/'+str_n+'_Y'
    dir3 = 'Data/chain'
    if str_n == "CSD":
        x_sim = io_mat.loadmat(dir1)
        y_sim = io_mat.loadmat(dir2)
        chain = io_mat.loadmat(dir3)['chain2']
    else:
        x_sim = np.loadtxt(dir1+'.txt')
        y_sim = np.loadtxt(dir2+'.txt')
        chain = io_mat.loadmat(dir3)['chain2']
    return {'x_sim': x_sim, 'chain': chain, 'y_sim': y_sim}
def generate_scenario2():
    """
    hybrid grey-box modeling

    return {'y_sim': y_sim, 'y_ind_sim': y_ind_sim, 't_sim': t_sim,
            'y_obs': y_obs, 'y_ind_obs': y_ind_obs, 't_obs': t_obs}
    """
def scale(inputx):
    """
    zero mean and unit standard deviation normalization
    """
    mu = np.mean(inputx, axis=0)
    sigma = np.std(inputx, axis=0)
    x_scale = (inputx-mu)/(sigma + 1.0e-6)
    return {'norm': x_scale, 'mu': mu, 'sigma': sigma}

def rescale(inputx, mu, sigma):
    x_rescaled = inputx*sigma+mu
    return x_rescaled
