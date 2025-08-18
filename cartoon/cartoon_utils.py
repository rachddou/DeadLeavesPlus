import numpy as np
from time import time
from numba import jit, njit

def chambolle_tv_l1(channel, nb_iter_max, tau, sigma, lambda_value, theta):

    u = channel.copy()
    print (u.shape)
    
    p1 = np.zeros(channel.shape)
    p2 = np.zeros(channel.shape)

        
    E = 0
    for it in range(nb_iter_max):
        u_old = u.copy()
        
        #compute the gradient
        t0 = time()
        gradx= u[1:,:] - u[:-1,:]
        grady = u[:,1:] - u[:,:-1]
        t1 = time()
        # print(f"Gradient computation time: {t1 - t0:.4f} seconds")

        gradx = np.pad(gradx, ((0, 1), (0, 0)), mode='constant')
        grady = np.pad(grady, ((0, 0), (0, 1)), mode='constant')
        
        # compute p
        tmp1 = p1+sigma*gradx
        tmp2 = p2+sigma*grady
        p1,p2 = prox_f_star(tmp1,tmp2)

        # t2 = time()
        # print(f"Proximal operator computation time: {t2 - t1:.4f} seconds")
        #compute u
        div = compute_Divergence(p1, p2)
        
        # t3 = time()
        # print(f"Divergence computation time: {t3 - t2:.4f} seconds")
        tmp = u + tau*div
        tmp = prox_g(tmp, channel,lambda_value, tau)
        
        # t4 = time()
        # print(f"Proximal operator g computation time: {t4 - t3:.4f} seconds")
        
        u = tmp+ theta * (tmp- u_old)
        
        # t5 = time()
        # print(f"Update step time: {t5 - t4:.4f} seconds")
        
        # print(f"full iteration time: {t5 - t0:.4f} seconds")
        
        
        if it%50 == 0:
            E_old = E
            # test_time = time()
            E = lambda_value*np.mean(np.abs(u - channel)) + np.mean(np.sqrt((gradx**2 + grady**2)))

            print(f"Energy: {E_old - E}")
            if abs(E_old - E) < 3e-4:
                print(f"Converged at iteration {it}")
                break
            # print(f"test time: {time()-test_time:.4f} seconds")
    return u

@njit(parallel=True )
def prox_f_star(im1, im2):
    
    norm_p = np.maximum(1.0,np.sqrt(im1**2 + im2**2))
    p1 = im1 / norm_p  
    p2 = im2 / norm_p 

    return p1, p2

@njit(parallel=True)
def prox_g(im1, channel, lambda_value, tau):
    diff_map = im1-channel
    output = np.zeros(im1.shape)
    
    ind1 = diff_map >lambda_value * tau
    ind2 = diff_map < -lambda_value * tau
    ind3 = np.abs(diff_map) < lambda_value * tau

    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            if diff_map[i, j] > lambda_value * tau:
                output[i, j] = im1[i, j] - lambda_value * tau
                
            elif diff_map[i, j] < -lambda_value * tau:
                output[i, j] = im1[i, j] + lambda_value * tau
            else:
                output[i, j] = channel[i, j]
    # output[ind1] = im1[ind1] - lambda_value * tau
    # output[ind2] = im1[ind2] + lambda_value * tau
    # output[ind3] = channel[ind3]

    return output

# @njit(parallel=True)
# def test(u,channel,gradx,grady,lambda_value):
#     E = lambda_value*np.sum(np.abs(u - channel)) + 0.5 * np.sum(np.sqrt((gradx**2 + grady**2)))
#     E = np.mean(E)
#     print(f"Energy: {E}")
#     return E

def compute_Divergence(p1, p2):
    
    new_p1 = np.pad(p1, ((1, 0), (0, 0)), mode='constant')
    new_p2 = np.pad(p2, ((0, 0), (1, 0)), mode='constant')

    div = new_p1[1:, :] - new_p1[:-1, :] + new_p2[:, 1:] - new_p2[:, :-1]
    return div



    