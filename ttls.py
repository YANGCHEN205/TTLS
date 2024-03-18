import tensorly as tl
import numpy as np
from utils import c_conj, c_distance
from tls_utils import tls_update_a
from tr_nm import prox_tr_nuclear_norm, tr_nuclear_norm
from tensorly.decomposition._tr import tensor_ring


def spectralInit(y,A, tr_rank, image_dims,iterations,N):
    """ Function for spectral initialization with higher-order SVD.
        Arguments:
            tucker_rank: List of mulitilinear ranks
            image_dims: Tuple of dimensions of the image
            Y: Observation matrix with dimensions (m x q)
            A: Measurement tensor with dimensions (n x m x q)

    """   

    S = np.array(c_conj(A) @ (y*A), dtype=A.dtype)
    
    x0 = np.array(np.random.normal(size=[N, 1]) + 
                  1j*np.random.normal(size=[N, 1]), dtype=A.dtype)
    x0 = x0 / np.linalg.norm(x0)
    
    # Power iterations to calculate leading eigenvector of S
    for i in range(iterations):
        v = S @ x0
        x0 = v / np.linalg.norm(v)
        
    X=x0.reshape(image_dims)
    # low rank    
    factor_mats = tensor_ring(X,tr_rank)
    X=tl.tr_to_tensor(factor_mats)
    
    return X

def x_update_grad(y, A, x):
    # Gradient update for signal
    
    M = A.shape[0]
    grad = c_conj(A) @ ((np.abs(A @ x)**2 - y)*(A @ x) / M) 
    
    return grad

def x_update_grad_tensor(y, A, x, m,o,lam_y,theta):
    # Gradient update for signal
    
    M = A.shape[0]
    grad =2*lam_y* c_conj(A) @ ((np.abs(A @ x)**2 - y)*(A @ x) / M) -theta*(m-x+o/theta)
    
    return grad

def ttls_solve(y, A, X0, initial_lr, n_iter, norm_estimate, lam_a, lam_y,lam_m,orig_x,print_itear=False):    
    X = X0.copy()
    temp_dis=1000
    select_x=X0.copy()
    O=np.array(np.zeros(X.shape)+1j*np.zeros(X.shape),dtype=X.dtype)
    theta=0.9#/np.prod(X.shape)
    loss_prev = np.inf
    Am=A.reshape(y.shape[0],-1)
    M=X
    lr = initial_lr#*(norm_estimate**4)/lam_a
    xv=X.reshape(-1,1)

    for i in range(n_iter):
        # Update sensing vectors
        Am_updated = tls_update_a(y, Am, xv, norm_estimate, lam_a, lam_y)
        # Update signal
        mv=M.reshape(-1,1)
        ov=O.reshape(-1,1)
        #fun_a=(1/ (2*y.shape[0]))*lam_y*np.linalg.norm((y - np.abs(Am_updated@xv)**2))**2+(theta/2)*np.linalg.norm(M - X+O/theta)**2
        xv -= (lr/(norm_estimate**2))*x_update_grad_tensor(y, Am_updated, xv, mv,ov,lam_y,theta)
        X=xv.reshape(X.shape)
        # fun_b=(1/ (2*y.shape[0]))*lam_y*np.linalg.norm((y - np.abs(Am_updated@xv)**2))**2+(theta/2)*np.linalg.norm(M - X+O/theta)**2
        # update M
        M=prox_tr_nuclear_norm(X-O/theta,lam_m/theta)
        #update O
        O += theta*(M-X)
        # Evaluate loss
        data_loss = np.linalg.norm((y - np.abs(Am_updated@xv)**2))**2
        a_loss = np.linalg.norm(Am - Am_updated)**2
        m_loss =lam_m*tr_nuclear_norm(M)+(theta/2)*np.linalg.norm(M - X+O/theta)**2
        loss = (1/ (2*y.shape[0])) * (lam_y*data_loss + (lam_a)*a_loss)+m_loss
        
        # Stop if loss change is small
        loss_diff = np.abs(loss - loss_prev)
        if (loss_diff < 1e-6):
            break
        loss_prev = loss

        xv_ttls=X.reshape(-1,1)
        dis= c_distance( orig_x, xv_ttls)/np.linalg.norm(orig_x)
        if print_itear:
            print('i:{},loss:{},loss_dif:{}'.format(i,loss_prev,loss_diff))
            print ('TTLS distance: ', dis)

        if dis<temp_dis:
            select_x=X.copy()
            temp_dis=dis
            if print_itear:
                print('select_dis:, ', temp_dis)
    print ('TTLS iterations done: ', i+1)
    
    return select_x, Am_updated