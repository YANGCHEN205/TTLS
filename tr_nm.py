import numpy as np
import tensorly as tl


def svt_tsn(mat, tau):
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    vec = s.copy()
    vec = s - tau
    vec[vec < 0] = 0
    return u @ np.diag(vec) @ v

def nuclear_norm(mat):
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    nm = s.sum()
    return nm
def inverseorder(order):
    # Calculate the inverse permutation order
    inverse_order = np.argsort(order)

    # Add 1 to the result to get the MATLAB-style index (1-based indexing)
    inverse_order += 1

    return inverse_order-1
def prox_tr_nuclear_norm(B, lambda_, alpha=None):
    # Proximal operator for tensor nuclear norm regularization

    # Input
    #     B: a D-th tensor of size I_1 * ... * I_D
    #     lambda_: a weight factor balancing the contribution of the nuclear norm term
    #               and the approximation error
    #     alpha: a vector of weight factors balancing the contribution of the nuclear
    #            norm of each mode. The default setting is 1/D for each mode if not assigned.

    # Output
    #     A: a low tensor ring rank approximation for B

    # Size and order of B
    dimB = B.shape
    D = len(dimB)
    L = int(np.ceil(D / 2))

    # Check if alpha exists
    if alpha is None:
        alpha = np.full(L, 1/L)

    # Check if alpha is a scalar
    if np.isscalar(alpha):
        alpha = np.full(L, alpha)

    # Check if the summation of alpha equals to 1
    if not np.isclose(np.sum(alpha), 1):
        alpha /= np.sum(alpha)

    if D == 2:
        return svt_tsn(B, lambda_)

    A = np.array(np.zeros(dimB) + 1j*np.zeros(dimB),dtype=B.dtype)

    for d in range(L):
        # Update m^(n)
        order = np.concatenate((np.arange(d, D), np.arange(d)))
        Z_temp = np.transpose(B, order) 
        dimz=Z_temp.shape
        Z = Z_temp.reshape(np.prod(dimz[:L]), -1)
        tau = alpha[d] * lambda_
        M = svt_tsn(Z, tau)
        M_temp = np.reshape(M, dimz)
        iorder=inverseorder(order)
        A += np.transpose(M_temp, iorder)

    A /= L
    return A
def tr_nuclear_norm(B,alpha=None):
    # tensor nuclear norm 

    # Input
    #     B: a D-th tensor of size I_1 * ... * I_D
    #     lambda_: a weight factor balancing the contribution of the nuclear norm term
    #               and the approximation error
    #     alpha: a vector of weight factors balancing the contribution of the nuclear
    #            norm of each mode. The default setting is 1/D for each mode if not assigned.

    # Output
    #     A: a low tensor ring rank approximation for B

    # Size and order of B
    dimB = B.shape
    D = len(dimB)
    L = int(np.ceil(D / 2))

    # Check if alpha exists
    if alpha is None:
        alpha = np.full(L, 1/L)

    # Check if alpha is a scalar
    if np.isscalar(alpha):
        alpha = np.full(L, alpha)

    # Check if the summation of alpha equals to 1
    if not np.isclose(np.sum(alpha), 1):
        alpha /= np.sum(alpha)

    if D == 2:
        return nuclear_norm(B)

    tnm=0;

    for d in range(L):
        # Update m^(n)
        order = np.concatenate((np.arange(d, D), np.arange(d)))
        Z_temp = np.transpose(B, order) 
        dimz=Z_temp.shape
        Z = Z_temp.reshape(np.prod(dimz[:L]), -1)
        znm = alpha[d] * nuclear_norm(Z)
        tnm += znm


    return tnm

def tr_rand(sz,R,dtype):
    D=len(sz)
    U= []
    for d in range(D):
        if dtype == 'complex64':
           U.append(np.array(np.random.normal(size=[R,sz[d],R]) + 1j*np.random.normal(size=[R,sz[d],R]), dtype='complex64'))
    X=tr_to_tensor(U)      
    return X

def tr_to_tensor(factors):
    """Returns the full tensor whose TR decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in TR format
        into the corresponding full tensor

    Parameters
    ----------
    factors : list of 3D-arrays
              TR factors (TR-cores)

    Returns
    -------
    output_tensor : ndarray
                   tensor whose TR decomposition was given by 'factors'
    """
    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (-1, factors[0].shape[2]))

    for factor in factors[1:-1]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    full_tensor = tl.reshape(
        full_tensor, (factors[-1].shape[2], -1, factors[-1].shape[0])
    )
    full_tensor = tl.moveaxis(full_tensor, 0, -1)
    full_tensor = tl.reshape(
        full_tensor, (-1, factors[-1].shape[0] * factors[-1].shape[2])
    )
    factor = tl.moveaxis(factors[-1], -1, 1)
    factor = tl.reshape(factor, (-1, full_shape[-1]))
    full_tensor = tl.dot(full_tensor, factor)
    return tl.reshape(full_tensor, full_shape)


# Helper function to compute tensor nuclear norm (similar to the nuclear_norm function in tensorly)
def tensor_nuclear_norm(X):
    return np.linalg.norm(X, ord='nuc')


