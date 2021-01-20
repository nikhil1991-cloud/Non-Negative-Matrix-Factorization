import numpy as np

def NNMF_MU(A,r,epochs,initial_W=None,initial_H=None):
    '''
    Runs multiplicative updates (MU) to perform non-negative matrix factorization on A.
    Problem: To minimize ||A - WH||_F (where ||    ||_F is the frobeneus norm) w.r.t W,H such that W,H>=0.
    
    Refer to https://papers.nips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf for details about multiplicative updates.
    
    Inputs:
          A: ndarray - m by n matrix to factorize.
          r: int - specifies the column size for W and row size for H s.t W is of shape m by r and H is of shape r by n (r < min(m,n) always).
          epochs: int - specifies the number of iterations to be run.
          initial_W: ndarray - m by r matrix for initial W (if None is specified, W is initialised with random numbers in interval [0,1)).
          initial_H: ndarray - r by n matrix for initial H (if None is specified, H is initialised with random numbers in interval [0,1)).
          
    Returns:
          W: ndarray - m by k matrix where k = dim.
          H: ndarray - k by n matrix where k = dim.
    '''
    
    #Initialize W and H
    if initial_W is None:
       W = np.random.rand(np.shape(A)[0], r)
    else:
       W = initial_W
    if initial_H is None:
       H = np.random.rand(r, np.shape(A)[1])
    else:
       H = initial_H

    frob_norm = [np.linalg.norm(A - np.matmul(W,H))]
    #Factorization
    for num in range (0,epochs):
        H = H*(np.matmul(W.T,A)/np.matmul(np.matmul(W.T,W),H))
        W = W*(np.matmul(A,H.T)/np.matmul(W,np.matmul(H,H.T)))
        frob_norm.append(np.linalg.norm(A - np.matmul(W,H)))
    return W,H,frob_norm

