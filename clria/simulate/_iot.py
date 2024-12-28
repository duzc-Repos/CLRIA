import numpy as np
from sklearn.decomposition import non_negative_factorization
from sklearn.utils.extmath import randomized_svd

def generate_coupling_matrix(n_row, n_col):
    """Generate coupling matrix for ligand or receptor

    Parameters
    ----------
    n_row : int
        Number of LR pairs
    n_col : int
        Number of ligand or receptor

    Returns
    -------
    (n_row, n_col) numpy.ndarray
        The binary matrix with at least non-zero element in each row and column.
        It records the composed relationship between LR pair and its ligand or receptor.
    """
    tmp1 = np.identity(n_col)
    tmp2 = np.random.random( size=(n_row-n_col, n_col) )
    tmp2 = (tmp2 >= tmp2.max(axis=1, keepdims=True)).astype(int)
    T = np.concatenate( (tmp1, tmp2), axis=0 )
    
    idx_row, idx_col = np.arange(n_row), np.arange(n_col)
    np.random.shuffle(idx_row)
    np.random.shuffle(idx_col)

    return T[idx_row, :][:, idx_col]

def simu_data(n_r=10, n1=123, n2=None, n_lr=500, 
              TL=None, TR=None,
              scale_factor=1e4, epsilon=1, 
              is_decomposition=False, n_d = 10, solver="nmf", eps=1e-100):
    """Generate systhesis data using inverse optimal transport (iOT)

    Parameters
    ----------
    n_r : int, optional
        Number of rank, by default 10
    n1 : int, optional
        Number of sender regions, by default 123
    n2 : None or int, optional
        Number of receiver regions, by default None
    n_lr : int, optional
        Number of LR pairs, by default 500
    TL : None, int or (n_lr, n_ligand) numpy.ndarray , optional
        The ligand coupling matrix (a binary matrix). 
        If None, it is a identity matrix; If int, it will generated a (n_lr, TL) binary matrix;
        If numpy.ndarray, use it for following generation. by default None
    TR : None, int or (n_lr, n_receptor) numpy.ndarray , optional
        The receptor coupling matrix (a binary matrix).
        If None, it is a identity matrix; If int, it will generated a (n_lr, TR) binary matrix;
        If numpy.ndarray, use it for following generation. by default None
    scale_factor : float or int, optional
        The factor to normalize the generated transport cost matrix M, by default 1e4
    epsilon : int, optional
        The coefficient for entropy regularization of generated transport cost matrix M, by default 1
    is_decomposition : bool, optional
        Whether to decompose transport cost matrix M, by default False
    n_d : int, optional
        Number of components, by default 10
    solver : str ("nmf" or "svd"), optional
        Numerical solver to perform matrix factorization of M, by default "nmf"
    eps : _type_, optional
        The coefficient to avoid zero in tensor factor A, B, C, by default 1e-100

    Returns
    -------
    n_r : int
        Number of rank. The same as input
    A : (n1, n_r) numpy.ndarray
        The generated sender loading matrix.
    B : (n2, n_r) numpy.ndarray
        The generated sender loading matrix.
    C : (n_lr, n_r) numpy.ndarray
        The generated sender loading matrix.
    L : (n1, n_ligand) numpy.ndarray
        The generated ligand expression matrix.
    R : (n2, n_receptor) numpy.ndarray
        The generated receptor expression matrix.
    TL : (n_lr, n_ligand) numpy.ndarray
        The generated ligand coupling matrix (a binary matrix).
    TR : (n_lr, n_receptor) numpy.ndarray
        The generated ligand coupling matrix (a binary matrix).
    M : (n1, n2) numpy.ndarray
        The transport cost matrix computed using inverse optimal transport.
    (M1, M2) : ( (n1, n_d) numpy.ndarray, (n2, n_d) numpy.ndarray ), optional
        The decomposed matrix from M. Only return when "is_decompostion is True".
    """
    ## check parameters
    if n2 is None:
        n2 = n1
    
    if TL is None:
        TL = np.identity(n=n_lr)
    elif isinstance(TL, int):
        assert TL <= n_lr, f"TL should less than or equal to n_lr, {TL}, {n_lr}"
        TL = generate_coupling_matrix(n_lr, TL)
    else:
        assert (TL.shape[0]==n_lr and TL.shape[1] <= n_lr), f"illegal data dimension, {TL.shape}, {n_lr}"
    
    if TR is None:
        TR = np.identity(n=n_lr)
    elif isinstance(TR, int):
        assert TR <= n_lr, f"TR should less than or equal to n_lr, {TR}, {n_lr}"
        TR = generate_coupling_matrix(n_lr, TR)
    else:
        assert (TR.shape[0]==n_lr and TR.shape[1] <= n_lr), f"illegal data dimension, {TR.shape}, {n_lr}"
    
    assert n_d <= max(n1, n2), "n_d should less than max{n1, n2}"

    ## true data
    A = np.random.random( size=(n1, n_r) )
    A /= A.sum(axis=0, keepdims=True)
    A = np.maximum(eps, A)
    B = np.random.random( size=(n2, n_r) )
    B /= B.sum(axis=0, keepdims=True)
    B = np.maximum(eps, B)
    C = np.maximum(eps, np.random.random( size=(n_lr, n_r) ))
    #print(A.shape, B.shape, C.shape)

    ## systhesis data
    L = np.dot(A * B.sum(axis=0, keepdims=True), np.dot(C.T, TL))
    R = np.dot(B * A.sum(axis=0, keepdims=True), np.dot(C.T, TR))
    
    ## stable softmax
    #M = np.exp(-np.dot(A * C.sum(axis=0, keepdims=True), B.T)/epsilon)
    #M = M * (scale_factor / M.sum())
    P = -np.dot(A * C.sum(axis=0, keepdims=True), B.T)/epsilon
    P -= np.max(P)
    log_M = np.log( scale_factor/np.exp(P).sum() ) + P
    M = np.exp(log_M)
    
    #print(M.shape, np.isnan(M).sum() )

    if is_decomposition:
        if solver == "nmf":
            M1, M2, _ = non_negative_factorization(M, n_components=n_d, solver="cd", tol=1e-8, max_iter=5000)
            M2 = M2.T
        elif solver == "svd":
            tmp1, s, tmp2 = randomized_svd(M, n_components=n_d)
            s = np.sqrt(s).reshape(1, -1)
            M1 = tmp1 * s
            M2 = tmp2.T * s
            print("using svd")
        else:
            pass

        return n_r, A, B, C, L, R, TL, TR, M, (M1, M2)
    else:
        return n_r, A, B, C, L, R, TL, TR, M


if __name__ == "__main__":
    
    ################################
    ## simulation test1
    ################################
    
    r, N1, N2, I_len = 20, 123, 124, 1000      # A, B, C dimension
    m, n = 300, 400                            # TL, TR, L, R dimension
    K, epsilon = 1e4, 1                        # M
    is_decom, d, solver = True, 10, "nmf"      # M1, M2

    print("### Simulation test1")
    r, A, B, C, L, R, TL, TR, M, (M1, M2) = simu_data(r, N1, N2, I_len, m, n, K, epsilon, is_decom, d, solver)
    err = np.abs(M - np.dot(M1, M2.T))
    print("True data", r, A.shape, B.shape, C.shape, sep="\t")
    print("Simu data", L.shape, R.shape, TL.shape, TR.shape, M.shape, M1.shape, M2.shape)
    print("Error", err.sum(), err.max())

    ################################
    ## simulation test2
    ################################
    #"""
    r, N1, N2, I_len = 20, 123, 124, 1000      # A, B, C dimension
    m, n = None, 400                           # TL, TR, L, R dimension
    K, epsilon = 1e4, 1                        # M
    is_decom, d, solver = True, 10, "svd"      # M1, M2

    print("### Simulation test2")
    r, A, B, C, L, R, TL, TR, M, (M1, M2) = simu_data(r, N1, N2, I_len, m, n, K, epsilon, is_decom, d, solver)
    err = np.abs(M - np.dot(M1, M2.T))
    print("True data", r, A.shape, B.shape, C.shape, sep="\t")
    print("Simu data", L.shape, R.shape, TL.shape, TR.shape, M.shape, M1.shape, M2.shape)
    print("Error", err.sum(), err.max())
    #"""

    ################################
    ## simulation test3
    ################################
    #"""
    r, N1, N2, I_len = 20, 123, 124, 1000                 # A, B, C dimension
    m, n = None, generate_coupling_matrix(I_len, 200)     # TL, TR, L, R dimension
    K, epsilon = 1e4, 1                                   # M
    is_decom, d, solver = True, 10, "nmf"                 # M1, M2

    print("### Simulation test3")
    r, A, B, C, L, R, TL, TR, M, (M1, M2) = simu_data(r, N1, N2, I_len, m, n, K, epsilon, is_decom, d, solver)
    err = np.abs(M - np.dot(M1, M2.T))
    print("True data", r, A.shape, B.shape, C.shape, sep="\t")
    print("Simu data", L.shape, R.shape, TL.shape, TR.shape, M.shape, M1.shape, M2.shape)
    print("Error", err.sum(), err.max())
    #"""

    ################################
    ## simulation test4
    ################################
    #"""
    r, N1, N2, I_len = 20, 123, 124, 1000                 # A, B, C
    m = generate_coupling_matrix(I_len, 400)              # TL, L
    n = generate_coupling_matrix(I_len, 300)              # TR, R
    K, epsilon = 1e4, 1                                   # M
    is_decom, d, solver = True, 10, "svd"                 # M1, M2

    print("### Simulation test4")
    r, A, B, C, L, R, TL, TR, M, (M1, M2) = simu_data(r, N1, N2, I_len, m, n, K, epsilon, is_decom, d, solver)
    err = np.abs(M - np.dot(M1, M2.T))
    print("True data", r, A.shape, B.shape, C.shape, sep="\t")
    print("Simu data", L.shape, R.shape, TL.shape, TR.shape, M.shape, M1.shape, M2.shape)
    print("Error", err.sum(), err.max())
    #"""

    pass

