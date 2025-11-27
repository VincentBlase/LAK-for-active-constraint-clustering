def adaptive_rbf_kernel(X, k=None, alpha=1.0, smooth=True,
                        clip_percentiles=(15,85), make_psd=True, eps=1e-10):
    """
    Compute an adaptive RBF kernel matrix K for data X with local bandwidths.
    - X: (n_samples, n_features)
    - k: neighbors used to estimate local scale. Default: max(5, min(50, int(0.01*n)))
    - alpha: multiplicative scale for sigma_i
    - smooth: median smoothing of sigma over k-neighbors
    - clip_percentiles: (low, high) percentiles to clip sigma to
    - make_psd: if True, perform eigenvalue clipping to make K PSD
    - eps: small value to avoid division by zero
    Returns:
    - K: (n,n) kernel matrix
    - sigma: (n,) local bandwidths
    """
    n, d = X.shape
    if k is None:
        k = max(5, min(200, int(0.01 * n)))
    k = min(k, n-1)

    # 1) k-NN distances
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)  # include self
    dists, idx = nn.kneighbors(X)                 # dists[:,0] == 0 (self)
 
    kth_dist = dists[:, -1]                                   

    # 2) raw sigma
    sigma = alpha * (kth_dist + eps)

    # 3) optional smoothing (median of neighbor sigmas)
    if smooth:
        neigh_sigma = sigma[idx[:, 1:]]  # exclude self
        sigma = np.median(np.concatenate([sigma.reshape(-1,1), neigh_sigma], axis=1), axis=1)

    # 4) clipping to avoid extremes
    lo, hi = np.percentile(sigma, [clip_percentiles[0], clip_percentiles[1]])
    sigma = np.clip(sigma, lo + eps, hi + eps)

    # 5) pairwise squared distances (could be optimized / approximated)
    sq = np.sum(X**2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)

    # 6) build kernel with geometric-mean sigma_ij = sqrt(sigma_i*sigma_j)
    Sigma = np.outer(sigma, sigma)
    denom = 2.0 * Sigma + eps
    K = np.exp(- d2 / denom)

    # 7) optional PSD correction (eigenvalue clipping)
    if make_psd:
        # compute symmetric eigen-decomposition
        vals, vecs = eigh(K)
        vals_clipped = np.clip(vals, a_min=0.0, a_max=None)
        K = (vecs * vals_clipped) @ vecs.T

    return K
