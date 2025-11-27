import numpy as np
from LAK.clustering import clustering
from mmd_critic.kernels import RBFKernel
from LAK.open_data import open_dataset
from LAK.compute_score import compute_score
from LAK.LAK import Locally_adaptive_kernel

name='IRIS'
X,y = open_dataset(name)
print('DATASET : ', name)

max_budget = 20

ARI_FALCON = []
NMI_FALCON = []
silhouette_FALCON = []

clusterer_FALCON = FALCON(X,y,Locally_adaptive_kernel, Locally_adaptive_kernel)
clustering_FALCON, intermediate_clustering = clusterer_FALCON.cluster(n=max_budget,k_proto=2, train_indices = None)
    
ARI_FALCON, NMI_FALCON, silhouette_FALCON = compute_score(intermediate_clustering,max_budget)
