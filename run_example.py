import numpy as np
from LAK.clustering import clustering
from mmd_critic.kernels import RBFKernel
from LAK.open_data import open_dataset
from LAK.compute_score import compute_score
from LAK.LAK import adaptive_rbf_kernel

name='IRIS'
X,y = open_dataset(name)
print('DATASET : ', name)

max_budget = 20

ARI_score = []
NMI_score = []
silhouette_score = []

clusterer = clustering(X,y,adaptive_rbf_kernel, adaptive_rbf_kernel)
clustering_final, intermediate_clustering = clusterer.cluster(n=max_budget,k_proto=2, train_indices = None)
    
ARI_score, NMI_score, silhouette_score = compute_score(intermediate_clustering,max_budget)
