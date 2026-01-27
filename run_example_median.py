import numpy as np
from LAK.clustering import clustering
from mmd_critic.kernels import RBFKernel
from LAK.open_data import open_dataset
from LAK.compute_score import compute_score

name='IRIS' #Enter dataset name
X,y = open_dataset(name)
print('DATASET : ', name)

max_budget = 200

ARI_score = []
NMI_score = []
silhouette_score = []

sigma = np.sqrt(np.median(pairwise_distances(X,X)))

clusterer = clustering(X,y,RBFKernel(sigma), RBFKernel(sigma))
clustering_final, intermediate_clustering = clusterer.cluster(n=max_budget,k_proto=2, train_indices = None)
    
ARI_score, NMI_score, silhouette_score = compute_score(intermediate_clustering,max_budget) #Compute score for each budget value


<-- Print table 2 values -->

print(ARI_score[25])
print(ARI_score[50])
print(ARI_score[100])
print(ARI_score[200])
