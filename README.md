## Name
Adaptive Local Kernel for Efficient Active Pairwise Constraint Clustering

## Description
We introduce a novel active pairwise constraint clustering approach that employs a locally adaptive kernel, automat-ically adjusting its bandwidth based on local data density. This local adaptation identifies regions that deviate from uniform or Gaussian assumptions, prioritizing them for constraint queries and thus minimizing user effort while improving clustering quality.


```
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
```


## Installation
We recommand to download files on python 3.10.4 (Should work with other versions)

## Usage
Active clustering with pairwise constraint

## License
For open source projects, say how it is licensed.

## Project status
Keep going
