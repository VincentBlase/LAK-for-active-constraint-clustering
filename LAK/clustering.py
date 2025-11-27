from mmd_critic import MMDCritic
from mmd_critic.kernels import RBFKernel

import numpy as np
import pandas as pd

import math
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances, adjusted_rand_score
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations, chain



class clustering:
    """
    
    Attributes:
        X (array-like): The dataset for which to apply protoKNN.
        y (array-like): The labels for which will be used for labelisation part.
    """
    
    def __init__(self, X, labels=None, prototype_kernel= None, criticism_kernel=None):
        """
        Initializes the MMD Critic class
    
        Args:
            X (array-like): The dataset
            labels (Optional[array-like]): The labels for the dataset.
            prototype_kernel (Kernel): The kernel to use for prototypes. Should extend the mmd_critic.kernels.Kernel class
            criticism_kernel (Optional[Kernel]): The kernel to use for criticisms. If None, uses the prototype kernel
        """

        if labels is None :
            raise ValueError('Need labels')
        
        self.X = np.asarray(X)
        self.labels = np.asarray(labels).reshape(-1) 
        self.prototype_kernel = prototype_kernel or RBFKernel(0.25)
        self.criticism_kernel = criticism_kernel or prototype_kernel
        self.budget = 0
        self.cluster_bags = dict()
        self.nb_bags = 0
        self.MMD = MMDCritic(self.X, self.prototype_kernel, self.criticism_kernel)
        self.intermediate_cluster = dict()

    def cluster(self, n, k_proto, train_indices=None):
        """
        Args:
            n: budget
            train_indices : indices of the train set

        Raises:
            ValueError on improper n.
        """


        if train_indices is None:
            train_indices = np.arange(len(self.X))
            
        cluster_points = self.X.copy()
        
        nb_proto = k_proto  # Number of inital prototypes
        
        protos_inds, centers, closest_centers_indices = self.calcul_cluster(cluster_points, n_proto = nb_proto, train_indices = train_indices, local = False) # Calculate initial cluster with initial prototypes
        
        if self.labels[protos_inds[0]] == self.labels[protos_inds[1]]:
            self.cluster_bags[self.nb_bags] = [protos_inds[0], protos_inds[1]]
            self.nb_bags += 1
            self.budget += 1
        else :
            self.cluster_bags[self.nb_bags] = [protos_inds[0]]
            self.cluster_bags[self.nb_bags +1] = [protos_inds[1]]
            self.nb_bags += 2
            self.budget += 1

        if len(protos_inds) > 2 :
            self.generate_link_cluster(protos_inds)
                    
                

        # While budget not reach
        cpt = 0
        nb_criti_cluster_bool_global = True
        
        while self.budget < n and (len(train_indices) > 2*len(protos_inds)  + 1) and nb_criti_cluster_bool_global :

            criti_inds =  self.calculCriticism(protos_inds, len(protos_inds) + 1 , train_indices)
            # Find cluster with most critiscism
            cluster_points, nb_criti_cluster, protos_inds, train_indices_cluster = self.find_cluster(centers, criti_inds, protos_inds, closest_centers_indices, train_indices = train_indices)

            # If cluster larger then number of critiscism
            if len(train_indices_cluster) > nb_criti_cluster :
                
                ## While local cluster selected larger then number of prototypes and critiscisms and while there is more then one critiscism inside a local cluster
                if nb_criti_cluster == 0 :
                    nb_criti_cluster_bool_global = False
                else :
                    nb_criti_cluster_bool_global = True
                
                    
                        
                protos_inds_cluster = self.calcul_cluster(cluster_points, n_proto = nb_criti_cluster + 1, train_indices = train_indices_cluster, local = True) # Calculate local cluster


                protos_inds = np.append(protos_inds, np.concatenate([np.where(np.all(self.X == row, axis=1))[0] for row in cluster_points[protos_inds_cluster]], axis=0))
                protos_inds = np.unique(protos_inds)
                
                

            centers, closest_centers_indices = self.calcul_new_cluster(protos_inds, train_indices) # Update cluster calcul with new prototypes found

            self.generate_link_cluster(protos_inds)

            y_pred = closest_centers_indices.copy()
            
            value_max = max(y_pred)
            for i,j in self.cluster_bags.items():
                for k in j :
                    label = y_pred[k]
                    for l in range(len(y_pred)):
                        if y_pred[l] == label:
                            y_pred[l] = i + value_max *2

            
            self.intermediate_cluster[self.budget] = y_pred

           
        while len(protos_inds) < len(train_indices) and self.budget < n :
            
            criti_inds =  self.calculCriticism(protos_inds, 1 , train_indices)
            # Find cluster with most critiscism
            cluster_points, nb_criti_cluster, protos_inds, train_indices_cluster = self.find_cluster(centers, criti_inds, protos_inds, closest_centers_indices, train_indices = train_indices)

            # If cluster larger then number of critiscism
            if len(train_indices_cluster) > nb_criti_cluster :
                      
                protos_inds_cluster = self.calcul_cluster(cluster_points, n_proto = nb_criti_cluster + 1, train_indices = train_indices_cluster, local = True) # Calculate local cluster


                protos_inds = np.append(protos_inds, np.concatenate([np.where(np.all(self.X == row, axis=1))[0] for row in cluster_points[protos_inds_cluster]], axis=0))
                protos_inds = np.unique(protos_inds)
                
            else :
                protos_inds = np.append(protos_inds, criti_inds)
                protos_inds = np.unique(protos_inds)


                
                
            centers, closest_centers_indices = self.calcul_new_cluster(protos_inds, train_indices) # Update cluster calcul with new prototypes found

            self.generate_link_cluster(protos_inds)

            y_pred = closest_centers_indices.copy()
            
            value_max = max(y_pred)
            for i,j in self.cluster_bags.items():
                for k in j :
                    label = y_pred[k]
                    for l in range(len(y_pred)):
                        if y_pred[l] == label:
                            y_pred[l] = i + value_max *2

            self.intermediate_cluster[self.budget] = y_pred


        value_max = max(closest_centers_indices)
        for i,j in self.cluster_bags.items():
            for k in j :
                label = closest_centers_indices[k]
                for l in range(len(closest_centers_indices)):
                    if closest_centers_indices[l] == label:
                        closest_centers_indices[l] = i + value_max *2
             
        self.intermediate_cluster[self.budget] = closest_centers_indices
        
        return closest_centers_indices, self.intermediate_cluster


    def find_cluster(self, centers, criti_inds, protos_inds, closest_centers_indices, train_indices):
        """
        Find cluster with the most critiscism indices

        Args : 
            X (array-like): The dataset
            centers : Values of prototypes which represent centers of clusters
            criti_inds : Indices of critiscisms 
            proto_inds : Indices of prototypes
            closest_centers_indices : Labels of clusters for each instances
            train_indices : Indices used for training

        Return :
            cluster_points : Values of clusters selected
            nb_criti_cluster : Number of critiscism in cluster selected
            protos_inds : Indices of prototypes updated
            train_indices_cluster : Indices used for training in cluster selected
        """
            
        
        ## Calculate number of criticism for each cluster
        closest_proto_ind = pairwise_distances(self.X[criti_inds,:],centers, metric='euclidean').argmin(axis=1)  
        freq_closest_proto_ind = np.bincount(closest_proto_ind)

        
        nb_criti_cluster = np.max(freq_closest_proto_ind) 
        if len(criti_inds) == 1 :
            best_indices = np.argwhere(freq_closest_proto_ind == np.amax(freq_closest_proto_ind)).flatten().tolist()[0]
            max_len_train_indices_cluster = len(np.where(np.isin(np.where(closest_centers_indices == best_indices)[0], train_indices))[0])
        else :
            max_len_train_indices_cluster = 0
            
            while max_len_train_indices_cluster < nb_criti_cluster + 1:
                list_best_indices = np.argwhere(freq_closest_proto_ind == np.amax(freq_closest_proto_ind)).flatten().tolist()
                if len(list_best_indices) == 1:
                    best_indices = list_best_indices[0]
                    max_len_train_indices_cluster = len(np.where(np.isin(np.where(closest_centers_indices == best_indices)[0], train_indices))[0])
                else :
                    max_len_train_indices_cluster = 0
                    for j in list_best_indices:
                        ## Select cluster with most critiscism
                        train_indices_cluster = np.where(np.isin(np.where(closest_centers_indices == j)[0], train_indices))[0]
                        if len(train_indices_cluster) > max_len_train_indices_cluster :
                            max_len_train_indices_cluster = len(train_indices_cluster)
                            best_indices = j
            
                nb_criti_cluster = np.max(freq_closest_proto_ind)
                freq_closest_proto_ind[list_best_indices] -= 1

        ## Select cluster with most critiscism
        cluster_points = self.X[closest_centers_indices == best_indices]
        train_indices_cluster = np.where(np.isin(np.where(closest_centers_indices == best_indices)[0], train_indices))[0]

        return cluster_points, nb_criti_cluster, protos_inds, train_indices_cluster


    def calculCriticism(self, protos_inds, nb_criti, train_indices):
        """
        Calcul new critiscisms 

        Args : 
            X (array-like): The dataset
            prototype_kernel : prototype kernel used for calculated prototypes
            criticism_kernel : criticism kernel used for calculated criticisms
            proto_inds : Indices of prototypes
            train_indices : Indices of train dataset

        Return :
            criti_inds : Indices of criticism updated
        """
        criti_inds = self.MMD.select_criticisms(nb_criti, self.X[protos_inds], train_indices = train_indices, argMax = True)

        return criti_inds
    
    def generate_link_cluster(self, protos_inds):
        """
        Calcul ML link

        Args : 
            proto_inds : Indices of prototypes
            budget : updated budget
            cluster_bags : Dict of cluster with prototypes inds in it
            nb_bags : Number of cluster

        Return :
            Updated 
        """
        for index in protos_inds :
            if index in list(chain.from_iterable(list(self.cluster_bags.values()))):
                continue
            else :
                ML = False
                cpt = 0
               # Calculer les moyennes des listes
                mean_values = {key: sum(self.X[i] for i in ind) / len(ind) for key, ind in self.cluster_bags.items()}
            
                # Calculer les distances par rapport au nouvel indice
                dist = {key: np.linalg.norm(self.X[index] - mean) for key, mean in mean_values.items()}
                
                # Trier les clés en fonction des distances
                ordered_key = sorted(dist, key=dist.get)
                while not ML and cpt < self.nb_bags:
                    if self.labels[index] == self.labels[self.cluster_bags[ordered_key[cpt]][0]]:
                        self.cluster_bags[ordered_key[cpt]].append(index)
                        ML = True
                        self.budget += 1
                    else :
                        cpt +=1
                        self.budget += 1

                if cpt == self.nb_bags :
                    self.cluster_bags[cpt] = [index]
                    self.nb_bags += 1 
                    


    def calcul_new_cluster(self, protos_inds, train_indices):
        
        """
        Calcul new critiscisms and prototypes centers with the prototypes indices updated

        Args : 
            X (array-like): The dataset
            prototype_kernel : prototype kernel used for calculated prototypes
            criticism_kernel : criticism kernel used for calculated criticisms
            proto_inds : Indices of prototypes
            train_indices : Indices of train dataset

        Return :
            cluster_points : Values of local cluster selected
            closest_proto_ind : Closest critiscisms indices for each prototypes
            protos_inds : Indices of prototypes updated
        """
            
        centers = self.X[protos_inds,:] ## Calculate centers values of each cluster created from prototypes
        closest_centers_indices = pairwise_distances(self.X,centers, metric='euclidean').argmin(axis=1) ## Calculate clusters from centers values with 1NN algorithm. 

        return centers, closest_centers_indices

    def calcul_cluster(self,cluster_points, n_proto, train_indices, local):
        """
        Calcul clustering from critiscm and prototype indices
        
        Args : 
            X (array-like): The dataset
            prototype_kernel : prototype kernel used for calculated prototypes
            n_proto : Number of prototypes to be calculated
            train_indices : Indices of train dataset
            local : Boolean if calcul on local or global data space

        Return :
            protos_inds : Indices of prototypes
            centers : Values of prototypes which represent centers of clusters
            closest_centers_indices : Labels of clusters for each instances
        """

        ## Calculate prototypes and critiscisms
        if local :
            
            critic = MMDCritic(cluster_points, self.prototype_kernel, self.criticism_kernel)
            protos_inds = critic.select_prototypes(n_proto,train_indices = train_indices)
            return protos_inds
            
        else :
            protos_inds = self.MMD.select_prototypes(n_proto,train_indices = train_indices)
            centers = cluster_points[protos_inds,:] ## Calculate centers values of each cluster created from prototypes
            closest_centers_indices = pairwise_distances(cluster_points,centers, metric='euclidean').argmin(axis=1) ## Calculate clusters from centers values with 1NN algorithm. 
    
            return protos_inds, centers, closest_centers_indices


    
    




        


    
