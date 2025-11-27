def compute_score(intermediate_clustering, maxbudget):

    ARI_FALCON = []
    NMI_FALCON = []
    silhouette_FALCON = []
  
    interaction = list(intermediate_clustering.keys())[0]
    
    cpt = 0
    list_budget = list(intermediate_clustering.keys())
    budget = list_budget[cpt]
    next_budget = list_budget[cpt + 1]
    for i in range(1,max_budget + 1):
        if i >= next_budget :
            if list_budget[-1] >  i:
                cpt += 1
                budget = next_budget
                next_budget = list_budget[cpt]
            else:
                budget = next_budget
        if i < interaction :
            ARI_score.append(0)
            NMI_score.append(0)
            silhouette_score.append(0)
        else:
            y_pred = intermediate_clustering[budget]
            ARI_score.append(metrics.adjusted_rand_score(np.array(y_pred), y))
            NMI_score.append(normalized_mutual_info_score(np.array(y_pred), y))
            if len(np.unique(np.array(y_pred))) > 1:
                silhouette_score.append(metrics.silhouette_score(X,np.array(y_pred))) 
            else:
                silhouette_score.append(0) 

    return ARI_score, NMI_score, silhouette_score

