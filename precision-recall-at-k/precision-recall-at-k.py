def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k=recommended[:k]
    top_k_relevant=0
    for i in top_k:
        if i in relevant:
            top_k_relevant+=1
    return [top_k_relevant/k,top_k_relevant/len(relevant)]