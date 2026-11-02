import numpy as np 

def edgelist2adj(edgelist, n=None, vec=1):
    """edgelist to adjacency matrix"""
    n = int(np.max(edgelist))+1 if n is None else n
    lhs = np.zeros((n, n))
    lhs[edgelist[:,0], edgelist[:,1]] = vec
    lhs[edgelist[:,1], edgelist[:,0]] = vec
    return lhs

def adj2edgelist(adj):
    """adjacency matrix to edgelist"""
    return np.stack(np.where(np.triu(adj)==1)).T

def incidence_mtx(edge_list, N:int, directed:bool=True):
    """
    convert an unweighted directed edge list of length M
    into a directed incidence matrix of size N x M
    N number of nodes, M number of edges

    Args:
        N (int): number of nodes
        edge_list (np.ndarray): edge list of length M, with elements (i,j) for i->j
    
    Returns:
        inc_mtx (np.ndarray): the unweighted directed incidence matrix of size N x M
    """
    M = len(edge_list)
    inc_mtx = np.zeros((N,M))
    s = -1 if directed else 1

    #the inefficient way
    for i in range(M):
        n1,n2 = edge_list[i]
        inc_mtx[n1, i] =  s #outgoing edge
        inc_mtx[n2, i] =  1 #incoming edge
    return inc_mtx 