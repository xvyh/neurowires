import numpy as np

def square_lattice_network(W:int, H:int, torus:bool=False):
    """
    Generate the adjacency matrix of an undirected unweighted network, 
    which tiles with squares 

    Input:
        W (int): number of nodes for width of lattice grid
        H (int): number of nodes for height of lattice grid
        torus(bool): whether to make the square lattice a torus

    Returns:
        A (np.ndarray): undirected unweighted adjacency matrix
    
    E.g., for W=5, H=2, the returning network has form
        0-1-2-3-4
        | | | | |
        5-6-7-8-9
    
        if torus, then the connections circle back
        0-1-2-3-4 - 0
        etc...
    """
    tot = W*H
    A = np.zeros((tot, tot))
    for i in range(tot-1):
        last_row= i>= tot-W #if last row
        last_col= i%W==W-1 #if last col
        # if last_row and last_col only happen at (tot,tot)
        if last_row:
            A[i,i+1] = A[i+1,i] = 1
        elif last_col:
            A[i,i+W] = A[i+W,i] = 1
        else:
            A[i,i+1] = A[i+1,i] = 1
            A[i,i+W] = A[i+W,i] = 1
    if torus: #square into donut; connect the ends
        for i in range(0,tot,W): #connecting rows start and end
            A[i, i+W-1] = A[i+W-1, i] = 1
        for i in range(W): #connecting colns start and end
            A[i, i+W*(H-1)] = A[i+W*(H-1), i] = 1
    return A

def sqlattice_convert_to_id(coordinates:list, w:int):
    """ assume coords is a list of 2 element tuples [(x1,y1), (x2,y2),...] """
    return [int(x+w*y) for x,y in coordinates]

def sqlattice_convert_to_coord(ids:list, w:int):
    return [(id%w, id//w) for id in ids]

def reorder_lattice_edgelist(edgelist, W:int, H:int):
    """
        ensure all edges are in accending order,
        i.e. (u,v) such that v>u, EXCEPT at boundaries 
        then it is (v,u)
    """
    for i, (u,v) in enumerate(edgelist):
        if u%W==0 and v==u+W-1: #boundary at hoizontal
            edgelist[i] = (v,u)
        elif u//W==0 and v==u+(H-1)*W:
            edgelist[i] = (v,u)
        elif u > v:
            edgelist[i] = (v,u)
    return edgelist