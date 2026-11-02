"""
Network node value update models
"""
from numba import njit
from ..dynamical_network import * 

#|>========================<[Linear System of Eqns]>=========================<|#
class NodeLinSys(NodeModel):
    def __init__(self):
        self._configured = False

    def configure(
            self,
            param:NetworkParam,
            input:InputState,
        ) -> None:
        """
        Configure the model object 
        to initialise the modified graph Laplacian
        for solving linear system of equations
        """
        n = param.nodenum
        m = len(input.electrode)

        self.kmtx = np.zeros((n+m, n+m))
        self.rhs = np.zeros(n+m)

        for i, this_elec in enumerate(input.electrode):
            self.kmtx[n+i, this_elec] = 1
            self.kmtx[this_elec, n+i] = 1

        self._param = param
        self._configured = True

        
    def update(
        self,
        state:NetworkState,
        input:InputState,
        dt:float,
        t:int
        ) -> None:
        """Solve system of linear equations"""
        n = self._param.nodenum
        edgelist = self._param.edgelist
        lhs = self.kmtx 
        rhs = self.rhs
        lhs[edgelist[:,0], edgelist[:,1]] = -state.cond 
        lhs[edgelist[:,1], edgelist[:,0]] = -state.cond 
        lhs[range(n), range(n)] = 0 #needs to be here for correct sum below
        lhs[range(n), range(n)] = -np.sum(lhs[:n,:n], axis=0)
        rhs[n:] = input.signal[t,:]
        sol = np.linalg.solve(lhs, rhs)
        state.nodeval = rs = sol[:n]
        state.edgeval = rs[edgelist[:,0]] - rs[edgelist[:,1]]

#|>=========================<[Local Approximation]>==========================<|#
class NodeLocal(NodeModel):
    def __init__(self, g0:float=1.0):
        self.g0 = g0
        self._configured = False

    def configure(
            self,
            param:NetworkParam,
            input:InputState,
        ) -> None:
        """
        Configure the model object 
        to initialise the base M0 matrix
        """
        n = param.nodenum
        m = len(input.electrode)

        self.kmtx = np.zeros((n+m, n+m))
        self.rhs = np.zeros(n+m)

        for i, this_elec in enumerate(input.electrode):
            self.kmtx[n+i, this_elec] = 1
            self.kmtx[this_elec, n+i] = 1

        # solve for base M
        self.kmtx[param.edgelist[:,0], param.edgelist[:,1]] = -1
        self.kmtx[param.edgelist[:,1], param.edgelist[:,0]] = -1
        self.degrees = -np.sum(self.kmtx[:n,:n], axis=0)
        self.kmtx[range(n),range(n)] = self.degrees
        Linv = np.linalg.inv(self.kmtx)
        self.M0 = Linv[:n,n:]

        self._param = param
        self._configured = True

    def update(
        self,
        state:NetworkState,
        input:InputState,
        dt:float,
        t:int
        ) -> None:

        edgenum = self._param.edgenum
        edgelist= self._param.edgelist
        dg = (state.cond - self.g0)/self.g0
        rin = input.signal[t,:]
        r0 = self.M0 @ rin 
        rs = r0.copy()
        self._numba_solve_local_first_order(
            r0, edgenum, edgelist,
            self.degrees,dg,rs)
        state.nodeval = rs
        state.edgeval = rs[edgelist[:,0]] - rs[edgelist[:,1]]

    @staticmethod
    @njit
    def _numba_solve_local_first_order(r0,E,edgelist,deg,dg,r):
        for e in range(E):
            i = edgelist[e,0]
            j = edgelist[e,1]
            v0 = r0[i] - r0[j]
            #first order
            c1 = dg[e]*v0
            r[i] -= c1/deg[i]
            r[j] += c1/deg[j]
        