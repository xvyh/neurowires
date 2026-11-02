"""
Simulation for dynamical networks
base classes
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from numpy.typing import NDArray
import numpy as np 

# optional dependency for progressbar
try:
    from tqdm.autonotebook import tqdm as pbar
except ImportError:
    def pbar(iterable, *args, **kwargs):
        return iterable 

#|>==============================<[parameter]>===============================<|#
@dataclass(frozen=True)
class NetworkParam:
    """
    Parameters:
        edgelist (array): edgelist of the specified graph
        
    Assumptions:
        for N nodes, the nodes in the edgelist are labelled 0,1,...,N-1
        for E edges, each edgelist entry is of form
            [edge_id:[source_node_id, target_node_id, *relaxed_node_ids]]
    """
    edgelist: NDArray[np.int32]
    nodenum:int|None = None
    edgenum:int|None = None 

    def __post_init__(self):
        if self.nodenum is None:
            object.__setattr__(
                self,
                "nodenum",
                np.max(self.edgelist) + 1
            )
        if self.edgenum is None:
            object.__setattr__(
                self,
                "edgenum",
                len(self.edgelist)
            )
#|>================================<[state]>=================================<|#
@dataclass
class NetworkState:
    nodeval: NDArray[np.float64]
    edgeval: NDArray[np.float64]
    flux: NDArray[np.float64]
    cond :NDArray[np.float64]

@dataclass 
class InputState:
    """
    Parameters:
        signals (array): input signals
        electrodes (list): list of nodes assigned as interfacing electrodes
    """
    signal: NDArray[np.float64]
    electrode: NDArray[np.int32]

#|>==============================<[base model]>==============================<|#
class NodeModel(ABC):
    """Node value model"""
    @abstractmethod
    def configure(
        self,
        param:NetworkParam,
        input:InputState,
        ) -> None:
        pass 
    @abstractmethod
    def update(
        self,
        state:NetworkState,
        input:InputState,
        dt:float
        ) -> None:
        pass

class FluxModel(ABC):
    """Flux model"""
    @abstractmethod
    def configure(
        self,
        param:NetworkParam
        ) -> None:
        pass 
    @abstractmethod
    def update(
        self,
        state:NetworkState,
        input:InputState,
        dt:float
        ) -> None:
        pass 

class CondModel(ABC):
    """Conductance model"""
    @abstractmethod
    def configure(
        self,
        param:NetworkParam
        ) -> None:
        pass 
    @abstractmethod
    def update(
        self,
        state:NetworkState
        ) -> None:
        pass


#|>========================<[base dynamical network]>========================<|#
class DynamicalNetwork:
    """
    Dynamical Network Class
    Base class for managing dynamical network simulations
    """
    def __init__(
            self,
            state:NetworkState,
            param:NetworkParam,
            nodemodel:NodeModel,
            fluxmodel:FluxModel,
            condmodel:CondModel,
    ) -> None:
        self.state = state
        self.param = param
        self.nodemodel = nodemodel
        self.fluxmodel = fluxmodel
        self.condmodel = condmodel

    def configure(self,input:InputState) -> None:
        self.fluxmodel.configure(self.param)
        self.condmodel.configure(self.param)
        self.nodemodel.configure(self.param,input)

    def update(self,
            input:InputState, dt:float, t:int) -> None:
        self.fluxmodel.update(
            state=self.state,
            input=input,
            dt=dt, t=t)
        self.condmodel.update(state=self.state)
        self.nodemodel.update(
            state=self.state,
            input=input,
            dt=dt, t=t)

    def simulate(self,
            dt:float, 
            steps:int, 
            input:InputState,
            augment:Callable|None=None,
            return_flux:bool=False,
            enable_pbar:bool=False):
        """
        Parameters:
            dt (float)  : timestep duration 
            steps (int) : total number of simulated steps
            augment (function): input signal augmentation mid-simulation
            return_flux (bool): whether to return the state values
            enable_pbar (bool): optional progress bar
        Returns:
            node_values (array): (steps x nodenum) sized array of node values
            optionally:
            state_values (array): (steps x edgenum x state_nums) sized array of internal states
        """
        node_values = np.zeros((steps, self.param.nodenum))
        if return_flux:
            edge_states = np.zeros((steps, self.param.edgenum))
        update_signal = augment is not None
        self.configure(input=input)

        for t in pbar(
            range(steps), desc='|Neurowires', 
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            disable= not enable_pbar):

            self.update(input=input,dt=dt, t=t)
            node_values[t,:] = self.state.nodeval
            
            if return_flux:
                edge_states[t,:] = self.state.flux 
            if update_signal:
                input = augment(self, input, t)
        if return_flux:
            return node_values, edge_states
        return node_values
    
