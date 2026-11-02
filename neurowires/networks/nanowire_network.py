"""
Nanowire network base classes
"""
from ..dynamical_network import *

#|>==============================<[parameter]>===============================<|#
@dataclass(frozen=True)
class NanowireNetworkParam(NetworkParam):
    """
    Parameters:
        ron (float): On resistance
        roff (float): Off resistance
        smax (float): largest filament gap size
        fluxmax (float): largest flux
        phi (float): energy barrier height in electron volts
        mass (float): mass of charge carriers, in electron masses
        area (float): area of memristive contact area
        eta (float): global time scaling parameter, serves similar role as dt
    """
    ron:float           = 1.0
    roff:float          = 1000.0
    fluxmax:float       = 1.0
    fluxcrit:float      = 2/3
    smax:float|NDArray  = 5e-9
    phi:float|NDArray   = 0.82
    mass:float|NDArray  = 1.0
    area:float|NDArray  = 0.17e-18
    eta:float|NDArray   = 1.0

#|>===========================<[Nanowire Network]>===========================<|#
class NanowireNetwork(DynamicalNetwork):
    """
    NanowireNetwork Class
    inherits from DynamicalNetwork class
    """
    def __init__(
            self,
            param:NanowireNetworkParam,
            nodemodel:NodeModel,
            fluxmodel:FluxModel,
            condmodel:CondModel,
    ) -> None:
        # initialising internal states
        nodeval = np.zeros(param.nodenum)
        edgeval = np.zeros(param.edgenum)
        flux    = np.zeros(param.edgenum)
        cond    = np.zeros(param.edgenum)
        state   = NetworkState(
            nodeval = nodeval,
            edgeval = edgeval,
            flux    = flux,
            cond    = cond
        )
        super().__init__(
            state=state,
            param=param,
            nodemodel=nodemodel,
            fluxmodel=fluxmodel,
            condmodel=condmodel)