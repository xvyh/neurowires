"""
Models for introducing stochasticity
"""
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np 

#|>============================<[Stochasticity]>=============================<|#
class StochModel(ABC):
    @abstractmethod
    def sample(self) -> NDArray[np.float64]:
        """return next noise sample"""
        pass 

class WhiteGaussianNoise(StochModel):
    def __init__(
            self,
            sigma: float | NDArray[np.float64],
            shape: tuple[int,...],
            rng: np.random.Generator | None = None,
            rng_seed: float = 0,
        ) -> None:
        self.sigma = sigma 
        self.shape = shape
        self.rng = rng or np.random.default_rng(rng_seed)

    def sample(self) -> NDArray[np.float64]:
        return self.rng.normal(
            loc=0.0, scale=self.sigma, size=self.shape) 

class OUProcess(StochModel):
    def __init__(
            self,
            theta:float,
            sigma:float,
            shape: tuple[int,...],
            dt:float,
            x0:float|NDArray[np.float64]=0.0,
            rng: np.random.Generator | None = None,
            rng_seed: float = 0.0
        ) -> None:
        self.theta = theta 
        self.sigma = sigma 
        self.x = np.full(shape, x0) 
        self.dt = dt 
        self.rng = rng or np.random.default_rng(rng_seed)

        self.expfac = np.exp(-theta*dt)
        self.stdfac = sigma * np.sqrt(
            (1-np.exp(-2.0*theta*dt))/(2.0*theta)
        )

    def sample(self) -> NDArray[np.float64]:
        self.x = (
            self.expfac * self.x 
            + self.stdfac * self.rng.normal(
                size=np.shape(self.x)
            )
        )
        return self.x 