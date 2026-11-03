"""
Memristor state and conductance update models
"""
from ..dynamical_network import *
from ..networks.nanowire_network import *
from .stochastic_models import * 

# Physical and mathematical constants
ELECTRON_MASS       = 9.1093837015e-31  # kg
ELEMENTARY_CHARGE   = 1.602176634e-19   # Coulomb
PLANCK_CONSTANT     = 6.62607015e-34    # Js
PI                  = 3.141592653589793

#|>==========================<[Conductance Models]>==========================<|#
class CondTanh(CondModel):
    def __init__(
            self,
            C3:float = 8,
            C4:float = 0.63,
            unidir:bool=True,
        ) -> None:
        self.C3 = C3 
        self.C4 = C4
        if unidir is True:
            self._update_implicit=self._update_unidir
        else:
            self._update_implicit=self._update_bidir

    def configure(
            self,
            param:NanowireNetworkParam
        ) -> None:
        self.C1 = ((1/param.ron) + (1/param.roff))/2 
        self.C2 = ((1/param.ron) - (1/param.roff))/2 
        self._param = param

    def update(self,
        state:NetworkState):
        return self._update_implicit(state)
    
    def _update_unidir(
        self,
        state:NetworkState
        ) -> None:
        
        state.cond = self.C2 * np.tanh(
            self.C3 * (state.flux/self._param.fluxmax - self.C4)
        ) + self.C1

    def _update_bidir(
        self,
        state:NetworkState
        ) -> None:
        
        state.cond = self.C2 * np.tanh(
            self.C3 * (np.abs(state.flux)/self._param.fluxmax - self.C4)
        ) + self.C1

class CondCosh(CondModel):
    def __init__(
            self,
            C3:float = 8,
            C4:float = 0.667,
            unidir:bool=True,
        ) -> None:
        self.C3 = C3 
        self.C4 = C4
        if unidir is True:
            self._update_implicit=self._update_unidir
        else:
            self._update_implicit=self._update_bidir

    def configure(
            self,
            param:NanowireNetworkParam
        ) -> None:
        self.C1 = (1/param.roff)
        self.C2 = (1/param.ron) - (1/param.roff)
        self._param = param

    def update(self,
        state:NetworkState):
        return self._update_implicit(state)
    
    def _update_unidir(
        self,
        state:NetworkState
        ) -> None:
        s = np.minimum(
            state.flux/self._param.fluxmax, self.C4)
        state.cond = self.C1 + self.C2 *(
            np.cosh(self.C3*(s-self.C4))**(-2)
        )

    def _update_bidir(
        self,
        state:NetworkState
        ) -> None:
        s = np.minimum(
            np.abs(state.flux)/self._param.fluxmax, self.C4)
        state.cond = self.C1 + self.C2 *(
            np.cosh(self.C3*(s-self.C4))**(-2)
        )

class CondTunnelSimmons(CondModel):
    """Tunnelling conductance via Simmons low voltage formula"""
    def configure(
        self,
        param:NanowireNetworkParam
        ) -> None:
        """
        zeta_a = 137_788_151_839_399.33
        zeta_b=       9_231_941_643.76233

            rt = zeta_a * s * np.exp(zeta_b*s)
            where
            s = (lambda_crit - abs(filamentState))*s_max/lambda_crit #m
            and s[s<0]=0
        """
        m_electron  = ELECTRON_MASS  # kg
        m_star      = param.mass*m_electron # kg
        elem_charge = ELEMENTARY_CHARGE # Coulomb
        phi_e       = param.phi         # phi=0.82eV => phi/elem_charge = 0.82 volts
        A           = param.area        # m^2
        h           = PLANCK_CONSTANT   # Js
        pi          = PI
        self.zeta_a = 1/(A*(((3*((2*m_star)**(0.5))\
                    *(elem_charge**(2.5))))/(2*h**2))*(phi_e**0.5))
        self.zeta_b= (4*pi*(2*m_star*elem_charge)**0.5)*(phi_e**0.5)/h

        self._param = param

    def get_conductance(self, flux:NDArray[np.float64]) -> NDArray[np.float64]:
        s_max   = self._param.smax
        xcrit   = self._param.fluxcrit
        ron     = self._param.ron 
        roff    = self._param.roff
        s = (xcrit - abs(flux))*s_max/xcrit #m
        s[s<0]=0
        rt = self.zeta_a * s * np.exp(self.zeta_b*s)
        conductance = 1/(rt + ron + (ron**2/(roff-ron))) + 1/roff
        return conductance
    
    def update(self, state:NetworkState):
        state.cond = self.get_conductance(flux=state.flux)

class CondTunnelLandauer(CondModel):
    """Tunnelling conductance via Landauer formalism"""
    def __init__(self,
            fermi_level:float=5.5 #electron volts
            ) -> None:
        self.fermi_level = fermi_level

    def configure(
        self,
        param:NanowireNetworkParam
        ) -> None:
        self.C1 = (1/param.roff)
        self.C2 = (1/param.ron) - (1/param.roff)
        self._param = param

    @staticmethod
    def transmission_rec_barrier(V,s,m,Vb):
        # 2*elem_charge/hbar**2 = 2.881297266732744e+49
        C1 = (Vb**2)/(4*V*np.abs(Vb-V))
        C2 = np.sqrt(np.abs(Vb-V)*m*2.881297266732744e+49)
        return 1/(1+C1*np.sinh(C2*s)**2)
    
    def get_conductance(
            self, 
            flux:NDArray[np.float64], 
            edgeval:NDArray[np.float64]
            ) -> NDArray[np.float64]:
        Vb      = self._param.phi 
        s_max   = self._param.smax
        xcrit   = self._param.fluxcrit
        m       = self._param.mass*ELECTRON_MASS
        Vf      = self.fermi_level 
        V       = np.abs(edgeval.copy())
        V[V>1.6]= 1.6 # temporary fix, approximation breaks down at large voltages
        s = (xcrit - abs(flux))*s_max/xcrit 
        s[s<0]=0
        G= self.C1 + self.C2*(
            self.transmission_rec_barrier(Vf-V/2,np.abs(s),m,Vb+Vf)
            + self.transmission_rec_barrier(Vf+V/2,np.abs(s),m,Vb+Vf))/2
        return G
    
    def update(self, state:NetworkState):
        state.cond = self.get_conductance(
            flux=state.flux,
            edgeval = state.edgeval
            )

#|>==========================<[Flux State Models]>===========================<|#

class FluxThresholdIntegrator(FluxModel):
    def __init__(
            self,
            vset:float  = 0.01,
            vreset:float= 0.005,
            boost:float = 10.0,
            noise:StochModel | None = None
        ) -> None:
        self.vset   = vset 
        self.vreset = vreset
        self.boost  = boost
        if self.noise:
            self.get_flux_change = self._get_flux_change_stoch
        else:
            self.get_flux_change = self._get_flux_change

    def configure(
        self,
        param:NanowireNetworkParam
        ) -> None:
        self._param = param

    def _get_flux_change_stoch(
        self,
        flux:NDArray[np.float64],
        edgeval:NDArray[np.float64],
        ) -> NDArray[np.float64]:
        dflux = (abs(edgeval) > self.vset) *\
                (abs(edgeval) - self.vset) *\
                np.sign(edgeval) 
        dflux = dflux + \
                (abs(edgeval) < self.vreset) *\
                (abs(edgeval) - self.vreset) *\
                np.sign(flux) * self.boost
        dflux += self.noise.sample()
        return dflux
    
    def _get_flux_change(
        self,
        flux:NDArray[np.float64],
        edgeval:NDArray[np.float64],
        ) -> NDArray[np.float64]:
        dflux = (abs(edgeval) > self.vset) *\
                (abs(edgeval) - self.vset) *\
                np.sign(edgeval) 
        dflux = dflux + \
                (abs(edgeval) < self.vreset) *\
                (abs(edgeval) - self.vreset) *\
                np.sign(flux) * self.boost
        return dflux
    
    def update(
        self,
        state:NetworkState,
        input:InputState,
        dt:float,
        t:int,
        ) -> None:
        xmax = self._param.fluxmax
        eta = self._param.eta
        dflux = self.get_flux_change(
            flux    = state.flux,
            edgeval = state.edgeval
        )
        flux = state.flux + dt*eta*dflux
        flux[abs(flux) > xmax] = \
            np.sign(flux[abs(flux) > xmax])*xmax
        
        state.flux = flux
        
class FluxLeakyIntegrator(FluxModel):
    def __init__(
            self,
            tauv:float|NDArray[np.float64] = 200/3,
            taux:float|NDArray[np.float64] = 20/3,
            noise:StochModel | None = None
        ) -> None:
        """
        tauv by default is 1/fluxmax = 1/0.015
        taux by default is 0.1/fluxmax = 0.1/0.015
        """
        self.tauv = tauv 
        self.taux = taux 
        self.noise = noise

        if self.noise:
            self.get_flux_change = self._get_flux_change_stoch
        else:
            self.get_flux_change = self._get_flux_change

    def configure(
        self,
        param:NanowireNetworkParam
        ) -> None:
        self._param = param

    def _get_flux_change_stoch(
        self,
        flux:NDArray[np.float64],
        edgeval:NDArray[np.float64],
        ) -> NDArray[np.float64]:
        dflux = self.tauv * edgeval \
            - self.taux * flux \
            + self.noise.sample()
        return dflux
    
    def _get_flux_change(
        self,
        flux:NDArray[np.float64],
        edgeval:NDArray[np.float64],
        ) -> NDArray[np.float64]:
        dflux = self.tauv * edgeval \
            - self.taux * flux 
        return dflux
    
    def update(
        self,
        state:NetworkState,
        input:InputState,
        dt:float,
        t:int,
        ) -> None:
        xmax = self._param.fluxmax
        eta = self._param.eta
        dflux = self.get_flux_change(
            flux    = state.flux,
            edgeval = state.edgeval
        )
        flux = state.flux + dt*eta*dflux

        flux[abs(flux) > xmax] = \
            np.sign(flux[abs(flux) > xmax])*xmax
        
        state.flux = flux
