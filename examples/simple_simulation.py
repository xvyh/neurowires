#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import neurowires as nws 

def main() -> None:
    # Set up a square lattice network
    W, H = 5, 5
    adj_mtx = nws.connectivity.square_lattice_network(W,H,torus=False)
    electrodes = nws.connectivity.sqlattice_convert_to_id(
        [(0,H//2), (W-1,H//2)], W)
    square_lattice_net = nws.utils.adj2edgelist(adj_mtx)

    param = nws.NanowireNetworkParam(
            edgelist=square_lattice_net
        )

    print(
        (f"Square lattice network with "
        f"{param.nodenum} nodes, " 
        f"{param.edgenum} edges."))

    # Initialise the input signal
    dt = 0.01
    T = 100
    freq = 0.25
    timevec = np.arange(0,dt*T, dt)
    input_signal = np.sin(2 * np.pi * freq * timevec).reshape((T,1))*0.2
    drain_signal = np.zeros((T,1))
    resin        = np.concatenate((input_signal,drain_signal),axis=1) 

    input_signal = nws.InputState( #input signal object
        signal      = resin,
        electrode   = electrodes
    )
    noise = nws.models.OUProcess( #stochastic model object
        theta=0.1,
        sigma=0.01,
        shape=param.edgenum,
        dt=dt,rng_seed=0)

    # Initialise the nanowire network object, and the chosen update models
    nanowirenet = nws.NanowireNetwork(
        param = param,
        nodemodel= nws.models.NodeLinSys(),
        fluxmodel= nws.models.FluxLeakyIntegrator(noise=noise),
        condmodel= nws.models.CondTanh()
    )

    # Simulate
    nodevals, fluxvals = nanowirenet.simulate(
        dt=dt,
        steps=100,
        input = input_signal,
        enable_pbar=True,
        return_flux=True
    )

if __name__ == "__main__": main()