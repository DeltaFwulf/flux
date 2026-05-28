"""Contains functions specific to the transient simulation of a 1-dimensional mesh."""

import numpy as np

from util import tdma
from plotter import plot_temp_1d



def simulate_1d() -> None:
    """
    Simulates the heat transfer in a 1D mesh, then animates results.

    - the curvature may be selected as either flat, cylindrical, or spherical, l = 0,1,2
    - the diffusivity is currently constant
    """

    mesh = {}
    mesh.update({'x':np.arange(0.0, 1.0, 0.05)})

    mask = np.ones((mesh['x'].size - 1), int)
    mask[[10, -1]] = 0
    mesh.update({'mask':mask})

    mesh.update({'diffusivity':1e-4})
    mesh.update({'init-temp':10.0})
    mesh.update({'lambda':0})

    # boundary conditions are now given left to right
    bc = []
    bc.append({'type':'d', 'value':10.0})
    bc.append({'type':'d', 'value':15.0})
    bc.append({'type':'d', 'value':20.0})
    bc.append({'type':'n', 'value':0.0})

    t, u = mesh_1d(mesh=mesh, bc=bc, tf=3000, theta=0.5, max_courant=0.5)

    plot_temp_1d(x=mesh['x'],
                 t=t,
                 u=u,
                 show_final=True,
                 freq=50.0,
                 regions=find_regions(mesh['mask']))



def mesh_1d(mesh:dict, bc:dict, tf:float, theta:float=0.5, max_courant:float=0.5) -> tuple[np.ndarray, np.ndarray]:  # pylint:disable=line-too-long
    """Solve the time-dependent temperature profile through a 1D mesh
       with given thermal properties."""

    x = mesh['x']
    dx = x[1] - x[0]

    alpha = mesh['diffusivity']*np.ones_like(x)

    dt = min(tf / 2, max_courant*dx**2 / np.max(alpha))
    t = np.arange(0.0, tf, dt)

    bx_c = alpha*mesh['lambda']*(1 - theta)*dt / (2*dx)
    bx_n = alpha*mesh['lambda']*theta*dt / (2*dx)
    bxx_c = alpha*(1 - theta)*dt / dx**2
    bxx_n = alpha*theta*dt / dx**2

    print(f"Timestep set to {dt:0.1f} s to maintain maximum Courant number of {max_courant}.")

    u = np.zeros((t.size, mesh['x'].size), float) + mesh['init-temp']

    regions = find_regions(mesh['mask'])

    for n in range(1, t.size):
        for m, region in enumerate(regions):
            l = region[0]
            r = region[1]
            tl = bc[2*m]['type']
            vl = bc[2*m]['value']
            tr = bc[2*m+1]['type']
            vr = bc[2*m+1]['value']

            a = np.r_[0.0, -bxx_n[l:r-1] + bx_n[l:r-1] / x[l+1:r], 0.0 if tr == 'd' else -1.0]
            b = np.r_[1.0 if tl == 'd' else -1.0, 1 + 2*bxx_n[l+1:r], 1.0]
            c = np.r_[0.0 if tl == 'd' else 1.0, -bxx_n[l+2:r+1] + bx_n[l+2:r+1] / x[l+1:r], 0.0]

            d = np.r_[vl*(1 if tl == 'd' else dx),\
                    (bx_c[l+2:r+1]*u[n-1,l+2:r+1] - bx_c[l:r-1]*u[n-1,l:r-1]) / x[l+1:r] +\
                    bxx_c[l:r-1]*u[n-1,l:r-1] +\
                    (1 - 2*bxx_c[l+1:r])*u[n-1,l+1:r] +\
                    bxx_c[l+2:r+1]*u[n-1,l+2:r+1],\
                    vr*(1 if tr == 'd' else dx)]

            u[n, l:r+1] = tdma(u[n - 1, l:r+1], a, b, c, d)

    return t, u



def find_regions(mask:np.ndarray) -> list[list[int]]:
    """Gives index pairs for all contiguous regions in a line, given a 1D mask."""

    conv = (mask == 0)[:-1] + mask[1:]
    edges = np.ravel(np.sort(np.hstack((np.where(conv == 2), np.where(conv == 0))))) + 1

    if mask[0] == 1:
        edges = np.insert(edges, 0, 0)

    return [[edges[i], edges[i+1]] for i in range(0, edges.size, 2)]



simulate_1d()
