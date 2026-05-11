"""Simulates diffusion of heat through a uniform, rectangular 2D mesh and animates the results."""

from math import copysign
import numpy as np

from solvers import tdma
from plotter import plot_temp_2d

# TODO: change unit for Neumann boundaries to accept flux, maybe create new boundary, 'q'
# TODO: implement piecewise variable diffusivity
# TODO: implement masked region method

def simulate() -> None:
    """Simulates the transient temperature of a 2D mesh using a finite difference solver,
       then animates results."""

    # set up the mesh geometry
    mesh = {}
    mesh.update({'x':np.arange(0.035, 0.111, 0.003)})
    mesh.update({'y':np.arange(0.0, 0.15, 0.03)})
    mesh.update({'init-temp':288.0})
    mesh.update({'diffusivity':0.000000166667})
    mesh.update({'max-courant':0.5})
    mesh.update({'lambda':1}) # curvature parameter

    # boundary conditions
    bc = {}
    bc.update({'w':{'type':'n', 'value':-57481.17489}})
    bc.update({'e':{'type':'d', 'value':288}})
    bc.update({'s':{'type':'n', 'value':0}})
    bc.update({'n':{'type':'n', 'value':0}})

    t, u = mesh_2d(mesh=mesh, bc=bc, tf=10e3, theta=0.5)
    plot_temp_2d(mesh=mesh, t=t, u=u, show_final=False)



def mesh_2d(*, mesh:dict, bc:dict, tf:float, theta:float) -> tuple[np.ndarray, np.ndarray]:  # pylint:disable=too-many-locals
    """
    Calculates the transient temperature of a 2D, rectangular mesh,
    with variable scheme and curvature options.

    - To adjust the FD scheme, set the value 'theta' between 0 and 1 
      (0 is forward explicit, 1 is backward implicit, 0.5 is CN).
    - To adjust the mesh curvature, set l = 0 for planar, 1 for cylindrical, 
      (spherical will be handled in future).
    """

    x = mesh['x']
    y = mesh['y']
    dx = mesh['x'][1] - mesh['x'][0]
    dy = mesh['y'][1] - mesh['y'][0]
    alpha = mesh['diffusivity']*np.ones((y.size, x.size), float)

    dt = np.min([tf / 10, mesh['max-courant']*dx**2 / np.max(alpha),\
                 mesh['max-courant']*dy**2 / np.max(alpha)])

    print(f"Timestep set to {dt:0.1f} to maintain Courant numbers below {mesh['max-courant']}.")
    t = np.arange(0.0, tf, dt)

    bxx_c = alpha*dt*(1 - theta) / dx**2
    bxx_n = alpha*dt*theta / dx**2
    byy_c = 0 if mesh['lambda'] > 1 else (alpha*(1 - theta)*dt / dy**2)
    byy_n = 0 if mesh['lambda'] > 1 else (alpha*theta*dt / dy**2)
    bx_c = mesh['lambda']*alpha*(1 - theta)*dt / (2*dx)
    bx_n = mesh['lambda']*alpha*theta*dt / (2*dx)

    u = np.zeros((y.size, x.size, t.size), float)
    u[:,:,0] += mesh['init-temp']

    for n in range(1, t.size):

        u_mid = np.zeros((y.size, x.size), float)
        for j in range(1, y.size - 1):

            a = np.r_[0.0, -bxx_n[j,:-2] + bx_n[j,:-2] / x[1:-1], 0.0 if bc['e']['type'] == 'd'\
                      else -1.0]
            b = np.r_[1.0 if bc['w']['type'] == 'd' else -1.0, 1 + 2*bxx_n[j, 1: -1], 1.0]
            c = np.r_[0.0 if bc['w']['type'] == 'd' else 1.0, -bxx_n[j,2:] - bx_n[j,2:] / x[1:-1],\
                      0.0]

            d = np.r_[bc['w']['value'] if bc['w']['type'] == 'd' else dx*bc['w']['value'],\

                      byy_c[j-1,1:-1]*u[j-1,1:-1,n-1] +\
                      (1 - 2*byy_c[j,1:-1])*u[j,1:-1,n-1] +\
                      byy_c[j+1,1:-1]*u[j+1,1:-1,n-1],\

                      bc['e']['value'] if bc['e']['type'] == 'd' else dx*bc['e']['value']]

            u_mid[j,:] = tdma(u[j,:,n - 1], a, b, c, d)

        # row boundary lines:
        u_mid[0,:] = bc['s']['value'] + u_mid[0,:] if bc['s']['type'] == 'd' else\
                     u_mid[1,:] - bc['s']['value']*dy

        u_mid[-1,:] = bc['n']['value'] + u_mid[-1,:] if bc['n']['type'] == 'd' else\
                      u_mid[-2,:] + bc['n']['value']*dy

        u[:,:,n] = u_mid
        for i in range(1, x.size - 1):

            a = np.r_[0.0, -byy_n[:-2,i], 0.0 if bc['n']['type'] == 'd' else -1.0]
            b = np.r_[1.0 if bc['s']['type'] == 'd' else -1.0, 1+2*byy_n[1:-1,i], 1.0]
            c = np.r_[0.0 if bc['s']['type'] == 'd' else 1.0, -byy_n[2:,i], 0.0]

            d = np.r_[bc['s']['value'] if bc['s']['type'] == 'd' else dy*bc['s']['value'],\

                      (bxx_c[1:-1,i + 1] + bx_c[1:-1,i + 1] / x[i])*u_mid[1:-1,i + 1] + \
                      (1 - 2*bxx_c[1:-1,i])*u_mid[1:-1,i] +\
                      (bxx_c[1:-1,i - 1] - bx_c[1:-1,i - 1] / x[i])*u_mid[1:-1,i - 1],\

                      bc['n']['value'] if bc['n']['type'] == 'd' else dy*bc['n']['value']]

            u[:,i,n] = tdma(u_mid[:,i], a, b, c, d)

        # column boundary lines
        u[:,0,n] = np.zeros(y.size, float) + bc['w']['value'] if bc['w']['type'] == 'd' else\
                   u[:,1,n] - bc['w']['value']*dx

        u[:,-1,n] = np.zeros(y.size, float) + bc['e']['value'] if bc['e']['type'] == 'd' else\
                    u[:,-2,n] + bc['e']['value']*dx

    return t, u



def find_regions(axis:str, index:int, lines:list) -> list[list]:
    """Calculates internal regions of a slice and bounding line indices.

    lines is a list of index pairs that define the boundary lines
    
    k is the index along the normal axis of the sweep line.

    returns an array with elements in format: [p0, p1, l0, l1],
    where p is the index of one of the bounding lines, and l is that line's
    index.

    p is the direction index along the sweep line, n is normal direction index
    l is the index of a boundary's associated line
    desired output is array of elements (p0, p1, l0, l1)
    """

    a = 0 if axis == 'y' else 1 # used for line check calcs
    bp = []
    bi = []

    # check each line to see if it's a bounding line, recording index
    for l, line in enumerate(lines):

        is_normal = line[0][a] == line[1][a]
        d0 = line[0][1 - a] - index
        d1 = line[1][1 - a] - index

        if is_normal and ((copysign(1, d0) != copysign(1, d1)) or (d0*d1 == 0)):
            bp.append(line[0][a])
            bi.append(l)

    # build the output array:
    regions = [i for sublist in zip(bp, bi) for i in sublist]

    return regions



simulate()
