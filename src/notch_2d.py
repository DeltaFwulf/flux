"""Simulates transient heat flow through a 2D mesh, then animates results."""

from math import copysign, ceil
import numpy as np

from solvers import tdma
from plotter import plot_temp_2d

# TODO: change unit for Neumann boundaries to accept flux, maybe create new boundary, 'q'
# TODO: implement piecewise variable diffusivity as material function
# TODO: create efficient method for blocking out rectangular subregions for plotting purposes

# TODO: save memory by only storing every n iterations. 
# Can save every 1 second but simulate to maintain max Courant number



def simulate() -> None:
    """Simulates the transient temperature of a 2D mesh using a finite difference solver,
       then animates results."""

    dx = 0.005
    dy = 0.005

    # each line is represented by a pair of coordinates: ((xa, ya), (xb, yb))
    # lines = [((0.0, 0.0), (1.0, 0.0)),
    #          ((1.0, 0.0), (1.0, 2.0)),
    #          ((1.0, 2.0), (0.0, 2.0)),
    #          ((0.0, 2.0), (0.0, 0.0)),
    #          ((0.2, 0.4), (0.8, 0.4)),
    #          ((0.8, 0.4), (0.8, 1.6)),
    #          ((0.8, 1.6), (0.2, 1.6)),
    #          ((0.2, 1.6), (0.2, 0.4))]

    # chamber brick with one notch
    lines = [((0.035, 0.0), (0.111, 0.0)),
             ((0.111, 0.0), (0.111, 0.210)),
             ((0.111, 0.210), (0.035, 0.210)),
             ((0.035, 0.210), (0.035, 0.192)),
             ((0.035, 0.192), (0.045, 0.192)),
             ((0.045, 0.192), (0.045, 0.102)),
             ((0.045, 0.102), (0.035, 0.102)),
             ((0.035, 0.102), (0.035, 0.0))]

    # snap lines to the grid and convert to mesh indices
    lines = [tuple((round(p[0] / dx), round(p[1] / dy)) for p in l) for l in lines]

    i_min = min(p[0] for l in lines for p in l)
    i_max = max(p[0] for l in lines for p in l)
    j_min = min(p[1] for l in lines for p in l)
    j_max = max(p[1] for l in lines for p in l)

    i_arr = np.arange(i_min, i_max + 1, 1)
    j_arr = np.arange(j_min, j_max + 1, 1)

    # step 5: define boundary conditions for each line (how tf to automate this?)
    bc = []
    # outer ring
    bc.append({'type':'n', 'value':0.0})
    bc.append({'type':'n', 'value':0.0})
    bc.append({'type':'n', 'value':0.0})
    bc.append({'type':'n', 'value':0.0})
    # inner ring
    bc.append({'type':'d', 'value':100.0})
    bc.append({'type':'d', 'value':100.0})
    bc.append({'type':'d', 'value':100.0})
    bc.append({'type':'n', 'value':0.0})

    # step 6: calculate regions
    regions_x = [find_regions_2d(p='x', ind_n=j, ind_p=i_arr, lines=lines) for j in j_arr]
    regions_y = [find_regions_2d(p='y', ind_n=i, ind_p=j_arr, lines=lines) for i in i_arr]

    # step 7: create the mesh
    mesh = {}
    mesh.update({'i_arr':i_arr, 'j_arr':j_arr})
    mesh.update({'x':dx*i_arr, 'y':dy*j_arr}) # TODO: just have plotter do this for us
    mesh.update({'lines':lines})
    mesh.update({'boundary-conditions':bc})
    mesh.update({'dx':dx, 'dy':dy})
    mesh.update({'regions_x':regions_x, 'regions_y':regions_y})
    mesh.update({'init-temp':0.0})
    mesh.update({'diffusivity':1e-4})
    mesh.update({'lambda':0}) # curvature factor

    # step 8: simulate mesh
    t, u = mesh_2d(mesh=mesh, bc=bc, tf=100, theta=0.5, max_courant=0.5, dt_storage=1.0)

    # step 9: animate results
    plot_temp_2d(mesh=mesh, t=t, u=u, show_final=False)



def mesh_2d(*, mesh:dict, bc:dict, tf:float, theta:float=0.5, max_courant:float=0.5, **kwargs) -> tuple[np.ndarray, np.ndarray]:  # pylint:disable=too-many-locals
    """
    Calculates the transient temperature of a 2D, rectangular mesh,
    with variable scheme and curvature options.

    - To adjust the FD scheme, set the value 'theta' between 0 and 1 
      (0 is forward explicit, 1 is backward implicit, 0.5 is CN).
    - To adjust the mesh curvature, set l = 0 for planar, 1 for cylindrical, 
      (spherical will be handled in future).
    """

    i_arr = mesh['i_arr']
    j_arr = mesh['j_arr']
    dx = mesh['dx']
    dy = mesh['dy']
    alpha = mesh['diffusivity']*np.ones((i_arr.size, j_arr.size), float)

    # Memory-saving storage
    # dt_str = kwargs.get('dt_storage')
    dt_sim = np.min([tf / 10, max_courant*dx**2 / np.max(alpha),\
                 max_courant*dy**2 / np.max(alpha)])

    # # set dt_sim as a simple fraction of dt_str, so we can use a loop counter
    # k_str = 1
    # if dt_str is not None:
    #     k_str = ceil(dt_str / dt_sim)
    #     dt_sim /= k_str

    print(f"Timestep set to {dt_sim:0.1f} to maintain Courant numbers below {max_courant}.")
    t = np.arange(0.0, tf, dt_sim)

    bxx_c = alpha*dt_sim*(1 - theta) / dx**2
    bxx_n = alpha*dt_sim*theta / dx**2
    byy_c = 0 if mesh['lambda'] > 1 else (alpha*(1 - theta)*dt_sim / dy**2)
    byy_n = 0 if mesh['lambda'] > 1 else (alpha*theta*dt_sim / dy**2)
    bx_c = mesh['lambda']*alpha*(1 - theta)*dt_sim / (2*dx)
    bx_n = mesh['lambda']*alpha*theta*dt_sim / (2*dx)

    u = np.zeros((i_arr.size, j_arr.size, t.size), float)
    u[:,:,0] = mesh['init-temp']

    for n in range(1, t.size):

        u_mid = np.zeros((i_arr.size, j_arr.size), float)

        # row slices (across x)
        for j in range(j_arr.size):
            for reg in mesh['regions_x'][j]:

                s = np.where(i_arr == reg['bounds'][0])[0][0]
                e = np.where(i_arr == reg['bounds'][1])[0][0]

                if reg['type'] == 'edge':
                    # solve appropriate boundary condition based on prev ts state
                    bc = mesh['boundary-conditions'][reg['bc']]

                    u_mid[s:e+1, j] = bc['value'] if bc['type'] == 'd' else\
                        u[s:e+1, j+reg['direction'], n-1] - bc['value']*dy*reg['direction']

                    continue

                # solve internal region using TDMA from s to e inclusive
                bc_s = mesh['boundary-conditions'][reg['bc_s']]
                bc_e = mesh['boundary-conditions'][reg['bc_e']]

                a = np.r_[0.0, -bxx_n[s:e-1, j] + bx_n[s:e-1, j] / (dx*i_arr[s+1:e]), 0.0 if\
                          bc_e['type'] == 'd' else -1.0]

                b = np.r_[1.0 if bc_s['type'] == 'd' else -1.0, 1 + 2*bxx_n[s+1:e,j], 1.0]

                c = np.r_[0.0 if bc_s['type'] == 'd' else 1.0, -bxx_n[s+2:e+1,j] -\
                          bx_n[s+2:e+1,j] / (dx*i_arr[s+1:e]), 0.0]

                d = np.r_[bc_s['value'] if bc_s['type'] == 'd' else dx*bc_s['value'],\

                          byy_c[s+1:e,j-1]*u[s+1:e,j-1,n-1] +\
                          (1 - 2*byy_c[s+1:e,j])*u[s+1:e,j,n-1] +\
                          byy_c[s+1:e,j+1]*u[s+1:e,j+1,n-1],\

                          bc_e['value'] if bc_e['type'] == 'd' else dx*bc_e['value']]

                u_mid[s:e+1,j] = tdma(u[s:e+1,j,n-1], a, b, c, d)

        # column slices (across y)
        u[:,:,n] = u_mid
        for i in range(i_arr.size):
            for reg in mesh['regions_y'][i]:

                s = np.where(j_arr == reg['bounds'][0])[0][0]
                e = np.where(j_arr == reg['bounds'][1])[0][0]

                if reg['type'] == 'edge':
                    bc = mesh['boundary-conditions'][reg['bc']]

                    u[i,s:e+1,n] = bc['value'] if bc['type'] == 'd' else\
                        u[i+reg['direction'], s:e+1, n-1] - bc['value']*dx*reg['direction']

                    continue

                bc_s = mesh['boundary-conditions'][reg['bc_s']]
                bc_e = mesh['boundary-conditions'][reg['bc_e']]

                a = np.r_[0.0, -byy_n[i, s:e-1], 0.0 if bc_e['type'] == 'd' else -1.0]

                b = np.r_[1.0 if bc_s['type'] == 'd' else -1.0, 1 + 2*byy_n[i, s+1:e], 1.0]

                c = np.r_[0.0 if bc_s['type'] == 'd' else 1.0, -byy_n[i,s+2:e+1], 0.0]

                d = np.r_[bc_s['value'] if bc_s['type'] == 'd' else dy*bc_s['value'],\

                    (bxx_c[i+1,s+1:e] + bx_c[i+1,s+1:e] / (dx*i_arr[i]))*u_mid[i+1,s+1:e] +\
                    (1 - 2*bxx_c[i,s+1:e])*u_mid[i,s+1:e] +\
                    (bxx_c[i-1,s+1:e] - bx_c[i-1,s+1:e] / (dx*i_arr[i]))*u_mid[i-1,s+1:e],

                    bc_e['value'] if bc_e['type'] == 'd' else dy*bc_e['value']]

                u[i,s:e+1,n] = tdma(u_mid[i,s:e+1], a, b, c, d)

    return t, u



def find_regions_2d(p:str, ind_n:int, ind_p:np.ndarray, lines:list) -> list[list]:
    """
    Uses a transition method to identify all regions in this slice.

    Regions are either 'i', internal or 'b', bounding.

    Internal regions are solved for in the TDMA steps.
    Bounding regions have boundary conditions applied in the BC step, in the correct direction.
    
    boundary lines have a direction, 'l' or 'r' in which to apply the boundary.
    'l' is descending index, 'r' is ascending index.
    
    ind_n is the slice's index in the normal axis
    ind_p is the array of indices along the parallel direction

    each region has a type and bounding points. If edge it has a direction and boundary number
    {'type':'internal', 'bounds':(pa, pb)}
    {'type':'edge', 'bounds':(pa, pb), 'direction':1, 'bc_index':5}
    direction is +1 for ascending order, -1 for descending. (towards next internal region)
    """

    a = 0 if p == 'x' else 1 # gives line coordinate to inspect for normality
    transitions = []
    m = 0

    for k in ind_p:

        for l, line in enumerate(lines):

            is_normal = (line[0][a] == line[1][a]) and k in (line[0][a], line[1][a])

            da = line[0][1 - a] - ind_n
            db = line[1][1 - a] - ind_n

            spans = copysign(1, da) != copysign(1, db) or ind_n in (line[0][1 - a], line[1][1 - a])

            if is_normal and spans:

                direction = -1

                if da*db != 0:
                    direction = 2
                elif da > ind_n or db > ind_n:
                    direction = 1

                if m > 1 and transitions[-1][3] == transitions[-2][3] == 2:
                    m = 0

                transitions.append([k, l, m, direction])
                m += 1
                break

    regions = []
    for i, transition in enumerate(transitions):  # construct regions from transition array

        if transition[2] == 0:
            continue

        pa = (transitions[i-1][0], ind_n) if a == 0 else (ind_n, transitions[i-1][0])
        pb = (transition[0], ind_n) if a == 0 else (ind_n, transition[0])
        reg = {'bounds':(pa[a], pb[a])}

        if 2 in (transition[-1], transitions[i-1][-1]) or\
                (len(regions) > 0 and regions[-1]['type'] == 'edge' and m > 1):

            reg.update({'type':'internal', 'bc_s':transitions[i-1][1], 'bc_e':transition[1]})
            regions.append(reg)
            continue

        d = transitions[i - 1][3] if transition[2] == 1 else -transitions[i - 1][3]

        for l, line in enumerate(lines):  # find the boundary condition (contains both pa, pb)
            if pa in line and pb in line:
                break

        reg.update({'type':'edge', 'direction':d, 'bc':l})
        regions.append(reg)

    return regions



simulate()
