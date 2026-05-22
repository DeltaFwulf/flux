"""Simulates multiple 2D meshes via linked boundary conditions."""

from os import getcwd
from os.path import join
import numpy as np
import yaml

from solvers import tdma, find_regions_2d
from plotter import plot_temp_2d



def simulate(config:str) -> None:
    """Sets up the mesh definitions and handles simulation and plotting."""

    # load runtime settings and mesh definitions
    with open(join(getcwd(), 'src', config), 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.SafeLoader)

    # process meshes
    for m in config['meshes'].values():

        # snap lines to the grid and convert to mesh indices
        m['lines'] = [tuple((round(p[0] / m['dx']), round(p[1] / m['dy'])) for p in l)\
                 for l in m['lines']]

        i_min = min(p[0] for l in m['lines'] for p in l)
        i_max = max(p[0] for l in m['lines'] for p in l)
        j_min = min(p[1] for l in m['lines'] for p in l)
        j_max = max(p[1] for l in m['lines'] for p in l)

        m.update({'i_arr':np.arange(i_min, i_max + 1, 1)})
        m.update({'j_arr':np.arange(j_min, j_max + 1, 1)})

        m.update({'regions_x':[find_regions_2d(p='x', ind_n=j, ind_p=m['i_arr'],\
            lines=m['lines']) for j in m['j_arr']]})
        m.update({'regions_y':[find_regions_2d(p='y', ind_n=i, ind_p=m['j_arr'],\
            lines=m['lines']) for i in m['i_arr']]})

        m.update({'x':m['dx']*m['i_arr']})
        m.update({'y':m['dy']*m['j_arr']})
        m.update({'u':np.zeros((m['i_arr'].size, m['j_arr'].size, 1), float) + m['u0']})

    # process conditions / whatever I want to call that

    t = np.array([0.0], float)
    t_now = 0.0

    while t_now <= config['tf']:
        # calculate next timestep given current diffusivities
        dt = min(config['max_courant']*min(m['dx'], m['dy'])**2 / np.max(m['diffusivity'])\
                 for m in config['meshes'].values())

        t_now += dt
        store = t_now - t[-1] >= config['dt_storage']
        if store:
            t = np.hstack((t, t_now))

        # calculate boundary condition values


        for m in config['meshes'].values():

            u_new = update_mesh(mesh=m,
                                 dt=dt,
                                 curv=config['lambda'],
                                 theta=config['theta'])

            if store:
                m['u'] = np.dstack((m['u'], u_new) )

            # update mesh diffusivity from material definition

    plot_temp_2d(meshes=config['meshes'], t=t)



def update_mesh(*, mesh:dict, dt:float, curv:int, theta:float=0.5) -> np.ndarray:
    """Updates the state of a single mesh over a single timestep via the ADI method."""

    i_arr = mesh['i_arr']
    j_arr = mesh['j_arr']
    dx = mesh['dx']
    dy = mesh['dy']

    # Calculate diffusivity from material, temperature
    alpha = mesh['diffusivity']*np.ones((i_arr.size, j_arr.size), float)

    # calculate mesh coefficients
    bxx_c = alpha*dt*(1 - theta) / dx**2
    bxx_n = alpha*dt*theta / dx**2
    byy_c = 0 if curv > 1 else (alpha*(1 - theta)*dt / dy**2)
    byy_n = 0 if curv > 1 else (alpha*theta*dt / dy**2)
    bx_c = curv*alpha*(1 - theta)*dt / (2*dx)
    bx_n = curv*alpha*theta*dt / (2*dx)

    u_in = mesh['u'][:, :, -1]
    u_mid = np.zeros((i_arr.size, j_arr.size), float)
    u_out = np.zeros((i_arr.size, j_arr.size), float)

    # row slices (across x)
    for j in range(j_arr.size):
        for reg in mesh['regions_x'][j]:

            s = np.where(i_arr == reg['bounds'][0])[0][0]
            e = np.where(i_arr == reg['bounds'][1])[0][0]

            if reg['type'] == 'edge':

                bc = mesh['bc'][reg['bc']]

                u_mid[s:e+1, j] = bc['value'] if bc['type'] == 'd' else\
                    u_in[s:e+1, j+reg['direction']] - bc['value']*dy*reg['direction']

                continue

            bc_s = mesh['bc'][reg['bc_s']]
            bc_e = mesh['bc'][reg['bc_e']]

            a = np.r_[0.0, -bxx_n[s:e-1, j] + bx_n[s:e-1, j] / (dx*i_arr[s+1:e]), 0.0 if\
                        bc_e['type'] == 'd' else -1.0]

            b = np.r_[1.0 if bc_s['type'] == 'd' else -1.0, 1 + 2*bxx_n[s+1:e,j], 1.0]

            c = np.r_[0.0 if bc_s['type'] == 'd' else 1.0, -bxx_n[s+2:e+1,j] -\
                        bx_n[s+2:e+1,j] / (dx*i_arr[s+1:e]), 0.0]

            d = np.r_[bc_s['value'] if bc_s['type'] == 'd' else dx*bc_s['value'],\

                        byy_c[s+1:e,j-1]*u_in[s+1:e,j-1] +\
                        (1 - 2*byy_c[s+1:e,j])*u_in[s+1:e,j] +\
                        byy_c[s+1:e,j+1]*u_in[s+1:e,j+1],\

                        bc_e['value'] if bc_e['type'] == 'd' else dx*bc_e['value']]

            u_mid[s:e+1,j] = tdma(u_in[s:e+1,j], a, b, c, d)

    # column slices (across y)
    u_out = u_mid
    for i in range(i_arr.size):
        for reg in mesh['regions_y'][i]:

            s = np.where(j_arr == reg['bounds'][0])[0][0]
            e = np.where(j_arr == reg['bounds'][1])[0][0]

            if reg['type'] == 'edge':
                bc = mesh['bc'][reg['bc']]

                u_out[i,s:e+1] = bc['value'] if bc['type'] == 'd' else\
                    u_in[i+reg['direction'], s:e+1] - bc['value']*dx*reg['direction']

                continue

            bc_s = mesh['bc'][reg['bc_s']]
            bc_e = mesh['bc'][reg['bc_e']]

            a = np.r_[0.0, -byy_n[i, s:e-1], 0.0 if bc_e['type'] == 'd' else -1.0]

            b = np.r_[1.0 if bc_s['type'] == 'd' else -1.0, 1 + 2*byy_n[i, s+1:e], 1.0]

            c = np.r_[0.0 if bc_s['type'] == 'd' else 1.0, -byy_n[i,s+2:e+1], 0.0]

            d = np.r_[bc_s['value'] if bc_s['type'] == 'd' else dy*bc_s['value'],\

                (bxx_c[i+1,s+1:e] + bx_c[i+1,s+1:e] / (dx*i_arr[i]))*u_mid[i+1,s+1:e] +\
                (1 - 2*bxx_c[i,s+1:e])*u_mid[i,s+1:e] +\
                (bxx_c[i-1,s+1:e] - bx_c[i-1,s+1:e] / (dx*i_arr[i]))*u_mid[i-1,s+1:e],

                bc_e['value'] if bc_e['type'] == 'd' else dy*bc_e['value']]

            u_out[i,s:e+1] = tdma(u_mid[i,s:e+1], a, b, c, d)

    return u_out



simulate('testrun.yaml')
