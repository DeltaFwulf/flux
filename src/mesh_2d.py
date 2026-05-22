"""Simulates multiple 2D meshes via linked boundary conditions."""

from os import getcwd
from os.path import join
import numpy as np
import yaml

from util import tdma, find_regions_2d, find_edges, bound_gradients
from plotter import plot_temp_2d

# FIXME: dirichlet boundary not updating properly, giving weird corner behaviour (top left)
# FIXME: non-equal mesh spacing leads to discontinous conduction temperature (check element areas)
# TODO: set all u values to NaN in void regions


def simulate(config:str) -> None:
    """Sets up the mesh definitions and handles simulation and plotting."""

    # load runtime settings and mesh definitions
    with open(config, 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.SafeLoader)

    with open(join(getcwd(), 'src', 'data', 'materials.yaml'), encoding='utf-8') as f:
        materials = yaml.load(stream=f, Loader=yaml.SafeLoader)

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

        m.update({'regions_x':[find_regions_2d(direction='x', ind_n=j, ind_p=m['i_arr'],\
            lines=m['lines']) for j in m['j_arr']]})
        m.update({'regions_y':[find_regions_2d(direction='y', ind_n=i, ind_p=m['j_arr'],\
            lines=m['lines']) for i in m['i_arr']]})

        # generate a mask array for this mesh

        # replace material string with material definition
        mat = materials.get(m['material'])
        if mat is None:
            print(f"Material {m['material']} is not found at /src/data/materials.yaml.")
            raise ValueError

        m.update({'material':mat})

        find_edges(m)

        m.update({'x':m['dx']*m['i_arr']})
        m.update({'y':m['dy']*m['j_arr']})

        # Meshes store 'u' for final results, u_latest for use in next timestep, u_last
        # for reference by other meshes.
        m.update({'u':np.zeros((m['i_arr'].size, m['j_arr'].size, 1), float) + m['u0']})
        m.update({'u_latest':m['u'][:, :, -1]})
        m.update({'u_last':m['u'][:, :, -1]})

        update_properties(m)

    t = np.array([0.0], float)
    t_now = 0.0

    while t_now < config['tf']:
        # calculate next timestep to satisfy courant constraint

        dt = min(config['dt_storage'], min(config['max_courant']*min(m['dx'],\
                 m['dy'])**2 / np.max(m['diffusivity']) for m in config['meshes'].values()))

        t_now += dt
        store = t_now - t[-1] >= config['dt_storage']
        if store:
            t = np.hstack((t, t_now))

        for key in config['meshes'].keys():

            m = config['meshes'][key]
            m['u_last'] = m['u_latest']
            update_properties(m)
            bound_gradients(config=config, mesh_name=key)

            m['u_latest'] = update_mesh(mesh=m,
                                        dt=dt,
                                        curv=m['curvature'],
                                        theta=config['theta'])

            if store:
                m['u'] = np.dstack((m['u'], m['u_latest']))

    plot_temp_2d(meshes=config['meshes'], t=t)



def update_mesh(*, mesh:dict, dt:float, curv:int, theta:float=0.5) -> np.ndarray:
    """Updates the state of a single mesh over a single timestep via the ADI method."""

    i_arr = mesh['i_arr']
    j_arr = mesh['j_arr']
    dx = mesh['dx']
    dy = mesh['dy']
    alpha = mesh['diffusivity']

    # calculate mesh coefficients
    bxx_c = alpha*dt*(1 - theta) / dx**2
    bxx_n = alpha*dt*theta / dx**2
    byy_c = 0 if curv > 1 else (alpha*(1 - theta)*dt / dy**2)
    byy_n = 0 if curv > 1 else (alpha*theta*dt / dy**2)
    bx_c = curv*alpha*(1 - theta)*dt / (2*dx)
    bx_n = curv*alpha*theta*dt / (2*dx)

    u_in = mesh['u_last']
    u_mid = np.zeros_like(u_in, float)

    # row slices (across x)
    for j in range(j_arr.size):
        for reg in mesh['regions_x'][j]:

            s = np.where(i_arr == reg['bounds'][0])[0][0]
            e = np.where(i_arr == reg['bounds'][1])[0][0]

            if reg['type'] == 'edge':

                edge = mesh['edges'][reg['bc']]
                bc = mesh['bc'][edge['line_index']]

                if bc['mode'] == 'dirichlet':
                    u_mid[s:e+1, j] = bc['value']
                else:
                    # apply gradient
                    d_edge = sum(edge['direction'])
                    u_mid[s:e+1,j] = u_in[s:e+1, j+d_edge] -\
                                     dy*mesh['gradients'][edge['line_index']]*d_edge

                continue

            bc_s = mesh['bc'][reg['bc_s']]
            bc_e = mesh['bc'][reg['bc_e']]
            grad_s = mesh['gradients'][reg['bc_s']][j - mesh['edges'][reg['bc_s']]['indices'][0]]
            grad_e = mesh['gradients'][reg['bc_e']][j - mesh['edges'][reg['bc_e']]['indices'][0]]

            a = np.r_[0.0, -bxx_n[s:e-1, j] + bx_n[s:e-1, j] / (dx*i_arr[s+1:e]), 0.0 if\
                        bc_e['mode'] == 'dirichlet' else -1.0]

            b = np.r_[1.0 if bc_s['mode'] == 'dirichlet' else -1.0, 1 + 2*bxx_n[s+1:e,j], 1.0]

            c = np.r_[0.0 if bc_s['mode'] == 'dirichlet' else 1.0, -bxx_n[s+2:e+1,j] -\
                        bx_n[s+2:e+1,j] / (dx*i_arr[s+1:e]), 0.0]

            d = np.r_[bc_s['value'] if bc_s['mode'] == 'dirichlet' else dx*grad_s,\

                        byy_c[s+1:e,j-1]*u_in[s+1:e,j-1] +\
                        (1 - 2*byy_c[s+1:e,j])*u_in[s+1:e,j] +\
                        byy_c[s+1:e,j+1]*u_in[s+1:e,j+1],\

                        bc_e['value'] if bc_e['mode'] == 'dirichlet' else dx*grad_e]

            u_mid[s:e+1,j] = tdma(u_in[s:e+1,j], a, b, c, d)

    # column slices (across y)
    u_out = u_mid
    for i in range(i_arr.size):
        for reg in mesh['regions_y'][i]:

            s = np.where(j_arr == reg['bounds'][0])[0][0]
            e = np.where(j_arr == reg['bounds'][1])[0][0]

            if reg['type'] == 'edge':

                edge = mesh['edges'][reg['bc']]
                bc = mesh['bc'][edge['line_index']]

                if bc['mode'] == 'dirichlet':
                    u_out[i,s:e+1] = bc['value']
                else:
                    d_edge = sum(edge['direction'])
                    u_out[i,s:e+1] = u_mid[i+d_edge, s:e+1] -\
                                     dx*mesh['gradients'][edge['line_index']]*d_edge

                continue

            bc_s = mesh['bc'][reg['bc_s']]
            bc_e = mesh['bc'][reg['bc_e']]
            grad_s = mesh['gradients'][reg['bc_s']][i - mesh['edges'][reg['bc_s']]['indices'][0]]
            grad_e = mesh['gradients'][reg['bc_e']][i - mesh['edges'][reg['bc_e']]['indices'][0]]

            a = np.r_[0.0, -byy_n[i, s:e-1], 0.0 if bc_e['mode'] == 'd' else -1.0]

            b = np.r_[1.0 if bc_s['mode'] == 'd' else -1.0, 1 + 2*byy_n[i, s+1:e], 1.0]

            c = np.r_[0.0 if bc_s['mode'] == 'd' else 1.0, -byy_n[i,s+2:e+1], 0.0]

            d = np.r_[bc_s['value'] if bc_s['mode'] == 'dirichlet' else dy*grad_s,\

                (bxx_c[i+1,s+1:e] + bx_c[i+1,s+1:e] / (dx*i_arr[i]))*u_mid[i+1,s+1:e] +\
                (1 - 2*bxx_c[i,s+1:e])*u_mid[i,s+1:e] +\
                (bxx_c[i-1,s+1:e] - bx_c[i-1,s+1:e] / (dx*i_arr[i]))*u_mid[i-1,s+1:e],

                bc_e['value'] if bc_e['mode'] == 'd' else dy*grad_e]

            u_out[i,s:e+1] = tdma(u_mid[i,s:e+1], a, b, c, d)

    return u_out



def update_properties(mesh:dict) -> None:
    """Updates the mesh's thermal properties (k, cp, rho) given temperature."""

    u = mesh['u_last']
    mat = mesh['material']

    k = np.interp(x=u, xp=mat['u'], fp=mat['k'])
    cp = np.interp(x=u, xp=mat['u'], fp=mat['cp'])
    rho = np.interp(x=u, xp=mat['u'], fp=mat['rho'])
    alpha = k / (rho*cp)

    mesh.update({'k':k, 'cp':cp, 'rho':rho, 'diffusivity':alpha})



simulate('/home/deltav/Documents/GitHub/flux/debug.yaml')
