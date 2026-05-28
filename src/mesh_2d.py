"""Simulates multiple 2D meshes via linked boundary conditions."""

import numpy as np

from util import tdma, calc_edge_states, update_properties



def simulate_2d(inputs:dict) -> None:
    """Sets up the mesh definitions and handles simulation and plotting."""

    t = np.array([0.0], float)
    t_now = 0.0

    while t_now < inputs['tf']:
        # calculate next timestep to satisfy courant constraint

        dt = min(inputs['dt_storage'], min(inputs['max_courant']*min(m['dx'],\
                 m['dy'])**2 / np.max(m['diffusivity']) for m in inputs['meshes'].values()))

        t_now += dt
        store = t_now - t[-1] >= inputs['dt_storage']
        if store:
            t = np.hstack((t, t_now))

        for m in inputs['meshes'].values():

            m['u_last'] = m['u_latest']
            update_properties(m)
            #bound_gradients(config=inputs, meshkey=key)
            calc_edge_states(cfg=inputs)

            m['u_latest'] = update_mesh(mesh=m,
                                        dt=dt,
                                        curv=m['curvature'],
                                        theta=inputs['theta'])

            if store:
                m['u'] = np.dstack((m['u'], m['u_latest']))

    outputs = {}
    outputs.update({'meshes':inputs['meshes']})
    outputs.update({'t':t})

    return outputs



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

    # FIXME: plug in new edge state logic

    # row slices (across x)
    for j in range(j_arr.size):
        for reg in mesh['regions_x'][j]:

            s = np.where(i_arr == reg['bounds'][0])[0][0]
            e = np.where(i_arr == reg['bounds'][1])[0][0]

            if reg['type'] == 'edge':

                edge = mesh['edges'][reg['bc']]
                # bc = mesh['bc'][edge['line_index']]

                # if bc['mode'] == 'dirichlet':
                #     u_mid[s:e+1, j] = bc['value']
                # else:
                #     # apply gradient
                #     d_edge = sum(edge['direction'])
                #     u_mid[s:e+1,j] = u_in[s:e+1, j+d_edge] -\
                #                      dy*mesh['gradients'][edge['line_index']]*d_edge

                edge_state = mesh['edge_states'][reg['bc']]

                if edge_state['type'] == 'direct':
                    u_mid[s:e+1, j] = edge_state['values']
                else:
                    d_edge = sum(edge['direction'])
                    u_mid[s:e+1, j] = u_in[s:e+1, j+d_edge] - dy*edge_state['values']*d_edge

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

            a = np.r_[0.0, -byy_n[i, s:e-1], 0.0 if bc_e['mode'] == 'dirichlet' else -1.0]

            b = np.r_[1.0 if bc_s['mode'] == 'dirichlet' else -1.0, 1 + 2*byy_n[i, s+1:e], 1.0]

            c = np.r_[0.0 if bc_s['mode'] == 'dirichlet' else 1.0, -byy_n[i,s+2:e+1], 0.0]

            d = np.r_[bc_s['value'] if bc_s['mode'] == 'dirichlet' else dy*grad_s,\

                (bxx_c[i+1,s+1:e] + bx_c[i+1,s+1:e] / (dx*i_arr[i]))*u_mid[i+1,s+1:e] +\
                (1 - 2*bxx_c[i,s+1:e])*u_mid[i,s+1:e] +\
                (bxx_c[i-1,s+1:e] - bx_c[i-1,s+1:e] / (dx*i_arr[i]))*u_mid[i-1,s+1:e],

                bc_e['value'] if bc_e['mode'] == 'dirichlet' else dy*grad_e]

            u_out[i,s:e+1] = tdma(u_mid[i,s:e+1], a, b, c, d)

    return u_out
