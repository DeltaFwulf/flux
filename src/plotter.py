"""Contains functions for plotting results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_temp_1d(*, x:np.ndarray, t:np.ndarray, u:np.ndarray, **kwargs) -> None:
    """Animates the transient temperature of a 1D mesh.
    
    To overlay the final mesh temperature throughout the animation, pass
    the keyword argument 'show_final' as True.

    Set the frequency of frames with the 'freq' keyword argument.

    If a mask is sent with the kwarg 'regions', only nodes in regions will
    be plotted.
    """


    def update(frame) -> None:
        ax.clear()

        regions = kwargs['regions'] if kwargs.get('regions') is not None else [[0, x.size - 1]]
        for i, r in enumerate(regions):
            ax.plot(x[r[0]:r[1]+1],
                    u[frame,r[0]:r[1]+1],
                    linestyle='-',
                    color='red',
                    label=f"transient, region {i}")

            if kwargs.get('show_final') is True:
                ax.plot(x[r[0]:r[1]+1],
                        u[-1,r[0]:r[1]+1],
                        linestyle='--',
                        color='b',
                        label=f"final, region {i}")

        ax.set_xlabel("x, m")
        ax.set_ylabel("u, K")
        ax.set_ylim(np.min(u), np.max(u)*1.05)

        ax.set_title(f"Temperature of a 1D Mesh @ t = {t[frame]:0.1f} s")
        ax.legend()
        ax.grid(True)


    fig, ax = plt.subplots()
    interval = 50 if not isinstance(kwargs.get('freq'), (int, float)) else 1000.0 / kwargs['freq']
    _ = animation.FuncAnimation(fig=fig, func=update, frames=t.size, interval=interval, blit=False)
    plt.show()



def plot_temp_2d(*, meshes:dict, t:np.ndarray, **kwargs) -> None:
    """Animates the temperatures of multiple meshes, formed from rectangular elements."""


    def update(frame):
        ax.clear()

        for k, m in meshes.items():
            ax.plot_surface(xm[k], ym[k], m['u'][:,:,frame].transpose(),
                            edgecolor='black',
                            cmap='magma',
                            norm=norm,
                            alpha=0.5,
                            linewidth=0.8)

            ax.contour(xm[k], ym[k], m['u'][:, :, frame].transpose(),\
                       zdir='z', offset=u_min, cmap='coolwarm')

            ax.contour(xm[k], ym[k], m['u'][:, :, frame].transpose(),\
                       zdir='x', offset=min(m['x']) - side_offset, cmap='coolwarm')

        ax.set_zlim(u_min, u_max)
        ax.set_aspect('equalxy')
        ax.set_xlabel('x, m')
        ax.set_ylabel('y, m')
        ax.set_zlabel('u, K')
        ax.set_title(f"Mesh Temperature at t = {t[frame]:0.1f} s")


    u_min = min(np.min(m['u']) for m in meshes.values())
    u_max = max(np.max(m['u']) for m in meshes.values())
    side_offset = max(m['dx'] for m in meshes.values())

    xm = {}
    ym = {}

    for k in meshes.keys():
        xk, yk = np.meshgrid(meshes[k]['x'], meshes[k]['y'])
        xm.update({k:xk})
        ym.update({k:yk})

    plt.style.use('dark_background')
    norm = plt.Normalize(u_min, u_max)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    fig.set_tight_layout(True)

    interval = 50 if kwargs.get('interval') is not float else kwargs['interval']
    _ = animation.FuncAnimation(fig, update, frames=t.size, interval=interval, blit=False)
    plt.show()



def plot2d_flat(meshes:dict, t:np.ndarray, **kwargs) -> None:
    """Plots transient mesh temperatures as pcolormesh, with mesh outlines."""

    u_min = min(np.min(m['u']) for m in meshes.values())
    u_max = max(np.max(m['u']) for m in meshes.values())
    norm = plt.Normalize(u_min, u_max)

    fig, ax_transient = plt.subplots()
    fig.set_tight_layout(True)

    xm = {}
    ym = {}
    for k, m in meshes.items():
        xk, yk = np.meshgrid(m['x'], m['y'])
        xm.update({k:xk})
        ym.update({k:yk})

    def update(frame):
        """Updates the axis frame."""
        ax_transient.clear()

        for k, m in meshes.items():
            ax_transient.pcolormesh(xm[k],
                                    ym[k],
                                    m['u'][:, :, frame].transpose(),
                                    cmap='magma',
                                    norm=norm)

            # draw the lines
            for line in m['lines']:
                ax_transient.plot(np.array([line[0][0], line[1][0]])*m['dx'],
                                  np.array([line[0][1], line[1][1]])*m['dy'],
                                  linestyle='-',
                                  color='black')

        ax_transient.set_aspect('equal')
        ax_transient.set_xlabel('x, m')
        ax_transient.set_ylabel('y, m')
        ax_transient.set_title(f"Mesh Temperature at t = {t[frame]:0.1f} s")


    interval = 50 if kwargs.get('interval') is not float else kwargs['interval']
    _ = animation.FuncAnimation(fig, update, frames=t.size, interval=interval, blit=False)
    plt.show()



def setup_visualiser(config:dict) -> None:
    """Plots mesh boundaries and links for visual inspection."""

    pass
