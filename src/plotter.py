"""Contains functions for plotting results."""

from os.path import join, exists
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



def animate_temp_2d(results:dict, **kwargs) -> None:
    """Animates the temperatures of multiple meshes, formed from rectangular elements."""


    def update(frame):
        ax.clear()

        for k, m in meshes.items():

            u_masked = np.ma.masked_where(masks[k], m['u'][:,:,frame]).transpose()
            ax.plot_surface(xm[k], ym[k], u_masked,
                            edgecolor=m['colour'],
                            cmap='magma',
                            norm=norm,
                            alpha=1.0,
                            linewidth=0.5)

            ax.contour(xm[k], ym[k], u_masked,\
                       zdir='x',
                       offset=min(m['x']) - side_offset,
                       cmap='coolwarm')

        ax.set_zlim(u_min, u_max)
        ax.set_aspect('equalxy')
        ax.set_xlabel('x, m')
        ax.set_ylabel('y, m')
        ax.set_zlabel('u, K')
        ax.set_title(f"Mesh Temperature at t = {t[frame]:0.1f} s")


    meshes = results['meshes']
    t = results['t']

    u_min = min(np.min(m['u']) for m in meshes.values())
    u_max = max(np.max(m['u']) for m in meshes.values())
    side_offset = max(m['dx'] for m in meshes.values())

    xm = {}
    ym = {}
    masks = {}
    for k, m in meshes.items():
        xk, yk = np.meshgrid(m['x'], m['y'])
        xm.update({k:xk})
        ym.update({k:yk})
        masks.update({k:mask_void_regions(m)})

    plt.style.use('dark_background')
    norm = plt.Normalize(u_min, u_max)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    fig.set_tight_layout(True)

    interval = 50 if kwargs.get('interval') is not float else kwargs['interval']
    ani = animation.FuncAnimation(fig, update, frames=t.size, interval=interval, blit=False)

    saved = False
    if kwargs.get('save'):
        saved = save_figure(ani)

    if not saved:
        plt.show()



def plot2d_flat(results:dict, **kwargs) -> None:
    """Plots transient mesh temperatures as pcolormesh, with mesh outlines."""

    t = results['t']

    u_min = min(np.min(m['u']) for m in results['meshes'].values())
    u_max = max(np.max(m['u']) for m in results['meshes'].values())

    norm = plt.Normalize(u_min, u_max)

    plt.style.use('dark_background')
    fig, ax_transient = plt.subplots(layout='compressed')

    xm = {}
    ym = {}
    masks = {}
    for k, m in results['meshes'].items():
        xk, yk = np.meshgrid(m['x'], m['y'])
        xm.update({k:xk})
        ym.update({k:yk})
        masks.update({k:mask_void_regions(m)})


    def plot_mesh(ax, xm:np.ndarray, ym:np.ndarray, u:np.ndarray):
        """Draws a pcolormesh and returns the object"""
        return ax.pcolormesh(xm, ym, u, shading='gouraud', cmap='magma', norm=norm)


    def update(frame):
        """Draws next animation frame."""

        ax_transient.clear()

        for k, m in results['meshes'].items():

            u_masked = np.ma.masked_where(masks[k], m['u'][:,:,frame]).transpose()
            plot_mesh(ax_transient, xm[k], ym[k], u_masked)

            for li in m['line_indices']:
                ax_transient.plot(np.array([li[0][0], li[1][0]])*m['dx'],
                                  np.array([li[0][1], li[1][1]])*m['dy'],
                                  linestyle='-',
                                  color=m['colour'])

        ax_transient.set_aspect('equal')
        ax_transient.set_xlabel('x, m')
        ax_transient.set_ylabel('y, m')
        ax_transient.set_title(f"Mesh Temperature at t = {t[frame]:0.1f} s")


    pcm = plot_mesh(ax_transient, xm[k], ym[k], m['u'][:,:,0].transpose())
    fig.colorbar(mappable=pcm).set_label("Temperature (K)")

    interval = 50 if kwargs.get('interval') is not float else kwargs['interval']
    ani = animation.FuncAnimation(fig, update, frames=t.size, interval=interval, blit=False)

    saved = False
    if kwargs.get('save'):
        saved = save_figure(ani)

    if not saved:
        plt.show()



def plot_total_powers(results:dict):
    """Temporary plotter for mesh total powers."""
    
    # TODO: make grid lines thinner

    fig, axs = plt.subplots(2, 2)
    ax_pwr = axs[0, 0]
    ax_energy = axs[1, 0]
    ax_mean_temp = axs[0, 1]
    ax_flux = axs[1, 1]
    dt = np.r_[0, results['t'][1:] - results['t'][:-1]]

    for mesh in results['meshes'].values():

        total_power = 0.0
        for p in mesh['edge_powers']:
            total_power += p

        ax_pwr.plot(results['t'], total_power, '-', label=mesh['label'])
        ax_energy.plot(results['t'], np.cumsum(dt*total_power), '-*', label=mesh['label'] + ' stored')
        ax_energy.plot(results['t'], mesh['net_energy'], '-', label=mesh['label'] + ' sub-stepped')
        ax_mean_temp.plot(results['t'], np.mean(mesh['u'], axis=(0, 1), dtype=np.float64), '-', label=mesh['label'])

        for l, flux in enumerate(mesh['edge_fluxes']):
            ax_flux.plot(results['t'], flux, label=l)

    ax_pwr.set_xlabel("Time, s")
    ax_pwr.set_ylabel("Power, W")
    ax_pwr.legend()
    ax_pwr.grid(True)

    ax_energy.set_xlabel("Time, s")
    ax_energy.set_ylabel("Energy Change, J")
    ax_energy.legend()
    ax_energy.grid(True)

    ax_mean_temp.set_xlabel("Time, s")
    ax_mean_temp.set_ylabel("Mean Temperature, K")
    ax_mean_temp.legend()
    ax_mean_temp.grid(True)

    ax_flux.set_xlabel("Time, s")
    ax_flux.set_ylabel("Flux by edge, W/m^2")
    ax_flux.legend()
    ax_flux.grid(True)

    fig.tight_layout()
    plt.show()


def plot_steady_slice(results:dict) -> None:
    """Plots the pipe's final temperature distribution.

    This is compared to the theoretical final temperature distribution
    to evaluate the simulation's accuracy.
    
    ONLY to be used with the pipe.yaml case!!!
    """

    fig, ax = plt.subplots()

    for k, m in results['meshes'].items():

        u_slice = m['u'][:, :, -1]
        u_slice = u_slice[:, np.shape(u_slice)[1] // 2]
        du = u_slice[0] - u_slice[-1]
        u_cyl = u_slice[0] - du*np.log(m['x'] / m['x'][0]) / np.log(m['x'][-1] / m['x'][0])

        ax.plot(m['x'], u_slice, '-', label=k + ' ADI')
        ax.plot(m['x'], u_cyl, '--', label=k + ' Theoretical')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Radius, m")
    ax.set_ylabel("Temperature, K")
    ax.set_title("Temperature vs Radius (simulation vs theory)")
    plt.show()



def mask_void_regions(mesh:dict) -> np.ndarray:
    """Masks a mesh's void regions so plotters ignore them.
    
    By scanning through all regions in one direction, the 'voids'
    between different regions can be located and a mask array created
    for plotters, so that meshes are plotted with clean boundaries.
    """

    mask = np.zeros((mesh['i'].size, mesh['j'].size,), bool)

    # iterate through all x slices and locate all void regions
    for j, row in enumerate(mesh['regions_x']):
        for reg in row:
            mask[:, j] = [not (reg['bounds'][0] <= i <= reg['bounds'][1]) for i in mesh['i']]

    return mask



def save_figure(ani) -> bool:
    """Saves an animation as a GIF.

    The user gives the containing folder absolute path, and the
    filename without the .gif extension.

    The function returns True if successful, else False.
    """

    while True:
        save_dir = input("Please specify containing directory: ")

        if not exists(save_dir):
            response = input(f"{save_dir} does not exist, try again? (y/n): ")
            if response.casefold() == 'n':
                print("save cancelled")
                return False

        else:
            break

    filename = input("Please give filename (without extension): ")
    path = join(save_dir, filename + '.gif')

    if exists(path) and input("A file already exists with this name, overwrite? (y/n): ").casefold() == 'n':
        print("save cancelled")
        return False

    ani.save(path, writer="pillow")
    print(f"Animation saved to: {path}")
    return True
