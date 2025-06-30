from matplotlib import pyplot as plt
import numpy as np
from .operators import xop_2d, yop_2d

def VisualizeGrid(xu,yu,xv,yv,xp,yp,xz,yz):
    '''Visualize the C-grid obtained, useful for confirmation
    when the number of cells is small.
    '''
    fgrid,axgrid = plt.subplots()
    axgrid.set_aspect('equal')
    axgrid.scatter(xp,yp,'k',marker='$p$')
    axgrid.scatter(xu,yu,'r',marker='$u$')
    axgrid.scatter(xv,yv,'b',marker='$v$')
    axgrid.scatter(xz,yz,'m',marker='$z$')
    axgrid.set_title('Arakawa C-grid layout')  
    return

def coLocateVelocity(u,v):
    '''calculate the velocity at cell centers for plotting'''
    M,N = u.shape; N-= 1
    uc,vc = np.zeros([M,N]),np.zeros([M,N])
    xop_2d(uc,u)
    yop_2d(vc,v)
    return uc,vc

def StartPlot(u,v,p,xe,xc,ye,yc,plevs, XarrowStep = 4, YarrowStep=4, arrow_scale=1, colormap = 'seismic',tracers=[]):
    fig,ax = plt.subplots()
    ax.set_xlim(xe[0],xe[-1])
    ax.set_ylim(ye[0],ye[-1])
    ax.set_aspect('equal')
    
    # plot pressure contours
    pcont=ax.contourf(xc,yc,p, cmap=colormap, vmin=plevs.min(), vmax=plevs.max(), levels=plevs)    # draw initial p

    # plot currents
    uc,vc = coLocateVelocity(u,v)
    arrows = ax.quiver(xc[::XarrowStep],
                       yc[::YarrowStep],
                       uc[::XarrowStep,::YarrowStep],
                       vc[::XarrowStep,::YarrowStep],
                       scale=arrow_scale)
    
    if len(tracers)>0:
        tracer_plot = plot_tracers(ax, tracers)

    # Add the colorbar
    cbar = fig.colorbar(pcont, ax=ax)

#    ucont=ax.contour(xe,yc,u)    # draw initial p
#    vcont=ax.contour(xc,ye,v)    # draw initial p
    pmin = p.min()
    pmax = p.max()
    titlestr = 'time:{},pmin:{:5.2e},pmax:{:5.2e}'.format(0.0,pmin,pmax)
    ax.set_title(titlestr)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    return fig,ax,pcont,arrows, tracer_plot

def StartPlotVort(vort, xe,ye, plevs):
    fig,ax = plt.subplots()
    ax.set_xlim(xe[0],xe[-1])
    ax.set_ylim(ye[0],ye[-1])
    ax.set_aspect('equal')
    
    pcont=ax.contourf(xe,ye,vort, cmap='coolwarm', vmin=-.1, vmax=.1, levels=plevs)    # draw initial p
    
    # Add the colorbar
    cbar = fig.colorbar(pcont, ax=ax)
    
    vortMin = vort.min()
    vortMax = vort.max()
    titlestr = 'time:{},pmin:{:5.2e},pmax:{:5.2e}'.format(0.0,vortMin,vortMax)
    ax.set_title(titlestr)
    
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    return fig,ax,pcont

def UpdatePlot(u,v,p,xe,xc,ye,yc,ax,pcont,plevs,timec,arrows,XarrowStep = 4, YarrowStep=4, colormap = 'seismic',tracers=[],tracer_plot=None):
    pcont.remove()
    pcont = ax.contourf(xc,yc,p, cmap=colormap, vmin=plevs.min(), vmax = plevs.max(),  alpha=.75, levels=plevs)
    uc,vc = coLocateVelocity(u,v)
    arrows.set_UVC(uc[::XarrowStep,::YarrowStep],
                   vc[::XarrowStep,::YarrowStep])             # updates arrows
    if len(tracers) > 0:
        tracer_plot = plot_tracers(ax, tracers,tracer_plot)
    plt.pause(1.e-2)
    # linep.set_ydata(p[0,:])
    # lineu.set_ydata(u[0,:])
    pmin = p.min()
    pmax = p.max()
    titlestr = 'time:{},pmin:{:5.2e},pmax:{:5.2e}'.format(np.round(timec,2),pmin,pmax)
    ax.set_title(titlestr)
    return pcont, tracer_plot

def UpdatePlotVort(vort,xe,ye,ax,pcont,plevs,timec):
    pcont.remove()
    pcont=ax.contourf(xe,ye,vort, cmap = 'coolwarm', vmin = -.5, vmax=.5, levels=plevs)
    plt.pause(1.e-2)
    vortMin = vort.min()
    vortMax = vort.max()
    titlestr = 'time:{},pmin:{:5.2e},pmax:{:5.2e}'.format(timec,vortMin,vortMax)
    ax.set_title(titlestr)
    return pcont

def StartPlotEnergy(include_enstrophy=True):
    fig,ax = plt.subplots()
    
    # plot pressure contours
    pline=ax.plot(0.0,1.0, color='dodgerblue', label='System Energy departure')    # draw initial energy
    if include_enstrophy:
        pline2=ax.plot(0.0,1.0, color='orange', label='System Enstrophy departure')

    titlestr = 'time:{},total energy %:{:5.2e}'.format(0.0,1.0)
    ax.set_title(titlestr)
    ax.set_ylim(0.89,1.001)

    ax.set_xlabel('Timestep')
    ax.legend()

    if include_enstrophy:
        return fig,ax,pline, pline2
    else:
        return fig,ax,pline

def UpdatePlotEnergy(ax,pline,pline2,times,elevel,strlevel,timec,include_enstrophy=True):
    pline[0].remove()
    if include_enstrophy:
        pline2[0].remove()
    pline=ax.plot(times,elevel, color='dodgerblue')
    if include_enstrophy:
        pline2=ax.plot(times,strlevel, color='orange')
    plt.pause(1.e-2)
    titlestr = 'time:{},total energy %:{:5.2e}'.format(timec,elevel[-1])
    ax.set_title(titlestr)

    if include_enstrophy:
        return pline, pline2
    else:
        return pline

def StartPlotULaplace(u,dxs):
    fig,ax = plt.subplots()
    ax.set_aspect('equal')

    u_visc = np.zeros((u.shape[0]+2,u.shape[1]+2)) #u-like matrix with ghosts
    u_visc[1:-1, 1:-1] = u.copy()                  #copy u values for interior points

    u_visc[   0, 1:-1] = u[ 0,:].copy()                 #Free-slip boundary, assign same vals
    u_visc[  -1, 1:-1] = u[-1,:].copy()
    u_visc[1:-1,    0] = u[:, 0].copy()
    u_visc[1:-1,   -1] = u[:,-1].copy()

    #now, u_visc is too wide by 2 for y operation, to tall by 2 for x operation!
    #thus, index it from 1:-1 in those directions

    laplacian_u = (u_visc[:-2,1:-1] - 2*u_visc[1:-1,1:-1] + u_visc[2:,1:-1])/dxs[1]**2     \
                + (u_visc[1:-1,:-2] - 2*u_visc[1:-1,1:-1] + u_visc[1:-1,2:])/dxs[0]**2
    
    laplacian_u *= np.sign(u)
    
    #laplacian needs to always be negative? No, laplacian needs to be always TOWARD ZERO. Laplacian should have opposite sign as vel


    threshold = 10**-12
    masked_laplace = np.ma.masked_where(np.abs(laplacian_u) < threshold, laplacian_u)

    # Plot with a perceptually uniform color map
    lcont = ax.contourf(masked_laplace)#, cmap='Greys', vmin=-threshold, vmax=0, levels=20)
    cbar = fig.colorbar(lcont, ax=ax)
    ax.set_title('Laplacian of u')

    return fig, ax, lcont

def UpdatePlotULaplace(ax, lcont, u,dxs):
    lcont.remove()

    u_visc = np.zeros((u.shape[0]+2,u.shape[1]+2)) #u-like matrix with ghosts
    u_visc[1:-1, 1:-1] = u.copy()                  #copy u values for interior points

    u_visc[   0, 1:-1] = u[ 0,:].copy()                 #Free-slip boundary, assign same vals
    u_visc[  -1, 1:-1] = u[-1,:].copy()
    u_visc[1:-1,    0] = u[:, 0].copy()
    u_visc[1:-1,   -1] = u[:,-1].copy()

    #now, u_visc is too wide by 2 for y operation, to tall by 2 for x operation!
    #thus, index it from 1:-1 in those directions

    laplacian_u = (u_visc[:-2,1:-1] - 2*u_visc[1:-1,1:-1] + u_visc[2:,1:-1])/dxs[1]**2     \
                + (u_visc[1:-1,:-2] - 2*u_visc[1:-1,1:-1] + u_visc[1:-1,2:])/dxs[0]**2
    
    laplacian_u *= np.sign(u)

    threshold = 10**-12
    masked_laplace = np.ma.masked_where(np.abs(laplacian_u) < threshold, laplacian_u)

    # Plot with a perceptually uniform color map
    lcont = ax.contourf(masked_laplace)#, cmap='Greys', vmin=-threshold, vmax=0, levels=20)
    ax.set_title('Laplacian of u')

    return lcont, laplacian_u, np.sum(laplacian_u)

def plot_tracers(ax, tracers, tracer_plot=None):
    """Plot or update tracers on the plot."""
    if tracer_plot is None:
        tracer_plot = ax.scatter(tracers[:, 0], tracers[:, 1], color='black', s=10)
    else:
        tracer_plot.set_offsets(tracers)
    return tracer_plot

def initialize_tracers(num_tracers, xmin, xmax, ymin, ymax):
    """Generate random initial positions for tracers within the domain."""
    tracer_x = np.random.rand(num_tracers) * (xmax - xmin) + xmin
    tracer_y = np.random.rand(num_tracers) * (ymax - ymin) + ymin
    return np.vstack((tracer_x, tracer_y)).T  # Shape (num_tracers, 2)

def update_tracers(tracers, u, v, xu, yu, xv, yv, dt, xmin, xmax, ymin, ymax):
    """Update tracer positions based on interpolated velocity and timestep."""
    new_tracers = []
    for tracer in tracers:
        u_interp, v_interp = interpolate_velocity(tracer, u, v, xu, yu, xv, yv)
        x_new = tracer[0] + u_interp * dt
        y_new = tracer[1] + v_interp * dt
        
        # Handle boundaries (wrap-around example)
        if x_new < xmin: x_new += (xmax - xmin)
        if x_new > xmax: x_new -= (xmax - xmin)
        if y_new < ymin: y_new += (ymax - ymin)
        if y_new > ymax: y_new -= (ymax - ymin)
        
        new_tracers.append((x_new, y_new))
    return np.array(new_tracers)

def interpolate_velocity(tracer, u, v, xu, yu, xv, yv):
    """Interpolate velocity (u, v) at a tracer's position using bilinear interpolation.
    Handles edge cases by using fewer points near boundaries."""
    x, y = tracer

    # Find indices for surrounding grid points
    i_u = np.searchsorted(xu[0, :], x) - 1
    j_u = np.searchsorted(yu[:, 0], y) - 1
    i_v = np.searchsorted(xv[0, :], x) - 1
    j_v = np.searchsorted(yv[:, 0], y) - 1

    # Ensure indices are within bounds
    i_u = max(0, min(i_u, xu.shape[1] - 2))
    j_u = max(0, min(j_u, yu.shape[0] - 2))
    i_v = max(0, min(i_v, xv.shape[1] - 2))
    j_v = max(0, min(j_v, yv.shape[0] - 2))

    # Interpolation for u
    if y < yu[0, 0] or y > yu[-1, 0]:  # Out of bounds in y-direction
        if y < yu[0, 0]:
            u_interp = u[0, i_u]  # Use the closest value at the bottom
        else:
            u_interp = u[-1, i_u]  # Use the closest value at the top
    elif x < xu[0, 0] or x > xu[0, -1]:  # Out of bounds in x-direction
        if x < xu[0, 0]:
            u_interp = u[j_u, 0]  # Use the closest value on the left
        else:
            u_interp = u[j_u, -1]  # Use the closest value on the right
    else:  # Bilinear interpolation within bounds
        u00, u10 = u[j_u, i_u], u[j_u, i_u + 1]
        u01, u11 = u[j_u + 1, i_u], u[j_u + 1, i_u + 1]
        dx_u = xu[0, i_u + 1] - xu[0, i_u]
        dy_u = yu[j_u + 1, 0] - yu[j_u, 0]
        u_interp = ((u00 * (xu[0, i_u + 1] - x) * (yu[j_u + 1, 0] - y) +
                     u10 * (x - xu[0, i_u]) * (yu[j_u + 1, 0] - y) +
                     u01 * (xu[0, i_u + 1] - x) * (y - yu[j_u, 0]) +
                     u11 * (x - xu[0, i_u]) * (y - yu[j_u, 0])) / (dx_u * dy_u))

    # Interpolation for v
    if y < yv[0, 0] or y > yv[-1, 0]:  # Out of bounds in y-direction
        if y < yv[0, 0]:
            v_interp = v[0, i_v]  # Use the closest value at the bottom
        else:
            v_interp = v[-1, i_v]  # Use the closest value at the top
    elif x < xv[0, 0] or x > xv[0, -1]:  # Out of bounds in x-direction
        if x < xv[0, 0]:
            v_interp = v[j_v, 0]  # Use the closest value on the left
        else:
            v_interp = v[j_v, -1]  # Use the closest value on the right
    else:  # Bilinear interpolation within bounds
        v00, v10 = v[j_v, i_v], v[j_v, i_v + 1]
        v01, v11 = v[j_v + 1, i_v], v[j_v + 1, i_v + 1]
        dx_v = xv[0, i_v + 1] - xv[0, i_v]
        dy_v = yv[j_v + 1, 0] - yv[j_v, 0]
        v_interp = ((v00 * (xv[0, i_v + 1] - x) * (yv[j_v + 1, 0] - y) +
                     v10 * (x - xv[0, i_v]) * (yv[j_v + 1, 0] - y) +
                     v01 * (xv[0, i_v + 1] - x) * (y - yv[j_v, 0]) +
                     v11 * (x - xv[0, i_v]) * (y - yv[j_v, 0])) / (dx_v * dy_v))

    return u_interp, v_interp