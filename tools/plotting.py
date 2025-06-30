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

def StartPlot(u,v,p,xe,xc,ye,yc,plevs, XarrowStep = 4, YarrowStep=4, arrow_scale=1, colormap = 'seismic'):
    fig,ax = plt.subplots()
    ax.set_xlim(xe[0],xe[-1])
    ax.set_ylim(ye[0],ye[-1])
    ax.set_aspect('equal')
    
    # plot pressure contours
    pcont=ax.contourf(xc,yc,p, cmap=colormap, vmin=plevs.min(), vmax=plevs.max(), levels=plevs, extend='both')    # draw initial p

    # plot currents
    uc,vc = coLocateVelocity(u,v)
    arrows = ax.quiver(xc[::XarrowStep],
                       yc[::YarrowStep],
                       uc[::XarrowStep,::YarrowStep],
                       vc[::XarrowStep,::YarrowStep],
                       scale=arrow_scale)

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

    return fig,ax,pcont,arrows

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

def UpdatePlot(u,v,p,xe,xc,ye,yc,ax,pcont,plevs,timec,arrows,XarrowStep = 4, YarrowStep=4, colormap = 'seismic'):
    pcont.remove()
    pcont = ax.contourf(xc,yc,p, cmap=colormap, vmin=plevs.min(), vmax = plevs.max(),  alpha=.75, levels=plevs, extend='both')
    uc,vc = coLocateVelocity(u,v)
    arrows.set_UVC(uc[::XarrowStep,::YarrowStep],
                   vc[::XarrowStep,::YarrowStep])             # updates arrows
    plt.pause(1.e-2)
    # linep.set_ydata(p[0,:])
    # lineu.set_ydata(u[0,:])
    pmin = p.min()
    pmax = p.max()
    displayTime = timec #/(24*3600) - this was for the one that runs to ten years out
    titlestr = 'time:{},pmin:{:5.2e},pmax:{:5.2e}'.format(np.round(displayTime,2),pmin,pmax)
    ax.set_title(titlestr)
    return pcont

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