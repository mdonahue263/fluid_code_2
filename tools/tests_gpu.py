import cupy as cp
from .operators_gpu import xop_2d, yop_2d


def StabilityCheck(gravity,depth,dt,dx):
    maxc = cp.sqrt(gravity*cp.abs(depth).max()) # max. gravity wave speed
    courant = maxc*dt/dx             # Courant number
    factors = 2.0*cp.sqrt(2.0)
    print('max. Courant number:',courant) #
       
    if courant > 1.76: # RK3 stability for C-grid
        print('Unstable gravity wave for RK3, timestep too large')
        maxdt = dx/maxc/factors
        print('Maximum RK3-timestep is:',maxdt)
    else:
        print('Stable gravity wave time step:')

def calculate_energy(u,v,p,depth,grav):
    """
    Calculate Kinetic + Potential Energy at Cell Center
    u,v,p: Current velocities and pressures
    depth: depth parameter
    grav: adjusted gravity parameter
    """
    
    #Calcualte KE by square velocities and averaging to cell center:
    u_cell = cp.zeros_like(p)
    v_cell = cp.zeros_like(p)
    
    xop_2d(u_cell, u**2)
    yop_2d(v_cell, v**2)
    
    system_ke = cp.sum((depth+p)*(u_cell + v_cell)/2)
    
    system_pe = cp.sum(grav*p**2/2)
    
    return system_ke + system_pe

def calculate_enstrophy(u,v,p,dxs,depth,coriolis,alin):
    # Water depth at p-points
    h = depth + alin * p  # water depth

    # Compute water depth at u-points
    hbx = cp.zeros_like(u)                  # create array
    xop_2d(hbx,h)                           # averaged depth to cell edges
    hbx[:,0] = h[:,0]; hbx[:,-1] = h[:,-1]  # boundary edges

    # Compute water depth at z-points
    hbxy = cp.zeros_like(coriolis)               # create array
    yop_2d(hbxy,hbx)                             # averaged depth to cell edges
    hbxy[0,:] = hbx[0,:]; hbxy[-1,:] = hbx[-1,:] # boundary edges
    
    depth_term = 1/hbxy
    
    #compute relative vorticity
    relvort=cp.zeros_like(coriolis)
    xop_2d(relvort,v,dxs[0],avg=False)
    yop_2d(relvort,u,-dxs[1],avg=False,add=True)
    
    q=cp.sum((alin*relvort + coriolis)*depth_term)
    
    return cp.sum(q**2 * h * 0.5)

def calculate_laplace_u(u, dxs):

    u_visc = cp.zeros((u.shape[0]+2,u.shape[1]+2)) #u-like matrix with ghosts
    u_visc[1:-1, 1:-1] = u.copy()                  #copy u values for interior points

    u_visc[   0, 1:-1] = u[ 0,:].copy()                 #Free-slip boundary, assign same vals
    u_visc[  -1, 1:-1] = u[-1,:].copy()
    u_visc[1:-1,    0] = u[:, 0].copy()
    u_visc[1:-1,   -1] = u[:,-1].copy()

    #now, u_visc is too wide by 2 for y operation, to tall by 2 for x operation!
    #thus, index it from 1:-1 in those directions

    laplacian_u = (u_visc[:-2,1:-1] - 2*u_visc[1:-1,1:-1] + u_visc[2:,1:-1])/dxs[1]**2     \
                + (u_visc[1:-1,:-2] - 2*u_visc[1:-1,1:-1] + u_visc[1:-1,2:])/dxs[0]**2
    
    return laplacian_u

def calculate_laplace_v(v,dxs):

    v_visc = cp.zeros((v.shape[0]+2,v.shape[1]+2)) #v-like matrix with ghosts
    v_visc[1:-1, 1:-1] = v.copy()                  #copy v values for interior points

    v_visc[   0,1:-1] = v[ 0,:].copy()                 #Free-slip boundary, assign same vals
    v_visc[  -1,1:-1] = v[-1,:].copy()
    v_visc[1:-1,   0] = v[:, 0].copy()
    v_visc[1:-1,  -1] = v[:,-1].copy()

    laplacian_v = (v_visc[:-2,1:-1] - 2*v_visc[1:-1,1:-1] + v_visc[2:,1:-1])/dxs[1]**2     \
                + (v_visc[1:-1,:-2] - 2*v_visc[1:-1,1:-1] + v_visc[1:-1,2:])/dxs[0]**2
    
    return laplacian_v