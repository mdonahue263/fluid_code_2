import numpy as np

def xop_2d(g,f,dx=1., avg=True,add=False):
    """
    Performs the following operation on a C-grid:
    g[i-igo, j] = s * g[i-igo, j] + a[0] * f[i, j] + a[1] * f[i-1, j]
    for i = 2, ..., M+igo and j = 1, ..., N
    
    Parameters:
        g (numpy.ndarray): Output array, modified in place.
        f (numpy.ndarray): Input array.
        dx (float): space-step, or 1 if averaging
        add (bool): if True, add to existing g array. Otherwise replace
        avg (bool): if True, averaging operator, else gradient operator
    """

    if add:
        s = 1.0
    else:
        s = 0.0

    if avg:
        a = np.array(([1/2,1/2]))
    else:
        a = np.array(([1.,-1.]))

    a /= dx

    Mg,Ng = g.shape # shape of array g
    Mf,Nf = f.shape # shape of arrays f
    if Ng > Nf:     # offset of array g
        igo,N = 0,Nf; # operation from p to u-points, set igo=0
                      # end points of g untouched and must be set by BC
    else:
        igo,N = 1,Ng; # operation from u to p-points set, igo=1
    g[:,1-igo:N] = s*g[:,1-igo:N] + a[0]*f[:,1:N+igo] + a[1]*f[:,0:N+igo-1]

def yop_2d(g,f, dy=1.,avg=True,add=False):
    """
    Performs the following operation on a C-grid:
    g[i-igo, j] = s * g[i-igo, j] + a[0] * f[i, j] + a[1] * f[i-1, j]
    for i = 2, ..., M+igo and j = 1, ..., N
    
    Parameters:
        g (numpy.ndarray): Output array, modified in place.
        f (numpy.ndarray): Input array.
        dy (float): y-step, or 1 if averaging
        add (bool): if True, add to existing g array. Otherwise replace
        avg (bool): if True, averaging operator, else gradient operator
    """

    if add:
        s = 1.0
    else:
        s = 0.0

    if avg:
        a = np.array(([1/2,1/2]))
    else:
        a = np.array(([1.,-1.]))

    a /= dy


    Mg,Ng = g.shape # shape of array g
    Mf,Nf = f.shape # shape of arrays f
    if Mg > Mf:     # offset of array g
        igo,M = 0,Mf; # operation from p to u-points, set igo=0
                      # end points of g untouched and must be set by BC
    else:
        igo,M = 1,Mg; # operation from u to p-points set, igo=1
    g[1-igo:M,:] = s*g[1-igo:M,:] + a[0]*f[1:M+igo,:] + a[1]*f[0:M+igo-1,:]

def Discrete1DGrid(smin,smax,Ns):
    '''Discretize the interval smin<= s <= smax into Ns intervals
    of size ds, 
    se: 1D array coordinate of cell edges with se[0]=smin, se[Ns]=smax.
    sc: 1D array coordinate of cell centers (mid-points)
    '''
    se,ds = np.linspace(smin,smax,Ns+1,retstep=True) # cell edges
    sc = 0.5*(se[0:-1]+se[1:])                       # cell centers
    return se,sc,ds

def RHSswe(u,v,p,depth,coriolis,gravity,dxs, timeCurrent, visc = 10**-3,bdc=0,tau_x=0,tau_y=0, alin=1, periodic=0):
    ''' Compute rhs of mass and mom conservation as
    ru,rv,rp=RHSswe(u,v,p,depth,coriolis,gravity,dxs, timeCurrent,visc,bdc,tau_x,tau_y, alin,periodic)
    bdc = bottom drag coefficient
    tau = wind stress
    alin=0 or 1, computes the linear or nonlinear SWE
    '''


    Mp,Np = p.shape # number of cells

# Water depth at p-points
    h = depth + alin * p  # water depth

# Compute water depth at u-points
    hbx = np.zeros_like(u)                  # create array
    xop_2d(hbx,h)                           # averaged depth to cell edges
    if periodic:
        hbx[:,-1] = hbx[:,1].copy()
        hbx[:, 0] = hbx[:,-2].copy()
    else:
        hbx[:,0] = h[:,0]; hbx[:,-1] = h[:,-1]  # boundary edges
    U = hbx*u                               # volume fluxes


# Compute water depth at v-points
    hby = np.zeros_like(v)                  # create array
    yop_2d(hby,h)                           # averaged depth to cell edges
    hby[0,:] = h[0,:]; hby[-1,:] = h[-1,:]  # boundary edges
    V = hby*v                               # volume fluxes

# Compute water depth at z-points
    hbxy = np.zeros_like(coriolis)               # create array
    yop_2d(hbxy,hbx)                             # averaged depth to cell edges
    hbxy[0,:] = hbx[0,:]; hbxy[-1,:] = hbx[-1,:] # boundary edges

####### compute total head ##############################################
    ru,rv = u**2,v**2    # square x-y velocities
    rp =-gravity*p       # pressure contribution to total head

    if alin: #no need to do anything if not alin.
        xop_2d(rp,ru,dx=-2,add=True) # add x-averaged u^2/2
        if periodic:
            rp[:,0] = rp[:,-1].copy() #shouldnt be needed if u is already correctly periodic
        yop_2d(rp,rv,dy=-2,add=True) # add y-averaged v^2/2
####### end- compute total head ##########################################



####### Potential Vorticity ##############################################
    q = coriolis.copy();                         # planetary vorticity
    if alin:                                     # no need to do anything if not alin
        xop_2d(q,v,  dxs[0], avg=False,add=True) # add relative vorticity due to zonal shear
            ##### issue here for periodic boundaries - east and west edge are missing relative vorticity!
        if periodic:
            q[:, 0] = q[:, -2].copy()
            q[:,-1] = q[:, 1].copy()
            ###### END periodic condition ##########
        yop_2d(q,u, -dxs[1], avg=False,add=True) # add relative vorticity due to meridional shear

    q /= hbxy; 
####### End Potential Vorticity ###########################################
####### Pressure gradients in x-y momentum equations#######################
    xop_2d(ru, rp, dxs[0], avg=False) # total head x-gradient
    yop_2d(rv, rp, dxs[1], avg=False) # total head y-gradient

        ###Periodicity adjustment - only needed when there is an X operation!
    if periodic:
        ru[:,0] = ru[:,-2].copy()
        ru[:,-1] = ru[:,1].copy()
        ###End periodicity adjustment

####### end- Pressure gradients in x-y momentum equations##################

###### Mass-flux divergence in mass equations##############################
    xop_2d(rp, U, -dxs[0], avg=False, add=False)# mass-flux divergence in x

    ###Periodicity adjustment - only needed when there is an X operation!
    if periodic:
        rp[:,0] = rp[:,-1].copy()
        # rp[:,-1] = rp[:,1].copy()
    ###End periodicity adjustment

    yop_2d(rp, V, -dxs[1], avg=False, add=True) # ADD mass-flux divergence in y

###### end- Mass-flux divergence in mass equations#########################

###### Add rotational terms ###############################################
    xop_2d(hbxy,V)                              # x-average of V
    if periodic: #add back correct mass flux to V points
        hbxy[:,0] = hbxy[:,-2].copy();
        hbxy[:,-1] = hbxy[:,1].copy();
    else:       #no V-motion to account for on edges 
        hbxy[:,0] = V[:,0]; hbxy[:,-1] = V[:,-1];
    hbxy = hbxy*q;                              # form q*Vbx
    yop_2d(ru, hbxy, add=True)                  # y-average and add to ru
    
    yop_2d(hbxy, U)                             # y-average of U
    hbxy[0,:] = U[0,:]; hbxy[-1,:] = U[-1,:];
    hbxy = hbxy*q;                              # form q*Uby
    xop_2d(rv, hbxy, -1, add=True)              # x-average and add to rv #HERE HERE HERER EHRE HEFR ERHER HERER HERE HERER HERERE HERE HERE

    if periodic:
        rv[:,0]=rv[:,-1].copy()
        # rv[:,0]=rv[:,-2].copy()
###### end- Add rotational terms ##########################################

##### Add Bottom Drag #####################################################
    if bdc != 0:
        ru -= (bdc * u)/hbx
        rv -= (bdc * v)/hby
        
##### end-Add Bottom Drag #################################################

##### Add Wind-Stress######################################################
    ru += tau_x/hbx
    rv += tau_y/hby
##### end- Add Wind Stress##################################################






###### Add U-Viscous Terms#################################################
###### Don't do any of this if viscosity is 0
    if visc != 0:
        u_visc = np.zeros((u.shape[0]+2,u.shape[1]+2)) #u-like matrix with ghosts
        u_visc[1:-1, 1:-1] = u.copy()                  #copy u values for interior points

        u_visc[   0, 1:-1] = u[ 0,:].copy()                 #Free-slip boundary, assign same vals
        u_visc[  -1, 1:-1] = u[-1,:].copy()
        

        if periodic:
            u_visc[1:-1,    0] = u[:, -3].copy()
            u_visc[1:-1,   -1] = u[:,2].copy()


        else:
            u_visc[1:-1,    0] = u[:, 0].copy()
            u_visc[1:-1,   -1] = u[:,-1].copy()

        #now, u_visc is too wide by 2 for y operation, to tall by 2 for x operation!
        #thus, index it from 1:-1 in those directions

        hbx = (u_visc[:-2,1:-1] - 2*u_visc[1:-1,1:-1] + u_visc[2:,1:-1])/dxs[1]**2     \
                    + (u_visc[1:-1,:-2] - 2*u_visc[1:-1,1:-1] + u_visc[1:-1,2:])/dxs[0]**2
        
        #could NOT make viscosity act correctly on boundaries when periodic - need to overwrite for now
        # if periodic:
        #     hbx[:,0:2] = hbx[:,-2:] = np.zeros_like(hbx[:,0:2])
        
        ru += visc*hbx
    ###### end- Add U-Viscous Terms############################################


    ###### Add V-Viscous Terms#################################################
        v_visc = np.zeros((v.shape[0]+2,v.shape[1]+2)) #v-like matrix with ghosts
        v_visc[1:-1, 1:-1] = v.copy()                  #copy v values for interior points

        v_visc[   0,1:-1] = v[ 0,:].copy()                 #Free-slip boundary, assign same vals
        v_visc[  -1,1:-1] = v[-1,:].copy()

        if periodic:
            v_visc[1:-1,   0] = v[:, -2].copy() 
            v_visc[1:-1,  -1] = v[:,0].copy()
        else:
            v_visc[1:-1,   0] = v[:, 0].copy()
            v_visc[1:-1,  -1] = v[:,-1].copy()

        hby = (v_visc[:-2,1:-1] - 2*v_visc[1:-1,1:-1] + v_visc[2:,1:-1])/dxs[1]**2     \
                    + (v_visc[1:-1,:-2] - 2*v_visc[1:-1,1:-1] + v_visc[1:-1,2:])/dxs[0]**2

        rv += visc*hby
    ###### end- Add U-Viscous Terms############################################



    #Correct the Boundaries!
    #North - South: Closed
    rv[0,:] = rv[-1,:] = 0.0  # closed boundary south-north BC
    if not periodic:
        ru[:,0] = ru[:,-1] = 0.0  # closed boundary west-east BC


    return ru,rv,rp

def RK3(u,v,p, depth, coriolis, gravity, dxs, dt, timeCurrent, visc=10**-3, bdc=0,tau_x=0,tau_y=0,alin=1.0, periodic=0):   
    """ Runge Kutta 3 integration """
    ru,rv,rp = RHSswe(u,v,p,depth,coriolis,gravity,dxs,timeCurrent,visc,bdc,tau_x,tau_y,alin,periodic)       # stage 1 tendency
    pt = p + rp*dt    # stage 1 solution
    ut = u + ru*dt    # stage 1 solution
    vt = v + rv*dt    # stage 1 solution

    ru,rv,rp = RHSswe(ut,vt,pt,depth,coriolis,gravity,dxs,timeCurrent+dt, visc,bdc,tau_x,tau_y, alin,periodic)  # stage 2 tendency
    pt = 0.75 * p + 0.25 * (pt + rp*dt) # stage 2 solution
    ut = 0.75 * u + 0.25 * (ut + ru*dt) # stage 2 solution
    vt = 0.75 * v + 0.25 * (vt + rv*dt) # stage 2 solution

    ru,rv,rp = RHSswe(ut,vt,pt,depth,coriolis,gravity,dxs,timeCurrent+0.5*dt, visc,bdc,tau_x,tau_y, alin,periodic) # stage 3 tendency
    p += 2.0 * (pt + rp*dt) # portion of final estimate
    u += 2.0 * (ut + ru*dt) # portion of final estimate
    v += 2.0 * (vt + rv*dt) # portion of final estimate

    p /= 3.0 # updated solution at next time step
    u /= 3.0 # updated solution at next time step
    v /= 3.0 # updated solution at next time step


def RK1(u, v, p, depth, coriolis, gravity, dxs, dt, timeCurrent, visc=10**-3, alin=1.0):   
    """ Forward Euler integration """
    """ DEFUNCT DUE TO EXTRA TERMS AND PERIODICITY"""
    # Calculate the tendencies
    ru, rv, rp = RHSswe(u, v, p, depth, coriolis, gravity, dxs, timeCurrent, visc, alin)
    
    # Update the variables using the tendencies
    u += ru * dt
    v += rv * dt
    p += rp * dt

def initialize_vortex(xu,xv,xp,yp,xz,yz,dx,dy,grav,f0,a,L,x_center,y_center):
    """
    Create initial vortex conditions
    xu,yu,xv,yv,xp,yp,xz,yz:
        arrays of points at which to define velocities, pressures, streamfunctions/rotation
    dx, dy is space steps
    depth is basin depth
    grav is adjusted gravity parameter (O(10^-2))
    f0 is coriolis parameter (O(10^-4))
    a = is peak displacement of layer interface (O(10^1m))
    L is decay length scale of Guassian vortex O(10^5m)
    x_center, y_center is position of vortex center
    """
    
    #calculate distances at relevant points
    rp = np.sqrt((xp-x_center)**2 + (yp-y_center)**2)
    rz = np.sqrt((xz-x_center)**2 + (yz-y_center)**2)


    #ETA term - on cell centers
    p = a*np.exp(-(rp**2)/(L**2))
    
    #CAP V Term - on cell corners
    vort_ratio = (8*grav)/(f0**2 * L**2)
    vort_inner =(1-vort_ratio*a*np.exp(-(rz**2)/(L**2)))**(1/2)
    V = ((f0*rz)/2)*(1-vort_inner)
    
    
 
    #u is y derivative of V
    #v is -x derivative of V
    u=np.zeros_like(xu)
    v=np.zeros_like(xv)
    
    yop_2d(u,V,dy,False,False)
    xop_2d(v,V,-dx,False,False)
    
    #edge velocities - currently basin closed
    u[:,0] = 0.0; u[:,-1] = 0.0;
    v[0,:] = 0.0; v[-1,:] = 0.0;
    
    return u,v,p

def current_vorticity(xz,u,v,dxs):
    #calculate vorticity on z-points, free-slip condition (no vorticity on edges)
    #dv/dx - du/dy


    vort_grid = np.zeros_like(xz)

    x_vort = np.zeros_like(xz[1:-1,1:-1])
    y_vort = x_vort.copy()

    #x deriv of v and y deriv of u
    xop_2d(x_vort, v[1:-1,:], dxs[0], avg=False,add=False)
    yop_2d(y_vort, u[:,1:-1], dxs[1], avg=False,add=False)

    vort_grid[1:-1,1:-1] = x_vort - y_vort

    return vort_grid

def RHS_AL81(u,v,p,depth,coriolis,gravity,dxs, timeCurrent, visc = 10**-3,bdc=0,tau_x=0,tau_y=0, alin=1, periodic=0):
    ''' Compute rhs of mass and mom and enstrophy conservation as
    ru,rv,rp=RHSswe(u,v,p,depth,coriolis,gravity,dxs, timeCurrent,visc,bdc,tau_x,tau_y, alin,periodic)
    bdc = bottom drag coefficient
    tau = wind stress
    alin=0 or 1, computes the linear or nonlinear SWE
    '''
    Mp,Np = p.shape # number of cells

    dx, dy=dxs


# Water depth at p-points
    h = depth + alin * p  # water depth

# Compute water depth at u-points
    hbx = np.zeros_like(u)                  # create array
    xop_2d(hbx,h)                           # averaged depth to cell edges
    if periodic:
        hbx[:,-1] = hbx[:,1].copy()
        hbx[:, 0] = hbx[:,-2].copy()
    else:
        hbx[:,0] = h[:,0]; hbx[:,-1] = h[:,-1]  # boundary edges
    U = hbx*u                               # volume fluxes


# Compute water depth at v-points
    hby = np.zeros_like(v)                  # create array
    yop_2d(hby,h)                           # averaged depth to cell edges
    hby[0,:] = h[0,:]; hby[-1,:] = h[-1,:]  # boundary edges
    V = hby*v                               # volume fluxes

# Compute water depth at z-points
    hbxy = np.zeros_like(coriolis)               # create array
    yop_2d(hbxy,hbx)                             # averaged depth to cell edges
    hbxy[0,:] = hbx[0,:]; hbxy[-1,:] = hbx[-1,:] # boundary edges

    ########################## hbx is no longer being used! - it is now a free uvar ######################
    ########################## hby is no longer being used! - it is now a free vvar ######################


####### compute total head ##############################################
    ru,rv = u**2,v**2    # square x-y velocities
    rp =-gravity*p       # pressure contribution to total head

    if alin: #no need to do anything if not alin.
        xop_2d(rp,ru,dx=-2,add=True) # add x-averaged u^2/2
        # if periodic:
        #     rp[:,0] = rp[:,-2].copy()
        #     rp[:,-1] = rp[:,1].copy()
        yop_2d(rp,rv,dy=-2,add=True) # add y-averaged v^2/2
####### end- compute total head ##########################################



####### Potential Vorticity ##############################################
    q = coriolis.copy();                         # planetary vorticity
    if alin:                                     # no need to do anything if not alin
        xop_2d(q,v,  dx, avg=False,add=True) # add relative vorticity due to zonal shear
            ##### issue here for periodic boundaries - east and west edge are missing relative vorticity!
        if periodic:
            q[:, 0] = q[:, -2].copy()
            q[:,-1] = q[:, 1].copy()
            ###### END periodic condition ##########
        yop_2d(q,u, -dy, avg=False,add=True) # add relative vorticity due to meridional shear

    #if not periodic, in this scheme I think we need to kill relative vorticity on boundaries.
    #but if it IS periodic, we still need to kill relative vorticity on north-south, since it does get the dv/dx influence
        q[0, :] = coriolis[0, :].copy()
        q[-1, :]=coriolis[-1, :].copy()
        if not periodic:
            q[:, 0] = coriolis[:, 0].copy()
            q[:,-1] = coriolis[:,-1].copy()

    q /= hbxy; 
    if periodic:
        assert((q[:,0] == q[:,-2]).all())
        assert((q[:,1] == q[:,-1]).all()) #check that q is periodic before using in later calculations
####### End Potential Vorticity ###########################################

####### Pressure gradients in x-y momentum equations#######################
    xop_2d(ru, rp, dx, avg=False) # total head x-gradient
    yop_2d(rv, rp, dy, avg=False) # total head y-gradient

        ###Periodicity adjustment - only needed when there is an X operation!
    if periodic:
        ru[:,0] = ru[:,-2].copy()
        ru[:,-1] = ru[:,1].copy()
        ###End periodicity adjustment
    # else: #may not need this part?
    #     ru[:,0] = ru[:,-1] = 0

    # rv[0,:] = rv[-1,0] = 0

####### end- Pressure gradients in x-y momentum equations##################

##### now, ru and rv just have the total head components, and we have hbx and hby as free uvar and vvar. ##########
##### have U, V, q, hbxy. I don't think we need hbxy, which is a free z variable right now (zpoint depths) ########

    pvar1 = np.zeros_like(p)
    pvar2 = pvar1.copy()
    pvar3 = pvar1.copy()
    pvar4 = pvar1.copy()


    xop_2d(pvar1, U, dx, avg=False,add=False) #### U gradient in x direction
    yop_2d(pvar2, V, dy, avg=False,add=False) #### V gradient in y direction

    # if periodic:
    #     pvar1[:,0] = pvar1[:,-2].copy()
    #     pvar1[:,-1] = pvar1[:,1].copy()

    rp = (-pvar1-pvar2).copy()                    #### p-Update is DONE ####


    xop_2d(hby, q,dx, avg=False, add=False)   #### x-gradient of potential vorticity (on v-points)
    if periodic:
        hby[:,0] = hby[:,-2].copy()
        hby[:,-1] = hby[:,1].copy()
    yop_2d(hbx, q, dy, avg=False, add=False)   #### y-gradient of potential vorticity (on u-points)

    yop_2d(pvar3, hby, dy, avg=False, add=False)# xy-gradient of potential vorticity (on p-points)

    if periodic: #spot test
        assert((pvar3[:,0] == pvar3[:,-2]).all())

    pvar4 = (1/48)*pvar2 * pvar3                         #### ygrad V * xy-grad q

    xop_2d(ru, pvar4, dx, avg=False, add = True) ### ru now has total head gradient and 1/48 term
    if periodic:
        ru[:,0] = ru[:,-2].copy()
        ru[:,-1] = ru[:,1].copy()

    pvar4 = (-1/48)*pvar1*pvar3                           #### grad U * xy-grad q
    if periodic: #spot test
        assert((pvar4[:,0] == pvar4[:,-2]).all())

    yop_2d(rv, pvar4, dy, avg=False, add = True) ### rv now has total head gradient and 1/48 term
    if periodic: #spot test
        assert((rv[:,0] == rv[:,-2]).all())

    #now, pvar 4 is free and pvar3 is free

    xop_2d(pvar3, hbx)                         #### x-average of y-gradient of potential vorticity
    if periodic:
        pvar3[:,0] = pvar3[:,-2].copy()
        pvar3[:,-1] = pvar3[:,1].copy()
    #### hbx is now free

    pvar4 = (-1/12)*pvar1*pvar3                         #### xavg ygrad q * xavg U
    if periodic: #spot test
        assert((pvar4[:,0] == pvar4[:,-2]).all())

    xop_2d(ru, pvar4, dx=1, avg=True, add=True)  ### ru now has total head gradient, 1/48 term, and one of the 1/12 terms
    if periodic:
        ru[:,0]  = ru[:,-2].copy()
        ru[:,-1] = ru[:,1].copy()

    #pvar1, pvar4 have been freed up

    yop_2d(pvar1, hby)                         #### y-average of x-gradient of potential vorticity
    #### hby is now free

    pvar4 = (1/12)*pvar1*pvar2                         #### yavg xgrad q * yavg V
    if periodic: #spot test
        assert((pvar4[:,0] == pvar4[:,-2]).all())

    yop_2d(rv,pvar4,dy=1, avg=True,add=True)  ### rv now has total head gradient, 1/48 term, and one of the 1/12 terms

    #now pvar2, 4 are free, pvar3, 1 are x, y avg of y,x gradient of potential vorticity


    xop_2d(pvar2, U)                        #### x-average of U
    if periodic:
        pvar2[:,0] = pvar2[:,-2].copy()
        pvar2[:,-1] = pvar2[:,1].copy()

    pvar4 = (-1/12)*pvar3 * pvar2          #### xavg U * x-average of y-gradient of potential vorticity

    xop_2d(ru, pvar4, dx=dx, avg=False, add=True)### ru now has total head gradient, 1/48 term, and both 1/12 terms
    if periodic:
        ru[:,0] = ru[:,-2].copy()
        ru[:,-1] = ru[:,1].copy()

    #pvar4 is free, pvar3 is free  

    yop_2d(pvar3, V)                        #### y-average of V

    pvar4 = (1/12)*(pvar1 * pvar3)                   #### y-average of V * y average of x gradient of q

    yop_2d(rv, pvar4, dy=dy, avg=False, add=True)### rv now has total head gradient, 1/48 term, and both 1/12 terms

    #pvar4 is free, pvar1 is free

    xop_2d(hby, q)                          ####  x-average of q
    yop_2d(pvar4, hby)                      #### xy-average of q

    pvar1 = pvar4 * pvar3                   #### xy-average of q *  y-average of V

    xop_2d(ru, pvar1, dx=1, avg=True, add=True)# ru done
    if periodic:
        ru[:,0] = ru[:,-2].copy()
        ru[:,-1] = ru[:,1].copy()
    #pvar1 free

    pvar1 = -pvar4*pvar2                     #### xy-average of q * x-average of U

    yop_2d(rv, pvar1, dy=1, avg=True, add=True)

    #all done!

    rv[0,:] = rv[-1,:] = 0.0  # closed boundary south-north BC
    if not periodic:
        ru[:,0] = ru[:,-1] = 0.0  # closed boundary west-east BC

    if periodic:
        rv[:,-1]=rv[:,1].copy()
        rv[:,0]=rv[:,-2].copy()

    if periodic: #spot test
        assert((ru[:,0] == ru[:,-2]).all())
        assert((rv[:,0] == rv[:,-2]).all())
        assert((rp[:,0] == rp[:,-2]).all())

    return ru,rv,rp

def RK3_AL81(u,v,p, depth, coriolis, gravity, dxs, dt, timeCurrent, visc=10**-3, bdc=0,tau_x=0,tau_y=0,alin=1.0, periodic=0):   
    """ Runge Kutta 3 integration """
    ru,rv,rp = RHS_AL81(u,v,p,depth,coriolis,gravity,dxs,timeCurrent,visc,bdc,tau_x,tau_y,alin,periodic)       # stage 1 tendency
    pt = p + rp*dt    # stage 1 solution
    ut = u + ru*dt    # stage 1 solution
    vt = v + rv*dt    # stage 1 solution

    ru,rv,rp = RHS_AL81(ut,vt,pt,depth,coriolis,gravity,dxs,timeCurrent+dt, visc,bdc,tau_x,tau_y, alin,periodic)  # stage 2 tendency
    pt = 0.75 * p + 0.25 * (pt + rp*dt) # stage 2 solution
    ut = 0.75 * u + 0.25 * (ut + ru*dt) # stage 2 solution
    vt = 0.75 * v + 0.25 * (vt + rv*dt) # stage 2 solution

    ru,rv,rp = RHS_AL81(ut,vt,pt,depth,coriolis,gravity,dxs,timeCurrent+0.5*dt, visc,bdc,tau_x,tau_y, alin,periodic) # stage 3 tendency
    p += 2.0 * (pt + rp*dt) # portion of final estimate
    u += 2.0 * (ut + ru*dt) # portion of final estimate
    v += 2.0 * (vt + rv*dt) # portion of final estimate

    p /= 3.0 # updated solution at next time step
    u /= 3.0 # updated solution at next time step
    v /= 3.0 # updated solution at next time step