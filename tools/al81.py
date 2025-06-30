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
        if periodic:
            rp[:,0] = rp[:,-2].copy()
            rp[:,-1] = rp[:,1].copy()
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

    q /= hbxy; 
####### End Potential Vorticity ###########################################

####### Pressure gradients in x-y momentum equations#######################
    xop_2d(ru, rp, dx, avg=False) # total head x-gradient
    yop_2d(rv, rp, dy, avg=False) # total head y-gradient

        ###Periodicity adjustment - only needed when there is an X operation!
    if periodic:
        ru[:,0] = ru[:,-2].copy()
        ru[:,-1] = ru[:,1].copy()
        ###End periodicity adjustment

####### end- Pressure gradients in x-y momentum equations##################

##### now, ru and rv just have the total head components, and we have hbx and hby as free uvar and vvar. ##########
##### have U, V, q, hbxy. I don't think we need hbxy, which is a free z variable right now (zpoint depths) ########

    pvar1 = np.zeros_like(p)
    pvar2 = pvar1.copy()
    pvar3 = pvar1.copy()
    pvar4 = pvar1.copy()


    xop_2d(pvar1, U, dx, avg=False,add=False) #### U gradient in x direction
    yop_2d(pvar2, V,dy, avg=False,add=False) #### V gradient in y direction

    rp = (-pvar1-pvar2).copy()                    #### p-Update is DONE ####


    xop_2d(hby, q,dx, avg=False, add=False)   #### x-gradient of potential vorticity (on v-points)
    yop_2d(hbx, q,dy, avg=False, add=False)   #### y-gradient of potential vorticity (on u-points)

    yop_2d(pvar3, hby, dy, avg=False, add=False)# xy-gradient of potential vorticity (on p-points)

    pvar4 = (1/48)*pvar2 * pvar3                         #### ygrad V * xy-grad q

    xop_2d(ru, pvar4, dx, avg=False, add = True) ### ru now has total head gradient and 1/48 term

    pvar4 = (-1/48)*pvar1*pvar3                           #### grad U * xy-grad q

    yop_2d(rv, pvar4, dy, avg=False, add = True) ### rv now has total head gradient and 1/48 term

    #now, pvar 4 is free and pvar3 is free

    xop_2d(pvar3, hbx)                         #### x-average of y-gradient of potential vorticity
    #### hbx is now free

    pvar4 = (-1/12)*pvar1*pvar3                         #### xavg ygrad q * xavg U

    xop_2d(ru, pvar4, dx=1, avg=True, add=True)  ### ru now has total head gradient, 1/48 term, and one of the 1/12 terms

    #pvar1, pvar4 have been freed up

    yop_2d(pvar1, hby)                         #### y-average of x-gradient of potential vorticity
    #### hby is now free

    pvar4 = (1/12)*pvar1*pvar2                         #### yavg xgrad q * yavg V

    yop_2d(rv,pvar4,dy=1, avg=True,add=True)  ### rv now has total head gradient, 1/48 term, and one of the 1/12 terms

    #now pvar2, 4 are free, pvar3, 1 are x, y avg of y,x gradient of potential vorticity


    xop_2d(pvar2, U)                        #### x-average of U

    pvar4 = (-1/12)*pvar3 * pvar2          #### xavg U * x-average of y-gradient of potential vorticity

    xop_2d(ru, pvar4, dx=dx, avg=False, add=True)### ru now has total head gradient, 1/48 term, and both 1/12 terms

    #pvar4 is free, pvar3 is free  

    yop_2d(pvar3, V)                        #### y-average of V

    pvar4 = (1/12)*(pvar1 * pvar3)                   #### y-average of V * y average of x gradient of q

    yop_2d(rv, pvar4, dx=dy, avg=False, add=True)### rv now has total head gradient, 1/48 term, and both 1/12 terms

    #pvar4 is free, pvar1 is free

    xop_2d(hbx, q)                          ####  x-average of q
    yop_2d(pvar4, hbx)                      #### xy-average of q

    pvar1 = pvar4 * pvar3                   #### xy-average of q *  y-average of V

    xop_2d(ru, pvar1, dx=1, avg=True, add=True)# ru done
    #pvar1 free

    pvar1 = -pvar4*pvar2                     #### xy-average of q * x-average of U

    yop_2d(rv, pvar1, dx=1, avg=True, add=True)

    #all done!


















###### Mass-flux divergence in mass equations##############################
    xop_2d(rp, U, -dxs[0], avg=False, add=False)# mass-flux divergence in x

    ###Periodicity adjustment - only needed when there is an X operation!
    if periodic:
        rp[:,0] = rp[:,-2].copy()
        rp[:,-1] = rp[:,1].copy()
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
    xop_2d(rv, hbxy, -1, add=True)              # x-average and add to rv

    if periodic:
        rv[:,-1]=rv[:,1].copy()
        rv[:,0]=rv[:,-2].copy()
###### end- Add rotational terms ##########################################