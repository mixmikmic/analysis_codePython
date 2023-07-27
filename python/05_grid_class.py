get_ipython().run_cell_magic('writefile', 'fdm_c.py', '\nimport numpy as np\nimport pdb\nimport scipy.sparse as sp\nfrom scipy.sparse.linalg import spsolve # to use its short name\nfrom collections import namedtuple\n\nclass InputError(Exception):\n    pass\n\ndef quivdata(Out, gr, iz=0):\n    """Returns coordinates and velocity components to show velocity with quiver\n    \n    Compute arrays to display velocity vectors using matplotlib\'s quiver.\n    The quiver is always drawn in the xy-plane for a specific layer. Default iz=0\n    \n    Parameters\n    ----------\n    `Out` : namedtuple holding arrays `Qx`, `Qy`, `Qz` as defined in `fdm3`\n        `Qx` : ndarray, shape: (Nz, Ny, Nx-1), [L3/T]\n            Interfacial flows in finite difference model in x-direction from `fdm3\'\n        `Qy` : ndarray, shape: (Nz, Ny-1, Nx), [L3/T]\n            Interfacial flows in finite difference model in y-direction from `fdm3`\n        `Qz` : ndarray, shape: (Nz-1, Ny, Nx), [L3/T]\n            Interfacial flows in finite difference model in z-direction from `fdm3`            \n    `gr` : `grid_object` generated by Grid\n    `iz` : int [-]\n            iz is the number of the layer for which the data are requested,\n            and all output arrays will be 2D for that layer.\n            if iz==None, then all outputs will be full 3D arrays and cover all layers\n            simultaneously\n\n    Returns\n    -------\n    `Xm` : ndarray, shape: (Nz, Ny, Nx), [L]\n        x-coordinates of cell centers\n    `Ym` : ndarray, shape: (Nz, Ny, Nx), [L]\n        y-coodinates of cell centers\n    `ZM` : ndarray, shape: (Nz, Ny, Nx), [L]\n        `z`-coordinates at cell centers\n    `U` : ndarray, shape: (Nz, Ny, Nx), [L3/d]\n        Flow in `x`-direction at cell centers\n    `V` : ndarray, shape: (Nz, Ny, Nx), [L3/T]\n        Flow in `y`-direction at cell centers\n    `W` : ndarray, shape: (Nz, Ny, Nx), [L3/T]\n        Flow in `z`-direction at cell centers.\n    \n    """\n    \n    X, Y = np.meshgrid(gr.xm, gr.ym) # coordinates of cell centers\n    \n    shp = (gr.Ny, gr.Nx) # 2D tuple to select a single layer\n    \n    # Flows at cell centers\n    U = np.concatenate((Out.Qx[iz, :, 0].reshape((1, gr.Ny, 1)), \\\n                        0.5 * (Out.Qx[iz, :, :-1].reshape((1, gr.Ny, gr.Nx-2)) +\\\n                               Out.Qx[iz, :, 1: ].reshape((1, gr.Ny, gr.Nx-2))), \\\n                        Out.Qx[iz, :, -1].reshape((1, gr.Ny, 1))), axis=2).reshape(shp)\n    V = np.concatenate((Out.Qy[iz, 0, :].reshape((1, 1, gr.Nx)), \\\n                        0.5 * (Out.Qy[iz, :-1, :].reshape((1, gr.Ny-2, gr.Nx)) +\\\n                               Out.Qy[iz, 1:,  :].reshape((1, gr.Ny-2, gr.Nx))), \\\n                        Out.Qy[iz, -1, :].reshape((1, 1, gr.Nx))), axis=1).reshape(shp)\n    return X, Y, U, V\n\n\ndef unique(x, tol=0.0001):\n    """return sorted unique values of x, keeping ascending or descending direction"""\n    if x[0]>x[-1]:  # vector is reversed\n        x = np.sort(x)[::-1]  # sort and reverse\n        return x[np.hstack((np.diff(x) < -tol, True))]\n    else:\n        x = np.sort(x)\n        return x[np.hstack((np.diff(x) > +tol, True))]\n\n    \ndef fdm3(gr, kx, ky, kz, FQ, HI, IBOUND):\n    \'\'\'Steady state 3D Finite Difference Model returning computed heads and flows.\n        \n    Heads and flows are returned as 3D arrays as specified under output parmeters.\n    \n    Parameters\n    ----------\n    \'gr\' : `grid_object`, generated by gr = Grid(x, y, z, ..)\n    `kx`, `ky`, `kz` : ndarray, shape: (Nz, Ny, Nx), [L/T]\n        hydraulic conductivities along the three axes, 3D arrays.\n    `FQ` : ndarray, shape: (Nz, Ny, Nx), [L3/T]\n        prescrived cell flows (injection positive, zero of no inflow/outflow)\n    `IH` : ndarray, shape: (Nz, Ny, Nx), [L]\n        initial heads. `IH` has the prescribed heads for the cells with prescribed head.\n    `IBOUND` : ndarray, shape: (Nz, Ny, Nx) of int\n        boundary array like in MODFLOW with values denoting\n        * IBOUND>0  the head in the corresponding cells will be computed\n        * IBOUND=0  cells are inactive, will be given value NaN\n        * IBOUND<0  coresponding cells have prescribed head\n    \n    outputs\n    -------    \n    `Out` : namedtuple containing heads and flows:\n        `Out.Phi` : ndarray, shape: (Nz, Ny, Nx), [L3/T] \n            computed heads. Inactive cells will have NaNs\n        `Out.Q`   : ndarray, shape: (Nz, Ny, Nx), [L3/T]\n            net inflow in all cells, inactive cells have 0\n        `Out.Qx   : ndarray, shape: (Nz, Ny, Nx-1), [L3/T] \n            intercell flows in x-direction (parallel to the rows)\n        `Out.Qy`  : ndarray, shape: (Nz, Ny-1, Nx), [L3/T] \n            intercell flows in y-direction (parallel to the columns)\n        `Out.Qz`  : ndarray, shape: (Nz-1, Ny, Nx), [L3/T] \n            intercell flows in z-direction (vertially upward postitive)\n        the 3D array with the final heads with `NaN` at inactive cells.\n    \n    TO 160905\n    \'\'\'\n\n    # define the named tuple to hold all the output of the model fdm3\n    Out = namedtuple(\'Out\',[\'Phi\', \'Q\', \'Qx\', \'Qy\', \'Qz\'])\n    Out.__doc__ = """fdm3 output, <namedtuple>, containing fields Phi, Qx, Qy and Qz\\n \\\n                    Use Out.Phi, Out.Q, Out.Qx, Out.Qy and Out.Qz"""                            \n                                \n    if kx.shape != gr.shape:\n        raise AssertionError("shape of kx {0} differs from that of model {1}".format(kx.shape,SHP))\n    if ky.shape != gr.shape:\n        raise AssertionError("shape of ky {0} differs from that of model {1}".format(ky.shape,SHP))\n    if kz.shape != gr.shape:\n        raise AssertionError("shape of kz {0} differs from that of model {1}".format(kz.shape,SHP))\n    \n    active = (IBOUND > 0).reshape(gr.Nod,)  # boolean vector denoting the active cells\n    inact  = (IBOUND ==0).reshape(gr.Nod,) # boolean vector denoting inacive cells\n    fxhd   = (IBOUND < 0).reshape(gr.Nod,)  # boolean vector denoting fixed-head cells\n\n    # reshaping shorthands\n    rx = lambda a : np.reshape(a, (1, 1, gr.Nx))\n    ry = lambda a : np.reshape(a, (1, gr.Ny, 1))\n    rz = lambda a : np.reshape(a, (gr.Nz, 1, 1))\n        \n    # half cell flow resistances\n    Rx = 0.5 * rx(gr.dx) / (ry(gr.dy) * rz(gr.dz)) / kx\n    Ry = 0.5 * ry(gr.dy) / (rz(gr.dz) * rx(gr.dx)) / ky\n    Rz = 0.5 * rz(gr.dz) / (rx(gr.dx) * ry(gr.dy)) / kz\n    \n    # set flow resistance in inactive cells to infinite\n    Rx = Rx.reshape(gr.Nod,); Rx[inact] = np.Inf; Rx=Rx.reshape(gr.shape)\n    Ry = Ry.reshape(gr.Nod,); Ry[inact] = np.Inf; Ry=Ry.reshape(gr.shape)\n    Rz = Rz.reshape(gr.Nod,); Rz[inact] = np.Inf; Rz=Rz.reshape(gr.shape)\n    \n    # conductances between adjacent cells\n    Cx = 1 / (Rx[:, :, :-1] + Rx[:, :, 1:])\n    Cy = 1 / (Ry[:, :-1, :] + Ry[:, 1:, :])\n    Cz = 1 / (Rz[:-1, :, :] + Rz[1:, :, :])\n    \n    IE = gr.NOD[:, :, 1:]  # east neighbor cell numbers\n    IW = gr.NOD[:, :, :-1] # west neighbor cell numbers\n    IN = gr.NOD[:, :-1, :] # north neighbor cell numbers\n    IS = gr.NOD[:, 1:, :]  # south neighbor cell numbers\n    IT = gr.NOD[:-1, :, :] # top neighbor cell numbers\n    IB = gr.NOD[1:, :, :]  # bottom neighbor cell numbers\n    \n    R = lambda x : x.ravel()  # generate anonymous function R(x) as shorthand for x.ravel()\n\n    # notice the call  csc_matrix( (data, (rowind, coind) ), (M,N))  tuple within tupple\n    # also notice that Cij = negative but that Cii will be postive, namely -sum(Cij)\n    A = sp.csc_matrix(( -np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),\\\n                        (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\\\n                         np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ),\\\n                      )),(gr.Nod,gr.Nod))\n    \n    # to use the vector of diagonal values in a call of sp.diags() we need to have it aa a \n    # standard nondimensional numpy vector.\n    # To get this:\n    # - first turn the matrix obtained by A.sum(axis=1) into a np.array by np.array( .. )\n    # - then take the whole column to loose the array orientation (to get a dimensionless numpy vector)\n    adiag = np.array(-A.sum(axis=1))[:,0]\n    \n    Adiag = sp.diags(adiag)  # diagonal matrix with a[i,i]\n    \n    RHS = FQ.reshape((gr.Nod,1)) - A[:,fxhd].dot(HI.reshape((gr.Nod,1))[fxhd]) # Right-hand side vector\n    \n    Out.Phi = HI.flatten() # allocate space to store heads\n    \n    Out.Phi[active] = spsolve( (A+Adiag)[active][:,active] ,RHS[active] ) # solve heads at active locations\n    \n    # net cell inflow\n    Out.Q  = (A+Adiag).dot(Out.Phi).reshape(gr.shape)\n\n    # set inactive cells to NaN\n    Out.Phi[inact] = np.NaN # put NaN at inactive locations\n    \n    # reshape Phi to shape of grid\n    Out.Phi = Out.Phi.reshape(gr.shape)\n    \n    #Flows across cell faces\n    Out.Qx =  -np.diff( Out.Phi, axis=2) * Cx\n    Out.Qy =  +np.diff( Out.Phi, axis=1) * Cy\n    Out.Qz =  +np.diff( Out.Phi, axis=0) * Cz\n    \n    return Out # all outputs in a named tuple for easy access')

import fdm_c
from importlib import reload
reload(fdm_c)

# Make sure that your modules like grid are in the sys.path
import sys

path2modules = './modules/'

if not path2modules in sys.path:
    sys.path.append(path2modules)

reload(mfgrid)

gr.z

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt # combines namespace of numpy and pyplot

import numpy as np
import mfgrid
Grid = mfgrid.Grid

# specify a rectangular grid
x = np.arange(-1000.,  1000.,  25.)
y = np.arange( 1000., -1000., -25.)
z = np.array([20, 0 ,-10, -100.])

gr = Grid(x, y, z) # generating a grid object for this model

k = 10.0 # m/d uniform conductivity
kx = k * gr.const(k) # using gr.const(value) to generate a full 3D array of the correct shape
ky = k * gr.const(k)
kz = k * gr.const(k)

IBOUND = gr.const(1)
IBOUND[:, -1, :] = -1  # last row of model heads are prescribed
IBOUND[:, 40:45, 20:70]=0 # inactive

FQ = gr.const(0.)    # all flows zero. Note SHP is the shape of the model grid
FQ[2, 30, 25] = -1200  # extraction in this cell

HI = gr.const(0.)

Out = fdm_c.fdm3(gr, kx, ky, kz, FQ, HI, IBOUND)

layer = 2 # contours for this layer
nc = 50   # number of contours in total

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title("Contours (%d in total) of the head in layer %d with inactive section" % (nc, layer))
plt.contour(gr.xm, gr.ym, Out.Phi[:,:,layer], nc) # using gr here also

#plt.quiver(X, Y, U, V) # show velocity vectors
#X, Y, U, V = fdm_c.quivdata(Out, gr, iz=0) # use function in fdm_c
X, Y, U, V = gr.quivdata(Out, iz=0) # use method in Grid
plt.quiver(X, Y, U, V)


