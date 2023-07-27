import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import cmocean as cmo
from matplotlib import cm

from scripts import catchmentErosion as eroCatch

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

dataTIN = eroCatch.catchmentErosion(folder='output',timestep=13)

dataTIN.regridTINdataSet()

dataTIN.plotdataSet(title='Elevation', data=dataTIN.z, color=cmo.cm.delta,crange=[-2000,2000])

dataTIN.plotdataSet(title='Slope', data=dataTIN.slp, color=cmo.cm.tempo,crange=[0.,1.])

dataTIN.plotdataSet(title='Erosion/Deposition [m]', data=dataTIN.dz, color=cmo.cm.balance,crange=[-300.,300.])

dataTIN.plotdataSet(title='Aspect', data=dataTIN.aspect, color=cmo.cm.haline,crange=[0.,2.],ctr='w')

dataTIN.plotdataSet(title='Horizontal curvature', data=dataTIN.hcurv, color=cmo.cm.balance,
                      crange=[-0.001,0.001])

dataTIN.plotdataSet(title='Vertical curvature', data=dataTIN.vcurv, color=cmo.cm.balance,
                      crange=[-0.0012,0.0012])

dataTIN.plotdataSet(title='Erosion/Deposition [m]', data=dataTIN.dz, color=cmo.cm.amp,  
                      crange=[0,500], erange=[570000,665000,4200000,4260000],
                      depctr=(50,150,300,500),size=(10,10))

dataTIN.getDepositedVolume(time=130000.,erange=[570000,665000,4200000,4260000])



