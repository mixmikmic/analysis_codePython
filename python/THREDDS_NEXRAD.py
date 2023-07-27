import matplotlib
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
get_ipython().magic('matplotlib inline')

# The archive of data on S3 URL did not work for me, despite .edu domain
#url = 'http://thredds-aws.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/'

#Trying motherlode URL
url = 'http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/'
from siphon.radarserver import RadarServer
rs = RadarServer(url)

from datetime import datetime, timedelta
query = rs.query()
query.stations('KLVX').time(datetime.utcnow())

rs.validate_query(query)

catalog = rs.get_catalog(query)

catalog.datasets

ds = list(catalog.datasets.values())[0]
ds.access_urls

from siphon.cdmr import Dataset
data = Dataset(ds.access_urls['CdmRemote'])

import numpy as np
def raw_to_masked_float(var, data):
    # Values come back signed. If the _Unsigned attribute is set, we need to convert
    # from the range [-127, 128] to [0, 255].
    if var._Unsigned:
        data = data & 255

    # Mask missing points
    data = np.ma.array(data, mask=data==0)

    # Convert to float using the scale and offset
    return data * var.scale_factor + var.add_offset

def polar_to_cartesian(az, rng):
    az_rad = np.deg2rad(az)[:, None]
    x = rng * np.sin(az_rad)
    y = rng * np.cos(az_rad)
    return x, y

sweep = 0
ref_var = data.variables['Reflectivity_HI']
ref_data = ref_var[sweep]
rng = data.variables['distanceR_HI'][:]
az = data.variables['azimuthR_HI'][sweep]

ref = raw_to_masked_float(ref_var, ref_data)
x, y = polar_to_cartesian(az, rng)

from metpy.plots import ctables  # For NWS colortable
ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

import matplotlib.pyplot as plt
import cartopy

def new_map(fig, lon, lat):
    # Create projection centered on the radar. This allows us to use x
    # and y relative to the radar.
    proj = cartopy.crs.LambertConformal(central_longitude=lon, central_latitude=lat)

    # New axes with the specified projection
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Add coastlines
    ax.coastlines('50m', 'black', linewidth=2, zorder=2)

    # Grab state borders
    state_borders = cartopy.feature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lines',
        scale='50m', facecolor='none')
    ax.add_feature(state_borders, edgecolor='black', linewidth=1, zorder=3)
    
    return ax

# Our specified time
#dt = datetime(2012, 10, 29, 15) # Superstorm Sandy
#dt = datetime(2016, 6, 18, 1)
dt = datetime(2016, 6, 8, 18) 
query = rs.query()
query.lonlat_point(-73.687, 41.175).time_range(dt, dt + timedelta(hours=1))

cat = rs.get_catalog(query)
cat.datasets

ds = list(cat.datasets.values())[0]
data = Dataset(ds.access_urls['CdmRemote'])
# Pull out the data of interest
sweep = 0
rng = data.variables['distanceR_HI'][:]
az = data.variables['azimuthR_HI'][sweep]
ref_var = data.variables['Reflectivity_HI']

# Convert data to float and coordinates to Cartesian
ref = raw_to_masked_float(ref_var, ref_var[sweep])
x, y = polar_to_cartesian(az, rng)

fig = plt.figure(figsize=(10, 10))
ax = new_map(fig, data.StationLongitude, data.StationLatitude)

# Set limits in lat/lon space
ax.set_extent([-77, -70, 38, 43])

# Add ocean and land background
ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', scale='50m',
                                            edgecolor='face',
                                            facecolor=cartopy.feature.COLORS['water'])
land = cartopy.feature.NaturalEarthFeature('physical', 'land', scale='50m',
                                           edgecolor='face',
                                           facecolor=cartopy.feature.COLORS['land'])

ax.add_feature(ocean, zorder=-1)
ax.add_feature(land, zorder=-1)
ax.pcolormesh(x, y, ref, cmap=ref_cmap, norm=ref_norm, zorder=0);



