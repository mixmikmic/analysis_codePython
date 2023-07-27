import pynbn
c = pynbn.connect('lionfish_pynbn','password');

sp = c.get_tvk(query='Bombus terrestris') #get the tvk (the "taxon version key" for buff tails)
keys = []
for res in sp['results']:
    k = res['ptaxonVersionKey']
    keys.append(str(k))
print "%d species match this query string" % len(keys)
print keys
tvk = keys[0]
print "We'll use the first key (%s)" % tvk
#we usually take the first item from this list (advice from the NBN hackday)
obs = c.get_observations(tvks=[tvk], start_year=1990, end_year=2010) #get observations
print "There are %d records for B. terrestris between 1990 and 2010" % len(obs)

import numpy as np
import bng
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

coords = []
for o in obs:    
    try:
        coords.append(bng.grid_ref_str_to_easting_northing(o['location']))
    except IndexError:
        pass

coords = np.array(coords) #convert to numpy array

fig, ax = plt.subplots(figsize=(10, 10))
ax.hexbin(coords[:,0],coords[:,1],gridsize=30,cmap='gray',bins='log')



