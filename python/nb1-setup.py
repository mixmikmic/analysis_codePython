# Complete set of Python 3.6 imports used for these examples

# Standard modules
import io
import logging
import lzma
import multiprocessing
import os
import ssl
import sys
import time
import urllib.request
import zipfile

# Third-party modules
import fastparquet      # Needs python-snappy and llvmlite
import graphviz         # To visualize Dask graphs 
import numpy as np
import pandas as pd
import psutil           # Memory stats
import dask
import dask.dataframe as dd
import bokeh.io         # For Dask profile graphs
import seaborn as sns   # For colormaps

# Support multiple lines of output in each cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Don't wrap tables
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.width = 300

# Show matplotlib and bokeh graphs inline in Jupyter notebook
get_ipython().magic('matplotlib inline')
bokeh.io.output_notebook()

print(sys.version)
np.__version__, pd.__version__, dask.__version__

task = ddf.head(n=2, npartitions=2, compute=False)

task.visualize()

task.dask

task._keys()







print(pd.DataFrame.__doc__)

print(dd.DataFrame.__doc__)

dd.from_pandas()



ddf = dd.from_pandas(df, chunksize=2)
task = ddf[ddf.a>2]

task.compute()

task.visualize()

print(dd.DataFrame.__doc__)

task._meta

task.npartitions
task.divisions

task._name

task.dask

task.dask[(task._name,0)]

task.dask[(task._name,1)]

get_ipython().magic('pinfo2 task.compute')

task2.compute()

task2.visualize()

task2.dask[(task2._name,0)]



