get_ipython().magic('pylab inline')

# Third party python libraries
import numpy

# Try and import from "The Virtual Brain"
from tvb.simulator.lab import *
LOG = get_logger('demo')
from tvb.datatypes.time_series import TimeSeriesRegion
import tvb.analyzers.fmri_balloon as bold
from tvb.simulator.plot import timeseries_interactive as timeseries_interactive

LOG.info("Configuring...")

#Initialise a Model, Coupling, and Connectivity.
lb = models.LarterBreakspear(QV_max=1.0, QZ_max=1.0, 
                             d_V=0.65, d_Z=0.65, 
                             aee=0.36, ani=0.4, ane=1.0, C=0.1)

lb.variables_of_interest = ["V", "W", "Z"]

white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([7.0])

white_matter_coupling = coupling.HyperbolicTangent(a=0.5*lb.QV_max, 
                                                   midpoint=lb.VT, 
                                                   sigma=lb.d_V)

#Initialise an Integrator
heunint = integrators.HeunDeterministic(dt=0.2)

#Initialise some Monitors with period in physical time
mon_tavg =  monitors.TemporalAverage(period=2.)
mon_bold  = monitors.Bold(period=2000.)
#Bundle them
what_to_watch = (mon_bold, mon_tavg)

#Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = simulator.Simulator(model = lb, 
                          connectivity = white_matter,
                          coupling = white_matter_coupling, 
                          integrator = heunint, 
                          monitors = what_to_watch)

sim.configure()

LOG.info("Starting simulation...")
#Perform the simulation
bold_data, bold_time = [], []
tavg_data, tavg_time = [], []

for raw, tavg in sim(simulation_length=480000):
    if not raw is None:
        bold_time.append(raw[0])
        bold_data.append(raw[1])
    
    if not tavg is None:
        tavg_time.append(tavg[0])
        tavg_data.append(tavg[1])

        
LOG.info("Finished simulation.")

#Make the lists numpy.arrays for easier use.
LOG.info("Converting result to array...")
TAVG_TIME = numpy.array(tavg_time)
BOLD_TIME = numpy.array(bold_time)
BOLD = numpy.array(bold_data)
TAVG = numpy.array(tavg_data)

#Create TimeSeries instance
tsr = TimeSeriesRegion(data = TAVG,
                       time = TAVG_TIME,
                       sample_period = 2.)
tsr.configure()

#Create and run the monitor/analyser
bold_model = bold.BalloonModel(time_series = tsr)
bold_data  = bold_model.evaluate()


bold_tsr = TimeSeriesRegion(connectivity = white_matter,
                            data = bold_data.data, 
                            time = bold_data.time)

#Prutty puctures...
tsi = timeseries_interactive.TimeSeriesInteractive(time_series = bold_tsr)
tsi.configure()
tsi.show()

