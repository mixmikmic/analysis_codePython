get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from gPhoton import gFind
from gPhoton import gAperture
from gPhoton import gMap
from gPhoton.gphoton_utils import read_lc
import datetime

from astropy.time import Time
from astropy import units as u
# from astropy.analytic_functions import blackbody_lambda #OLD!
from astropy.modeling.blackbody import blackbody_lambda
from gatspy.periodic import LombScargleFast

import extinction

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})

ra = 301.5644
dec = 44.45684

exp_data = gFind(band='NUV', skypos=[ra, dec], exponly=True)

exp_data

(exp_data['NUV']['t0'] - exp_data['NUV']['t0'][0]) / (60. * 60. * 24. * 365.)

# step_size = 20. # the time resolution in seconds

target = 'KIC8462852'

# phot_rad = 0.0045 # in deg
# ap_in = 0.0050 # in deg
# ap_out = 0.0060 # in deg

# print(datetime.datetime.now())
# for k in range(len(exp_data['NUV']['t0'])):
#     photon_events = gAperture(band='NUV', skypos=[ra, dec], stepsz=step_size, radius=phot_rad,
#                               annulus=[ap_in, ap_out], verbose=3, csvfile=target+ '_' +str(k)+"_lc.csv",
#                               trange=[int(exp_data['NUV']['t0'][k]), int(exp_data['NUV']['t1'][k])+1], 
#                               overwrite=True)
    
#     print(datetime.datetime.now(), k)



med_flux = np.array(np.zeros(4), dtype='float')
med_flux_err = np.array(np.zeros(4), dtype='float')

time_big = np.array([], dtype='float')
mag_big = np.array([], dtype='float')
flux_big = np.array([], dtype='float')

for k in range(4):
    data = read_lc(target+ '_' +str(k)+"_lc.csv")
    med_flux[k] = np.nanmedian(data['flux_bgsub'])
    med_flux_err[k] = np.std(data['flux_bgsub'])

    time_big = np.append(time_big, data['t_mean'])
    flux_big = np.append(flux_big, data['flux_bgsub'])
    mag_big = np.append(mag_big, data['mag'])
    
#     t0k = Time(int(data['t_mean'][0]) + 315964800, format='unix').mjd
    flg0 = np.where((data['flags'] == 0))[0]
    
    # for Referee: convert GALEX time -> MJD
    t_unix = Time(data['t_mean'] + 315964800, format='unix')
    mjd_time = t_unix.mjd
    t0k = (mjd_time[0])
    
    plt.figure()
    plt.errorbar((mjd_time - t0k)*24.*60.*60., data['flux_bgsub']/(1e-15), yerr=data['flux_bgsub_err']/(1e-15), 
             marker='.', linestyle='none', c='k', alpha=0.75, lw=0.5, markersize=2)
    
    plt.errorbar((mjd_time[flg0] - t0k)*24.*60.*60., data['flux_bgsub'][flg0]/(1e-15), 
                 yerr=data['flux_bgsub_err'][flg0]/(1e-15), 
             marker='.', linestyle='none')
#     plt.xlabel('GALEX time (sec - '+str(t0k)+')')
    plt.xlabel('MJD - '+ format(t0k, '9.3f') +' (seconds)')
    plt.ylabel('NUV Flux \n'
               r'(x10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ ${\rm\AA}^{-1}$)')
    plt.savefig(target+ '_' +str(k)+"_lc.pdf", dpi=150, bbox_inches='tight', pad_inches=0.25)

    
    flagcol = np.zeros_like(mjd_time)
    flagcol[flg0] = 1
    dfout = pd.DataFrame(data={'MJD':mjd_time, 
                               'flux':data['flux_bgsub']/(1e-15), 
                               'fluxerr':data['flux_bgsub_err']/(1e-15),
                               'flag':flagcol})
    dfout.to_csv(target+ '_' +str(k)+'data.csv', index=False, columns=('MJD', 'flux','fluxerr', 'flag'))
    



# k=2
# data = read_lc(target+ '_' +str(k)+"_lc.csv")

# t0k = int(data['t_mean'][0])
# plt.figure(figsize=(14,5))
# plt.errorbar(data['t_mean'] - t0k, data['flux_bgsub'], yerr=data['flux_bgsub_err'], marker='.', linestyle='none')
# plt.xlabel('GALEX time (sec - '+str(t0k)+')')
# plt.ylabel('NUV Flux')

# try cutting on flags=0
flg0 = np.where((data['flags'] == 0))[0]
plt.figure(figsize=(14,5))
plt.errorbar(data['t_mean'][flg0] - t0k, data['flux_bgsub'][flg0]/(1e-15), yerr=data['flux_bgsub_err'][flg0]/(1e-15), 
             marker='.', linestyle='none')
plt.xlabel('GALEX time (sec - '+str(t0k)+')')
# plt.ylabel('NUV Flux')
plt.ylabel('NUV Flux \n' 
           r'(x10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ ${\rm\AA}^{-1}$)')
plt.title('Flags = 0')



minper = 10 # my windowing
maxper = 200000
nper = 1000
pgram = LombScargleFast(fit_offset=False)
pgram.optimizer.set(period_range=(minper,maxper))

pgram = pgram.fit(time_big - min(time_big), flux_big - np.nanmedian(flux_big))

df = (1./minper - 1./maxper) / nper
f0 = 1./maxper

pwr = pgram.score_frequency_grid(f0, df, nper)

freq = f0 + df * np.arange(nper)
per = 1./freq

##
plt.figure()
plt.plot(per, pwr, lw=0.75)
plt.xlabel('Period (seconds)')
plt.ylabel('L-S Power')
plt.xscale('log')
plt.xlim(10,500)
plt.savefig('periodogram.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

t_unix = Time(exp_data['NUV']['t0'] + 315964800, format='unix')
mjd_time_med = t_unix.mjd
t0k = (mjd_time[0])

plt.figure(figsize=(9,5))
plt.errorbar(mjd_time_med - mjd_time_med[0], med_flux/1e-15, yerr=med_flux_err/1e-15,
             linestyle='none', marker='o')
plt.xlabel('MJD - '+format(mjd_time[0], '9.3f')+' (days)')
# plt.ylabel('NUV Flux')
plt.ylabel('NUV Flux \n' 
           r'(x10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ ${\rm\AA}^{-1}$)')
# plt.title(target)
plt.savefig(target+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

# average time of the gPhoton data
print(np.mean(exp_data['NUV']['t0']))
t_unix = Time(np.mean(exp_data['NUV']['t0']) + 315964800, format='unix')
t_date = t_unix.yday
print(t_date)

mjd_date = t_unix.mjd
print(mjd_date)

plt.errorbar([10, 14], [16.46, 16.499], yerr=[0.01, 0.006], linestyle='none', marker='o')
plt.xlabel('Quarter (approx)')
plt.ylabel(r'$m_{NUV}$ (mag)')
plt.ylim(16.52,16.44)

gck_time = Time(1029843320.995 + 315964800, format='unix')
gck_time.mjd

# and to push the comparison to absurd places...
# http://astro.uchicago.edu/~bmontet/kic8462852/reduced_lc.txt

df = pd.read_table('reduced_lc.txt', delim_whitespace=True, skiprows=1, 
                   names=('time','raw_flux', 'norm_flux', 'model_flux'))

# time = BJD-2454833
# *MJD = JD - 2400000.5

plt.figure()
plt.plot(df['time'] + 2454833 - 2400000.5, df['model_flux'], c='grey', lw=0.2)

gtime = [mjd_date, gck_time.mjd]
gmag = np.array([16.46, 16.499])
gflux = np.array([1, 10**((gmag[1] - gmag[0]) / (-2.5))])
gerr = np.abs(np.array([0.01, 0.006]) * np.log(10) / (-2.5) * gflux)

plt.errorbar(gtime, gflux, yerr=gerr, 
             linestyle='none', marker='o')
plt.ylim(0.956,1.012)
plt.xlabel('MJD (days)')
plt.ylabel('Relative Flux')

# plt.savefig(target+'_compare.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

####################
# add in WISE
plt.figure()
plt.plot(df['time'] + 2454833 - 2400000.5, df['model_flux'], c='grey', lw=0.2)
plt.errorbar(gtime, gflux, yerr=gerr, 
             linestyle='none', marker='o')
# the WISE W1-band results from another notebook
wise_time = np.array([55330.86838, 55509.906929000004])
wise_flux = np.array([ 1.,0.98627949])
wise_err = np.array([ 0.02011393,  0.02000256])
plt.errorbar(wise_time, wise_flux, yerr=wise_err,
             linestyle='none', marker='o')
plt.ylim(0.956,1.025)
plt.xlabel('MJD (days)')
plt.ylabel('Relative Flux')

# plt.savefig(target+'_compare2.png', dpi=150, bbox_inches='tight', pad_inches=0.25)

ffi_file = '8462852.txt'
ffi = pd.read_table(ffi_file, delim_whitespace=True, names=('mjd', 'flux', 'err'))


plt.figure()
# plt.plot(df['time'] + 2454833 - 2400000.5, df['model_flux'], c='grey', lw=0.2)

plt.errorbar(ffi['mjd'], ffi['flux'], yerr=ffi['err'], linestyle='none', marker='s', c='gray', 
             zorder=0, alpha=0.7)


gtime = [mjd_date, gck_time.mjd]
gmag = np.array([16.46, 16.499])
gflux = np.array([1, 10**((gmag[1] - gmag[0]) / (-2.5))])
gerr = np.abs(np.array([0.01, 0.006]) * np.log(10) / (-2.5) * gflux)

plt.errorbar(gtime, gflux, yerr=gerr, 
             linestyle='none', marker='o', zorder=1, markersize=10)

plt.xlabel('MJD (days)')
plt.ylabel('Relative Flux')

# plt.errorbar(mjd_time_med, med_flux/np.mean(med_flux), yerr=med_flux_err/np.mean(med_flux),
#              linestyle='none', marker='o', markerfacecolor='none', linewidth=0.5)
# print(np.sqrt(np.sum((med_flux_err / np.mean(med_flux))**2) / len(med_flux)))


plt.ylim(0.956,1.012)
# plt.ylim(0.9,1.1)


plt.savefig(target+'_compare.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

print('gflux: ', gflux, gerr)

# considering extinction... 
# w/ thanks to the Padova Isochrone page for easy shortcut to getting these extinction values:
# http://stev.oapd.inaf.it/cgi-bin/cmd

A_NUV = 2.27499 # actually A_NUV / A_V, in magnitudes, for R_V = 3.1
A_Kep = 0.85946 # actually A_Kep / A_V, in magnitudes, for R_V = 3.1
A_W1 = 0.07134 # actually A_W1 / A_V, in magnitudes, for R_V = 3.1

wave_NUV = 2556.69 # A
wave_Kep = 6389.68 # A
wave_W1 = 33159.26 # A


print('nuv')
## use the Long Cadence data.
frac_kep = (np.median(df['model_flux'][np.where((np.abs(df['time']+ 2454833 - 2400000.5 -gtime[0])) < 25)[0]]) - 
            np.median(df['model_flux'][np.where((np.abs(df['time']+ 2454833 - 2400000.5 -gtime[1])) < 25)[0]]))
## could use the FFI data, but it slightly changes the extinction coefficients and they a pain in the butt 
## to adjust manually because I was an idiot how i wrote this
# frac_kep = (np.median(ffi['flux'][np.where((np.abs(ffi['mjd'] -gtime[0])) < 75)[0]]) -
#             np.median(ffi['flux'][np.where((np.abs(ffi['mjd'] -gtime[1])) < 75)[0]]))
    
print(frac_kep)

mag_kep = -2.5 * np.log10(1.-frac_kep)
print(mag_kep)

mag_nuv = mag_kep / A_Kep * A_NUV
print(mag_nuv)

frac_nuv = 10**(mag_nuv / (-2.5))
print(1-frac_nuv)



frac_kep_w = (np.median(df['model_flux'][np.where((np.abs(df['time']+ 2454833 - 2400000.5 -wise_time[0])) < 25)[0]]) - 
              np.median(df['model_flux'][np.where((np.abs(df['time']+ 2454833 - 2400000.5 -wise_time[1])) < 25)[0]]))
print('w1')
print(frac_kep_w)
mag_kep_w = -2.5 * np.log10(1.-frac_kep_w)
print(mag_kep_w)
mag_w1 = mag_kep_w / A_Kep * A_W1
print(mag_w1)
frac_w1 = 10**(mag_w1 / (-2.5))
print(1-frac_w1)


plt.errorbar([wave_Kep, wave_NUV], [1-frac_kep, gflux[1]], yerr=[0, np.sqrt(np.sum(gerr**2))], 
             label='Observed', marker='o')
plt.plot([wave_Kep, wave_NUV], [1-frac_kep, frac_nuv], '--o', label=r'$R_V$=3.1 Model')
plt.legend(fontsize=10, loc='lower right')
plt.xlabel(r'Wavelength ($\rm\AA$)')
plt.ylabel('Relative Flux Decrease')
plt.ylim(0.93,1)
# plt.savefig(target+'_extinction_model_1.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

plt.errorbar([wave_Kep, wave_W1], [1-frac_kep_w, wise_flux[1]], yerr=[0, np.sqrt(np.sum(wise_err**2))], 
             label='Observed', marker='o', c='purple')
plt.plot([wave_Kep, wave_W1], [1-frac_kep_w, frac_w1], '--o', label='Extinction Model', c='green')
plt.legend(fontsize=10, loc='lower right')
plt.xlabel(r'Wavelength ($\rm\AA$)')
plt.ylabel('Relative Flux')
plt.ylim(0.93,1.03)
# plt.savefig(target+'_extinction_model_2.png', dpi=150, bbox_inches='tight', pad_inches=0.25)

plt.errorbar([wave_Kep, wave_NUV], [1-frac_kep, gflux[1]], yerr=[0, np.sqrt(np.sum(gerr**2))], 
             label='Observed1', marker='o')
plt.plot([wave_Kep, wave_NUV], [1-frac_kep, frac_nuv], '--o', label='Extinction Model1')


plt.errorbar([wave_Kep, wave_W1], [1-frac_kep_w, wise_flux[1]], yerr=[0, np.sqrt(np.sum(wise_err**2))], 
             label='Observed2', marker='o', c='purple')
plt.plot([wave_Kep, wave_W1], [1-frac_kep_w, frac_w1], '--o', label='Extinction Model2')
plt.legend(fontsize=10, loc='upper left')
plt.xlabel(r'Wavelength ($\rm\AA$)')
plt.ylabel('Relative Flux')
plt.ylim(0.93,1.03)
plt.xscale('log')
# plt.savefig(target+'_extinction_model_2.png', dpi=150, bbox_inches='tight', pad_inches=0.25)

# the "STANDARD MODEL" for extinction
A_V = 0.0265407
R_V = 3.1

ext_out = extinction.ccm89(np.array([wave_Kep, wave_NUV]), A_V, R_V)

# (ext_out[1] - ext_out[0]) / ext_out[1]
print(10**(ext_out[0]/(-2.5)), (1-frac_kep)) # these need to match (within < 1%)
print(10**(ext_out[1]/(-2.5)), gflux[1]) # and then these won't, as per our previous plot

# print(10**((ext_out[1] - ext_out[0])/(-2.5)) / 10**(ext_out[0]/(-2.5)))

# now find an R_V (and A_V) that gives matching extinctions in both bands
# start by doing a grid over plasible A_V values at each R_V I care about... we doing this brute force!

ni=50
nj=50
di = 0.2
dj = 0.0003
ext_out_grid = np.zeros((2,ni,nj))
for i in range(ni):
    R_V = 1.1 + i*di
    for j in range(nj):
        A_V = 0.02 + j*dj

        ext_out_ij = extinction.ccm89(np.array([wave_Kep, wave_NUV]), A_V, R_V)
        ext_out_grid[:,i,j] = 10**(ext_out_ij/(-2.5))
        
R_V_grid = 1.1 + np.arange(ni)*di
A_V_grid = 0.02 + np.arange(nj)*dj

# now plot where the Kepler extinction (A_Kep) matches the measured value, for each R_V

plt.figure()
plt.contourf( A_V_grid, R_V_grid, ext_out_grid[0,:,:], origin='lower' )
cb = plt.colorbar()
cb.set_label('A_Kep (flux)')

A_V_match = np.zeros(ni)
ext_NUV = np.zeros(ni)
for i in range(ni):
    xx = np.interp(1-frac_kep, ext_out_grid[0,i,:][::-1], A_V_grid[::-1])
    plt.scatter(xx, R_V_grid[i], c='r', s=10)
    A_V_match[i] = xx
    ext_NUV[i] = 10**(extinction.ccm89(np.array([wave_NUV]),xx, R_V_grid[i]) / (-2.5))
    
plt.ylabel('R_V')
plt.xlabel('A_V (mag)')
plt.show()

# Finally: at what R_V do we both match A_Kep (as above), and *now* A_NUV?

RV_final = np.interp(gflux[1], ext_NUV, R_V_grid)
print(RV_final)

# this is the hacky way to sorta do an error propogation.... 
RV_err = np.mean(np.interp([gflux[1] + np.sqrt(np.sum(gerr**2)), 
                            gflux[1] - np.sqrt(np.sum(gerr**2))], 
                           ext_NUV, R_V_grid)) - RV_final
print(RV_err)

AV_final = np.interp(gflux[1], ext_NUV, A_V_grid)
print(AV_final)

plt.plot(R_V_grid, ext_NUV)
plt.errorbar(RV_final, gflux[1], yerr=np.sqrt(np.sum(gerr**2)), xerr=RV_err, marker='o')
plt.xlabel('R_V')
plt.ylabel('A_NUV (flux)')


plt.errorbar([wave_Kep, wave_NUV], [1-frac_kep, gflux[1]], yerr=[0, np.sqrt(np.sum(gerr**2))], 
             label='Observed', marker='o', linestyle='none', zorder=0, markersize=10)

plt.plot([wave_Kep, wave_NUV], [1-frac_kep, gflux[1]], label=r'$R_V$=5.0 Model', c='r', lw=3, alpha=0.7,zorder=1)
plt.plot([wave_Kep, wave_NUV], [1-frac_kep, frac_nuv], '--', label=r'$R_V$=3.1 Model',zorder=2)

plt.legend(fontsize=10, loc='lower right')
plt.xlabel(r'Wavelength ($\rm\AA$)')
plt.ylabel('Relative Flux Decrease')
plt.ylim(0.93,1)

plt.savefig(target+'_extinction_model_2.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

# For referee: compute how many sigma away the Rv=3.1 model is from the Rv=5
print( (gflux[1] - frac_nuv) / np.sqrt(np.sum(gerr**2)), np.sqrt(np.sum(gerr**2)) )

print( (gflux[1] - frac_nuv) / 3., 3. * np.sqrt(np.sum(gerr**2)) )

# how much Hydrogen would you need to cause this fading?
# http://www.astronomy.ohio-state.edu/~pogge/Ast871/Notes/Dust.pdf
# based on data from Rachford et al. (2002) http://adsabs.harvard.edu/abs/2002ApJ...577..221R

A_Ic = extinction.ccm89(np.array([8000.]), AV_final, RV_final)
N_H = A_Ic / ((2.96 - 3.55 * ((3.1 / RV_final)-1)) * 1e-22)

print(N_H[0] , 'cm^-2')

# see also http://adsabs.harvard.edu/abs/2009MNRAS.400.2050G for R_V=3.1 only
print(2.21e21 * AV_final, 'cm^-2')

1-gflux[1]

# do simple thing first: a grid of temperatures starting at T_eff of the star (SpT = F3, T_eff = 6750)
temp0 = 6750 * u.K
wavelengths = [wave_Kep, wave_NUV] * u.AA
wavegrid = np.arange(wave_NUV, wave_Kep) * u.AA

flux_lam0 = blackbody_lambda(wavelengths, temp0)
flux_lamgrid = blackbody_lambda(wavegrid, temp0)

plt.plot(wavegrid, flux_lamgrid/1e6)
plt.scatter(wavelengths, flux_lam0/1e6)

Ntemps = 50
dT = 5 * u.K
flux_lam_out = np.zeros((2,Ntemps))

for k in range(Ntemps):
    flux_new = blackbody_lambda(wavelengths, (temp0 - dT*k) )
    flux_lam_out[:,k] = flux_new

    

# [1-frac_kep, gflux[1]]
yy = flux_lam_out[0,:] / flux_lam_out[0,0]
xx = temp0 - np.arange(Ntemps)*dT

temp_new = np.interp(1-frac_kep, yy[::-1], xx[::-1] )
# this is the hacky way to sorta do an error propogation.... 
err_kep = np.mean(ffi['err'][np.where((np.abs(ffi['mjd'] -gtime[0])) < 50)[0]])

temp_err = (np.interp([1-frac_kep - err_kep, 
                              1-frac_kep + err_kep], 
                              yy[::-1], xx[::-1]))
temp_err = (temp_err[1] - temp_err[0])/2.

print(temp_new, temp_err)


yy2 = flux_lam_out[1,:] / flux_lam_out[1,0]

NUV_new = np.interp(temp_new, xx[::-1], yy2[::-1])

print(NUV_new)
print(gflux[1], np.sqrt(np.sum(gerr**2)))

plt.plot(temp0 - np.arange(Ntemps)*dT, flux_lam_out[0,:]/flux_lam_out[0,0], label='Blackbody model (Kep)')
plt.plot(temp0 - np.arange(Ntemps)*dT, flux_lam_out[1,:]/flux_lam_out[1,0],ls='--', label='Blackbody model (NUV)')

plt.errorbar(temp_new, gflux[1], yerr=np.sqrt(np.sum(gerr**2)), marker='o', label='Observed NUV' )



plt.scatter([temp_new], [1-frac_kep], s=60, marker='s')
plt.scatter([temp_new], [NUV_new], s=60, marker='s')

plt.legend(fontsize=10, loc='upper left')
plt.xlim(6650,6750)
plt.ylim(.9,1)
plt.ylabel('Fractional flux')
plt.xlabel('Temperature')
# plt.title('Tuned to Kepler Dimming')
plt.savefig(target+'_blackbody.png', dpi=150, bbox_inches='tight', pad_inches=0.25)



