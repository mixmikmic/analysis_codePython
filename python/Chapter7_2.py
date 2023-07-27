#importing modules
import math
from __future__ import division

#Variable declaration
E0=8.86*10**-12
mu0=4*3.14*10**-7
H=1

#Calculations
E=H*(math.sqrt(mu0/E0))

#Result
print"The Magnitude of E for a plane wave in free space is %3.1f"%E

#importing modules
import math
from __future__ import division

#Variable declaration
mu0=4*3.14*10**-7
mur=1
Er=2
E0=8.86*10**-12
E01=5
c=3*10**8

#Calculations
Z=math.sqrt((mu0*mur)/(E0*Er))
H0=(E01/Z)*10
v=((c)/math.sqrt(mur*Er))*10**-8

#Result
print"The Impedence of the Medium is %3.1f"%Z
print"The Peak Magnetic Field Intensity is %1.3f"%H0,"A/m"
print"The Velocity of the wave is %1.2f"%v,"*10**8 m/s"

#importing modules
import math
from __future__ import division

#Variable declaration
c=3*10**8
f=3*10**11
E0=50

#Calculations
lambdaa=(c/f)
B0=(E0/c)*10**7

#Result
print"The Wavelength is",lambdaa,"m or 10**-3 m"
print"The Amplitude of the oscillating magnetic field is %1.2f"%B0,"*10**-7 T"

#importing modules
import math
from __future__ import division

#Variable declaration
R=1.5*10**11   #Average distance between sun & Earth
P=3.8*10**26   #Power Radiated by sun


#Calculations
S=((P*60)/(4*3.14*(R**2)*4.2*100))*10**-2

#Result
print"The Average solar energy incident on earth is %1.2f"%S,"cal/cm**2/min"

#importing modules
import math
from __future__ import division

#Variable declaration
S=2     #solar energy
EH=1400
Z=376.6

#Calculations
E=math.sqrt(EH*Z)
H=math.sqrt(EH/Z)
E0=E*math.sqrt(2)
H0=H*math.sqrt(2)

#Result
print"The Amplitude of Electric field is %i"%E0,"V/m"
print"The Amplitude of Magnetic field per turn is %1.2f"%H0,"A-turn/m"

#importing modules
import math
from __future__ import division

#Variable declaration
EH=(1000/(16*3.14))
Z=376.6

#Calculations
E=math.sqrt(EH*Z)
H=math.sqrt(EH/Z)

#Result
print"The Intensity of Electric field is %2.2f"%E,"V/m"
print"The Intensity of Magnetic Field is %0.3f"%H,"A-turn/m"

#importing modules
import math
from __future__ import division

#Variable declaration
Er=2.22  #Dielectric Constant
D=3.87   #Outer Diameter
d=0.6    #Inner Diameter

from numpy.lib.scimath import logn
from math import e

#Calculations
Z=((60/math.sqrt(Er))*logn(e,(D/d)))

#Result
print"The Intensity of Electric field is %i"%Z,"Ohm"

#importing modules
import math
from __future__ import division

#Variable declaration
C=70*10**-12     #Cable Capacitance
L=0.39*10**-6    #Cable Inductance

#Calculations
Z0=(math.sqrt(L/C))

#Result
print"The Characteristic Impedence is %2.2f"%Z0,"Ohm"

#importing modules
import math
from __future__ import division

#Variable declaration
VF=0.62  #Velocity Factor of coaxial Cable

#Calculations
Er=(1/(VF**2))

#Result
print"The Dielectric Constant of the insulation used is %1.1f"%Er

