from math import pi
import math   
# given
Bc=0.8
Hc=510
Bg=0.8
A=12.566 
lg=0.0015
lc=0.36
N=500
# calculations
Fg=Bg/A*(2*lg)
Fc=Hc*lc
F=Fc+Fg
i=F/N
Pre=Bc/Hc
RelPre=Pre/A
F=Hc*lc
i=F /N #current
print 'The current in circuit is ',i

from math import pi
A=12.566 
lc=360
N=500
i=4
lg=2*10**-3
m=-A*(lc/lg)
c=(N*i*A)/(lg)
Hc=(N*i)/(lc)  #flux density
print 'The flux density is',Hc

import math
#given
N1=500
I1=10
N2=500
I2=10
Ibafe=3*52*10**-2
A=12.566
b=1200
Ag=4*10^-4
Ac=4*10^-4
lg=5*10^-3
Ibecore=0.515
c=0.0002067
d=0.0004134
#calculations
F1=N1*I1
F2=N2*I2
Pre=1200*A
Rbafe=(Ibafe)/(Pre*Ac)
Rg=lg/(A*Ag)
Rbecore=Ibecore/(Pre*Ac)
Bg=d/(Ag)
Hg=Bg/A    # airgap ﬂux
print 'The airgap flux value is',Hg

from math import pi
# given 
Irad=20
Orad=25
Dia=22.5
N=250
i=2.5
B=1.225
# calculations
l=2*pi*Dia*10**-2
radius=1/2*(Irad+Orad)
H=(N*i)/l
A=pi*((Orad -Irad)/2)**2*10**-4
z=(1.225)*(pi*6.25*10**-4)
y=(N*z)
L=(y/i)
core=(B/H)
l=(2*pi*22.5*10**-2)
Rcore=(l)/(core*A)
L=(N**2)/(Rcore)
print 'The magnetic flux is',L

import math
# given
n=500
E=100
A=0.001
b=1/120
f=1.2
#calculations
max1=(E/1000)*(b)
max2=(f*A)
E=(120*n*max2*2) # result
print 'flux density',E

from math import pi
#given
lg=0.4*10**-2
Bg=0.8
Hm=42*10**3
A=4*pi*10**-7
Ag=2.5*10**-4
Bm=0.95
#calculations
Hg=Bg/A
lm=(lg/Hm)*Hg
Am=(Bg*Ag)/(Bm)
print 'The dimension Am is',Am



