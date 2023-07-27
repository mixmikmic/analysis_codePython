#importing modules
import math
from __future__ import division

#Variable declaration
ec=4*10**-4;    #electrical conductivity of intrinsic silicon at room temperature(ohm^-1 m^-1)
me=0.14;    #The electron mobility(m^2 V^-1 s^-1)
mh=0.04;   #The hole mobility(m^2 V^-1 s^-1)
e=1.6*10**-19;   #charge of electron(c)

#Calculation
ni=ec/(e*(me+mh));    #The intrinsic carrier concentration at room temperature(m^-3)

#Result
print "The intrinsic carrier concentration at room temperature is",round(ni/10**16,4),"*10**16 m^-3"

#importing modules
import math
from __future__ import division

#Variable declaration
d=2.37*10**19;   #The intrinsic carrier density at room temperature(m^-3)
me=0.38;   #The electron mobility(m^2 V^-1 s^-1)
mh=0.18;   #The hole mobility(m^2 V^-1 s^-1)
e=1.6*10**-19;  #charge of electron(c)

#Calculation
r=1/(d*e*(me+mh));    #The resistivity of intrinsic carrier(ohm m)

#Result
print "The resistivity of intrinsic carrier is",round(r,4),"ohm m"

#importing modules
import math
from __future__ import division

#Variable declaration
r=2*10**-4;   #the resistivity of In-Sb(ohm m)
me=6;    #The electron mobility(m^2 V^-1 s^-1)
mh=0.2;  #The hole mobility(m^2 V^-1 s^-1)
e=1.6*10**-19;   #charge of electron(c)

#Calculation
d=1/(r*e*(me+mh));    #The intrinsic carrier density at room tepmerature(m^-3)

#Result
print "The intrinsic carrier density at room tepmerature is",round(d/10**21,2),"*10**21 m^-3"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=1.1*1.6*10**-19;    #The energy gap of silicon(J)
me=0.48;    #The electron mobility(m^2 V^-1 s^-1)
mh=0.13;    #The hole mobility(m^2 V^-1 s^-1)
h=6.625*10**-34;   #Planck's constant(m^2 Kg/sec)
e=1.6*10**-19;   #charge of electron(c)
m=9.11*10**-31;  #mass of an electron
kb=1.38*10**-23;  #Boltzmann's constant(m^2 Kg s^-2 k^-1)
t=300;   #temperature(K)

#Calculation
ni=2*(2*math.pi*m*kb*t/h**2)**(3/2)*math.exp(-Eg/(2*kb*t));    #intrinsic carrier concentration(m^-3)
ec=ni*e*(me+mh);   #The electrical conductivity at room temperature(ohm^-1 m^-1 *10^-3)

#Result
print "The electrical conductivity at room temperature is",round(ec*10**3,4),"*10**-3 ohm^-1 m^-1"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=1.43*1.6*10**-19;    #The energy gap of intrinsic GaAs(J)
xe=0.85;    #The electron mobility(m^2 V^-1 s^-1)
xh=0.04;    #The hole mobility(m^2 V^-1 s^-1)
me=0.068*9.11*10**-31;   #effective mass of electron(m)
mh=0.5*9.11*10**-31;   #effective mass of hole(m)
h=6.625*10**-34;     #Planck's constant(m^2 Kg/sec)
e=1.6*10**-19;    #charge of electron(c)
m=9.11*10**-31;   #mass of an electron(kg)
kb=1.38*10**-23;  #Boltzmann's constant(m^2 Kg s^-2 k^-1)
t=300;    #temperature(K)

#Calculation
ni=2*(2*math.pi*kb*t/h**2)**(3/2)*(me*mh)**(3/4)*math.exp(-Eg/(2*kb*t));    #intrinsic carrier concentration(m^-3)
ec=ni*e*(xe+xh);    #The electrical conductivity at room temperature(ohm^-1 m^-1)

#Result
print "The intrinsic carrier concentration is",round(ni/10**12,3),"*10**12 m^-3"
print "The electrical conductivity at room temperature is",round(ec*10**7,4),"*10**-7 ohm^-1 m^-1"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=1.12*1.6*10**-19;     #Energy gap of Si semi conductor(J)
me=0.12*9.11*10**-31;   #The electron mobility(m^2 V^-1 s^-1)
mh=0.28*9.11*10**-31;   #The hole mobility(m^2 V^-1 s^-1)
t=300;    #temperature of fermi level(K)
kb=1.38*10**-23;   #Boltzmann's constant(m^2 Kg s^-2 k^-1)
m=9.11*10**-31;   #mass of an electron(Kg)

#Calculation
Ef=(Eg/2)+((3*kb*t/4)*math.log(mh/me));     #position of the fermi level(J)

#Result
print "The position of the fermi level is",round(Ef*10**20,6),"*10**-20 J or",round(Ef/(1.6*10**-19),4),"eV"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=1*1.6*10**-19;   #Energy gap(J)
E=0.1*1.6*10**-19;    #Fermi level is shifted by 10%(J)
me=1*9.11*10**-31;    #The electron mobility(m^2 V^-1 s^-1)
mh=4*9.11*10**-31;    #Effective mass of holes is 4 times that of electrons(m^2 V^-1 s^-1)
m=9.11*10**-31;    #mass of an electron(kg)
kb=1.38*10**-23;   #Boltzmann's constant(m^2 Kg s^-2 k^-1)

#Calculation
T=4*E/(3*kb*math.log(4));   #The Temperature of the fermi level shifted by 10%(K)

#Result
print "The Temperature of the fermi level shifted by 10% is",round(T,3),"K"

#importing modules
import math
from __future__ import division

#Variable declaration
l=1*10**-2;   #length of the intrinsic Ge rod(m)
b=1*10**-3;   #breadth of the intrinsic Ge rod(m)
t=1*10**-3;   #thickness of the intrinsic Ge rod(m)
T=300;      #temperature of the intrinsic Ge rod(K)
me=0.39;    #The electron mobility(m^2 V^-1 s^-1)
mh=0.19;    #The hole mobility(m^2 V^-1 s^-1)
ni=2.5*10**19;    #intrinsic carrier conduction(m^3)
e=1.6*10**-19;    #charge of electron(c)

#Calculation
ec=ni*e*(me+mh);    #The electrical conductivity at room temperature(ohm^-1 m^-1)
A=b*t;      #area(m^2)
R=l/(ec*A);    #The resistance of an intrinsic Ge rod(ohm)

#Result
print "The resistance of an intrinsic Ge rod is",int(R),"ohm"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=1.2*1.6*10**-19;    #The energy gap of intrinsic semiconductor(J)
T1=600;    #Temperature(K)
T2=300;    #Temperature(K)
e=1.6*10**-19;    #charge of electron(c)
kb=1.38*10**-23;     #Boltzmann's constant(m^2 Kg s^-2 k^-1)

#Calculation
x=math.exp((-Eg/(2*kb))*((1/T1)-(1/T2)));    #The ratio of conductiveness

#Result
print "The ratio of conductiveness is",round(x/10**5,2),"*10**5"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=0.72*1.6*10**-19;    #The band gap of Ge(J)
T1=293;    #Temperature(K)
T2=313;    #Temperature(K)
x1=2;     #The conductivity of Ge at T1(ohm^-1 m^-1)
e=1.6*10**-19;   #charge of electron(c)
kb=1.38*10**-23;   #Boltzmann's constant(m^2 Kg s^-2 k^-1)

#Calculation
x2=x1*math.exp((Eg/(2*kb))*((1/T1)-(1/T2)));    #The ratio of conductiveness

#Result
print "The conductivity of Ge at T2 is",round(x2,6),"ohm^-1 m^-1"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg1=0.36;    #The energy gap of intrinsic semiconductor A(eV)
Eg2=0.72;    #The energy gap of intrinsic semiconductor B(eV)
T1=300;     #Temperature of semiconductor A(K)
T2=300;     #Temperature of semiconductor B(K)
m=9.11*10**-31;   #mass of an electron(kg)
KT=0.026;     #kt(eV)

#Calculation
x=math.exp((Eg2-Eg1)/(2*KT));    #The intrinsic carrier density of A to B

#Result
print "The intrinsic carrier density of A to B is",int(x)

#importing modules
import math
from __future__ import division

#Variable declaration
T1=293;   #Temperature(K)
T2=373;   #Temperature(K)
x1=250;   #The conductivity of semiconductor at T1(ohm^-1 m^-1)
x2=1100;  #The conductivity of semiconductor at T2(ohm^-1 m^-1)
e=1.6*10**-19;   #charge of electron(c)
kb=1.38*10**-23;   #Boltzmann's constant(m^2 Kg s^-2 k^-1)

#Calculation
Eg=2*kb*math.log(x2/x1)*(T1*T2/(T2-T1));   #The band gap of semiconductor(J)

#Result
print "The band gap of semiconductor is",round(Eg*10**20,4),"*10**-20 J or",round(Eg/(1.6*10**-19),3),"eV"

#importing modules
import math
from __future__ import division

#Variable declaration
me=50;    #The electron mobility of pure semi conductor(m^2 V^-1 s^-1)
t1=4.2;   #temperature of pure semi conductor(K)
t2=300;   #temperature(K)

#Calculation
m=me*((t2**(-3/2))/(t1**(-3/2)));    #mobility of pure semi conductor(m^2 V^-1 s^-1)

#Result
print "mobility of pure semi conductor is",round(m,5),"m^2 V^-1 s^-1"

#importing modules
import math
from __future__ import division

#Variable declaration
ec1=19.96;    #The electrical conductivity of an intrinsic semi conductor(ohm^-1 m^-1)
ec2=79.44;    #The increasing electrical conductivity of an intrinsic semi conductor(ohm^-1 m^-1)
t1=333;   #temperature of an intrinsic semi conductor(K)
t2=373;   #increasing temperature of an intrinsic semi conductor(K)
kb=1.38*10**-23;   #Boltzmann's constant(m^2 Kg s^-2 k^-1)

#Calculation
Eg=2*kb*math.log(ec2/ec1)*((t1*t2)/(t2-t1));    #The band gap of an intrinsic semi conductor(J)

#Result
print "The band gap of an intrinsic semi conductor is",round(Eg*10**19,6),"*10**-19 J or",round(Eg/(1.6*10**-19),4),"eV"

