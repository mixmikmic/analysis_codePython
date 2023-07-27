#importing modules
import math
from __future__ import division

#Variable declaration   
e=1.6*10**-19;           #charge of electron(coulomb)
r0=2.81*10**-10;             #distance between ions(m)
A=1.748;      #constant
x=9*10**9;         #let x=1/(4*math.pi*epsilon0)
n=9;          

#Calculations
U0=-x*A*e**2*(1-(1/n))/(e*r0);       #potential energy per ion pair(eV)

#Result
print "potential energy per ion pair is",round(U0/2,3),"eV"
print "answer in the book varies due to rounding off errors"

