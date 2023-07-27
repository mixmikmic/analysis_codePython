#solution
#given
lh=25#mm#lower limit of hole
uh=25.02#mm#upper limit of hole
ls=24.95#mm#lower limit of shaft
us=24.97#mm#upper limit of shaft
h=uh-lh#mm#hole tolerance
s=us-ls#mm#shaft tolerance
a=lh-us#mm#alownce
print "the hole tolerance is,%f mm" %h
print ("the shaft tolerance is,%f mm " %s)
print ("the allowance is,%f mm" %a)


import math
#solution
#given
#shaft is 40 H8/f7
#since 40 mm lies in the diameter steps of 30 to 50 mm, therefore the mean diameter ie geometric mean of them
D=math.sqrt(30*50)#mm
i=0.45*((D)**(1/3))+(0.001*D)#mm#standard tolerance unit
#therfore,standard tolerance is
x=25*i*0.001#mm#standard tolerance for grade 8
x1=16*i*0.001#mm#standard tolerance for grade 7
#fundamental deviation
es=-5.5*(D)**0.41*0.001#mm
ei=es-0.025#mm
#limit of size
bs=40#mm#basic size
uh=40+0.039#mm#upper limitt of hole=lower limit  for hole+tolerance for hole
us=40-0.025#mm#uppr limit of shaft is lower limit of hole-fundamental deviation
ls=us-0.025#mm
print ("the standard tolernce for IT8 is,%f mm" %x)
print ("the satndard tolerance for IT7 is,%f mm" %x1)
print ("the fundamental upper deviation for shaft is,%f mm" %es)
print ("the fundamental lower deavtion for shaft is,%f mm " %ei)
print ("the basic size is,%f mm" %bs)
print ("upper limit for hole is,%f mm" %uh)
print ("the upper limit of shaft is,%f mm" %us)
print ("the lower limit of shaft is,%f mm" %ls)

#a.)12 mm elctric motion
#12 mm lies between 10 and 18,therefore
D=math.sqrt(10*18)#mm
i=0.45*(D)**0.33+0.001*D#standard tolrence unit
IT8=25*i*0.001#mm#standard tolerance for IT8
es=-11*(D)**0.41*0.001#mm#upper deviation for shaft
ei=es-IT8#mm#lower deviation for shaft
print ("the standard tolerance for shaft and hole of grade 8 is,%f mm" %IT8)
print ("the upper deviation for shaft is, %f mm" %es)
print ("the upper deviation for shaft is,%f mm" %ei)

import math
#75 mm basic size 
#since 75 lies betweenn 50 and 80
D=math.sqrt(50*80)#mm
i=0.45*(D)**0.33+0.001*D#standard tolerance unit
IT8=25*i*0.001#mm
IT7=16*i*0.001#mm
es=-2.5*(D)**0.34#mm#upper deviation of shaft
ei=es-IT7#mm#lower deviation fot hole
bs=75#mm#basic size
uh=75+IT8#upper limit of hole
us=75-0.01#mm#upper limit of shft
ls=us-0.03#mm
MxC=uh-ls#mm#maximum clearance
miC=75-us#mm
print ("maximum clearance is,%f mm"%MxC)
print ("minimum clearance is,%f mm"%miC)



