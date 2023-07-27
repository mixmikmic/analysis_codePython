#Variables

IB1 = 20.0 *10**-6                  #Base current with ac o/p shorted (in Ampere)
IC1 = 1.0 *10**-3                   #Collector current with ac o/p shorted (in Ampere)
VBC1 = 22.0 * 10**-3                #Base-collector voltage with ac o/p shorted (in volts)
VCE1 = 0                            #Collector-emitter voltage wwith ac o/p shorted (in volts)

IB2 = 0                             #Base current with ac i/p open-circuited (in Ampere)
VBE2 = 0.25 *10**-3                 #Base-emitter voltage with ac i/p open-circuited (in volts)
IC2 = 30.0 * 10**-6                 #Collector current with ac i/p open-circuited (in Ampere)              
VCE2 = 1                            #Collector-emitter voltage with ac i/p open-circuited (in volts)

#Calculation

hie = VBC1/IB1                      #hie (in ohm)
hfe = IC1/IB1                       #Current gain in CE
hre = VBE2/VCE2                     #hre 
hoe = IC2/VCE2                      #hoe (in Siemen)

#Result

print "hie : ",hie*10**-3,"kilo-ohm."
print "hfe : ",hfe,"."
print "hre : ",hre,"."
print "hoe : ",hoe * 10**6,"micro-S."

#Variables

hfe = 50.0                    #hfe
hie = 0.83 * 10**3            #hie (in ohm)

#Calculation

hfb =  -hfe/(1 + hfe)         #Current gain
hib = hie/(1 + hfe)           #Input impedance (in ohm)    

#Result

print "hfb : ",round(hfb,2),".\nhib : ",round(hib,2),"ohm."

#Variables

hfe = 100                          #hfe
hre = 0.02 * 10**-2                #hre
hoe = 5 * 10**-6                   #hoe (in Siemens) 
hic = hie = 2600.0                 #hie (in ohm) 

#Calculation

hfc = -(1 + hfe)                   #hfc                   
hrc = 1 - hre                      #hrc
hoc = hoe                          #hoe (in Siemens) 

#Result

print "hic :",hic,"ohm."
print "hfc :",hfc,"."
print "hrc :",round(hrc),"."
print "hoc :",hoc,"S."

#Variables

hie = 2000.0                    #hie (in ohm)
hre = 1.6 * 10**-4              #hre
hfe = 49                        #Current gain 
hoe = 50 * 10**-6               #hoe (in Ampere per volt)
RL = 30.0 * 10**3               #Load resistance (in ohm)
RS = 600.0                      #Source resistance (in ohm)

#Calculation

Ai = - hfe/(1 + hoe*RL)         #Current gain
Rin = hie - hre*hfe/(hoe + 1/RL)#Input resistance (in ohm)
Av = -hfe/((hoe + 1/RL)*Rin)    #Voltage gain 
Avs = Av*Rin/(Rin + RS)         #Overall voltage gain 
Ais = Ai*RS/(Rin + RS)          #Overall current gain
Gout = hoe - hfe*hre/(hie + RS) #Output conductance (in Siemens)
Rout = 1/Gout                   #Output resistance (in ohm)

#Result

print "Current gain :",Ai,"."
print "Input resistance :",Rin,"ohm."
print "Voltage gain :",round(Av,1),"."
print "Overall voltage gain :",round(Avs),"."
print "Overall current gain :",round(Ais,1),"."
print "Output conductance :",Gout,"S."
print "Output resistance :",round(Rout),"ohm."

#Slight variations due to higher precision.

#Variables

hie = 1.1 * 10**3              #hie (in ohm)
hre = 0.25 * 10**-3            #hre
hfe = 50                       #Current gain
hoe = 25.0 * 10**-6            #hoe (in Siemens)
RL = 1.0 * 10**3               #Load resistance (in ohm)
RS = 1.0 * 10**3               #Series resistance (in ohm) 

#Calculation

Ai = - hfe/(1 + hoe*RL)         #Current gain
Rin = hie - hre*hfe/(hoe + 1/RL)#Input resistance (in ohm)
Av = -hfe/((hoe + 1/RL)*Rin)    #Voltage gain 
Avs = Av*Rin/(Rin + RS)         #Overall voltage gain 
Ais = Ai*RS/(Rin + RS)          #Overall current gain

#Result

print "Current gain :",round(Ai,2),"."
print "Input resistance :",round(Rin,1),"ohm."
print "Voltage gain :",round(Av,2),"."
print "Overall voltage gain :",round(Avs,2),"."
print "Overall current gain :",round(Ais,3),"."

#Variables

hie = 1.0 * 10**3            #hie (in ohm)
hfe = 100                    #Current gain
RL = 2.0 * 10**3             #Load resistance (in ohm)
hre = hoe = 0                #hre 

#Calculation

Ai = - hfe/(1 + hoe*RL)         #Current gain
Rin = hie - hre*hfe/(hoe + 1/RL)#Input resistance (in ohm)
Av = -hfe/((hoe + 1/RL)*Rin)    #Voltage gain 

#Result

print "Current gain :",round(Ai,2),"."
print "Input resistance :",round(Rin,1),"ohm."
print "Voltage gain :",round(Av,2),"."

#Variables

RS = 200.0                  #internal resistance (in ohm)
RL = 1200.0                 #Load resistance (in ohm)
hib = 24.0                  #hib (in ohm)
hrb = 4.0 * 10**-4          #hrb
hfb = -0.98                 #hfb
hob = 0.6 * 10**-6          #hob (in Ampere per volt)

#Calculation

Ai = - hfb/(1 + hob*RL)         #Current gain
Rin = hib + hrb*Ai*RL           #Input resistance (in ohm)
Av = Ai*RL/Rin                  #Voltage gain 
Avs = Av*Rin/(Rin + RS)         #Overall voltage gain 
Ais = Ai*RS/(Rin + RS)          #Overall current gain

#Result

print "Current gain :",round(Ai,3),"."
print "Input resistance :",round(Rin,2),"ohm."
print "Voltage gain :",round(Av,2),"."
print "Overall voltage gain :",round(Avs,2),"."
print "Overall current gain :",round(Ais,3),"."

#Variables

IE = 1.2 * 10**-3          #Emitter current (in Ampere)
beta = 120.0               #Current gain
ro = 40.0 * 10**3          #O/p resistance (in ohm)
hre = 0                    #hre  

#Calculation

hfe = beta                 #hfe
hoe = 1/ro                 #hoe (in Siemen)
hie = 25.0*10**-3/IE*beta  #hie (in ohm)
alpha = beta/(1 + beta)    #Current gain in CB
hob = hoe/(1 + beta)       #hob (in Siemen)  
hib = 25 * 10**-3/IE       #hib (in ohm)

#Result

print "hfe :",hfe,"."
print "hoe :",hoe,"S."
print "hie :",hie*10**-3,"kilo-ohm."
print "Current amplification factor :",round(alpha,2),"."
print "hob :",hob,"."
print "hib :",round(hib,2),"ohm."

#Variables

hic = hie = 2.0 * 10**3            #hic (in ohm)
hfe = 100.0                        #Current gain in CE
hre = 2.5 * 10**-4                 #hre
hoe = 25.0 * 10**-6                #hoe (in Ampere per volt)
RS = 1.0 * 10**3                   #Source resistance (in ohm)
RL = 500.0                         #Load resistance (in ohm)

#Calculation

hfc = -(1 + hfe)                   #hfc
hrc = 1 - hre                      #hrc
hoc = hoe                          #hoc (in Siemens)
Ai = -hfc/(1 + hoc*RL)             #Current gain
Rin = hic - hrc*hfc/(hoc + 1/RL)   #Input resistance (in ohm)
Av = -hfc/((hoc + 1/RL)*Rin)       #Voltage gain
Avs = Av*Rin/(Rin + RS)            #Overall voltage gain
Ais = Ai*RS/(Rin + RS)             #Overall current gain
Go = hoc  -(hfc*hrc/(hic + RS))    #O/P conductance (in Siemens)
Ro = 1/Go                          #O/P resistance (in ohm)

#Result

print "Current gain :",round(Ai,2),"."
print "Input resistance :",round(Rin,3),"ohm."
print "Voltage gain :",round(Av,4),"."
print "Overall voltage gain :",round(Avs,4),"."
print "Overall current gain :",round(Ais,3),"."
print "Output resistance :",round(Ro,2),"ohm."
print "Output conductance :",round(Go,4),"Siemen."

#Slight variations due to higher precision.

#Variables

hie = 2.0 * 10**3              #hie (in ohm)
hoe = 25.0 * 10**-6            #hoe (in Siemens)
hfe = 55.0                     #Current gain in CE
Pin = 10.0 * 10**-3             #Output power (in watt)
RB = 80.0 * 10**3              #Base resistance (in ohm)
RC = 10.0 * 10**3              #Collector resitance (in ohm)
RL = 10.0 * 10**3              #Load resistance (in ohm)
RS = 5.0 * 10**3               #Source resistance (in ohm)  

#Calculation

Zb = hie                       #Zb (in ohm)
Zin = RB                       #Impedance (in ohm)
ZS = RS + Zin                  #Imput impedance (in ohm)
Zout = RC/hoe*(1/(RC + 1/hoe)) #Output impedance (in ohm)
Rac = Zout*RL/(Zout + RL)      #AC load resistance (in ohm)
Vout = -34.3*0.29              #Output voltage (in volts)
Pout = Vout**2/RL              #Output power (in watt) 
E = 0.29                       #EMF (in volts)
Ap = Pin/0.29**2*6.95*10**3    #Power gain

#Result

print "Power gain : ",round(Ap),"."
print "EMF E : ",E,"V."

#Variables

hie = 1.0 * 10**3            #hie (in ohm)
hfe = 100.0                  #Current gain   
R1 = 20.0 * 10**3            #Resistance1 (in ohm)
R2 = 10 * 10**3              #Resistance2 (in ohm)
hoe = 25.0 * 10**-6          #hoe (in Siemens)
RC = 2* 10**3                #Collector resistance (in ohm)
RL = 2* 10**3                #Load resistance (in ohm)

#Calculation

Zb = hie                                   #Zb (in ohm)   
Zin = Zb*R1*R2/(Zb*R1 + Zb*R2 + R1*R2)     #Input impedance (in ohm)
Zout = 1/hoe*RC/(RC + 1/hoe)               #Output impedance (in ohm)
Av = -(RC*RL)/(RC + RL)*hfe/hie            #Voltage gain
RB = R1*R2/(R1 + R2)                       #Base resistance (in ohm)
Ai = -hfe*RB*RC/((RC + RL)*(RB + Zb))      #Current gain

#Result

print "Input impedance : ",round(Zin * 10**-3,2),"kilo-ohm."
print "Output impedance : ",round(Zout * 10**-3,1),"kilo-ohm"
print "Current gain : ",round(Ai,1),"."
print "Voltage gain : ",Av,"."

#Variables

hie = 1100.0                          #hie (in ohm)
hre = 0                               #hre
hfe = 50.0                            #Current gain 
hoe = 100.0                           #hoe  
R1 = 100.0 * 10**3                    #Resistance1 (in ohm)
R2 = 10.0 * 10**3                     #Resistance2 (n ohm)
RE = 1.0 * 10**3                      #Emitter resistance (in ohm)
RL = 5.0 * 10**3                      #Load resistance (in ohm) 
RS = 10.0 * 10**3                     #Source resistance (in ohm)  

#Calculation

RB = hie + (1 + hfe)*RE                   #Base resistance (in ohm)
Rin = RB*R1*R2/((RB*R1 + RB*R2 + R1*R2))  #Input resistance (in ohm)
Ai = -hoe                                 #Current gain
Av = -hoe*RL/(hie + (1 + hfe)*RE)         #Voltage gain
Avs = Av * Rin/(Rin + RS)                 #Overall voltage gain

#Result

print "Ai : ",Ai,"."
print "Av : ",round(Av,3),"."
print "Avs : ",round(Avs,2),"."
print "Rin : ",round(Rin*10**-3,2),"kilo-ohm."

#Slight variation due to higher precision.

#Variables

RE = 100.0                #Emitter resistance (in ohm) 
RC = 1.0 * 10**3          #Collector resistance (in ohm)
VBE = 0.7                 #Base-emitter voltage (in volts)
RB = 420.0 * 10**3        #Base resistance (in ohm)
beta = 100                #Current gain in CE
VCC = 5.0                 #Collector supply voltage (in volts)  

#Calculation

IB = (VCC -VBE)/(RB + (beta + 1)*RE)      #Base current (in Ampere)
ICQ = beta * IB                           #Q-point collector current (in Ampere)
IE = (beta + 1)*IB                        #Emitter current (in Ampere)
r1e = 25.0*10**-3/IE                      #Resistance (in ohm)  
Rin = RB*(beta*r1e)/(RB + beta*r1e)       #Input resistance (in ohm)
Rout = RC                                 #Output resistance (in ohm)
Av = -ICQ/IB*Rout/Rin                     #Small signal voltage gain 
swing = VCC/(RC + RE)                     #Max. possible swing (in Ampere) 

#Result

print "Quiescent collector current : ",round(ICQ*10**3,3),"mA."
print "Small signal voltage gain : ",round(Av,2),"."
print "Maximum possible swing of collector current : ",round(swing*10**3,2),"mA."

#Slight variation due to high precision.

#Variables

beta = hfe = 130                  #Current gain in CE
R1 = 510.0 * 10**3                #Resistance1 (in ohm)
R2 = 510.0 * 10**3                #Resistance2 (n ohm)
RE = 7.5 * 10**3                  #Emitter resistance (in ohm)
RC = 9.1 * 10**3                  #Collector resistance (in ohm)
VCC = 18.0                        #Collector supply voltage (in volts)
VBE = 0                           #Base-Emitter voltage (in volts)
hie = 1.0 * 10**3                 #hie (in ohm)

#Calculation

Rth = R1*R2/(R1 + R2)             #Thevenin's eq. resistance (in ohm)
Vth = VCC * R2/(R1 + R2)          #Thevenin's eq. voltage (in volts)
IB = (Vth - VBE)/(Rth + (beta + 1)*RE)           #Base current (in Ampere)
IC = beta*IB                      #Collector current (in Ampere)
ICQ = IC                          #Q-point IC (in Ampere)
IE = (beta + 1)*IB                #Emitter current (in Ampere)
VCEQ = VCC - ICQ*RC - IE*RE       #Q-point VCE (in Ampere)    

IB1 = (VCC - VBE)/(R1 + (beta + 1)*RE)           #Base current1 (in Ampere)
IC1 = beta*IB1                    #Collector current1 (in Ampere) 
ICQ1 = IC1                        #Q-point IC (in Ampere)
IE1 = (beta + 1)*IB1              #Emitter current1 (in Ampere)
VCEQ1 = VCC - ICQ1*RC - IE1*RE    #Q-point VCE (in Ampere)    

Rin = (R1*R2*hie)/(R1*R2 + hie*R2 + hie*R1)      #Input resistance (in ohm)
Av = -50/hie*RC                  #Voltage gain        

#Result
print IB1,IC1,ICQ1,IE1
print "ICQ : ",round(ICQ*10**3,3),"mA and VCEQ : ",round(VCEQ,3),"V."
print "VCE when R2 is open circuited : ",round(VCEQ1,3),"V."
print "AV : ",round(Av,3),"."
print "Rin : ",round(Rin*10**-3,2),"kilo-ohm."

#Mistake in book for the value of hfe in calculation of Av.

#Variables

hfe = 110                     #Current gain in CE
hie = 1.6 * 10**3             #hie (in ohm)
hre = 2 * 10**-4              #hre
hoe = 20.0 * 10**-6           #hoe (in Ampere per volt)      
RB = 470.0 * 10**3            #Base resistance (in ohm)
RC = 4.7 * 10**3              #Collector resistance (in ohm)

#Calculation

Zin = RB*hie/(RB + hie)       #Input impedance (in ohm)
Zout = RC*1/hoe/(RC + 1/hoe)  #Output impedance (in ohm)
Av = -RC*hfe/hie              #Voltage gain  

#Result

print "Zin : ",round(Zin*10**-3,3)," kilo-ohm."
print "Zout : ",round(Zout*10**-3,3)," kilo-ohm."
print "Av : ",round(Av,3),"."

#Variables

hib = 25.0                            #hie (in ohm)
hfb = -0.98                           #Current gain in CB 
hob = 0.5 * 10**-6                    #hob (in Siemens)  
R1 = 20.0 * 10**3                     #Resistance1 (in ohm)
R2 = 5.0 * 10**3                      #Resistance2 (n ohm)
RE = 4.0 * 10**3                      #Emitter resistance (in ohm)
RL = 6.0 * 10**3                      #Load resistance (in ohm) 
RC = 8.0 * 10**3                      #Collector resistance (in ohm)  

#Calculation

Zin = hib*RE/(hib + RE)               #Input impedance (in ohm)
Zout = RC*1/hob/(RC + 1/hob)          #Output impedance (in ohm)
Av = -(RC*RL)/(RC+RL)*hfb/hib         #Voltage gain  

#Result

print "Zin : ",round(Zin,2)," ohm."
print "Zout : ",round(Zout*10**-3,2)," kilo-ohm."
print "Av : ",round(Av,3),"."

#Slight variation due to higher precision.

#Variables

hie = 2000.0                          #hie (in ohm)
hfe = 100.0                           #Current gain  
R1 = 10.0 * 10**3                     #Resistance1 (in ohm)
R2 = 10.0 * 10**3                     #Resistance2 (n ohm)
RE = 5.0 * 10**3                      #Emitter resistance (in ohm)
RL = 5.0 * 10**3                      #Load resistance (in ohm) 
RS = 1.0 * 10**3                      #Source resistance (in ohm)  

#Calculation

hic = hie                             #hic
hfc = -(1 + hfe)                      #hfc
Zb = hic - hfc*(RE*RL)/(RE + RL)      #ZB (in ohm)
Zin = Zb*R1*R2/(Zb*R1 + R1*R2 + Zb*R2)#Input impedance (in ohm)
Ze = -(hic + (R1*R2*RS/(R1*R2 + R2*RS + R1*RS)))/hfc    #Ze (in ohm)
Zout = Ze*RE/(Ze + RE)                #Output impedance (in ohm)  
Av = 1                                #Coltage gain
RB = R1*R2/(R1 + R2)                  #Base resistance (in ohm)
Ai = -hfc                             #Current gain
Ap = Ai                               #Power gain

#Result

print "Input impedance : ",round(Zin * 10**-3,1),"kilo-ohm."
print "Outpur impedance : ",round(Zout),"ohm."
print "Voltage gain : ",Av,"."
print "Current gain : ",Ai,"."
print "Power gain : ",Ap,"."

#Variables

RL = 5.0 * 10**3                      #Load resistance (in ohm) 
RS = 0.5 * 10**3                      #Source resistance (in ohm)  
hie = 1000.0                          #hie (in ohm)
hfe = 50.0                            #Current gain  
hoe = 25.0 * 10**-6                   #hor (in Siemens)  

#Calculation

hic = hie                             #hie (in ohm)
hrc = 1                               #hrc
hfc = -(1 + hfe)                      #hfc 
hoc = hoe                             #hoe (in Siemens)
Ai = -hfc/(1 + hoc*RL)                #Current gain
Ri = hic - hrc*hfc/(hoc + 1/RL)       #Input resistance (in ohm)
Av = Ai*RL/Ri                         #Voltage gain

#Result

print "Input impedance : ",round(Ri * 10**-3,3),"kilo-ohm."
print "Voltage gain : ",round(Av,4),"."
print "Current gain : ",round(Ai,2),"."

#Variables

VCC = 15.0               #Collector supply voltage (in volts)
RB = 100.0 * 10**3       #Base resistance (in ohm)
RE = 1.0 * 10**3         #Emitter resistance (in ohm)
hie = 1100.0             #hie (in ohm)
hfe = 50                 #hfe

#Calculation

hic = hie                #hic (in ohm)
hfc = -(1 + hfe)         #hfc
Zin = (hic - hfc*RE)*RB/((hic - hfc*RE) + RB)     #Input impedance (in ohm)

Zout = RE*(-hic/hfc)/(RE - hic/hfc)               #Output impedance (in ohm)
Av = -hfc*RE/(hic - hfc*RE)                       #Voltage gain
Ai = Av*Zin/RE                                    #Current gain 

#Result

print "Input impedance : ",round(Zin * 10**-3,3),"kilo-ohm."
print "Outpur impedance : ",round(Zout,1),"ohm."
print "Voltage gain : ",round(Av,4),"."
print "Current gain : ",round(Ai,2),"."

#Calculation mistake in the value of Zout in the book.

#Variables

hre = hoe = 0                   #hre
hie = 1.0 * 10**3               #hie (in ohm)
hfe = 100.0                     #hfe
VCC = 5.0                       #Collector supply voltage (in volts) 
R1 = 2.2 * 10**3                #Resistance1 (in ohm)
R2 = 2.2 * 10**3                #Resistance2 (in ohm)
RE = 1.0 * 10**3                #Emitter resistance (in ohm)

#Calculation

hic = hie                       #hic (in ohm)
hfc = -(1 + hfe)                #hfc  
hrc = 1 - hre                   #hrc
hoc = hoe = 0                   #hoc
Zin = (hic - hfc*RE)*R1*R2/(((hic - hfc*RE)*(R1+R2))+R1*R2)     #Input impedance (in ohm)
Av = -hfc*RE/(hic - hfc*RE)     #Voltage gain 

#Result

print "Zin : ",round(Zin*10**-3,4),"kilo-ohm."
print "Av : ",round(Av,2),"."

