#Variables

IC = IL = 1.2                 #Collector current (in Ampere)
Vout = 7.5                    #Voltage (in volts)
VBE = 0.5                     #Base-emitter voltage (in volts)
beta = 50.0                   #Current gain
VCC = 15.0                    #Supply voltage (in volts) 
IZmin = 10.0 * 10**-3         #Minimum zener current (in Ampere) 

#Calculation

IB = IC/beta                  #Base current (in Ampere)
VZ = Vout + VBE               #Zener diode breakdown voltage (in volts)
VR = VCC - VZ                 #Voltage drop in resistor R (in volts)
IR = IB + IZmin               #Current through R (in AMpere)                   
R = VR/IR                     #Resistance R (in ohm)

#Result

print "Breakdown voltage : ",VZ,"V.\nResistor R : ",round(R),"ohm."

#Variables

Vout = 10                    #Output voltage (in volts)
VBE = 0.4                    #Base-emitter voltage (in volts)
IL = 100.0 * 10**-3          #Load current (in Ampere)
Vinmin = 11.25               #Min. input voltage (in volts)
Vinmax = 13.75               #Max. input voltage (in volts)

#Calculation

VZ = Vout - VBE              #Zener breakdown voltage (in volts)
VRSE = Vinmax - Vout         #Max. voltage drop in series resistor (in volts)
Imax = IL                    #Series resistor current (in Ampere)
RSE = VRSE/Imax              #Series resistor (in ohm)

#Result

print "Breakdown voltage : ",VZ,"V.\nSeries Resistor RSE : ",round(RSE,1),"ohm."

