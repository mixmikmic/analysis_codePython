#Find specific resistance

#variable declaration
L =12  #meter
A=0.01*10**-4   #m**2
R=0.2  #ohm

#calculation
p=R*A/L   #specific resistance

#result
print "specific resistance = " , p, "ohm-metre"

#Calculate resistance

#variable declaration
a0=0.0043 
t1=27 #degree celsius
t2=40
R1=1.5 #ohm

#calculation
R2=R1*(1+a0*t2)/(1+a0*t1) #ohm

#result
print "The resistance of armature winding at 40 degree celcius =" , round(R2,3) ,"ohm"

#Find resistance , current and voltage

#variable decdlaration
R1=5 #ohm   
R2=10
R3=15
V=120 #volt

#calculation
R=R1+R2+R3 #ohm
I=V/R  # ampere
V1=I*R1  #volt
V2=I*R2
V3=I*R3

#result
print "Resistance = " , R , "ohm"
print "cureent = " , I , "amperes"
print "Voltage V1 = " , V1 , "volts"
print "Voltage V2 = " , V2 , "volts"
print "Voltage V3 = " , V3 , "volts"

#Find equivalent resistance

#varaiable declaration
Rab =(2.0*4.0)/(2+4) #ohms
Rbc =(6.0*8.0)/(6+8)

#calculation
Rac = Rab+Rbc #ohms

#result
print "resistance across AC = " , round(Rac,2) , "ohms"

#Find effective resistance

#variable declaration
Rab=4 #ohm
Rbc=(12.0*8.0)/(12+8)
Rcd=(3.0*6.0)/(3+6)

#calculation
Rad=Rab+Rbc+Rcd #ohm

#result
print "resistance across AC = " , Rad, "ohms"

#Find resistance

#variable declaration
R1=8 #ohms
R=6
#calculations
R2 = 48/2 # R = R1*R2/(R1+R2)

#result
print " Resistance R2 = " , R2, "ohms"

#Find values of current

#variable declaration
I=12.0 #ampere
R1=6.0 #ohms
R2=8.0

#calculations
I1=I*R2/(R1+R2) #amperes
I2=I*R1/(R1+R2)

#result
print "I1= " , round(I1,3) , "amperes" 
print "I2= " , round(I2,2) , "amperes"

#round off error in book

#Find current values

#variable declaration
R1=0.02 #ohms
R2=0.03
I = 10 #amperes

#Calculations
I1=(I*R2)/(R1+R2)
I2=(I*R1)/(R1+R2)

#result
print " I1= " , I1 , "amperes "
print " I2= " , I2 , "amperes " 

#Calculate resistance of each coil

#variable declaration
V=200.0 #volts
I=25.0 #amperes
P1=1500.0 #watts

#calculations 
R1=(V*V)/P1 #ohms
R=V/I       #total resistance
R2=R*R1/(R1-R) #ohms

#result
print "R1 = " ,round(R1,2) , "ohms" 
print "R2 = " , round(R2,2) , "ohms" 

#Calculate power dissipaed in each coil

#variable declaration
V=100.0 #volts 
P=1500.0 #watts

#calculations
R=(V**2/P)/2 #ohms
Ra=R
Rb=R
Rc=R
R1=((Ra*Rc)/(Ra+Rc))+Rb
I=V/R1 #amperes
I1=(I*Ra)/(Ra+Rc)
I2=(I*Ra)/(Ra+Rc)
Pb=I*I*Ra #watts
Pa=I1*I1*Rb
Pc=I2*I2*Rc

#result
print "power dissipated in coil  Pa = " , round(Pa,2) , "watts"
print "power dissipated in coil  Pb = " , round(Pb,2), "watts"
print "power dissipated in coil  Pc = " , round(Pc,2) , "watts"

#Round off error in book 

#Calculate amount of bill

#variable declaration
L=3600  #six lamp 1000 watt each for six days
H=3000  # heater
M=735.5 # single phase motor
F=2400  #four fans 75W
C=0.9 #cost of energy

#Calculations
T=L+H+M+F #total energy consumed in watt 
TE=T*30/1000
B=TE*C #Bill amount in Rs

#result
print "Bill amount = Rs " , round(B) 

#Calculate resistance value

#variable declaration
Rry=4.0 #ohm
Ryb=1.0
Rbr=5.0

#calculation
Rr=(Rbr*Rry)/(Rry+Rbr+Ryb)
Ry=(Rry*Ryb)/(Rry+Rbr+Ryb)
Rb=(Rbr*Ryb)/(Rry+Rbr+Ryb)

#result
print "Rr = " , Rr , "ohms"
print "Ry = " , Ry , "ohms"
print "Rb= " , Rb , "ohms"

#Value of Rr in book is printed wrong 

#Calculate resistance

#variable declaration
Rr=2.0 #ohms
Ry=0.67
Rb=1.0

#calculations
Rry=(Rr*Ry)+(Ry*Rb)+(Rb*Rr)/Rb
Ryb=((Rr*Ry)+(Ry*Rb)+(Rb*Rr))/Rr
Rbr=((Rr*Ry)+(Ry*Rb)+(Rb*Rr))/Ry

#result
print "Rry = " , round(Rry) , "ohms"
print "Ryb = " , round(Ryb) , "ohms"
print "Rbr = " , round(Rbr) , "ohms"



