from symint_bank import symh2georot,imph2georot
from function_bank import hyper2poin2d,hyper2poin3d,formatvec2d,unformatvec2d,formatvec3d,unformatvec3d,boostx2d,boosty2d,rotz2d,boostx3d,boosty3d,boostz3d,rotx3d,roty3d,rotz3d,motionmat2dh,motionmat3dh,py2dhfreetrans,xfunc2dhtrans,yfunc2dhtrans,zfunc2dhtrans,py3dhfreetrans,xfunc3dhtrans,yfunc3dhtrans,zfunc3dhtrans,wfunc3dhtrans,py3dhfreerot,xfunc3dhrot,yfunc3dhrot,zfunc3dhrot,wfunc3dhrot,py3dhgrav,py3dhefreetrans,xfunc3dhetrans,yfunc3dhetrans,zfunc3dhetrans,wfunc3dhetrans,py3dhefreerot,xfunc3dherot,yfunc3dherot,zfunc3dherot,wfunc3dherot,py3dhspring,py3dhespring
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
ai=.5
bi=.0*np.pi/2.
adi=.0
bdi=.5
delT=.01
maxT=10+delT

# Geodesic Trajectories

# Do the process for both parameterization at the same time to see how they compare
# Position is given in rotational parameterization
abri = array([ai, bi])
# Position in translational parameterization
abti = array([arcsinh(sinh(abri[0])*cos(abri[1])), arctanh(tanh(abri[0])*sin(abri[1]))])
posi = array([sinh(abri[0])*cos(abri[1]), sinh(abri[0])*sin(abri[1]), cosh(abri[0])])
posguess = posi
# Velocity given in translational parameterization
abdotti = array([adi, bdi])
# Velocity in rotational parameterization
abdotri = array([(abdotti[0]*sinh(abti[0])*cosh(abti[1]) + abdotti[1]*cosh(abti[0])*sinh(abti[1]))/sinh(abri[0]), (cos(abri[1])*(abdotti[0]*sinh(abti[0])*sinh(abti[1]) + abdotti[1]*cosh(abti[0])*cosh(abti[1])) - sin(abri[1])*(abdotti[0]*cosh(abti[0])))/sinh(abri[0])])
abdottguess = abdotti
abdotrguess = abdotri
nump=maxT/delT
timearr=np.arange(0,maxT,delT)

q = 0
testmr = []
gxr = []
gyr = []
gzr = []
testpr = []
gar = []
gbr = []	

gxr=append(gxr, posi[0])
gyr=append(gyr, posi[1])
gzr=append(gzr, posi[2])
gar=append(gar, abri[0])
gbr=append(gbr, abri[1])

testmr.append(symh2georot(posi, posguess, abdotri, abdotrguess, delT))
testpr.append(imph2georot(abri, abri, abdotri, abdotrguess, delT))		

gxr=append(gxr, testmr[0][0][0])
gyr=append(gyr, testmr[0][0][1])
gzr=append(gzr, testmr[0][0][2])
gar=append(gar, testpr[0][0][0])
gbr=append(gbr, testpr[0][0][1])

q=q+1
while(q < nump-1):
	nextposmr = array([testmr[q - 1][0][0], testmr[q - 1][0][1], testmr[q - 1][0][2]])
	nextdotmr = array([testmr[q - 1][0][3], testmr[q - 1][0][4]])
	nextpospr = array([testpr[q - 1][0][0], testpr[q - 1][0][1]])
	nextdotpr = array([testpr[q - 1][0][2], testpr[q - 1][0][3]])	

	testmr.append(symh2georot(nextposmr, nextposmr, nextdotmr, nextdotmr, delT))
	testpr.append(imph2georot(nextpospr, nextpospr, nextdotpr, nextdotpr, delT))
	
	gxr=append(gxr, testmr[q][0][0])
	gyr=append(gyr, testmr[q][0][1])
	gzr=append(gzr, testmr[q][0][2])
	gar=append(gar, testpr[q][0][0])
	gbr=append(gbr, testpr[q][0][1])

	q=q+1

gur=[]
gvr=[]
gpur=[]
gpvr=[]

for b in range(len(gxr)):
    gur=append(gur,gxr[b]/(gzr[b] + 1.))
    gvr=append(gvr,gyr[b]/(gzr[b] + 1.)) 
    gpur=append(gpur,sinh(gar[b])*cos(gbr[b])/(cosh(gar[b]) + 1.))
    gpvr=append(gpvr,sinh(gar[b])*sin(gbr[b])/(cosh(gar[b]) + 1.)) 	    	     

#Lie generator trajectories

checkhdata=[]
checkemdata=[]
liemag=array([sqrt(abdotti[0]**2. + cosh(abti[0])**2.*abdotti[1]**2.),0])
for c in timearr:

	checkhdata.append(hyper2poin2d(unformatvec2d(matmul(rotz2d(abri[1]),matmul(boostx2d(abri[0]),matmul(rotz2d(-abri[1]),matmul(rotz2d(arctan2(abdotti[1], abdotti[0])),formatvec2d(array([sinh(liemag[0]*c), cosh(liemag[0]*c)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)])))))))))
	checkemdata.append(unformatvec2d(matmul(rotz2d(abri[1]),matmul(boostx2d(abri[0]),matmul(rotz2d(-abri[1]),matmul(rotz2d(arctan2(abdotti[1], abdotti[0])),formatvec2d(array([sinh(liemag[0]*c), cosh(liemag[0]*c)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)]))))))))


#This formats the data to be compared with the integrator results

checku = []
checkv = []
checkx = []
checky = []
checkz = []	
for d in range(len(timearr)):
	checku.append(checkhdata[d][0])
	checkv.append(checkhdata[d][1])
	checkx.append(checkemdata[d][0])
	checky.append(checkemdata[d][1])
	checkz.append(checkemdata[d][2])		

#This is for the horizon circle.
# Theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# Compute x1 and x2 for horizon
xc = np.cos(theta)
yc = np.sin(theta)    		

spremerr=[]
slremerr=[]
lpremerr=[]

# Using the Euclidean distance function to determine error in embedding space
for j in range(len(checku)):
	spremerr.append(sqrt((gxr[j]-sinh(gar[j])*cos(gbr[j]))**2. + (gyr[j]-sinh(gar[j])*sin(gbr[j]))**2. + (gzr[j]-cosh(gar[j]))**2.))
	slremerr.append(sqrt((gxr[j]-checkx[j])**2. + (gyr[j]-checky[j])**2. + (gzr[j]-checkz[j])**2.))
	lpremerr.append(sqrt(abs((checkx[j]-sinh(gar[j])*cos(gbr[j]))**2. + (checky[j]-sinh(gar[j])*sin(gbr[j]))**2. + (checkz[j]-cosh(gar[j]))**2.)))	

#This is the particle trajectories in the Poincare model
    
# #Plot
fig , ((ax1,ax2)) = plt.subplots(1,2,figsize=(8,4))

ax1.plot(xc,yc)
ax1.plot(gur,gvr,label="sym R")
ax1.plot(gpur,gpvr,label="para R")
ax1.plot(checku,checkv,label="lie")
ax1.legend(loc='lower left')

ax2.plot(timearr,spremerr,label="sym/par R")
ax2.plot(timearr,slremerr,label="sym/lie R")
ax2.plot(timearr,lpremerr,label="lie/par R")
ax2.set_yscale("log",basey=10)	
ax2.legend(loc='lower right')	

fig.tight_layout()	

plt.show()





