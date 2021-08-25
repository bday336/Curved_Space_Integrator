from symint_bank import symh2geotrans,imph2geotrans,imprk4h2geotrans
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
testmt = []
gxt = []
gyt = []
gzt = []
testpt = []
gat = []
gbt = []

gxt=append(gxt, posi[0])
gyt=append(gyt, posi[1])
gzt=append(gzt, posi[2])
gat=append(gat, abti[0])
gbt=append(gbt, abti[1])

testmt.append(symh2geotrans(posi, posguess, abdotti, abdottguess, delT))
testpt.append(imph2geotrans(abti, abti, abdotti, abdottguess, delT))	
	
gxt=append(gxt, testmt[0][0][0])
gyt=append(gyt, testmt[0][0][1])
gzt=append(gzt, testmt[0][0][2])
gat=append(gat, testpt[0][0][0])
gbt=append(gbt, testpt[0][0][1])

q=q+1
while(q < nump-1):
	nextposmt = array([testmt[q - 1][0][0], testmt[q - 1][0][1], testmt[q - 1][0][2]])
	nextdotmt = array([testmt[q - 1][0][3], testmt[q - 1][0][4]])
	nextpospt = array([testpt[q - 1][0][0], testpt[q - 1][0][1]])
	nextdotpt = array([testpt[q - 1][0][2], testpt[q - 1][0][3]])

	testmt.append(symh2geotrans(nextposmt, nextposmt, nextdotmt, nextdotmt, delT))
	testpt.append(imph2geotrans(nextpospt, nextpospt, nextdotpt, nextdotpt, delT))
	
	gxt=append(gxt, testmt[q][0][0])
	gyt=append(gyt, testmt[q][0][1])
	gzt=append(gzt, testmt[q][0][2])
	gat=append(gat, testpt[q][0][0])
	gbt=append(gbt, testpt[q][0][1])

	q=q+1

gut=[]
gvt=[]
gput=[]
gpvt=[]

for b in range(len(gxt)):
    gut=append(gut,gxt[b]/(gzt[b] + 1.))
    gvt=append(gvt,gyt[b]/(gzt[b] + 1.)) 
    gput=append(gput,sinh(gat[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.))
    gpvt=append(gpvt,cosh(gat[b])*sinh(gbt[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.))	    	     

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

sptemerr=[]
sltemerr=[]
lptemerr=[]

# Using the Euclidean distance function to determine error in embedding space
for j in range(len(checku)):
	sptemerr.append(sqrt((gxt[j]-sinh(gat[j]))**2. + (gyt[j]-cosh(gat[j])*sinh(gbt[j]))**2. + (gzt[j]-cosh(gat[j])*cosh(gbt[j]))**2.))
	sltemerr.append(sqrt((gxt[j]-checkx[j])**2. + (gyt[j]-checky[j])**2. + (gzt[j]-checkz[j])**2.))
	lptemerr.append(sqrt(abs((checkx[j]-sinh(gat[j]))**2. + (checky[j]-cosh(gat[j])*sinh(gbt[j]))**2. + (checkz[j]-cosh(gat[j])*cosh(gbt[j]))**2.)))

# Checking the late stage error
oerr=[]
for k in range(len(timearr)):
	oerr.append(10.**(-16.)*np.exp(.5*k))


#This is the particle trajectories in the Poincare model
    
# #Plot
fig , ((ax1,ax2)) = plt.subplots(1,2,figsize=(8,4))

ax1.plot(xc,yc)
ax1.plot(gut,gvt,label="sym T")
ax1.plot(gput,gpvt,label="para T")
ax1.plot(checku,checkv,label="lie")
ax1.legend(loc='lower left')

ax2.plot(timearr,sptemerr,label="sym/par T")
ax2.plot(timearr,sltemerr,label="sym/lie T")
ax2.plot(timearr,lptemerr,label="lie/par T")
# ax2.plot(timearr,oerr,label="orth")
ax2.set_yscale("log",basey=10)	
ax2.legend(loc='lower right')	

fig.tight_layout()	

plt.show()





