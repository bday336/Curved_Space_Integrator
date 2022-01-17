from symint_bank import symh2xegeotrans,imph2xegeotrans
from function_bank import hyper2poin2d,hyper2poin3d,formatvec2d,unformatvec2d,formatvec3d,unformatvec3d,boostx2d,boosty2d,rotz2d,boostx3d,boosty3d,boostz3d,rotx3d,roty3d,rotz3d,motionmat2dh,motionmat3dh,py2dhfreetrans,xfunc2dhtrans,yfunc2dhtrans,zfunc2dhtrans,py3dhfreetrans,xfunc3dhtrans,yfunc3dhtrans,zfunc3dhtrans,wfunc3dhtrans,py3dhfreerot,xfunc3dhrot,yfunc3dhrot,zfunc3dhrot,wfunc3dhrot,py3dhgrav,py3dhefreetrans,xfunc3dhetrans,yfunc3dhetrans,zfunc3dhetrans,wfunc3dhetrans,py3dhefreerot,xfunc3dherot,yfunc3dherot,zfunc3dherot,wfunc3dherot,py3dhspring,py3dhespring
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
ai=.0
bi=0.*np.pi/2.
gi=0.
adi=2.11584 #1.49427
bdi=.0
gdi=1 
delT=.01
maxT=1+delT

# ai=.5
# bi=0.*np.pi/2.
# gi=0.
# adi=.0
# bdi=.5
# gdi=.0 
# delT=.01
# maxT=10+delT


#Do the process
abgr = array([ai, bi, gi])
abgt = array([
	arcsinh(sinh(abgr[0])*cos(abgr[1])), 
	arctanh(tanh(abgr[0])*sin(abgr[1])), 
	abgr[2]
	])
posi = array([
	sinh(abgr[0])*cos(abgr[1]), 
	sinh(abgr[0])*sin(abgr[1]), 
	abgr[2],
	cosh(abgr[0])
	])
posguess = posi
abgdotti = array([adi, bdi, gdi])
abgdotri = array([
    (abgdotti[0]*sinh(abgt[0])*cosh(abgt[1]) + abgdotti[1]*cosh(abgt[0])*sinh(abgt[1]))/sinh(abgr[0]),
    (cos(abgr[1])*(abgdotti[0]*sinh(abgt[0])*sinh(abgt[1]) + abgdotti[1]*cosh(abgt[0])*cosh(abgt[1])) - sin(abgr[1])*(abgdotti[0]*cosh(abgt[0])))/sinh(abgr[0]),
    abgdotti[2]
	])
abgdotguess = abgdotti
nump=maxT/delT
timearr=np.arange(0,maxT,delT)

q = 0
testm = []
gx = []
gy = []
gz = []
gw = []
testp = []
ga = []
gb = []
gg = []	

gx=append(gx, posi[0])
gy=append(gy, posi[1])
gz=append(gz, posi[2])
gw=append(gw, posi[3])
ga=append(ga, abgt[0])
gb=append(gb, abgt[1])
gg=append(gg, abgt[2])
testm.append(symh2xegeotrans(posi, posguess, abgdotti, abgdotguess, delT))
testp.append(imph2xegeotrans(abgt, abgt, abgdotti, abgdotguess, delT))		
gx=append(gx, testm[0][0][0])
gy=append(gy, testm[0][0][1])
gz=append(gz, testm[0][0][2])
gw=append(gw, testm[0][0][3])
ga=append(ga, testp[0][0][0])
gb=append(gb, testp[0][0][1])
gg=append(gg, testp[0][0][2])		
q=q+1
while(q < nump-1):
	nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
	nextdotm = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6]])
	nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
	nextdotp = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])				
	testm.append(symh2xegeotrans(nextposm, nextposm, nextdotm, nextdotm, delT))
	testp.append(imph2xegeotrans(nextposp, nextposp, nextdotp, nextdotp, delT))		
	gx=append(gx, testm[q][0][0])
	gy=append(gy, testm[q][0][1])
	gz=append(gz, testm[q][0][2])
	gw=append(gw, testm[q][0][3])
	ga=append(ga, testp[q][0][0])
	gb=append(gb, testp[q][0][1])
	gg=append(gg, testp[q][0][2])				
	q=q+1

gu=[]
gv=[]
gs=[]
gpu=[]
gpv=[]
gps=[]	
for b in range(len(gx)):
    gu=append(gu,gx[b]/(gw[b] + 1.))
    gv=append(gv,gy[b]/(gw[b] + 1.))
    gs=append(gs,gz[b]) 
    gpu=append(gpu,sinh(ga[b])/(cosh(ga[b])*cosh(gb[b]) + 1.))
    gpv=append(gpv,cosh(ga[b])*sinh(gb[b])/(cosh(ga[b])*cosh(gb[b]) + 1.))
    gps=append(gps,gg[b]) 	    	     

#Lie generator trajectories

checkhdata=[]
checkemdata=[]
liemag=array([sqrt(abgdotti[0]**2. + cosh(abgt[0])**2.*abgdotti[1]**2.),0.,0.])
for c in timearr:
	checkhdata.append(hyper2poin2d(unformatvec2d(matmul(rotz2d(abgr[1]),matmul(boostx2d(abgr[0]),matmul(rotz2d(-abgr[1]),matmul(rotz2d(arctan2(abgdotti[1], abgdotti[0])),formatvec2d(array([sinh(liemag[0]*c), cosh(liemag[0]*c)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)])))))))))
	checkemdata.append(unformatvec2d(matmul(rotz2d(abgr[1]),matmul(boostx2d(abgr[0]),matmul(rotz2d(-abgr[1]),matmul(rotz2d(arctan2(abgdotti[1], abgdotti[0])),formatvec2d(array([sinh(liemag[0]*c), cosh(liemag[0]*c)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)]))))))))

	#checkhdata.append(hyper2poin3d(unformatvec3d(matmul(rotz3d(abgr[2]),matmul(roty3d(-abgr[1]),matmul(boostx3d(abgr[0]),matmul(roty3d(abgr[1]),matmul(rotz3d(-abgr[2]),matmul(rotz3d(arctan2(abgdotti[1],abgdotti[0])),matmul(roty3d(-arctan2(abgdotti[2],abgdotti[0])),formatvec3d(array([sinh(liemag[0]*c), cosh(liemag[0]*c)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)*cosh(0.)]))))))))))))


#This formats the data to be compared with the integrator results

checku = []
checkv = []
checks = []
checkx = []
checky = []
checkz = []
checkw = []	
for d in range(len(timearr)):
	checku.append(checkhdata[d][0])
	checkv.append(checkhdata[d][1])
	checks.append(abgt[2]+abgdotti[2]*timearr[d])
	checkx.append(checkemdata[d][0])
	checky.append(checkemdata[d][1])
	checkz.append(abgt[2]+abgdotti[2]*timearr[d])
	checkw.append(checkemdata[d][2])		
   		

spemerr=[]
slemerr=[]
lpemerr=[]		
for j in range(len(checku)):
	spemerr.append(sqrt((gx[j]-sinh(ga[j]))**2. + (gy[j]-cosh(ga[j])*sinh(gb[j]))**2. + (gz[j]-gg[j])**2. + (gw[j]-cosh(ga[j])*cosh(gb[j]))**2.))
	slemerr.append(sqrt((gx[j]-checkx[j])**2. + (gy[j]-checky[j])**2. + (gz[j]-checkz[j])**2.+ (gw[j]-checkw[j])**2.))
	lpemerr.append(sqrt(abs((checkx[j]-sinh(ga[j]))**2. + (checky[j]-cosh(ga[j])*sinh(gb[j]))**2. + (checkz[j]-gg[j])**2. + (checkw[j]-cosh(ga[j])*cosh(gb[j]))**2.)))		

#This is the particle trajectories in the Poincare model
    
#Plot
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121, projection='3d')
# ax1.set_aspect("equal")

#draw sphere
u, v = np.mgrid[-2.:2.+.2:.2, 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
x = np.cos(v)
y = np.sin(v)
z = u
ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
ax1.set_xlim3d(-1,1)
ax1.set_ylim3d(-1,1)
ax1.set_zlim3d(-1,1)

#draw trajectory
ax1.plot3D(gu,gv,gs, label="sym")
ax1.plot3D(gpu,gpv,gps, label="parameter")
ax1.plot3D(checku,checkv,checks, label="lie")
ax1.legend(loc= 'lower left')

ax2=fig.add_subplot(122)
ax2.plot(timearr,spemerr,label="sym/par")
ax2.plot(timearr,slemerr,label="sym/lie")
ax2.plot(timearr,lpemerr,label="lie/par")
# plt.plot(timearr,pprk4emerr,label="prk4/par")
ax2.set_yscale("log",basey=10)
ax2.legend(loc='upper left')		

fig.tight_layout()	

plt.show()	