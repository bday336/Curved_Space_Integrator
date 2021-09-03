from symint_bank import sympbhgeo,imppbhgeo
from function_bank import hyper2poin2d,hyper2poin3d,formatvec2d,unformatvec2d,formatvec3d,unformatvec3d,boostx2d,boosty2d,rotz2d,boostx3d,boosty3d,boostz3d,rotx3d,roty3d,rotz3d,motionmat2dh,motionmat3dh,py2dhfreetrans,xfunc2dhtrans,yfunc2dhtrans,zfunc2dhtrans,py3dhfreetrans,xfunc3dhtrans,yfunc3dhtrans,zfunc3dhtrans,wfunc3dhtrans,py3dhfreerot,xfunc3dhrot,yfunc3dhrot,zfunc3dhrot,wfunc3dhrot,py3dhgrav,py3dhefreetrans,xfunc3dhetrans,yfunc3dhetrans,zfunc3dhetrans,wfunc3dhetrans,py3dhefreerot,xfunc3dherot,yfunc3dherot,zfunc3dherot,wfunc3dherot,py3dhspring,py3dhespring
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
ai=.25
bi=np.pi/2.
gi=0.
adi=-.5
bdi=.0
gdi=.2 
delT=.01
maxT=10+delT

#Do the process
abg = array([ai, bi, gi])
posi = array([exp(abg[0])*sin(abg[1])*cos(abg[2]), exp(abg[0])*sin(abg[1])*sin(abg[2]), exp(abg[0])*cos(abg[1]), abg[0]])
posguess = posi
abgdoti = array([adi, bdi, gdi])
abgdotguess = abgdoti
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
ga=append(ga, abg[0])
gb=append(gb, abg[1])
gg=append(gg, abg[2])
testm.append(sympbhgeo(posi, posguess, abgdoti, abgdotguess, delT))
testp.append(imppbhgeo(abg, abg, abgdoti, abgdotguess, delT))		
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
	testm.append(sympbhgeo(nextposm, nextposm, nextdotm, nextdotm, delT))
	testp.append(imppbhgeo(nextposp, nextposp, nextdotp, nextdotp, delT))		
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
    gu=append(gu,gx[b])
    gv=append(gv,gy[b])
    gs=append(gs,gz[b]) 
    gpu=append(gpu,exp(ga[b])*sin(gb[b])*cos(gg[b]))
    gpv=append(gpv,exp(ga[b])*sin(gb[b])*sin(gg[b]))
    gps=append(gps,exp(ga[b])*cos(gb[b])) 	    	     		   		

spemerr=[]	
for j in range(len(gu)):
	spemerr.append(sqrt((gx[j]-exp(ga[j])*sin(gb[j])*cos(gg[j]))**2. + (gy[j]-exp(ga[j])*sin(gb[j])*sin(gg[j]))**2. + (gz[j]-exp(ga[j])*cos(gb[j]))**2. + (gw[j]-ga[j])**2.))		

#This is the particle trajectories in the Poincare model
    
#Plot
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121, projection='3d')
#ax1.set_aspect("equal")

#draw sphere
# u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
# x = np.sin(u)*np.cos(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(u)
# ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
ax1.set_xlim3d(-4,4)
ax1.set_ylim3d(-4,4)
ax1.set_zlim3d(-4,4)

#draw trajectory
ax1.plot3D(gu,gv,gs, label="sym")
ax1.plot3D(gpu,gpv,gps, label="parameter")
ax1.legend(loc= 'lower left')

ax2=fig.add_subplot(122)
ax2.plot(timearr,spemerr,label="sym/par")
ax2.set_yscale("log",basey=10)	
ax2.legend(loc='upper left')		

fig.tight_layout()	

plt.show()	





