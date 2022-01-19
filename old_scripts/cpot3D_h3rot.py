from symint_bank import symh3cprot,imph3cprot
from function_bank import hyper2poin2d,hyper2poin3d,formatvec2d,unformatvec2d,formatvec3d,unformatvec3d,boostx2d,boosty2d,rotz2d,boostx3d,boosty3d,boostz3d,rotx3d,roty3d,rotz3d,motionmat2dh,motionmat3dh,py2dhfreetrans,xfunc2dhtrans,yfunc2dhtrans,zfunc2dhtrans,py3dhfreetrans,xfunc3dhtrans,yfunc3dhtrans,zfunc3dhtrans,wfunc3dhtrans,py3dhfreerot,xfunc3dhrot,yfunc3dhrot,zfunc3dhrot,wfunc3dhrot,py3dhgrav,py3dhefreetrans,xfunc3dhetrans,yfunc3dhetrans,zfunc3dhetrans,wfunc3dhetrans,py3dhefreerot,xfunc3dherot,yfunc3dherot,zfunc3dherot,wfunc3dherot,py3dhspring,py3dhespring
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
ai=1.
bi=np.pi/2.
gi=0.
adi=.0
bdi=.0
gdi=2. 
delT=.01
maxT=10+delT
massvec=array([10.,1.])


#Do the process
abgr = array([ai, bi, gi])
abgt = array([
	arcsinh(sinh(abgr[0])*sin(abgr[1])*cos(abgr[2])), 
	arcsinh((sinh(abgr[0])*sin(abgr[1])*sin(abgr[2]))/cosh(arcsinh(sinh(abgr[0])*sin(abgr[1])*cos(abgr[2])))), 
	arctanh(tanh(abgr[0])*cos(abgr[1]))
	])
posi = array([
	sinh(abgr[0])*sin(abgr[1])*cos(abgr[2]), 
	sinh(abgr[0])*sin(abgr[1])*sin(abgr[2]), 
	sinh(abgr[0])*cos(abgr[1]), 
	cosh(abgr[0])
	])
posguess = posi
abgdotti = array([adi, bdi, gdi])
abgdotri = array([
	(abgdotti[0]*sinh(abgt[0])*cosh(abgt[1])*cosh(abgt[2]) + abgdotti[1]*cosh(abgt[0])*sinh(abgt[1])*cosh(abgt[2]) + abgdotti[2]*cosh(abgt[0])*cosh(abgt[1])*sinh(abgt[2]))/sinh(abgr[0]), 
	((abgdotti[0]*sinh(abgt[0])*cosh(abgt[1])*cosh(abgt[2]) + abgdotti[1]*cosh(abgt[0])*sinh(abgt[1])*cosh(abgt[2]) + abgdotti[2]*cosh(abgt[0])*cosh(abgt[1])*sinh(abgt[2]))*cosh(abgr[0])*cos(abgr[1]) - (abgdotti[0]*sinh(abgt[0])*cosh(abgt[1])*sinh(abgt[2]) + abgdotti[1]*cosh(abgt[0])*sinh(abgt[1])*sinh(abgt[2]) + abgdotti[2]*cosh(abgt[0])*cosh(abgt[1])*cosh(abgt[2]))*sinh(abgr[0]))/(sinh(abgr[0])*sinh(abgr[0])*sin(abgr[1])), 
	((abgdotti[0]*sinh(abgt[0])*sinh(abgt[1]) + abgdotti[1]*cosh(abgt[0])*cosh(abgt[1]))*cos(abgr[2]) - abgdotti[0]*cosh(abgt[0])*sin(abgr[2]))/(sinh(abgr[0])*sin(abgr[1]))
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
ga=append(ga, abgr[0])
gb=append(gb, abgr[1])
gg=append(gg, abgr[2])
testm.append(symh3cprot(posi, posguess, abgdotti, abgdotguess, delT, massvec[0], massvec[1]))
testp.append(imph3cprot(abgr, abgr, abgdotti, abgdotguess, delT, massvec[0], massvec[1]))		
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
	testm.append(symh3cprot(nextposm, nextposm, nextdotm, nextdotm, delT, massvec[0], massvec[1]))
	testp.append(imph3cprot(nextposp, nextposp, nextdotp, nextdotp, delT, massvec[0], massvec[1]))		
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
g2u=[]
g2v=[]
g2pu=[]
g2pv=[]
for b in range(len(gx)):
    gu=append(gu,gx[b]/(gw[b] + 1.))
    gv=append(gv,gy[b]/(gw[b] + 1.))
    gs=append(gs,gz[b]/(gw[b] + 1.)) 
    gpu=append(gpu,sinh(ga[b])*sin(gb[b])*cos(gg[b])/(cosh(ga[b]) + 1.))
    gpv=append(gpv,sinh(ga[b])*sin(gb[b])*sin(gg[b])/(cosh(ga[b]) + 1.))
    gps=append(gps,sinh(ga[b])*cos(gb[b])/(cosh(ga[b]) + 1.))  
    g2u=append(g2u,gx[b]/(gw[b] + 1.))
    g2v=append(g2v,gy[b]/(gw[b] + 1.))
    g2pu=append(g2pu,sinh(ga[b])*sin(gb[b])*cos(gg[b])/(cosh(ga[b]) + 1.))
    g2pv=append(g2pv,sinh(ga[b])*sin(gb[b])*sin(gg[b])/(cosh(ga[b]) + 1.))  

#This is for the horizon circle.
# Theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# Compute x1 and x2 for horizon
xc = np.cos(theta)
yc = np.sin(theta) 	     

#This formats the data to be compared with the integrator results	
   		

spemerr=[]	
for j in range(len(gx)):
	spemerr.append(sqrt((gx[j]-sinh(ga[j])*sin(gb[j])*cos(gg[j]))**2. + (gy[j]-sinh(ga[j])*sin(gb[j])*sin(gg[j]))**2. + (gz[j]-sinh(ga[j])*cos(gb[j]))**2. + (gw[j]-cosh(ga[j]))**2.))
	

#This is the particle trajectories in the Poincare model
    
#Plot
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131, projection='3d')
# ax1.set_aspect("equal")

#draw sphere
u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
x = np.sin(u)*np.cos(v)
y = np.sin(u)*np.sin(v)
z = np.cos(u)
ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
ax1.set_xlim3d(-1,1)
ax1.set_ylim3d(-1,1)
ax1.set_zlim3d(-1,1)

#draw trajectory
ax1.plot3D(gu,gv,gs, label="sym")
ax1.plot3D(gpu,gpv,gps, label="parameter")
ax1.legend(loc= 'lower left')

#draw error
ax2=fig.add_subplot(132)
ax2.plot(timearr,spemerr,label="sym/par")
# plt.plot(timearr,pprk4emerr,label="prk4/par")
ax2.set_yscale("log",basey=10)
ax2.legend(loc='upper left')

#draw 2d trajectory
ax3=fig.add_subplot(133)
ax3.plot(xc,yc,color="k")
ax3.plot(g2u,g2v, label="sym")
ax3.plot(g2pu,g2pv, label="parameter")
ax3.legend(loc= 'lower left')		

fig.tight_layout()	

plt.show()	