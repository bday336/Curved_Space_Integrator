from symint_bank import symh2esptrans,imph2esptrans
from function_bank import hyper2poin2d,hyper2poin3d,formatvec2d,unformatvec2d,formatvec3d,unformatvec3d,boostx2d,boosty2d,rotz2d,boostx3d,boosty3d,boostz3d,rotx3d,roty3d,rotz3d,motionmat2dh,motionmat3dh,py2dhfreetrans,xfunc2dhtrans,yfunc2dhtrans,zfunc2dhtrans,py3dhfreetrans,xfunc3dhtrans,yfunc3dhtrans,zfunc3dhtrans,wfunc3dhtrans,py3dhfreerot,xfunc3dhrot,yfunc3dhrot,zfunc3dhrot,wfunc3dhrot,py3dhgrav,py3dhefreetrans,xfunc3dhetrans,yfunc3dhetrans,zfunc3dhetrans,wfunc3dhetrans,py3dhefreerot,xfunc3dherot,yfunc3dherot,zfunc3dherot,wfunc3dherot,py3dhspring,py3dhespring
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
a1i=-.5
b1i=0.
g1i=0.
ad1i=0.
bd1i=.5
gd1i=.2 

a2i=.5
b2i=0.
g2i=0.
ad2i=0.
bd2i=.5
gd2i=.2 

massvec=array([1.,1.])
sprcon=1.
eqdist=1.

delT=.01
maxT=10+delT


abg1 = array([a1i, b1i, g1i])
pos1i = array([sinh(abg1[0]), cosh(abg1[0])*sinh(abg1[1]), abg1[2], cosh(abg1[0])*cosh(abg1[1])])
pos1guess = pos1i
abgdot1i = array([ad1i, bd1i, gd1i])
abgdot1guess = abgdot1i

abg2 = array([a2i, b2i, g2i])
pos2i = array([sinh(abg2[0]), cosh(abg2[0])*sinh(abg2[1]), abg2[2], cosh(abg2[0])*cosh(abg2[1])])
pos2guess = pos2i
abgdot2i = array([ad2i, bd2i, gd2i])
abgdot2guess = abgdot2i	

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

q = 0
testm = []
g1x = []
g1y = []
g1z = []
g1w = []
g2x = []
g2y = []
g2z = []
g2w = []
testp = []
g1a = []
g1b = []
g1g = []
g2a = []
g2b = []
g2g = []	

g1x=append(g1x, pos1i[0])
g1y=append(g1y, pos1i[1])
g1z=append(g1z, pos1i[2])
g1w=append(g1w, pos1i[3])
g2x=append(g2x, pos2i[0])
g2y=append(g2y, pos2i[1])
g2z=append(g2z, pos2i[2])
g2w=append(g2w, pos2i[3])
g1a=append(g1a, abg1[0])
g1b=append(g1b, abg1[1])
g1g=append(g1g, abg1[2])
g2a=append(g2a, abg2[0])
g2b=append(g2b, abg2[1])
g2g=append(g2g, abg2[2])

testm.append(symh2esptrans(pos1i, pos1guess, pos2i, pos2guess, abgdot1i, abgdot1guess, abgdot2i, abgdot2guess, delT, massvec, sprcon, eqdist))
testp.append(imph2esptrans(abg1, abg1, abg2, abg2, abgdot1i, abgdot1guess, abgdot2i, abgdot2guess, delT, massvec, sprcon, eqdist))

g1x=append(g1x, testm[0][0][0])
g1y=append(g1y, testm[0][0][1])
g1z=append(g1z, testm[0][0][2])
g1w=append(g1w, testm[0][0][3])	
g2x=append(g2x, testm[0][0][4])
g2y=append(g2y, testm[0][0][5])
g2z=append(g2z, testm[0][0][6])
g2w=append(g2w, testm[0][0][7])

g1a=append(g1a, testp[0][0][0])
g1b=append(g1b, testp[0][0][1])
g1g=append(g1g, testp[0][0][2])	
g2a=append(g2a, testp[0][0][3])
g2b=append(g2b, testp[0][0][4])
g2g=append(g2g, testp[0][0][5])	

q=q+1
while(q < nump-1):

	nextpos1m = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
	nextdot1m = array([testm[q - 1][0][8], testm[q - 1][0][9], testm[q - 1][0][10]])
	nextpos2m = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6], testm[q - 1][0][7]])
	nextdot2m = array([testm[q - 1][0][11], testm[q - 1][0][12], testm[q - 1][0][13]])
	nextpos1p = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
	nextdot1p = array([testp[q - 1][0][6], testp[q - 1][0][7], testp[q - 1][0][8]])
	nextpos2p = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])
	nextdot2p = array([testp[q - 1][0][9], testp[q - 1][0][10], testp[q - 1][0][11]])

	testm.append(symh2esptrans(nextpos1m, nextpos1m, nextpos2m, nextpos2m, nextdot1m, nextdot1m, nextdot2m, nextdot2m, delT, massvec, sprcon, eqdist))
	testp.append(imph2esptrans(nextpos1p, nextpos1p, nextpos2p, nextpos2p, nextdot1p, nextdot1p, nextdot2p, nextdot2p, delT, massvec, sprcon, eqdist))

	g1x=append(g1x, testm[q][0][0])
	g1y=append(g1y, testm[q][0][1])
	g1z=append(g1z, testm[q][0][2])
	g1w=append(g1w, testm[q][0][3])
	g2x=append(g2x, testm[q][0][4])
	g2y=append(g2y, testm[q][0][5])
	g2z=append(g2z, testm[q][0][6])
	g2w=append(g2w, testm[q][0][7])

	g1a=append(g1a, testp[q][0][0])
	g1b=append(g1b, testp[q][0][1])
	g1g=append(g1g, testp[q][0][2])
	g2a=append(g2a, testp[q][0][3])
	g2b=append(g2b, testp[q][0][4])
	g2g=append(g2g, testp[q][0][5])

	q=q+1

g1u=[]
g1v=[]
g1s=[]
g1pu=[]
g1pv=[]
g1ps=[]

g2u=[]
g2v=[]
g2s=[]
g2pu=[]
g2pv=[]
g2ps=[]

#This is the particle trajectories in the Poincare model
for b in range(len(g1a)):

    g1u=append(g1u,g1x[b]/(g1w[b] + 1.))
    g1v=append(g1v,g1y[b]/(g1w[b] + 1.))
    g1s=append(g1s,g1z[b]) 
    g2u=append(g2u,g2x[b]/(g2w[b] + 1.))
    g2v=append(g2v,g2y[b]/(g2w[b] + 1.))
    g2s=append(g2s,g2z[b]) 
    g1pu=append(g1pu,sinh(g1a[b])/(cosh(g1a[b])*cosh(g1b[b]) + 1.))
    g1pv=append(g1pv,cosh(g1a[b])*sinh(g1b[b])/(cosh(g1a[b])*cosh(g1b[b]) + 1.))
    g1ps=append(g1ps,g1g[b]) 	
    g2pu=append(g2pu,sinh(g2a[b])/(cosh(g2a[b])*cosh(g2b[b]) + 1.))
    g2pv=append(g2pv,cosh(g2a[b])*sinh(g2b[b])/(cosh(g2a[b])*cosh(g2b[b]) + 1.))
    g2ps=append(g2ps,g2g[b]) 	        	     
		   		

spemerr1=[]	
for j in range(len(g1x)):
	spemerr1.append(sqrt((g1x[j]-sinh(g1a[j]))**2. + (g1y[j]-cosh(g1a[j])*sinh(g1b[j]))**2. + (g1z[j]-g1g[j])**2. + (g1w[j]-cosh(g1a[j])*cosh(g1b[j]))**2.))			
    
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121, projection='3d')
# ax1.set_aspect("equal")

#draw sphere
u, v = np.mgrid[-2.:2.+.2:.2, 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
x = np.cos(v)
y = np.sin(v)
z = u
ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
ax1.set_xlim3d(-2,2)
ax1.set_ylim3d(-2,2)
ax1.set_zlim3d(-2,2)

#draw trajectory
ax1.plot3D(g1u,g1v,g1s, label="sym", color="r")
ax1.plot3D(g2u,g2v,g2s, color="r")
# ax1.plot3D(g1pu,g1pv,g1ps, label="parameter", color="b")
# ax1.plot3D(g2pu,g2pv,g2ps, color="b")
ax1.legend(loc= 'lower left')

ax2=fig.add_subplot(122)
ax2.plot(timearr,spemerr1,label="sym/par")
# plt.plot(timearr,pprk4emerr,label="prk4/par")		
ax2.legend(loc='upper left')		

fig.tight_layout()	

plt.show()	





