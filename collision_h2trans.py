from symint_bank import imph2geotrans
from function_bank import hyper2poin2d,hyper2poin3d,formatvec2d,unformatvec2d,formatvec3d,unformatvec3d,boostx2d,boosty2d,rotz2d,boostx3d,boosty3d,boostz3d,rotx3d,roty3d,rotz3d,motionmat2dh,motionmat3dh,py2dhfreetrans,xfunc2dhtrans,yfunc2dhtrans,zfunc2dhtrans,py3dhfreetrans,xfunc3dhtrans,yfunc3dhtrans,zfunc3dhtrans,wfunc3dhtrans,py3dhfreerot,xfunc3dhrot,yfunc3dhrot,zfunc3dhrot,wfunc3dhrot,py3dhgrav,py3dhefreetrans,xfunc3dhetrans,yfunc3dhetrans,zfunc3dhetrans,wfunc3dhetrans,py3dhefreerot,xfunc3dherot,yfunc3dherot,zfunc3dherot,wfunc3dherot,py3dhspring,py3dhespring,geodesicflow_x
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

def convert_rot2trans(rot_vec):
	return array([arcsinh(sinh(rot_vec[0])*cos(rot_vec[1])), arctanh(tanh(rot_vec[0])*sin(rot_vec[1]))])

#Initial position (position in rotational / velocity in chosen parameterization)
#The initial conditions are given as an array of the data for each of the particles being
# initialized. Each element is a particle with each element having 6 components
# { ai , bi , adi , bdi , mass , radius }

#Initialize the particles in the simulation
particles=array([
    [.75,0.,-.2,0.,1.,.2],        #particle 1
    [.75,np.pi/2.,0.,-.2,1.,.2]   #particle 2
    ])

#Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT

# Geodesic Trajectories

# Position in translational parameterization
positions = array([convert_rot2trans(particles[0][:2]),convert_rot2trans(particles[1][:2])])
# Velocity given in translational parameterization
velocities = array([particles[0][2:4], particles[1][2:4]])

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []

# Include the intial data
gat=append(gat, array([positions[0][0],positions[1][0]]))
gbt=append(gbt, array([positions[0][1],positions[1][1]]))

# Numerical Integration step
step_data=array([
	imph2geotrans(positions[0], positions[0], velocities[0], velocities[0], delT), 
	imph2geotrans(positions[1], positions[1], velocities[1], velocities[1], delT)
	])

# Include the first time step
gat=append(gat, array([step_data[0][0],step_data[1][0]]))
gbt=append(gbt, array([step_data[0][1],step_data[1][1]]))

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
	nextpos = array([step_data[0][:2], step_data[1][:2]])
	nextdot = array([step_data[0][2:4], step_data[1][2:4]])

	step_data=array([
		imph2geotrans(nextpos[0], nextpos[0], nextdot[0], nextdot[0], delT), 
		imph2geotrans(nextpos[1], nextpos[1], nextdot[1], nextdot[1], delT)
		])

	gat=append(gat, array([step_data[0][0],step_data[1][0]]))
	gbt=append(gbt, array([step_data[0][1],step_data[1][1]]))

	q=q+1

gut=[]
gvt=[]

for b in range(len(timearr)):
    gut=append(gut,sinh(gat[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.))
    gvt=append(gvt,cosh(gat[b])*sinh(gbt[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.))	    	     		

#This is for the horizon circle.
# Theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# Compute x1 and x2 for horizon
xc = np.cos(theta)
yc = np.sin(theta)    		


#This is the particle trajectories in the Poincare model
    
# #Plot
plt.figure(figsize=(5,5))

plt.plot(xc,yc)
plt.plot(gut[0::2],gvt[0::2],label="particle 1")
plt.plot(gut[1::2],gvt[1::2],label="particle 2")
plt.legend(loc='lower left')	

plt.show()





