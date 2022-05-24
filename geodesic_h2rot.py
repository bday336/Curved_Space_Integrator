from symint_bank import imph2georot
from function_bank import hyper2poinh2,h2dist,boostxh2,rotzh2,hypercirch2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

# Initial position (position / velocity in rotational parameterization)
# The initial conditions are given as an array of the data for each of the particles being
# initialized. Each element is a particle with each element having 6 components
# { ai , bi , adi , bdi , mass , radius }

#Initialize the particles in the simulation
particles=array([
    [.5,0*pi/2.,1.,.0,1.,.2]        #particle 1
    ])

#Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT

# Geodesic Trajectories

# Position in translational parameterization then rotational
positions = array([particles[0][:2]])
# Velocity given in translational parameterization then rotational
velocities = array([particles[0][2:4]])

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []

# Include the intial data
gat=append(gat, positions[0][0])
gbt=append(gbt, positions[0][1])

# Numerical Integration step
step_data=array([
    # From testing the two methods it seems that both are of the same accuracy
    # This leads me to wonder if we should just stick with the more straight forward 
    # rk4 method if are are going to approximate the connected tangent bundle since the
    # accuracy would be better than the second order.

    # WARNING - Cannot use the rk4 method presented here since we need to have time reversal
    # which implies we need an implicit algorithm like the solver I have developed.

	#imprk4h2geotrans(positions[0], velocities[0], delT) 
    imph2georot(positions[0], velocities[0], delT)
	])

# Include the first time step
gat=append(gat, step_data[0][0])
gbt=append(gbt, step_data[0][1])

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
    nextpos = array([step_data[0][:2]])
    nextdot = array([step_data[0][2:4]])

    step_data=array([
        #imprk4h2geotrans(nextpos[0], nextdot[0], delT) 
        imph2georot(nextpos[0], nextdot[0], delT)
        ])

    gat=append(gat, step_data[0][0])
    gbt=append(gbt, step_data[0][1])

    q=q+1

# Generate geodesic flow to compare results
checkdata=[]
for c in timearr:
    checkdata=append(checkdata, array([sinh(positions[0][0]+velocities[0][0]*c)*cos(positions[0][1]+velocities[0][1]*c),sinh(positions[0][0]+velocities[0][0]*c)*sin(positions[0][1]+velocities[0][1]*c),cosh(positions[0][0]+velocities[0][0]*c)]))
'''rotzh2(particles[0][1]) @ boostxh2(particles[0][0]) @ rotzh2(-particles[0][1]) @'''

# Compare error between solver and geodesic flow
errordatah2=[]
errordatae3=[]
for d in range(len(gat)):
    # The hyperbolic distance error is modified so that even is the argument of the arccosh is less than 1 it still returns a none nan value
    # Hence why i am taking the absolute value of the complex valued arccosh result (might not be the best solution but the values are so
    # close to 1 in the argument I doubt it matters too much)
    errordatah2=append(errordatah2,abs(arccosh(complex(-sinh(gat[d])*cos(gbt[d])*checkdata[0::3][d]-cosh(gat[d])*sin(gbt[d])*checkdata[1::3][d]+cosh(gat[d])*checkdata[2::3][d]))))
    errordatae3=append(errordatae3,sqrt((sinh(gat[d])*cos(gbt[d])-checkdata[0::3][d])**2.+(cosh(gat[d])*sin(gbt[d])-checkdata[1::3][d])**2.+(cosh(gat[d])-checkdata[2::3][d])**2.))

# Transform into Poincare disk model for plotting
gut=[]
gvt=[]
checkplot=[]
for b in range(len(gat)):
    gut=append(gut,sinh(gat[b])*cos(gbt[b])/(cosh(gat[b]) + 1.))
    gvt=append(gvt,sinh(gat[b])*sin(gbt[b])/(cosh(gat[b]) + 1.))
    checkplot=append(checkplot,array([checkdata[0::3][b]/(checkdata[2::3][b] + 1.),checkdata[1::3][b]/(checkdata[2::3][b] + 1.)]))	    	     		

#This is for the horizon circle.
# Theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# Compute x1 and x2 for horizon
xc = np.cos(theta)
yc = np.sin(theta)    		


#####################
#  PLOTTING SECTION #
#####################

# ------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model ###
# ------------------------------------------------------------------

#This is the particle trajectories in the Poincare model
    
# #Plot
# plt.figure(figsize=(5,5))

# part1x,part1y=hypercirch2(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]),particles[0][5])

# plt.plot(xc,yc)
# plt.plot(gut,gvt,label="solver")
# plt.plot(checkplot[0::2],checkplot[1::2],label="geoflow")
# plt.plot(part1x,part1y)
# plt.legend(loc='lower left')	

# plt.show()

# -----------------------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model with error plots ###
# -----------------------------------------------------------------------------------


# Plot Trajectory with error
fig , ((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(12,4))

part1x,part1y=hypercirch2(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]),particles[0][5])

ax1.plot(xc,yc)
ax1.plot(gut,gvt,label="solver")
ax1.plot(checkplot[0::2],checkplot[1::2],label="geoflow")
ax1.plot(part1x,part1y)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend(loc='lower left')

ax2.plot(timearr,errordatah2,label="h2 error")
#ax2.set_yscale("log",basey=10)	
ax2.set_xlabel('time (s)')
ax2.legend(loc='lower right')	

ax3.plot(timearr,errordatae3,label="e3 error")
ax3.set_yscale("log",basey=10)
ax3.set_xlabel('time (s)')	
ax3.legend(loc='lower right')	

fig.tight_layout()	

plt.show()

# ------------------------------------------------------------------
### Uncomment to just generate gif of trajectory of the particle ###
# ------------------------------------------------------------------

# #Generate gif
# # create empty lists for the x and y data
# x1 = []
# y1 = []

# # First set up the figure, the axis, and the plot element we want to animate
# fig, ax = plt.subplots(figsize=(6,6))

# ax.set_xlim([ -1, 1])
# ax.set_xlabel('X')
# ax.set_ylim([-1, 1])
# ax.set_ylabel('Y')

# ax.plot(xc,yc)
# part1x,part1y=hypercirc(array([sinh(gat[0::1][0]),cosh(gat[0::1][0])*sinh(gbt[0::1][0]),cosh(gat[0::1][0])*cosh(gbt[0::1][0])]),particles[0][5])
# circ1,=ax.plot(part1x,part1y)

# # animation function. This is called sequentially
# frames=50
# def animate(i):
# 	ax.plot(gut[0::1][:int(len(timearr)*i/frames)],gvt[0::1][:int(len(timearr)*i/frames)])
# 	part1x,part1y=hypercirc(array([sinh(gat[0::1][int(len(timearr)*i/frames)]),cosh(gat[0::1][int(len(timearr)*i/frames)])*sinh(gbt[0::1][int(len(timearr)*i/frames)]),cosh(gat[0::1][int(len(timearr)*i/frames)])*cosh(gbt[0::1][int(len(timearr)*i/frames)])]),particles[0][5])
# 	circ1.set_data([part1x],[part1y])

# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# # call the animator. blit=True means only re-draw the parts that 
# # have changed.
# anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

# anim.save('./h2collision_test.gif', writer='imagemagick')




