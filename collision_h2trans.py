from symint_bank import imph2geotrans
from function_bank import hyper2poinh2,h2dist,boostxh2,rotzh2,hypercirch2,collisionh2,convert_rot2transh2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

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
positions = array([convert_rot2transh2(particles[0][:2]),convert_rot2transh2(particles[1][:2])])
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
	imph2geotrans(positions[0], velocities[0], delT), 
	imph2geotrans(positions[1], velocities[1], delT)
	])

# Include the first time step
gat=append(gat, array([step_data[0][0],step_data[1][0]]))
gbt=append(gbt, array([step_data[0][1],step_data[1][1]]))

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
	# Collision detection check
	dist=h2dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])])
	if dist<=particles[0][-1]+particles[1][-1]:
		print("collided")
		nextpos = array([step_data[0][:2], step_data[1][:2]])
		nextdot= collisionh2(step_data[0][:2], step_data[1][:2],step_data[0][2:4], step_data[1][2:4],particles[0][4],particles[1][4],dist)
	else:
		nextpos = array([step_data[0][:2], step_data[1][:2]])
		nextdot = array([step_data[0][2:4], step_data[1][2:4]])

	step_data=array([
		imph2geotrans(nextpos[0], nextdot[0], delT), 
		imph2geotrans(nextpos[1], nextdot[1], delT)
		])

	gat=append(gat, array([step_data[0][0],step_data[1][0]]))
	gbt=append(gbt, array([step_data[0][1],step_data[1][1]]))

	q=q+1

# Transform into Poincare disk model for plotting
gut=[]
gvt=[]

for b in range(len(gat)):
    gut=append(gut,sinh(gat[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.))
    gvt=append(gvt,cosh(gat[b])*sinh(gbt[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.))	    	     		

#This is for the horizon circle.
# Theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# Compute x1 and x2 for horizon
xc = np.cos(theta)
yc = np.sin(theta)    


#####################
#  PLOTTING SECTION #
#####################

# --------------------------------------------------------------------
### Uncomment to just plot trajectories in the Poincare disk model ###
# --------------------------------------------------------------------


#This is the particle trajectories in the Poincare model
    
#Plot
plt.figure(figsize=(5,5))

part1x,part1y=hypercirch2(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])]),particles[0][5])
part2x,part2y=hypercirch2(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]),particles[1][5])

plt.plot(xc,yc)
plt.plot(gut[0::2],gvt[0::2],label="particle 1")
plt.plot(gut[1::2],gvt[1::2],label="particle 2")
plt.plot(part1x,part1y)
plt.plot(part2x,part2y)
plt.legend(loc='lower left')	

plt.show()

# ---------------------------------------------------------------------
### Uncomment to just generate gif of trajectories of the particles ###
# ---------------------------------------------------------------------

# #Generate gif
# # create empty lists for the x and y data
# x1 = []
# y1 = []
# x2 = []
# y2 = []

# # First set up the figure, the axis, and the plot element we want to animate
# fig, ax = plt.subplots(figsize=(6,6))

# ax.set_xlim([ -1, 1])
# ax.set_xlabel('X')
# ax.set_ylim([-1, 1])
# ax.set_ylabel('Y')

# ax.plot(xc,yc)
# part1x,part1y=hypercirch2(array([sinh(gat[0::2][0]),cosh(gat[0::2][0])*sinh(gbt[0::2][0]),cosh(gat[0::2][0])*cosh(gbt[0::2][0])]),particles[0][5])
# part2x,part2y=hypercirch2(array([sinh(gat[1::2][1]),cosh(gat[1::2][1])*sinh(gbt[1::2][1]),cosh(gat[1::2][1])*cosh(gbt[1::2][1])]),particles[1][5])
# circ1,=ax.plot(part1x,part1y)
# circ2,=ax.plot(part2x,part2y)

# # animation function. This is called sequentially
# frames=50
# def animate(i):
# 	ax.plot(gut[0::2][:int(len(timearr)*i/frames)],gvt[0::2][:int(len(timearr)*i/frames)])
# 	ax.plot(gut[1::2][:int(len(timearr)*i/frames)],gvt[1::2][:int(len(timearr)*i/frames)])
# 	part1x,part1y=hypercirch2(array([sinh(gat[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*sinh(gbt[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*cosh(gbt[0::2][int(len(timearr)*i/frames)])]),particles[0][5])
# 	part2x,part2y=hypercirch2(array([sinh(gat[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*sinh(gbt[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*cosh(gbt[1::2][int(len(timearr)*i/frames)])]),particles[1][5])
# 	circ1.set_data([part1x],[part1y])
# 	circ2.set_data([part2x],[part2y])

# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# # call the animator. blit=True means only re-draw the parts that 
# # have changed.
# anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

# anim.save('./test.gif', writer='imagemagick')




