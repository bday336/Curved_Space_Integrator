from symint_bank import imph2sptrans2
from function_bank import hyper2poinh2,h2dist,boostxh2,rotzh2,convert_rot2transh2,hypercirch2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
#The initial conditions are given as an array of the data for each of the particles being
# initialized. Each element is a particle with each element having 8 components
# { ai , bi , adi , bdi , mass , radius }

#Initialize the particles in the simulation
particles=array([
    [.5,np.pi/2.,.5,.0,1.,.2],          #particle 1
    [.5,3.*np.pi/2.,.5,.0,1.,.2]        #particle 2
    ])

# Initialize the parameters of what I will consider the
# sping system. This can be expanded on in the future with
# something like a class object similar to the particle
# table. The elements of each spring are given as
# { particle1 , particle2 , spring constant (k) , equilibrium length of spring (l_{eq}) }
spring=array([particles[0],particles[1],.5,1.])

#Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT

# Geodesic Trajectories

# Position in translational parameterization
positions = array([convert_rot2transh2(particles[0][:2]),convert_rot2transh2(particles[1][:2])])
# Velocity given in translational parameterization
velocities = array([particles[0][2:5], particles[1][2:5]])

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []
dist = []

# Include the intial data
gat=append(gat, array([positions[0][0],positions[1][0]]))
gbt=append(gbt, array([positions[0][1],positions[1][1]]))

# Distance between masses
dist=append(dist,h2dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]))

# Numerical Integration step
step_data=array([
	imph2sptrans2(positions[0], positions[1], velocities[0], velocities[1], delT, spring[0][4], spring[1][4], spring[2], spring[3])
	])

# Include the first time step
gat=append(gat, array([step_data[0][0],step_data[0][2]]))
gbt=append(gbt, array([step_data[0][1],step_data[0][3]]))
print(step_data[0][4],step_data[0][5])

# Distance between masses
dist=append(dist,h2dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]))

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
    # Collision detection check
    #dist=append(dist,h3dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])]))
    if False: #dist<=particles[0][-1]+particles[1][-1]:
        print("collided")
        nextpos = array([step_data[0][:3], step_data[1][:3]])
        nextdot= collisionh3(step_data[0][:3], step_data[1][:3],step_data[0][3:6], step_data[1][3:6],particles[0][6],particles[1][6],dist)
    else:
        nextpos = array([step_data[0][0:2], step_data[0][2:4]])
        nextdot = array([step_data[0][4:6], step_data[0][6:]])

    step_data=array([
        imph2sptrans2(nextpos[0], nextpos[1], nextdot[0], nextdot[1], delT, spring[0][4], spring[1][4], spring[2], spring[3])
        ])

    gat=append(gat, array([step_data[0][0],step_data[0][2]]))
    gbt=append(gbt, array([step_data[0][1],step_data[0][3]]))
    print(step_data[0][4],step_data[0][5])

    # Distance between masses
    dist=append(dist,h2dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]))

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

# ------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model ###
# ------------------------------------------------------------------

#Plot
plt.figure(figsize=(5,5))

part1x,part1y=hypercirch2(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])]),particles[0][5])
part2x,part2y=hypercirch2(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]),particles[1][5])

plt.plot(xc,yc)
plt.plot(gut[0::2],gvt[0::2], label="particle 1")
plt.plot(gut[1::2],gvt[1::2], label="particle 2")
plt.plot(part1x,part1y)
plt.plot(part2x,part2y)
plt.legend(loc='lower left')	

plt.show()

# --------------------------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------------------------

# # Plot Trajectory with error
# fig = plt.figure(figsize=(12,4))

# part1x,part1y=hypercirch2(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])]),particles[0][5])
# part2x,part2y=hypercirch2(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]),particles[1][5])

# ax1=fig.add_subplot(1,3,1)

# #draw trajectory
# ax1.plot(xc,yc)
# ax1.plot(gut[0::2],gvt[0::2], label="particle 1")
# ax1.plot(gut[1::2],gvt[1::2], label="particle 2")
# ax1.plot(part1x,part1y)
# ax1.plot(part2x,part2y)
# ax1.legend(loc= 'lower left')

# # Displacement Plot
# ax2=fig.add_subplot(1,3,2)

# ax2.plot(timearr,(dist-spring[3]),label="displacement")
# #ax2.axhline(y=spring[3]+sqrt(2.*1./spring[2]*.5*.5), color='b', linestyle='-')
# ax2.axhline(y=((dist.max()-spring[3])+(dist.min()-spring[3]))/2., color='r', linestyle='-')
# #ax2.set_yscale("log",basey=10)	
# #ax2.set_ylabel('displacement (m)')
# ax2.set_xlabel('time (s)')
# ax2.legend(loc='lower right')	

# # Force Plot
# ax3=fig.add_subplot(1,3,3)

# ax3.plot(timearr,(dist-spring[3])*spring[2],label="force")
# # ax3.set_yscale("log",basey=10)
# ax3.set_xlabel('time (s)')	
# ax3.legend(loc='lower right')	

# fig.tight_layout()	

# plt.show()

# ------------------------------------------------------------------
### Uncomment to just generate gif of trajectory of the particle ###
# ------------------------------------------------------------------

# #Generate gif
# # create empty lists for the x and y data
# x1 = []
# y1 = []
# x2 = []
# y2 = []

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure(figsize=(8,8))
# ax1 = fig.add_subplot(111,)
# # ax1.set_aspect("equal")
# ax1.plot(xc,yc)

# part1x,part1y=hypercirch2(array([sinh(gat[0::2][0]),cosh(gat[0::2][0])*sinh(gbt[0::2][0]),cosh(gat[0::2][0])*cosh(gbt[0::2][0])]),particles[0][5])
# part2x,part2y=hypercirch2(array([sinh(gat[1::2][1]),cosh(gat[1::2][1])*sinh(gbt[1::2][1]),cosh(gat[1::2][1])*cosh(gbt[1::2][1])]),particles[1][5])
# ax1.plot(part1x,part1y, color="b")
# ax1.plot(part2x,part2y, color="b")

# # animation function. This is called sequentially
# frames=50
# def animate(i):
#     ax1.cla()
#     ax1.plot(xc,yc)
#     ax1.plot(gut[0::2][:int(len(timearr)*i/frames)],gvt[0::2][:int(len(timearr)*i/frames)])
#     ax1.plot(gut[1::2][:int(len(timearr)*i/frames)],gvt[1::2][:int(len(timearr)*i/frames)])
#     part1x,part1y=hypercirch2(array([sinh(gat[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*sinh(gbt[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*cosh(gbt[0::2][int(len(timearr)*i/frames)])]),particles[0][5])
#     part2x,part2y=hypercirch2(array([sinh(gat[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*sinh(gbt[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*cosh(gbt[1::2][int(len(timearr)*i/frames)])]),particles[1][5])
#     ax1.plot(part1x,part1y, color="b")
#     ax1.plot(part2x,part2y, color="b")

# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# # call the animator. blit=True means only re-draw the parts that 
# # have changed.
# anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

# anim.save('./h2spring_test.gif', writer='imagemagick')

