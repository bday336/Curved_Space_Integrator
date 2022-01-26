from symint_bank import imph3cprot
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convert_rot2transh3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
#The initial conditions are given as an array of the data for each of the particles being
# initialized. Each element is a particle with each element having 8 components
# { ai , bi , gi , adi , bdi , gdi , mass , radius }

#Initialize the particles in the simulation
sourcemass = 1.
particles=array([
    [1.75,np.pi/2.,0.,.0,.0,.05,1.,.2]        #particle 1
    ])

#Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT

# Geodesic Trajectories

# Position in translational parameterization
positions = array([particles[0][:3]])
# Velocity given in translational parameterization
velocities = array([particles[0][3:6]])

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []
ggt = []

# Include the intial data
gat=append(gat, array([positions[0][0]]))
gbt=append(gbt, array([positions[0][1]]))
ggt=append(ggt, array([positions[0][2]]))

# Numerical Integration step
step_data=array([
	imph3cprot(positions[0], velocities[0], delT, sourcemass, particles[0][6])
	])

# Include the first time step
gat=append(gat, array([step_data[0][0]]))
gbt=append(gbt, array([step_data[0][1]]))
ggt=append(ggt, array([step_data[0][2]]))

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
    # Collision detection check
    #dist=h3dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])])
    if False:#dist<=particles[0][-1]+particles[1][-1]:
        print("collided")
        nextpos = array([step_data[0][:3], step_data[1][:3]])
        nextdot= collisionh3(step_data[0][:3], step_data[1][:3],step_data[0][3:6], step_data[1][3:6],particles[0][6],particles[1][6],dist)
    else:
        nextpos = array([step_data[0][:3]])
        nextdot = array([step_data[0][3:6]])

    step_data=array([
        imph3cprot(nextpos[0], nextdot[0], delT, sourcemass, particles[0][6])
        ])

    gat=append(gat, array([step_data[0][0]]))
    gbt=append(gbt, array([step_data[0][1]]))
    ggt=append(ggt, array([step_data[0][2]]))

    q=q+1

# Transform into Poincare disk model for plotting
gut=[]
gvt=[]
grt=[]

for b in range(len(gat)):
    gut=append(gut,sinh(gat[b])*sin(gbt[b])*cos(ggt[b])/(cosh(gat[b]) + 1.))
    gvt=append(gvt,sinh(gat[b])*sin(gbt[b])*sin(ggt[b])/(cosh(gat[b]) + 1.))
    grt=append(grt,sinh(gat[b])*cos(gbt[b])/(cosh(gat[b]) + 1.))    	     		

#####################
#  PLOTTING SECTION #
#####################

# ------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model ###
# ------------------------------------------------------------------

#Plot
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111, projection='3d')
# ax1.set_aspect("equal")

#draw sphere
u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
x = np.sin(u)*np.cos(v)
y = np.sin(u)*np.sin(v)
z = np.cos(u)
ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
ax1.set_xlim3d(-1,1)
ax1.set_xlabel('X')
ax1.set_ylim3d(-1,1)
ax1.set_ylabel('Y')
ax1.set_zlim3d(-1,1)
ax1.set_zlabel('Z')

part1x,part1y,part1z=hypercirch3(array([sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]),particles[0][7])

#draw trajectory
ax1.plot3D(gut[0::1],gvt[0::1],grt[0::1], label="solver")
ax1.legend(loc= 'lower left')

ax1.plot_surface(part1x, part1y, part1z, color="b")
	
plt.show()

# ------------------------------------------------------------------
### Uncomment to just generate gif of trajectory of the particle ###
# ------------------------------------------------------------------

# #Generate gif
# # create empty lists for the x and y data
# x1 = []
# y1 = []
# z1 = []

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure(figsize=(8,8))
# ax1 = fig.add_subplot(111, projection='3d')
# # ax1.set_aspect("equal")

# #draw sphere
# u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
# x = np.sin(u)*np.cos(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(u)
# ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
# ax1.set_xlim3d(-1,1)
# ax1.set_xlabel('X')
# ax1.set_ylim3d(-1,1)
# ax1.set_ylabel('Y')
# ax1.set_zlim3d(-1,1)
# ax1.set_zlabel('Z')

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::1][0]),cosh(gat[0::1][0])*sinh(gbt[0::1][0]),cosh(gat[0::1][0])*cosh(gbt[0::1][0])*sinh(ggt[0::1][0]),cosh(gat[0::1][0])*cosh(gbt[0::1][0])*cosh(ggt[0::1][0])]),particles[0][7])
# ball1=[ax1.plot_surface(part1x,part1y,part1z, color="b")]

# # animation function. This is called sequentially
# frames=50
# def animate(i):
#     ax1.plot3D(gut[0::1][:int(len(timearr)*i/frames)],gvt[0::1][:int(len(timearr)*i/frames)],grt[0::1][:int(len(timearr)*i/frames)])
#     part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::1][int(len(timearr)*i/frames)]),cosh(gat[0::1][int(len(timearr)*i/frames)])*sinh(gbt[0::1][int(len(timearr)*i/frames)]),cosh(gat[0::1][int(len(timearr)*i/frames)])*cosh(gbt[0::1][int(len(timearr)*i/frames)])*sinh(ggt[0::1][int(len(timearr)*i/frames)]),cosh(gat[0::1][int(len(timearr)*i/frames)])*cosh(gbt[0::1][int(len(timearr)*i/frames)])*cosh(ggt[0::1][int(len(timearr)*i/frames)])]),particles[0][7])
#     ball1[0].remove()
#     ball1[0]=ax1.plot_surface(part1x,part1y,part1z, color="b")

# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# # call the animator. blit=True means only re-draw the parts that 
# # have changed.
# anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

# anim.save('./h3collision_test.gif', writer='imagemagick')

