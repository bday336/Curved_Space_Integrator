from symint_bank import imph3sprot3
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convertpos_hyp2roth3,convertpos_rot2hyph3,initial_con
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

# Initial position (position / velocity in rotational parameterization)
# The initial conditions are given as an array of the data for each of the particles being
# initialized. Each element is a particle with each element having 8 components
# The initial positions are given by the coordinates in the rotational parameterization
# and the initial velocities are given through using a killing field of generic loxodromic
# geodesic flow along the x-axis.
# { ai , bi , gi , adi , bdi , gdi , mass , radius }

# Initialize the particles in the simulation
# Check Equilibrium (verified to 10^-16)
# particles=array([
#     np.concatenate((array([.5,np.pi/2.,0.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,0.*2.*np.pi/3.]),.0,.0),[1.,.2])),       #particle 1
#     np.concatenate((array([.5,np.pi/2.,1.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,1.*2.*np.pi/3.]),.0,.0),[1.,.2])),       #particle 2
#     np.concatenate((array([.5,np.pi/2.,2.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,2.*2.*np.pi/3.]),.0,.0),[1.,.2]))        #particle 3
#     ])

# Vertex Oriented Initial Velocities
# particles=array([
#     np.concatenate((array([.5,np.pi/2.,0.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,0.*2.*np.pi/3.]),.5,.0),[1.,.2])),       #particle 1
#     np.concatenate((array([.5,np.pi/2.,1.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,1.*2.*np.pi/3.]),.5,.0),[1.,.2])),       #particle 2
#     np.concatenate((array([.5,np.pi/2.,2.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,2.*2.*np.pi/3.]),.5,.0),[1.,.2]))        #particle 3
#     ])

# Edge Oriented Initial Velocities
# particles=array([
#     np.concatenate((array([.5,np.pi/2.,0.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,0.*2.*np.pi/3.]),-.5,.0),[1.,.2])),       #particle 1
#     np.concatenate((array([.5,np.pi/2.,1.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,1.*2.*np.pi/3.]),-.5,.0),[1.,.2])),       #particle 2
#     np.concatenate((array([.5,np.pi/2.,2.*2.*np.pi/3.]),initial_con(array([.5,np.pi/2.,2.*2.*np.pi/3.]),-.5,.0),[1.,.2]))        #particle 3
#     ])

# Face Oriented Initial Velocities
particles=array([
    np.concatenate((convertpos_hyp2roth3(rotyh3(np.pi/2.) @ convertpos_rot2hyph3([.5,np.pi/2.,0.*2.*np.pi/3.+np.pi/2.])),initial_con(convertpos_hyp2roth3(rotyh3(np.pi/2.) @ convertpos_rot2hyph3([.5,np.pi/2.,0.*2.*np.pi/3.+np.pi/2.])),.5,.0),[1.,.2])),       #particle 1
    np.concatenate((convertpos_hyp2roth3(rotyh3(np.pi/2.) @ convertpos_rot2hyph3([.5,np.pi/2.,1.*2.*np.pi/3.+np.pi/2.])),initial_con(convertpos_hyp2roth3(rotyh3(np.pi/2.) @ convertpos_rot2hyph3([.5,np.pi/2.,1.*2.*np.pi/3.+np.pi/2.])),.5,.0),[1.,.2])),       #particle 2
    np.concatenate((convertpos_hyp2roth3(rotyh3(np.pi/2.) @ convertpos_rot2hyph3([.5,np.pi/2.,2.*2.*np.pi/3.+np.pi/2.])),initial_con(convertpos_hyp2roth3(rotyh3(np.pi/2.) @ convertpos_rot2hyph3([.5,np.pi/2.,2.*2.*np.pi/3.+np.pi/2.])),.5,.0),[1.,.2]))        #particle 3
    ])

# Initialize the parameters of what I will consider the
# sping system. This can be expanded on in the future with
# something like a class object similar to the particle
# table. The elements of each spring are given as
# { spring constant (k) , equilibrium length of spring (l_{eq}) }
# The value for equilibrium length was calculated on mathematica
spring_arr=array([
    [1.,0.874436528313447],    #spring 12
    [1.,0.874436528313447],    #spring 13
    [1.,0.874436528313447],    #spring 23
    ])

# Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT
nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Positions in rotational parameterization
positions = array([
    particles[0][:3],
    particles[1][:3],
    particles[2][:3]])
# Velocities in rotational parameterization
velocities = array([
    particles[0][3:6],
    particles[1][3:6],
    particles[2][3:6]])
# Masses for each particle
masses = array([
    particles[0][6],
    particles[1][6],
    particles[2][6]])

# Containers for trajectory data
q = 0
gat = []
gbt = []
ggt = []
dist12 = []
dist13 = []
dist23 = []

# Include the intial data
gat=append(gat, array([positions[0][0],positions[1][0],positions[2][0]]))
gbt=append(gbt, array([positions[0][1],positions[1][1],positions[2][1]]))
ggt=append(ggt, array([positions[0][2],positions[1][2],positions[2][2]]))

# Distance between masses
dist12=append(dist12,h3dist([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])],[sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])]))
dist13=append(dist13,h3dist([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))
dist23=append(dist23,h3dist([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))

# Numerical Integration step
step_data=array([
	imph3sprot3(positions, velocities, delT, masses, spring_arr)
	])

# Include the first time step
gat=append(gat, array([step_data[0][0],step_data[0][3],step_data[0][6]]))
gbt=append(gbt, array([step_data[0][1],step_data[0][4],step_data[0][7]]))
ggt=append(ggt, array([step_data[0][2],step_data[0][5],step_data[0][8]]))

# Distance between masses
dist12=append(dist12,h3dist([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])],[sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])]))
dist13=append(dist13,h3dist([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))
dist23=append(dist23,h3dist([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))

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
        nextpos = array([step_data[0][0:3], step_data[0][3:6], step_data[0][6:9]])
        nextdot = array([step_data[0][9:12], step_data[0][12:15], step_data[0][15:]])

    step_data=array([
        imph3sprot3(nextpos, nextdot, delT, masses, spring_arr)
        ])

    gat=append(gat, array([step_data[0][0],step_data[0][3],step_data[0][6]]))
    gbt=append(gbt, array([step_data[0][1],step_data[0][4],step_data[0][7]]))
    ggt=append(ggt, array([step_data[0][2],step_data[0][5],step_data[0][8]]))

    # Distance between masses
    dist12=append(dist12,h3dist([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])],[sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])]))
    dist13=append(dist13,h3dist([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))
    dist23=append(dist23,h3dist([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))

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

# #Plot
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

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])]),particles[0][7])
# part2x,part2y,part2z=hypercirch3(array([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])]),particles[1][7])
# part3x,part3y,part3z=hypercirch3(array([sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]),particles[2][7])

# #draw trajectory
# ax1.plot3D(gut[0::3],gvt[0::3],grt[0::3], label="particle 1")
# ax1.plot3D(gut[1::3],gvt[1::3],grt[1::3], label="particle 2")
# ax1.plot3D(gut[2::3],gvt[2::3],grt[2::3], label="particle 3")
# ax1.legend(loc= 'lower left')

# ax1.plot_surface(part1x, part1y, part1z, color="b")
# ax1.plot_surface(part2x, part2y, part2z, color="r")
# ax1.plot_surface(part3x, part3y, part3z, color="k")

# plt.show()

# --------------------------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------------------------

# Plot Trajectory with error
fig = plt.figure(figsize=(12,4))

ax1=fig.add_subplot(1,3,1,projection='3d')

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

part1x,part1y,part1z=hypercirch3(array([sinh(gat[-3])*sin(gbt[-3])*cos(ggt[-3]),sinh(gat[-3])*sin(gbt[-3])*sin(ggt[-3]),sinh(gat[-3])*cos(gbt[-3]),cosh(gat[-3])]),particles[0][7])
part2x,part2y,part2z=hypercirch3(array([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])]),particles[1][7])
part3x,part3y,part3z=hypercirch3(array([sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]),particles[2][7])

#draw trajectory
ax1.plot3D(gut[0::3],gvt[0::3],grt[0::3], label="particle 1", color="b")
ax1.plot3D(gut[1::3],gvt[1::3],grt[1::3], label="particle 2", color="r")
ax1.plot3D(gut[2::3],gvt[2::3],grt[2::3], label="particle 3", color="k")
ax1.legend(loc= 'lower left')

ax1.plot_surface(part1x, part1y, part1z, color="b")
ax1.plot_surface(part2x, part2y, part2z, color="r")
ax1.plot_surface(part3x, part3y, part3z, color="k")

# Displacement Plot
ax2=fig.add_subplot(1,3,2)

ax2.plot(timearr,(dist12-spring_arr[0][1]),label="Spring 12 Displacement")
ax2.plot(timearr,(dist13-spring_arr[1][1]),label="Spring 13 Displacement")
ax2.plot(timearr,(dist23-spring_arr[2][1]),label="Spring 23 Displacement")
#ax2.axhline(y=spring[3]+sqrt(2.*1./spring[2]*.5*.5), color='b', linestyle='-')
#ax2.axhline(y=((dist.max()-spring[3])+(dist.min()-spring[3]))/2., color='r', linestyle='-')
#ax2.set_yscale("log",basey=10)	
#ax2.set_ylabel('displacement (m)')
ax2.set_xlabel('time (s)')
ax2.legend(loc='lower right')	

# Force Plot
ax3=fig.add_subplot(1,3,3)

ax3.plot(timearr,(dist12-spring_arr[0][1])*spring_arr[0][0],label="Spring 12 Force")
ax3.plot(timearr,(dist13-spring_arr[1][1])*spring_arr[1][0],label="Spring 13 Force")
ax3.plot(timearr,(dist23-spring_arr[2][1])*spring_arr[2][0],label="Spring 23 Force")
# ax3.set_yscale("log",basey=10)
ax3.set_xlabel('time (s)')	
ax3.legend(loc='lower right')	

fig.tight_layout()	

plt.show()

# ------------------------------------------------------------------
### Uncomment to just generate gif of trajectory of the particle ###
# ------------------------------------------------------------------

# # Generate gif
# # create empty lists for the x and y data
# x1 = []
# y1 = []
# z1 = []
# x2 = []
# y2 = []
# z2 = []
# x3 = []
# y3 = []
# z3 = []

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

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::3][0])*sin(gbt[0::3][0])*cos(ggt[0::3][0]),sinh(gat[0::3][0])*sin(gbt[0::3][0])*sin(ggt[0::3][0]),sinh(gat[0::3][0])*cos(gbt[0::3][0]),cosh(gat[0::3][0])]),particles[0][7])
# part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::3][1])*sin(gbt[1::3][1])*cos(ggt[1::3][1]),sinh(gat[1::3][1])*sin(gbt[1::3][1])*sin(ggt[1::3][1]),sinh(gat[1::3][1])*cos(gbt[1::3][1]),cosh(gat[1::3][1])]),particles[1][7])
# part3x,part3y,part3z=hypercirch3(array([sinh(gat[2::3][2])*sin(gbt[2::3][2])*cos(ggt[2::3][2]),sinh(gat[2::3][2])*sin(gbt[2::3][2])*sin(ggt[2::3][2]),sinh(gat[2::3][2])*cos(gbt[2::3][2]),cosh(gat[2::3][2])]),particles[2][7])
# ball1=[ax1.plot_surface(part1x,part1y,part1z, color="b")]
# ball2=[ax1.plot_surface(part2x,part2y,part2z, color="r")]
# ball3=[ax1.plot_surface(part3x,part3y,part3z, color="k")]

# # animation function. This is called sequentially
# frames=50
# def animate(i):
#     ax1.plot3D(gut[0::3][:int(len(timearr)*i/frames)],gvt[0::3][:int(len(timearr)*i/frames)],grt[0::3][:int(len(timearr)*i/frames)])
#     ax1.plot3D(gut[1::3][:int(len(timearr)*i/frames)],gvt[1::3][:int(len(timearr)*i/frames)],grt[1::3][:int(len(timearr)*i/frames)])
#     ax1.plot3D(gut[2::3][:int(len(timearr)*i/frames)],gvt[2::3][:int(len(timearr)*i/frames)],grt[2::3][:int(len(timearr)*i/frames)])
#     part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::3][int(len(timearr)*i/frames)])*sin(gbt[0::3][int(len(timearr)*i/frames)])*cos(ggt[0::3][int(len(timearr)*i/frames)]),sinh(gat[0::3][int(len(timearr)*i/frames)])*sin(gbt[0::3][int(len(timearr)*i/frames)])*sin(ggt[0::3][int(len(timearr)*i/frames)]),sinh(gat[0::3][int(len(timearr)*i/frames)])*cos(gbt[0::3][int(len(timearr)*i/frames)]),cosh(gat[0::3][int(len(timearr)*i/frames)])]),particles[0][7])
#     part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::3][int(len(timearr)*i/frames)])*sin(gbt[1::3][int(len(timearr)*i/frames)])*cos(ggt[1::3][int(len(timearr)*i/frames)]),sinh(gat[1::3][int(len(timearr)*i/frames)])*sin(gbt[1::3][int(len(timearr)*i/frames)])*sin(ggt[1::3][int(len(timearr)*i/frames)]),sinh(gat[1::3][int(len(timearr)*i/frames)])*cos(gbt[1::3][int(len(timearr)*i/frames)]),cosh(gat[1::3][int(len(timearr)*i/frames)])]),particles[1][7])
#     part3x,part3y,part3z=hypercirch3(array([sinh(gat[2::3][int(len(timearr)*i/frames)])*sin(gbt[2::3][int(len(timearr)*i/frames)])*cos(ggt[2::3][int(len(timearr)*i/frames)]),sinh(gat[2::3][int(len(timearr)*i/frames)])*sin(gbt[2::3][int(len(timearr)*i/frames)])*sin(ggt[2::3][int(len(timearr)*i/frames)]),sinh(gat[2::3][int(len(timearr)*i/frames)])*cos(gbt[2::3][int(len(timearr)*i/frames)]),cosh(gat[2::3][int(len(timearr)*i/frames)])]),particles[2][7])
#     ball1[0].remove()
#     ball1[0]=ax1.plot_surface(part1x,part1y,part1z, color="b")
#     ball2[0].remove()
#     ball2[0]=ax1.plot_surface(part2x,part2y,part2z, color="r")
#     ball3[0].remove()
#     ball3[0]=ax1.plot_surface(part3x,part3y,part3z, color="k")

# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# # call the animator. blit=True means only re-draw the parts that 
# # have changed.
# anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

# anim.save('./h3spring3_test.gif', writer='imagemagick')

