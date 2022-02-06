from symint_bank import imph3boxtrans
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convert_rot2transh3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arcsin,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
#The initial conditions are given as an array of the data for each of the particles being
# initialized. Each element is a particle with each element having 8 components
# { ai , bi , gi , adi , bdi , gdi , mass , radius }

#Initialize the particles in the simulation
particles=array([
    [.5,np.pi/2. - arcsin(1./sqrt(3)) + 2.*arcsin(1./sqrt(3)),np.pi*3./4.,.0,.0,.5,1.,.2],      #particle 1
    [.5,np.pi/2. - arcsin(1./sqrt(3)) + 2.*arcsin(1./sqrt(3)),np.pi*5./4.,.0,.0,.5,1.,.2],      #particle 2
    [.5,np.pi/2. - arcsin(1./sqrt(3)) + 2.*arcsin(1./sqrt(3)),np.pi*1./4.,.0,.0,.5,1.,.2],      #particle 3
    [.5,np.pi/2. - arcsin(1./sqrt(3)) + 2.*arcsin(1./sqrt(3)),np.pi*7./4.,.0,.0,.5,1.,.2],      #particle 4
    [.5,np.pi/2. - arcsin(1./sqrt(3)),np.pi*3./4.,.0,.0,.5,1.,.2],                              #particle 5
    [.5,np.pi/2. - arcsin(1./sqrt(3)),np.pi*5./4.,.0,.0,.5,1.,.2],                              #particle 6
    [.5,np.pi/2. - arcsin(1./sqrt(3)),np.pi*1./4.,.0,.0,.5,1.,.2],                              #particle 7
    [.5,np.pi/2. - arcsin(1./sqrt(3)),np.pi*7./4.,.0,.0,.5,1.,.2]                               #particle 8
    ])

# Initialize the parameters of what I will consider the
# sping system. This can be expanded on in the future with
# something like a class object similar to the particle
# table. The elements of each spring are given as
# { spring constant (k) , equilibrium length of spring (l_{eq}) }
# The value for equilibrium length was calculated on mathematica
spring_arr=array([
    [1.,0.5929828561590165],    #spring 12
    [1.,0.5929828561590165],    #spring 13
    [1.,0.5929828561590165],    #spring 15
    [1.,0.5929828561590165],    #spring 24
    [1.,0.5929828561590165],    #spring 26
    [1.,0.5929828561590165],    #spring 34
    [1.,0.5929828561590165],    #spring 37
    [1.,0.5929828561590165],    #spring 48
    [1.,0.5929828561590165],    #spring 56
    [1.,0.5929828561590165],    #spring 57
    [1.,0.5929828561590165],    #spring 68
    [1.,0.5929828561590165]     #spring 78
    ])

#Intialize the time stepping for the integrator.
delT=.01
maxT=1+delT

# Position in translational parameterization
positions = array([
    convert_rot2transh3(particles[0][:3]),
    convert_rot2transh3(particles[1][:3]),
    convert_rot2transh3(particles[2][:3]),
    convert_rot2transh3(particles[3][:3]),
    convert_rot2transh3(particles[4][:3]),
    convert_rot2transh3(particles[5][:3]),
    convert_rot2transh3(particles[6][:3]),
    convert_rot2transh3(particles[7][:3])
    ])
# Velocity given in translational parameterization
velocities = array([
    particles[0][3:6],
    particles[1][3:6],
    particles[2][3:6],
    particles[3][3:6],
    particles[4][3:6],
    particles[5][3:6],
    particles[6][3:6],
    particles[7][3:6]
    ])

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []
ggt = []

# Include the intial data
gat=append(gat, array([positions[0][0],positions[1][0],positions[2][0],positions[3][0],positions[4][0],positions[5][0],positions[6][0],positions[7][0]]))
gbt=append(gbt, array([positions[0][1],positions[1][1],positions[2][1],positions[3][1],positions[4][1],positions[5][1],positions[6][1],positions[7][1]]))
ggt=append(ggt, array([positions[0][2],positions[1][2],positions[2][2],positions[3][2],positions[4][2],positions[5][2],positions[6][2],positions[7][2]]))

# Numerical Integration step
# Change the mass array if want different masses (update in future) maybe something like mass_arr=particles[:,6]
step_data=array([
	imph3boxtrans(positions, velocities, delT, array([1.,1.,1.,1.,1.,1.,1.,1.]), spring_arr)
	])

# Include the first time step
gat=append(gat, array([step_data[0][0],step_data[0][3],step_data[0][6],step_data[0][9],step_data[0][12],step_data[0][15],step_data[0][18],step_data[0][21]]))
gbt=append(gbt, array([step_data[0][1],step_data[0][4],step_data[0][7],step_data[0][10],step_data[0][13],step_data[0][16],step_data[0][19],step_data[0][22]]))
ggt=append(ggt, array([step_data[0][2],step_data[0][5],step_data[0][8],step_data[0][11],step_data[0][14],step_data[0][17],step_data[0][20],step_data[0][23]]))

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
    # Collision detection check
    #dist=h3dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])])
    if False: #dist<=particles[0][-1]+particles[1][-1]:
        print("collided")
        nextpos = array([step_data[0][:3], step_data[1][:3]])
        nextdot= collisionh3(step_data[0][:3], step_data[1][:3],step_data[0][3:6], step_data[1][3:6],particles[0][6],particles[1][6],dist)
    else:
        nextpos = array([step_data[0][0:3],step_data[0][3:6],step_data[0][6:9],step_data[0][9:12],step_data[0][12:15],step_data[0][15:18],step_data[0][18:21],step_data[0][21:24]])
        nextdot = array([step_data[0][24:27],step_data[0][27:30],step_data[0][30:33],step_data[0][33:36],step_data[0][36:39],step_data[0][39:42],step_data[0][42:45],step_data[0][45:48]])

    step_data=array([
        imph3boxtrans(nextpos, nextdot, delT, array([1.,1.,1.,1.,1.,1.,1.,1.]), spring_arr)
        ])

    gat=append(gat, array([step_data[0][0],step_data[0][3],step_data[0][6],step_data[0][9],step_data[0][12],step_data[0][15],step_data[0][18],step_data[0][21]]))
    gbt=append(gbt, array([step_data[0][1],step_data[0][4],step_data[0][7],step_data[0][10],step_data[0][13],step_data[0][16],step_data[0][19],step_data[0][22]]))
    ggt=append(ggt, array([step_data[0][2],step_data[0][5],step_data[0][8],step_data[0][11],step_data[0][14],step_data[0][17],step_data[0][20],step_data[0][23]]))

    q=q+1

# Transform into Poincare disk model for plotting
gut=[]
gvt=[]
grt=[]


for b in range(len(gat)):
    gut=append(gut,sinh(gat[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))
    gvt=append(gvt,cosh(gat[b])*sinh(gbt[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))
    grt=append(grt,cosh(gat[b])*cosh(gbt[b])*sinh(ggt[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))	    	     		

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

part1x,part1y,part1z=hypercirch3(array([sinh(gat[-8]),cosh(gat[-8])*sinh(gbt[-8]),cosh(gat[-8])*cosh(gbt[-8])*sinh(ggt[-8]),cosh(gat[-8])*cosh(gbt[-8])*cosh(ggt[-8])]),particles[0][7])
part2x,part2y,part2z=hypercirch3(array([sinh(gat[-7]),cosh(gat[-7])*sinh(gbt[-7]),cosh(gat[-7])*cosh(gbt[-7])*sinh(ggt[-7]),cosh(gat[-7])*cosh(gbt[-7])*cosh(ggt[-7])]),particles[1][7])
part3x,part3y,part3z=hypercirch3(array([sinh(gat[-6]),cosh(gat[-6])*sinh(gbt[-6]),cosh(gat[-6])*cosh(gbt[-6])*sinh(ggt[-6]),cosh(gat[-6])*cosh(gbt[-6])*cosh(ggt[-6])]),particles[2][7])
part4x,part4y,part4z=hypercirch3(array([sinh(gat[-5]),cosh(gat[-5])*sinh(gbt[-5]),cosh(gat[-5])*cosh(gbt[-5])*sinh(ggt[-5]),cosh(gat[-5])*cosh(gbt[-5])*cosh(ggt[-5])]),particles[3][7])
part5x,part5y,part5z=hypercirch3(array([sinh(gat[-4]),cosh(gat[-4])*sinh(gbt[-4]),cosh(gat[-4])*cosh(gbt[-4])*sinh(ggt[-4]),cosh(gat[-4])*cosh(gbt[-4])*cosh(ggt[-4])]),particles[4][7])
part6x,part6y,part6z=hypercirch3(array([sinh(gat[-3]),cosh(gat[-3])*sinh(gbt[-3]),cosh(gat[-3])*cosh(gbt[-3])*sinh(ggt[-3]),cosh(gat[-3])*cosh(gbt[-3])*cosh(ggt[-3])]),particles[5][7])
part7x,part7y,part7z=hypercirch3(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])]),particles[6][7])
part8x,part8y,part8z=hypercirch3(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])]),particles[7][7])

#draw trajectory
ax1.plot3D(gut[0::8],gvt[0::8],grt[0::8], label="particle 1")
ax1.plot3D(gut[1::8],gvt[1::8],grt[1::8], label="particle 2")
ax1.plot3D(gut[2::8],gvt[2::8],grt[2::8], label="particle 3")
ax1.plot3D(gut[3::8],gvt[3::8],grt[3::8], label="particle 4")
ax1.plot3D(gut[4::8],gvt[4::8],grt[4::8], label="particle 5")
ax1.plot3D(gut[5::8],gvt[5::8],grt[5::8], label="particle 6")
ax1.plot3D(gut[6::8],gvt[6::8],grt[6::8], label="particle 7")
ax1.plot3D(gut[7::8],gvt[7::8],grt[7::8], label="particle 8")
ax1.legend(loc= 'lower left')

ax1.plot_surface(part1x, part1y, part1z, color="b")
ax1.plot_surface(part2x, part2y, part2z, color="b")
ax1.plot_surface(part3x, part3y, part3z, color="b")
ax1.plot_surface(part4x, part4y, part4z, color="b")
ax1.plot_surface(part5x, part5y, part5z, color="b")
ax1.plot_surface(part6x, part6y, part6z, color="b")
ax1.plot_surface(part7x, part7y, part7z, color="b")
ax1.plot_surface(part8x, part8y, part8z, color="b")

plt.show()

# ------------------------------------------------------------------
### Uncomment to just generate gif of trajectory of the particle ###
# ------------------------------------------------------------------

# #Generate gif
# # create empty lists for the x and y data
# x1 = []
# y1 = []
# z1 = []
# x2 = []
# y2 = []
# z2 = []

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

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::2][0]),cosh(gat[0::2][0])*sinh(gbt[0::2][0]),cosh(gat[0::2][0])*cosh(gbt[0::2][0])*sinh(ggt[0::2][0]),cosh(gat[0::2][0])*cosh(gbt[0::2][0])*cosh(ggt[0::2][0])]),particles[0][7])
# part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::2][1]),cosh(gat[1::2][1])*sinh(gbt[1::2][1]),cosh(gat[1::2][1])*cosh(gbt[1::2][1])*sinh(ggt[1::2][1]),cosh(gat[1::2][1])*cosh(gbt[1::2][1])*cosh(ggt[1::2][1])]),particles[1][7])
# ball1=[ax1.plot_surface(part1x,part1y,part1z, color="b")]
# ball2=[ax1.plot_surface(part2x,part2y,part2z, color="b")]

# # animation function. This is called sequentially
# frames=50
# def animate(i):
#     ax1.plot3D(gut[0::2][:int(len(timearr)*i/frames)],gvt[0::2][:int(len(timearr)*i/frames)],grt[0::2][:int(len(timearr)*i/frames)])
#     ax1.plot3D(gut[1::2][:int(len(timearr)*i/frames)],gvt[1::2][:int(len(timearr)*i/frames)],grt[1::2][:int(len(timearr)*i/frames)])
#     part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*sinh(gbt[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*cosh(gbt[0::2][int(len(timearr)*i/frames)])*sinh(ggt[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*cosh(gbt[0::2][int(len(timearr)*i/frames)])*cosh(ggt[0::2][int(len(timearr)*i/frames)])]),particles[0][7])
#     part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*sinh(gbt[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*cosh(gbt[1::2][int(len(timearr)*i/frames)])*sinh(ggt[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*cosh(gbt[1::2][int(len(timearr)*i/frames)])*cosh(ggt[1::2][int(len(timearr)*i/frames)])]),particles[1][7])
#     ball1[0].remove()
#     ball1[0]=ax1.plot_surface(part1x,part1y,part1z, color="b")
#     ball2[0].remove()
#     ball2[0]=ax1.plot_surface(part2x,part2y,part2z, color="b")

# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# # call the animator. blit=True means only re-draw the parts that 
# # have changed.
# anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

# anim.save('./h3spring_test.gif', writer='imagemagick')

