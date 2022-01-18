from symint_bank import imph3geotrans
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

# Transform from rotational to translational parameterization
def convert_rot2trans(rot_vec):
	return array([arcsinh(sinh(rot_vec[0])*sin(rot_vec[1])*cos(rot_vec[2])), arcsinh(sinh(rot_vec[0])*sin(rot_vec[1])*sin(rot_vec[2])/cosh(arcsinh(sinh(rot_vec[0])*sin(rot_vec[1])*cos(rot_vec[2])))), arctanh(tanh(rot_vec[0])*cos(rot_vec[1]))])

# Potentially can look into make correction for when the distance is less than the combined radii of the spheres
# maybe add a tolerance to help?
# maybe add correction to placement when the midpoint is centered at origin so that the COMs are the same distance from it? <- done
def collision(pos1,pos2,vel1,vel2,mass1,mass2):
    pos1hyp=array([
        sinh(pos1[0]),
        cosh(pos1[0])*sinh(pos1[1]),
        cosh(pos1[0])*cosh(pos1[1])*sinh(pos1[2]),
        cosh(pos1[0])*cosh(pos1[1])*cosh(pos1[2])])
    pos2hyp=array([
        sinh(pos2[0]),
        cosh(pos2[0])*sinh(pos2[1]),
        cosh(pos2[0])*cosh(pos2[1])*sinh(pos2[2]),
        cosh(pos2[0])*cosh(pos2[1])*cosh(pos2[2])])
    vel1hyp=array([
        vel1[0]*cosh(pos1[0]),
        vel1[0]*sinh(pos1[0])*sinh(pos1[1])+vel1[1]*cosh(pos1[0])*cosh(pos1[1]),
        vel1[0]*sinh(pos1[0])*cosh(pos1[1])*sinh(pos1[2])+vel1[1]*cosh(pos1[0])*sinh(pos1[1])*sinh(pos1[2])+vel1[2]*cosh(pos1[0])*cosh(pos1[1])*cosh(pos1[2]),
        vel1[0]*sinh(pos1[0])*cosh(pos1[1])*cosh(pos1[2])+vel1[1]*cosh(pos1[0])*sinh(pos1[1])*cosh(pos1[2])+vel1[2]*cosh(pos1[0])*cosh(pos1[1])*sinh(pos1[2])])
    vel2hyp=array([
        vel2[0]*cosh(pos2[0]),
        vel2[0]*sinh(pos2[0])*sinh(pos2[1])+vel2[1]*cosh(pos2[0])*cosh(pos2[1]),
        vel2[0]*sinh(pos2[0])*cosh(pos2[1])*sinh(pos2[2])+vel2[1]*cosh(pos2[0])*sinh(pos2[1])*sinh(pos2[2])+vel2[2]*cosh(pos2[0])*cosh(pos2[1])*cosh(pos2[2]),
        vel2[0]*sinh(pos2[0])*cosh(pos2[1])*cosh(pos2[2])+vel2[1]*cosh(pos2[0])*sinh(pos2[1])*cosh(pos2[2])+vel2[2]*cosh(pos2[0])*cosh(pos2[1])*sinh(pos2[2])])
    trans12xyp1= rotxh3(-arctan2(pos1hyp[2],pos1hyp[1])) @ pos1hyp
    trans12xyp2= rotxh3(-arctan2(pos1hyp[2],pos1hyp[1])) @ pos2hyp
    trans12xyv1= rotxh3(-arctan2(pos1hyp[2],pos1hyp[1])) @ vel1hyp
    trans12xyv2= rotxh3(-arctan2(pos1hyp[2],pos1hyp[1])) @ vel2hyp
    trans12op1= rotxh3(arctan2(pos1hyp[2],pos1hyp[1])) @ rotzh3(arctan2(trans12xyp1[1],trans12xyp1[0])) @ boostxh3(-arccosh(trans12xyp1[3])) @ rotzh3(-arctan2(trans12xyp1[1],trans12xyp1[0])) @ trans12xyp1
    trans12op2= rotxh3(arctan2(pos1hyp[2],pos1hyp[1])) @ rotzh3(arctan2(trans12xyp1[1],trans12xyp1[0])) @ boostxh3(-arccosh(trans12xyp1[3])) @ rotzh3(-arctan2(trans12xyp1[1],trans12xyp1[0])) @ trans12xyp2
    trans12ov1= rotxh3(arctan2(pos1hyp[2],pos1hyp[1])) @ rotzh3(arctan2(trans12xyp1[1],trans12xyp1[0])) @ boostxh3(-arccosh(trans12xyp1[3])) @ rotzh3(-arctan2(trans12xyp1[1],trans12xyp1[0])) @ trans12xyv1
    trans12ov2= rotxh3(arctan2(pos1hyp[2],pos1hyp[1])) @ rotzh3(arctan2(trans12xyp1[1],trans12xyp1[0])) @ boostxh3(-arccosh(trans12xyp1[3])) @ rotzh3(-arctan2(trans12xyp1[1],trans12xyp1[0])) @ trans12xyv2
    trans22xyp1= rotxh3(-arctan2(trans12op2[2],trans12op2[1])) @ trans12op1
    trans22xyp2= rotxh3(-arctan2(trans12op2[2],trans12op2[1])) @ trans12op2
    trans22xyv1= rotxh3(-arctan2(trans12op2[2],trans12op2[1])) @ trans12ov1
    trans22xyv2= rotxh3(-arctan2(trans12op2[2],trans12op2[1])) @ trans12ov2
    trans22xp1= rotzh3(-arctan2(trans22xyp2[1],trans22xyp2[0])) @ trans22xyp1
    trans22xp2= rotzh3(-arctan2(trans22xyp2[1],trans22xyp2[0])) @ trans22xyp2
    trans22xv1= rotzh3(-arctan2(trans22xyp2[1],trans22xyp2[0])) @ trans22xyv1
    trans22xv2= rotzh3(-arctan2(trans22xyp2[1],trans22xyp2[0])) @ trans22xyv2
    transm2op1= boostxh3(-.5*dist) @ trans22xp1
    transm2op2= boostxh3(-.5*dist) @ trans22xp2
    transm2ov1= boostxh3(-.5*dist) @ trans22xv1
    transm2ov2= boostxh3(-.5*dist) @ trans22xv2
    transv2ov1= boostxh3(arccosh(transm2op1[3])) @ transm2ov1
    transv2ov2= boostxh3(-arccosh(transm2op1[3])) @ transm2ov2
    postcolv1= array([transv2ov1[0]+2.*mass2/(mass1+mass2)*(transv2ov2[0]-transv2ov1[0]),transv2ov1[1],transv2ov1[2],0])
    postcolv2= array([transv2ov2[0]+2.*mass1/(mass1+mass2)*(transv2ov1[0]-transv2ov2[0]),transv2ov2[1],transv2ov2[2],0])
    newvel1= rotxh3(arctan2(pos1hyp[2],pos1hyp[1])) @ rotzh3(arctan2(trans12xyp1[1],trans12xyp1[0])) @ boostxh3(arccosh(trans12xyp1[3])) @ rotzh3(-arctan2(trans12xyp1[1],trans12xyp1[0])) @ rotxh3(-arctan2(pos1hyp[2],pos1hyp[1])) @ rotxh3(arctan2(trans12op2[2],trans12op2[1])) @ rotzh3(arctan2(trans22xyp2[1],trans22xyp2[0])) @ boostxh3(.5*dist) @ boostxh3(-arccosh(transm2op1[3])) @ postcolv1
    newvel2= rotxh3(arctan2(pos1hyp[2],pos1hyp[1])) @ rotzh3(arctan2(trans12xyp1[1],trans12xyp1[0])) @ boostxh3(arccosh(trans12xyp1[3])) @ rotzh3(-arctan2(trans12xyp1[1],trans12xyp1[0])) @ rotxh3(-arctan2(pos1hyp[2],pos1hyp[1])) @ rotxh3(arctan2(trans12op2[2],trans12op2[1])) @ rotzh3(arctan2(trans22xyp2[1],trans22xyp2[0])) @ boostxh3(.5*dist) @ boostxh3(-arccosh(transm2op1[3])) @ postcolv2
    newvelpara1= array([newvel1[0]/cosh(pos1[0]),(newvel1[1]-newvel1[0]*tanh(pos1[0])*sinh(pos1[1]))/(cosh(pos1[0]*cosh(pos1[1]))),(newvel1[2]*cosh(pos1[2])-newvel1[3]*sinh(pos1[2]))/(cosh(pos1[0]*cosh(pos1[1])))])
    newvelpara2= array([newvel2[0]/cosh(pos2[0]),(newvel2[1]-newvel2[0]*tanh(pos2[0])*sinh(pos2[1]))/(cosh(pos2[0]*cosh(pos2[1]))),(newvel2[2]*cosh(pos2[2])-newvel2[3]*sinh(pos2[2]))/(cosh(pos2[0]*cosh(pos2[1])))])
    return newvelpara1,newvelpara2

# Plot the hyperbolic 2-sphere
def hypercirch3(center,rad):
    u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
    x = sinh(rad)*sin(u)*cos(v)
    y = sinh(rad)*sin(u)*sin(v)
    z = sinh(rad)*cos(u)
    w = np.full(u.shape,cosh(rad))

    for b in range(u.shape[1]):
        for a in range(u.shape[0]):
            # This method currently does not preserve the orientation of the sphere (need to update if we wanted to have some applied texture)
            testarr=rotxh3(arctan2(center[2],center[1])) @ rotzh3(arctan2((rotxh3(-arctan2(center[2],center[1])) @ center)[1],(rotxh3(-arctan2(center[2],center[1])) @ center)[0])) @ boostxh3(arccosh(center[3])) @ array([x[b,a],y[b,a],z[b,a],w[b,a]])
            x[b,a],y[b,a],z[b,a],w[b,a]=testarr[0],testarr[1],testarr[2],testarr[3]

    return x/(w+1.),y/(w+1.),z/(w+1.)



#Initial position (position in rotational / velocity in chosen parameterization)
#The initial conditions are given as an array of the data for each of the particles being
# initialized. Each element is a particle with each element having 8 components
# { ai , bi , gi , adi , bdi , gdi , mass , radius }

#Initialize the particles in the simulation
particles=array([
    [.75,np.pi/2.+.1,0.,-.2,0.,0.,1.,.2],        #particle 1
    [.75,np.pi/2.-.1,np.pi/2.,0.,-.2,0.,1.,.2]   #particle 2
    ])

#Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT

# Geodesic Trajectories

# Position in translational parameterization
positions = array([convert_rot2trans(particles[0][:3]),convert_rot2trans(particles[1][:3])])
# Velocity given in translational parameterization
velocities = array([particles[0][3:6], particles[1][3:6]])

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []
ggt = []

# Include the intial data
gat=append(gat, array([positions[0][0],positions[1][0]]))
gbt=append(gbt, array([positions[0][1],positions[1][1]]))
ggt=append(ggt, array([positions[0][2],positions[1][2]]))

# Numerical Integration step
step_data=array([
	imph3geotrans(positions[0], positions[0], velocities[0], velocities[0], delT), 
	imph3geotrans(positions[1], positions[1], velocities[1], velocities[1], delT)
	])

# Include the first time step
gat=append(gat, array([step_data[0][0],step_data[1][0]]))
gbt=append(gbt, array([step_data[0][1],step_data[1][1]]))
ggt=append(ggt, array([step_data[0][2],step_data[1][2]]))

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
    # Collision detection check
    dist=h3dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])])
    if dist<=particles[0][-1]+particles[1][-1]:
        print("collided")
        nextpos = array([step_data[0][:3], step_data[1][:3]])
        nextdot= collision(step_data[0][:3], step_data[1][:3],step_data[0][3:6], step_data[1][3:6],particles[0][6],particles[1][6])
    else:
        nextpos = array([step_data[0][:3], step_data[1][:3]])
        nextdot = array([step_data[0][3:6], step_data[1][3:6]])

    step_data=array([
        imph3geotrans(nextpos[0], nextpos[0], nextdot[0], nextdot[0], delT), 
        imph3geotrans(nextpos[1], nextpos[1], nextdot[1], nextdot[1], delT)
        ])

    gat=append(gat, array([step_data[0][0],step_data[1][0]]))
    gbt=append(gbt, array([step_data[0][1],step_data[1][1]]))
    ggt=append(ggt, array([step_data[0][2],step_data[1][2]]))

    q=q+1

# Transform into Poincare disk model for plotting
gut=[]
gvt=[]
grt=[]


for b in range(len(gat)):
    gut=append(gut,sinh(gat[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))
    gvt=append(gvt,cosh(gat[b])*sinh(gbt[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))
    grt=append(grt,cosh(gat[b])*cosh(gbt[b])*sinh(ggt[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))	    	     		

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
ax1.set_ylim3d(-1,1)
ax1.set_zlim3d(-1,1)

part1x,part1y,part1z=hypercirch3(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])]),particles[0][7])
part2x,part2y,part2z=hypercirch3(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])]),particles[1][7])

#draw trajectory
ax1.plot3D(gut[0::2],gvt[0::2],grt[0::2], label="particle 1")
ax1.plot3D(gut[1::2],gvt[1::2],grt[1::2], label="particle 2")
ax1.legend(loc= 'lower left')

ax1.plot_surface(part1x, part1y, part1z, color="b")
ax1.plot_surface(part2x, part2y, part2z, color="b")
	

plt.show()

#Generate gif
# create empty lists for the x and y data
x1 = []
y1 = []
z1 = []
x2 = []
y2 = []
z2 = []

# First set up the figure, the axis, and the plot element we want to animate
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

part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::2][0]),cosh(gat[0::2][0])*sinh(gbt[0::2][0]),cosh(gat[0::2][0])*cosh(gbt[0::2][0])*sinh(ggt[0::2][0]),cosh(gat[0::2][0])*cosh(gbt[0::2][0])*cosh(ggt[0::2][0])]),particles[0][7])
part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::2][1]),cosh(gat[1::2][1])*sinh(gbt[1::2][1]),cosh(gat[1::2][1])*cosh(gbt[1::2][1])*sinh(ggt[1::2][1]),cosh(gat[1::2][1])*cosh(gbt[1::2][1])*cosh(ggt[1::2][1])]),particles[1][7])
ball1=[ax1.plot_surface(part1x,part1y,part1z, color="b")]
ball2=[ax1.plot_surface(part2x,part2y,part2z, color="b")]

# animation function. This is called sequentially
frames=50
def animate(i):
    ax1.plot3D(gut[0::2][:int(len(timearr)*i/frames)],gvt[0::2][:int(len(timearr)*i/frames)],grt[0::2][:int(len(timearr)*i/frames)])
    ax1.plot3D(gut[1::2][:int(len(timearr)*i/frames)],gvt[1::2][:int(len(timearr)*i/frames)],grt[1::2][:int(len(timearr)*i/frames)])
    part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*sinh(gbt[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*cosh(gbt[0::2][int(len(timearr)*i/frames)])*sinh(ggt[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*cosh(gbt[0::2][int(len(timearr)*i/frames)])*cosh(ggt[0::2][int(len(timearr)*i/frames)])]),particles[0][7])
    part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*sinh(gbt[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*cosh(gbt[1::2][int(len(timearr)*i/frames)])*sinh(ggt[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*cosh(gbt[1::2][int(len(timearr)*i/frames)])*cosh(ggt[1::2][int(len(timearr)*i/frames)])]),particles[1][7])
    ball1[0].remove()
    ball1[0]=ax1.plot_surface(part1x,part1y,part1z, color="b")
    ball2[0].remove()
    ball2[0]=ax1.plot_surface(part2x,part2y,part2z, color="b")

# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

# call the animator. blit=True means only re-draw the parts that 
# have changed.
anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

anim.save('./h3collision_test.gif', writer='imagemagick')

