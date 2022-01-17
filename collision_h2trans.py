from symint_bank import imph2geotrans
from function_bank import hyper2poinh2,h2dist,boostxh2,rotzh2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

# Transform from rotational to translational parameterization
def convert_rot2trans(rot_vec):
	return array([arcsinh(sinh(rot_vec[0])*cos(rot_vec[1])), arctanh(tanh(rot_vec[0])*sin(rot_vec[1]))])

# Potentially can look into make correction for when the distance is less than the combined radii of the spheres
# maybe add a tolerance to help?
# maybe add correction to placement when the midpoint is centered at origin so that the COMs are the same distance from it? <- done
def collision(pos1,pos2,vel1,vel2,mass1,mass2):
	pos1hyp=array([sinh(pos1[0]),cosh(pos1[0])*sinh(pos1[1]),cosh(pos1[0])*cosh(pos1[1])])
	pos2hyp=array([sinh(pos2[0]),cosh(pos2[0])*sinh(pos2[1]),cosh(pos2[0])*cosh(pos2[1])])
	vel1hyp=array([vel1[0]*cosh(pos1[0]),vel1[0]*sinh(pos1[0])*sinh(pos1[1])+vel1[1]*cosh(pos1[0])*cosh(pos1[1]),vel1[0]*sinh(pos1[0])*cosh(pos1[1])+vel1[1]*cosh(pos1[0])*sinh(pos1[1])])
	vel2hyp=array([vel2[0]*cosh(pos2[0]),vel2[0]*sinh(pos2[0])*sinh(pos2[1])+vel2[1]*cosh(pos2[0])*cosh(pos2[1]),vel2[0]*sinh(pos2[0])*cosh(pos2[1])+vel2[1]*cosh(pos2[0])*sinh(pos2[1])])
	trans12op1= rotzh2(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2(-arccosh(pos1hyp[2])) @ rotzh2(-arctan2(pos1hyp[1],pos1hyp[0])) @ pos1hyp
	trans12op2= rotzh2(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2(-arccosh(pos1hyp[2])) @ rotzh2(-arctan2(pos1hyp[1],pos1hyp[0])) @ pos2hyp
	trans12ov1= rotzh2(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2(-arccosh(pos1hyp[2])) @ rotzh2(-arctan2(pos1hyp[1],pos1hyp[0])) @ vel1hyp
	trans12ov2= rotzh2(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2(-arccosh(pos1hyp[2])) @ rotzh2(-arctan2(pos1hyp[1],pos1hyp[0])) @ vel2hyp
	trans22xp1= rotzh2(-arctan2(trans12op2[1],trans12op2[0])) @ trans12op1
	trans22xp2= rotzh2(-arctan2(trans12op2[1],trans12op2[0])) @ trans12op2
	trans22xv1= rotzh2(-arctan2(trans12op2[1],trans12op2[0])) @ trans12ov1
	trans22xv2= rotzh2(-arctan2(trans12op2[1],trans12op2[0])) @ trans12ov2
	transm2op1= boostxh2(-.5*dist) @ trans22xp1
	transm2op2= boostxh2(-.5*dist) @ trans22xp2
	transm2ov1= boostxh2(-.5*dist) @ trans22xv1
	transm2ov2= boostxh2(-.5*dist) @ trans22xv2
	transv2ov1= boostxh2(arccosh(transm2op1[2])) @ transm2ov1
	transv2ov2= boostxh2(-arccosh(transm2op1[2])) @ transm2ov2
	postcolv1= array([transv2ov1[0]+2.*mass2/(mass1+mass2)*(transv2ov2[0]-transv2ov1[0]),transv2ov1[1],0])
	postcolv2= array([transv2ov2[0]+2.*mass1/(mass1+mass2)*(transv2ov1[0]-transv2ov2[0]),transv2ov2[1],0])
	newvel1= rotzh2(-arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2(arccosh(pos1hyp[2])) @ rotzh2(arctan2(pos1hyp[1],pos1hyp[0])) @ rotzh2(arctan2(trans12op2[1],trans12op2[0])) @ boostxh2(.5*dist) @  boostxh2(-arccosh(transm2op1[2])) @ postcolv1
	newvel2= rotzh2(-arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2(arccosh(pos1hyp[2])) @ rotzh2(arctan2(pos1hyp[1],pos1hyp[0])) @ rotzh2(arctan2(trans12op2[1],trans12op2[0])) @ boostxh2(.5*dist) @ boostxh2(arccosh(transm2op1[2])) @ postcolv2
	newvelpara1= array([newvel1[0]/cosh(pos1[0]),(newvel1[1]*cosh(pos1[1])-newvel1[2]*sinh(pos1[1]))/cosh(pos1[0])])
	newvelpara2= array([newvel2[0]/cosh(pos2[0]),(newvel2[1]*cosh(pos2[1])-newvel2[2]*sinh(pos2[1]))/cosh(pos2[0])])
	return newvelpara1,newvelpara2

# Plot the hyperbolic sphere
def hypercirc(center,rad):
	theta = np.linspace(0, 2*np.pi, 100)

	xc = sinh(rad)*cos(theta)
	yc = sinh(rad)*sin(theta)
	zc = np.full(len(theta),cosh(rad))

	circ_traj=array([])
	for a in range(len(theta)):
		circ_traj=append(circ_traj,hyper2poinh2(rotzh2(arctan2(center[1],center[0])) @ boostxh2(arccosh(center[2])) @ array([xc[a],yc[a],zc[a]])))
	return circ_traj[0::2],circ_traj[1::2]



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
	# Collision detection check
	dist=h2dist([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])],[sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])])
	if dist<=particles[0][-1]+particles[1][-1]:
		print("collided")
		nextpos = array([step_data[0][:2], step_data[1][:2]])
		nextdot= collision(step_data[0][:2], step_data[1][:2],step_data[0][2:4], step_data[1][2:4],particles[0][4],particles[1][4])
	else:
		nextpos = array([step_data[0][:2], step_data[1][:2]])
		nextdot = array([step_data[0][2:4], step_data[1][2:4]])

	step_data=array([
		imph2geotrans(nextpos[0], nextpos[0], nextdot[0], nextdot[0], delT), 
		imph2geotrans(nextpos[1], nextpos[1], nextdot[1], nextdot[1], delT)
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


#This is the particle trajectories in the Poincare model
    
#Plot
plt.figure(figsize=(5,5))

part1x,part1y=hypercirc(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])]),particles[0][5])
part2x,part2y=hypercirc(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])]),particles[1][5])

plt.plot(xc,yc)
plt.plot(gut[0::2],gvt[0::2],label="particle 1")
plt.plot(gut[1::2],gvt[1::2],label="particle 2")
plt.plot(part1x,part1y)
plt.plot(part2x,part2y)
plt.legend(loc='lower left')	

plt.show()

#Generate gif
# create empty lists for the x and y data
x1 = []
y1 = []
x2 = []
y2 = []

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(figsize=(5,5))

ax.set_xlim([ -1, 1])
ax.set_ylim([-1, 1])

ax.plot(xc,yc)
part1x,part1y=hypercirc(array([sinh(gat[0::2][0]),cosh(gat[0::2][0])*sinh(gbt[0::2][0]),cosh(gat[0::2][0])*cosh(gbt[0::2][0])]),particles[0][5])
part2x,part2y=hypercirc(array([sinh(gat[1::2][1]),cosh(gat[1::2][1])*sinh(gbt[1::2][1]),cosh(gat[1::2][1])*cosh(gbt[1::2][1])]),particles[1][5])
circ1,=ax.plot(part1x,part1y)
circ2,=ax.plot(part2x,part2y)

# animation function. This is called sequentially
frames=50
def animate(i):
	ax.plot(gut[0::2][:int(len(timearr)*i/frames)],gvt[0::2][:int(len(timearr)*i/frames)])
	ax.plot(gut[1::2][:int(len(timearr)*i/frames)],gvt[1::2][:int(len(timearr)*i/frames)])
	part1x,part1y=hypercirc(array([sinh(gat[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*sinh(gbt[0::2][int(len(timearr)*i/frames)]),cosh(gat[0::2][int(len(timearr)*i/frames)])*cosh(gbt[0::2][int(len(timearr)*i/frames)])]),particles[0][5])
	part2x,part2y=hypercirc(array([sinh(gat[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*sinh(gbt[1::2][int(len(timearr)*i/frames)]),cosh(gat[1::2][int(len(timearr)*i/frames)])*cosh(gbt[1::2][int(len(timearr)*i/frames)])]),particles[1][5])
	circ1.set_data([part1x],[part1y])
	circ2.set_data([part2x],[part2y])

# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

# call the animator. blit=True means only re-draw the parts that 
# have changed.
anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

anim.save('./animation.gif', writer='imagemagick')




