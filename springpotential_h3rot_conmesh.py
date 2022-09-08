from asyncio import new_event_loop
from dbm import dumb
from symint_bank import imph3sprot3,imph3sprot3_condense_econ, imph3sprot_conmesh
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convertpos_hyp2roth3,convertpos_rot2hyph3,initial_con
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
import copy as cp
from mesh_bank import *
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

# Here we take a mesh data structure as an initial condition for the positions
# and connections acting as the rods or springs for the edges. The generic structure is
# given to be taken in the form of
# [ 
#   array of vertex position (x,y,z) , 
#   array of tuples of vertex indices forming egdes (1,2) ,
#   array of 3-tuple of vertex indices forming facets (1,2,3)
#  ]
# Need to take care to scale it an appropriate size since the euclidean coordinates
# will be interpreted as coordinates in Poincare disk (probably not best choice, but
# that is what we will use for now). (Rotational parameterization for position and
# velocity values). Preset meshes can be loaded in from the mesh_bank file. The mesh
# is taken to be in a relaxed state with no internal stress.

# The initial velocities of the vertices will be given through the killing field 
# initializer which describes a general loxodromic killing field along the x-axis.
# This will form the second array needed to initialize the problem. With the third 
# array containing the data concerning the mass of each vertex. The fourth and fifth
# arrays will contain the data for the edges of the mesh and will determine is the edge
# is a rigid rod or a spring. The fourth array is the spring array which contains the
# stiffness and rest length of the spring edge. The fifth array is the rigid rod array
# which contains the fixed length of the edge and the value of the multiplier of that
# edge (think this will be useful for visualizing the stress on that edge of the mesh)

# Parameters for convienence (if uniform throughout mesh)
v, w, k, x = 1., 0., 1., 1.

# Manually generate mesh data
mesh=equilateral_triangle_mesh(x)

# Format mesh for use in the solver. Mainly about ordering the edge connection
# array to make it easier to calculate the jacobian in solver. Have the lower
# value vertex index on the left of the tuple.
for a in mesh[1]:
    a.sort()
mesh[1].sort()

# Extract mesh information for indexing moving forward
vert_num=len(mesh[0])
edge_num=len(mesh[1])

# List of connecting vertices for each vertex. This is used in the velocity
# update constraints in the solver. Entries are grouped by source vertex (ordered
# by the order given in mesh[0]) with
# each sub entry being the vertex it is connected to and the information on the
# edge (spoke) connecting them.
# (index of spoke vertex in mesh[0], index of constraint for spoke in sparr or rigarr)
conn_list=[]
for b in range(vert_num):
    conn_list.append([])
for c in range(vert_num):
    for d in mesh[1]:
        if c in d:
            if d.index(c)==0:
                conn_list[c].append([d[1],mesh[1].index(d)])
            else:
                conn_list[c].append([d[0],mesh[1].index(d)])

# Generate velocity and mass array
velocities=[]
masses=[]
for e in mesh[0]:
    velocities.append(initial_con(e,v,w).tolist())
    masses.append(1.)

# Determine what constraint the mesh edges will have. This is done by making 
# a mask of mesh[1] which can be indexed to see what constraint to apply. Since
# we have two options we can use a simple boolean mask. We will take True to be
# a spring connection and False to be a rigid rod connection.
conn_mask=[]
for mas in range(edge_num):
    conn_mask.append(True) # Set to be all springs for now
rig_count=sum(conn_mask) # Number of rigid constraints

# Generate parameter arrays (we will implicitly take all the vertices to have
# mass of 1 so not included in this array. Can add later if needed). The rest
# length of the springs and fixed length of the rigid rods for the edges are 
# calculated from the mesh vertices separation values.

# For spring parameter array:
# (spring constant k , rest length l_eq)
# For rigid rod parameter array:
# (lagrange multiplier lambda , fixed length l_eq)
# by default lamba is initialized to zero since the rigid is initialized with no
# stress

para_arr=[]
for f in mesh[1]:
    if conn_mask[mesh[1].index(f)]:
        para_arr.append([
            k,
            h3dist(convertpos_rot2hyph3(mesh[0][f[0]]),convertpos_rot2hyph3(mesh[0][f[1]]))
            ])
    else:
        para_arr.append([
            0,
            h3dist(convertpos_rot2hyph3(mesh[0][f[0]]),convertpos_rot2hyph3(mesh[0][f[1]]))
            ])


# Intialize the time stepping for the integrator.
delT=.01
maxT=1+delT
nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []
ggt = []
dist = [] # contains distance data for all spring connections - index in multiples of total springs)
energy_dat=[] 

### Add initial conditions to the containers
# Position and kinetic energy data
kin_energy=0.
for g in range(vert_num):
    gat.append(mesh[0][g][0])
    gbt.append(mesh[0][g][1])
    ggt.append(mesh[0][g][2])
    kin_energy+=.5*masses[g]*( velocities[g][0]**2. + sinh(mesh[0][g][0])**2.*velocities[g][1]**2. + sinh(mesh[0][g][0])**2.*sin(mesh[0][g][1])**2.*velocities[g][2]**2. )

# Distance between vertices and potential energy (Will need to probably include the energy being stretching the rods)
pot_energy=0.
for h in range(edge_num):
    dist.append(para_arr[h][1])
    if conn_mask[h]:
        pot_energy+=.5*para_arr[h][0]*( para_arr[h][1] - para_arr[h][1] )**2.


# Energy of system
energy_dat.append(kin_energy+pot_energy)

# Copy of mesh (position of vertices will be changed with the integration)
mesh_copy=cp.deepcopy(mesh)

# Make container for the updating velocities
velocities_copy=cp.deepcopy(velocities)

# Numerical Integration step
step_data=[
	imph3sprot_conmesh(mesh_copy, velocities_copy, delT, masses, para_arr, conn_list, conn_mask)
	]

# Copy of mesh and velocity array (position of vertices and velocity values will be changed with the integration)
new_mesh=cp.deepcopy(mesh)
new_velocities=cp.deepcopy(velocities)

# Store data from first time step
a_dat=step_data[0][0:3*vert_num:3]
b_dat=step_data[0][1:3*vert_num:3]
g_dat=step_data[0][2:3*vert_num:3]
ad_dat=step_data[0][3*vert_num+0:2*3*vert_num:3]
bd_dat=step_data[0][3*vert_num+1:2*3*vert_num:3]
gd_dat=step_data[0][3*vert_num+2:2*3*vert_num:3]

# Position and velocity data
for i in range(vert_num):
    gat.append(a_dat[i])
    gbt.append(b_dat[i])
    ggt.append(g_dat[i])
    new_mesh[0][i]=[a_dat[i],b_dat[i],g_dat[i]]
    new_velocities[i]=[ad_dat[i],bd_dat[i],gd_dat[i]]

# Distance between vertices
for j in range(edge_num):
    dist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][1]])))

### Add initial conditions to the containers
# Position and kinetic energy data
kin_energy=0.
for g in range(vert_num):
    kin_energy+=.5*masses[g]*( new_velocities[g][0]**2. + sinh(new_mesh[0][g][0])**2.*new_velocities[g][1]**2. + sinh(new_mesh[0][g][0])**2.*sin(new_mesh[0][g][1])**2.*new_velocities[g][2]**2. )

# Distance between vertices and potential energy (Will need to probably include the energy being stretching the rods)
pot_energy=0.
for h in range(edge_num):
    if conn_mask[h]:
        pot_energy+=.5*para_arr[h][0]*( dist[-1*edge_num+h] - para_arr[h][1] )**2.

# Energy of system
energy_dat.append(kin_energy+pot_energy)

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):
    # print(q)

    step_data=array([
        imph3sprot_conmesh(new_mesh, new_velocities, delT, masses, para_arr, conn_list, conn_mask)
        ])

    # Copy of mesh and velocity array (position of vertices and velocity values will be changed with the integration)
    new_mesh=cp.deepcopy(new_mesh)
    new_velocities=cp.deepcopy(new_velocities)

    # Store data from first time step
    a_dat=step_data[0][0:3*vert_num:3]
    b_dat=step_data[0][1:3*vert_num:3]
    g_dat=step_data[0][2:3*vert_num:3]
    ad_dat=step_data[0][3*vert_num+0:2*3*vert_num:3]
    bd_dat=step_data[0][3*vert_num+1:2*3*vert_num:3]
    gd_dat=step_data[0][3*vert_num+2:2*3*vert_num:3]

    # Position and velocity data
    for k in range(vert_num):
        gat.append(a_dat[k])
        gbt.append(b_dat[k])
        ggt.append(g_dat[k])
        new_mesh[0][k]=[a_dat[k],b_dat[k],g_dat[k]]
        new_velocities[k]=[ad_dat[k],bd_dat[k],gd_dat[k]]

    # Distance between vertices
    for l in range(edge_num):
        dist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][l][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][l][1]])))

    ### Add initial conditions to the containers
    # Position and kinetic energy data
    kin_energy=0.
    for g in range(vert_num):
        kin_energy+=.5*masses[g]*( new_velocities[g][0]**2. + sinh(new_mesh[0][g][0])**2.*new_velocities[g][1]**2. + sinh(new_mesh[0][g][0])**2.*sin(new_mesh[0][g][1])**2.*new_velocities[g][2]**2. )

    # Distance between vertices and potential energy
    pot_energy=0.
    for h in range(edge_num):
        if conn_mask[h]:
            pot_energy+=.5*para_arr[h][0]*( dist[-1*edge_num+h] - para_arr[h][1] )**2.

    # Energy of system
    energy_dat.append(kin_energy+pot_energy)

    q=q+1


# Transform into Poincare disk model for plotting
gut=[]
gvt=[]
grt=[]


for i in range(len(gat)):
    gut.append(sinh(gat[i])*sin(gbt[i])*cos(ggt[i])/(cosh(gat[i]) + 1.))
    gvt.append(sinh(gat[i])*sin(gbt[i])*sin(ggt[i])/(cosh(gat[i]) + 1.))
    grt.append(sinh(gat[i])*cos(gbt[i])/(cosh(gat[i]) + 1.))	

# Save the time series data
#np.savetxt("dist1_data.csv",dist)
# np.savetxt("dist12_data.csv",dist12) 
# np.savetxt("dist13_data.csv",dist13) 
# np.savetxt("dist23_data.csv",dist23)    	     		

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

# Particle Plot data
part_plot=[]
for a in range(vert_num):
    part_plot.append(hypercirch3(array([sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*cos(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*sin(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*cos(gbt[-(vert_num-a)]),cosh(gat[-(vert_num-a)])]),.1))
    ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")

#draw trajectory
for b in range(vert_num):
    ax1.plot3D(gut[b::vert_num],gvt[b::vert_num],grt[b::vert_num], label="particle {}".format(b))
ax1.legend(loc= 'lower left')

plt.show()

# --------------------------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------------------------

# # Plot Trajectory with error
# fig = plt.figure(figsize=(14,6))

# ax1=fig.add_subplot(1,3,1,projection='3d')

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

# # Particle Plot data
# part_plot=[]
# for a in range(vert_num):
#     part_plot.append(hypercirch3(array([sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*cos(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*sin(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*cos(gbt[-(vert_num-a)]),cosh(gat[-(vert_num-a)])]),.1))
#     ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")

# #draw trajectory
# for b in range(vert_num):
#     ax1.plot3D(gut[b::vert_num],gvt[b::vert_num],grt[b::vert_num], label="particle []".format(b))
# ax1.legend(loc= 'lower left')

# # ax1.plot_surface(part1x, part1y, part1z, color="b")
# # ax1.plot_surface(part2x, part2y, part2z, color="r")
# # ax1.plot_surface(part3x, part3y, part3z, color="k")

# # Displacement Plot
# # ax2=fig.add_subplot(1,3,2)

# # ax2.plot(timearr,(dist12-spring_arr[0][1]),label="Spring 12 Displacement")
# # ax2.plot(timearr,(dist13-spring_arr[1][1]),label="Spring 13 Displacement")
# # ax2.plot(timearr,(dist23-spring_arr[2][1]),label="Spring 23 Displacement")
# # #ax2.axhline(y=spring[3]+sqrt(2.*1./spring[2]*.5*.5), color='b', linestyle='-')
# # #ax2.axhline(y=((dist.max()-spring[3])+(dist.min()-spring[3]))/2., color='r', linestyle='-')
# # #ax2.set_yscale("log",basey=10)	
# # #ax2.set_ylabel('displacement (m)')
# # ax2.set_xlabel('time (s)')
# # ax2.legend(loc='lower right')

# # Distance Plot
# ax2=fig.add_subplot(1,3,2)

# for c in range(sp_num):
#     ax2.plot(timearr,dist[c::sp_num],label="Spring [] Distance".format(c))

# # ax2.plot(timearr,dist[0::3],label="Spring 12 Distance")
# # ax2.plot(timearr,dist[1::3],label="Spring 13 Distance")
# # ax2.plot(timearr,dist[2::3],label="Spring 23 Distance")
# #ax2.axhline(y=spring[3]+sqrt(2.*1./spring[2]*.5*.5), color='b', linestyle='-')
# #ax2.axhline(y=((dist.max()-spring[3])+(dist.min()-spring[3]))/2., color='r', linestyle='-')
# #ax2.set_yscale("log",basey=10)	
# #ax2.set_ylabel('displacement (m)')
# ax2.set_xlabel('time (s)')
# ax2.legend(loc='lower right')

# # Energy Plot
# ax3=fig.add_subplot(1,3,3)

# ax3.plot(timearr,energy_dat,label="Energy")
# # ax3.set_yscale("log",basey=10)
# ax3.set_xlabel('time (s)')	
# ax3.legend(loc='lower right')

# # Force Plot
# # ax3=fig.add_subplot(1,3,3)

# # ax3.plot(timearr,(dist12-spring_arr[0][1])*spring_arr[0][0],label="Spring 12 Force")
# # ax3.plot(timearr,(dist13-spring_arr[1][1])*spring_arr[1][0],label="Spring 13 Force")
# # ax3.plot(timearr,(dist23-spring_arr[2][1])*spring_arr[2][0],label="Spring 23 Force")
# # # ax3.set_yscale("log",basey=10)
# # ax3.set_xlabel('time (s)')	
# # ax3.legend(loc='lower right')	

# fig.tight_layout()	

# plt.show()

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

# # Particle Plot data
# part_plot=[]
# balls=[]
# for a in range(vert_num):
#     part_plot.append(hypercirch3(array([sinh(gat[a])*sin(gbt[a])*cos(ggt[a]),sinh(gat[a])*sin(gbt[a])*sin(ggt[a]),sinh(gat[a])*cos(gbt[a]),cosh(gat[a])]),.1))
#     balls.append([ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")])

# # #draw trajectory
# # for b in range(vert_num):
# #     ax1.plot3D(gut[b::vert_num],gvt[b::vert_num],grt[b::vert_num], label="particle []".format(b))
# # ax1.legend(loc= 'lower left')

# # part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::3][0])*sin(gbt[0::3][0])*cos(ggt[0::3][0]),sinh(gat[0::3][0])*sin(gbt[0::3][0])*sin(ggt[0::3][0]),sinh(gat[0::3][0])*cos(gbt[0::3][0]),cosh(gat[0::3][0])]),particles[0][7])
# # part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::3][1])*sin(gbt[1::3][1])*cos(ggt[1::3][1]),sinh(gat[1::3][1])*sin(gbt[1::3][1])*sin(ggt[1::3][1]),sinh(gat[1::3][1])*cos(gbt[1::3][1]),cosh(gat[1::3][1])]),particles[1][7])
# # part3x,part3y,part3z=hypercirch3(array([sinh(gat[2::3][2])*sin(gbt[2::3][2])*cos(ggt[2::3][2]),sinh(gat[2::3][2])*sin(gbt[2::3][2])*sin(ggt[2::3][2]),sinh(gat[2::3][2])*cos(gbt[2::3][2]),cosh(gat[2::3][2])]),particles[2][7])
# # ball1=[ax1.plot_surface(part1x,part1y,part1z, color="b")]
# # ball2=[ax1.plot_surface(part2x,part2y,part2z, color="r")]
# # ball3=[ax1.plot_surface(part3x,part3y,part3z, color="k")]

# # animation function. This is called sequentially
# frames=50
# def animate(i):
#     for b in range(vert_num):
#         ax1.plot3D(gut[b::vert_num][:int(len(timearr)*i/frames)],gvt[b::vert_num][:int(len(timearr)*i/frames)],grt[b::vert_num][:int(len(timearr)*i/frames)], label="particle {}".format(b))
#     # ax1.plot3D(gut[0::3][:int(len(timearr)*i/frames)],gvt[0::3][:int(len(timearr)*i/frames)],grt[0::3][:int(len(timearr)*i/frames)])
#     # ax1.plot3D(gut[1::3][:int(len(timearr)*i/frames)],gvt[1::3][:int(len(timearr)*i/frames)],grt[1::3][:int(len(timearr)*i/frames)])
#     # ax1.plot3D(gut[2::3][:int(len(timearr)*i/frames)],gvt[2::3][:int(len(timearr)*i/frames)],grt[2::3][:int(len(timearr)*i/frames)])
#         part_plot.append(hypercirch3(array([sinh(gat[b::vert_num][int(len(timearr)*i/frames)])*sin(gbt[b::vert_num][int(len(timearr)*i/frames)])*cos(ggt[b::vert_num][int(len(timearr)*i/frames)]),sinh(gat[b::vert_num][int(len(timearr)*i/frames)])*sin(gbt[b::vert_num][int(len(timearr)*i/frames)])*sin(ggt[b::vert_num][int(len(timearr)*i/frames)]),sinh(gat[b::vert_num][int(len(timearr)*i/frames)])*cos(gbt[b::vert_num][int(len(timearr)*i/frames)]),cosh(gat[b::vert_num][int(len(timearr)*i/frames)])]),.1))
#         balls[b][0].remove()
#         balls[b][0]=ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")
#     # part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::3][int(len(timearr)*i/frames)])*sin(gbt[0::3][int(len(timearr)*i/frames)])*cos(ggt[0::3][int(len(timearr)*i/frames)]),sinh(gat[0::3][int(len(timearr)*i/frames)])*sin(gbt[0::3][int(len(timearr)*i/frames)])*sin(ggt[0::3][int(len(timearr)*i/frames)]),sinh(gat[0::3][int(len(timearr)*i/frames)])*cos(gbt[0::3][int(len(timearr)*i/frames)]),cosh(gat[0::3][int(len(timearr)*i/frames)])]),particles[0][7])
#     # part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::3][int(len(timearr)*i/frames)])*sin(gbt[1::3][int(len(timearr)*i/frames)])*cos(ggt[1::3][int(len(timearr)*i/frames)]),sinh(gat[1::3][int(len(timearr)*i/frames)])*sin(gbt[1::3][int(len(timearr)*i/frames)])*sin(ggt[1::3][int(len(timearr)*i/frames)]),sinh(gat[1::3][int(len(timearr)*i/frames)])*cos(gbt[1::3][int(len(timearr)*i/frames)]),cosh(gat[1::3][int(len(timearr)*i/frames)])]),particles[1][7])
#     # part3x,part3y,part3z=hypercirch3(array([sinh(gat[2::3][int(len(timearr)*i/frames)])*sin(gbt[2::3][int(len(timearr)*i/frames)])*cos(ggt[2::3][int(len(timearr)*i/frames)]),sinh(gat[2::3][int(len(timearr)*i/frames)])*sin(gbt[2::3][int(len(timearr)*i/frames)])*sin(ggt[2::3][int(len(timearr)*i/frames)]),sinh(gat[2::3][int(len(timearr)*i/frames)])*cos(gbt[2::3][int(len(timearr)*i/frames)]),cosh(gat[2::3][int(len(timearr)*i/frames)])]),particles[2][7])
#     # ball1[0].remove()
#     # ball1[0]=ax1.plot_surface(part1x,part1y,part1z, color="b")
#     # ball2[0].remove()
#     # ball2[0]=ax1.plot_surface(part2x,part2y,part2z, color="r")
#     # ball3[0].remove()
#     # ball3[0]=ax1.plot_surface(part3x,part3y,part3z, color="k")

# # equivalent to rcParams['animation.html'] = 'html5'
# rc('animation', html='html5')

# # call the animator. blit=True means only re-draw the parts that 
# # have changed.
# anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

# anim.save('./h3springmesh_test.gif', writer='imagemagick')

