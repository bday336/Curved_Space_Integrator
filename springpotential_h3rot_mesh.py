from asyncio import new_event_loop
from symint_bank import imph3sprot3,imph3sprot3_condense_econ, imph3sprot_mesh
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
# and connections acting as the springs for the edges. The generic structure is
# given to be taken in the form of
# [ 
#   array of vertex position (x,y,z) , 
#   array of tuples of vertex indices forming egdes (1,2) ,
#   array of 3-tuple of vertex indices forming facets (1,2,3)
#  ]
# Needs to take care to scale it an appropriate size since the euclidean coordinates
# will be interpreted as coordinates in Poincare disk (probably not best choice, but
# that is what we will use for now). (Rotational parameterization for position and
# velocity values). Preset meshes can be loaded in from the mesh_bank file.

# The initial velocities of the vertices will be given through the killing field 
# initializer which describes a general loxodromic killing field along the x-axis.
# This will form the second array needed to initialize the problem. With the third 
# array containing the data concerning the mass of each vertex, and values for stiffness
# and rest length of the spring/edges between each vertex.

# Manually generate mesh data
mesh=equilateral_triangle_onaxis_mesh(1)

# Format mesh for use in the solver. Mainly about ordering the edge connection
# array to make it easier to calculate the jacobian in solver. Have the lower
# value vertex index on the left of the tuple.
for a in mesh[1]:
    a.sort()
mesh[1].sort()

# Extract mesh information for indexing moving forward
vert_num=len(mesh[0])
sp_num=len(mesh[1])

# List of connecting vertices for each vertex. This is used in the velocity
# update constraints in the solver.
# (index of stoke vertex in mesh[0], index of spring for stoke in sparr)
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
    velocities.append(initial_con(e,1.,.0).tolist())
    masses.append(1.)

# Generate parameter array (we will implicitly take all the vertices to have
# mass of 1 so not included in this array. Can add later if needed). The rest
# length of the springs for the edges are calculated from the mesh vertices
# separation values.
# (spring constant k (all set to 1 here), rest length l_eq)
sparr=[]
for f in mesh[1]:
    sparr.append([
        1.,
        h3dist(convertpos_rot2hyph3(mesh[0][f[0]]),convertpos_rot2hyph3(mesh[0][f[1]]))
        ])

# Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT
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

# Distance between vertices and potential energy
pot_energy=0.
for h in range(sp_num):
    dist.append(sparr[h][1])
    pot_energy+=.5*sparr[h][0]*( sparr[h][1] - sparr[h][1] )**2.

# Energy of system
energy_dat.append(kin_energy+pot_energy)

# Copy of mesh (position of vertices will be changed with the integration)
mesh_copy=cp.deepcopy(mesh)

# Make container for the updating velocities
velocities_copy=cp.deepcopy(velocities)

# Numerical Integration step
step_data=[
	imph3sprot_mesh(mesh_copy, velocities_copy, delT, masses, sparr, energy_dat, conn_list)
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
for j in range(sp_num):
    dist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][1]])))

# Energy of system
energy_dat.append(step_data[0][-1])

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):

    step_data=array([
        imph3sprot_mesh(new_mesh, new_velocities, delT, masses, sparr, energy_dat[-1], conn_list)
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
    for l in range(sp_num):
        dist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][l][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][l][1]])))

    # Energy of system
    energy_dat.append(step_data[0][-1])

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
np.savetxt("dist6_data.csv",dist)
# np.savetxt("dist12_data.csv",dist12) 
# np.savetxt("dist13_data.csv",dist13) 
# np.savetxt("dist23_data.csv",dist23)    	     		

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

# # Particle Plot data
# part_plot=[]
# for a in range(vert_num):
#     part_plot.append(hypercirch3(array([sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*cos(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*sin(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*cos(gbt[-(vert_num-a)]),cosh(gat[-(vert_num-a)])]),.1))
#     ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")

# #draw trajectory
# for b in range(vert_num):
#     ax1.plot3D(gut[b::vert_num],gvt[b::vert_num],grt[b::vert_num], label="particle []".format(b))
# ax1.legend(loc= 'lower left')

# plt.show()

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

