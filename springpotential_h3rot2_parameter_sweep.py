from symint_bank import imph3sprot2,imph3sprot2_condense
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convertpos_hyp2roth3,convertpos_rot2hyph3,initial_con
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

# These are the parameter sweep loops over initial velocity, equilibrium length, spring stiffness
tot_max_dist_arr=[]     # Array of maximum distance of masses
tot_max_disp_arr=[]     # Array of maximum displacement of masses from equilibrium length of spring
tot_max_elong_arr=[]    # Array of maximum elongation percent
for v in arange(1.,11.,1.): # max for each was set to 11
    for l in arange(1.,11.,1.): # 10 entries
        for k in arange(0,11.,1.): # 11 entries
            print("On v={}, l={}, and k={}".format(v,l,k))

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
            #     np.concatenate((array([.5,np.pi/2.,1.*np.pi/2.]),initial_con(array([.5,np.pi/2.,1.*np.pi/2.]),.0,.0),[1.,.2])),      #particle 1
            #     np.concatenate((array([.5,np.pi/2.,3.*np.pi/2.]),initial_con(array([.5,np.pi/2.,3.*np.pi/2.]),.0,.0),[1.,.2]))       #particle 2
            #     ])

            # Parallel Initial Velocities
            particles=array([
                np.concatenate((array([l/2.,np.pi/2.,1.*np.pi/2.]),initial_con(array([l/2.,np.pi/2.,1.*np.pi/2.]),v,.0),[1.,.2])),      #particle 1
                np.concatenate((array([l/2.,np.pi/2.,3.*np.pi/2.]),initial_con(array([l/2.,np.pi/2.,3.*np.pi/2.]),v,.0),[1.,.2]))       #particle 2
                ])

            # Anti-Parallel Initial Velocities
            # particles=array([
            #     np.concatenate((array([.5,np.pi/2.,1.*np.pi/2.]),initial_con(array([.5,np.pi/2.,1.*np.pi/2.]),-v,.0),[1.,.2])),      #particle 1
            #     np.concatenate((array([.5,np.pi/2.,3.*np.pi/2.]),initial_con(array([.5,np.pi/2.,3.*np.pi/2.]),v,.0),[1.,.2]))       #particle 2
            #     ])

            # Initialize the parameters of what I will consider the
            # sping system. This can be expanded on in the future with
            # something like a class object similar to the particle
            # table. The elements of each spring are given as
            # { spring constant (k) , equilibrium length of spring (l_{eq}) }
            # The value for equilibrium length was calculated on mathematica
            spring_arr=array([
                [k,l]    #spring 12
                ])

            # Intialize the time stepping for the integrator.
            delT=.01
            maxT=5+delT
            nump=maxT/delT
            timearr=np.arange(0,maxT,delT)

            # Positions in rotational parameterization
            positions = array([
                particles[0][:3],
                particles[1][:3]])
            # Velocities in rotational parameterization
            velocities = array([
                particles[0][3:6],
                particles[1][3:6]])
            # Masses for each particle
            masses = array([
                particles[0][6],
                particles[1][6]])

            # Containers for trajectory data
            q = 0
            gat = []
            gbt = []
            ggt = []
            dist12 = []

            # Include the intial data
            gat=append(gat, array([positions[0][0],positions[1][0]]))
            gbt=append(gbt, array([positions[0][1],positions[1][1]]))
            ggt=append(ggt, array([positions[0][2],positions[1][2]]))

            # Distance between masses
            dist12=append(dist12,h3dist([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))

            # Numerical Integration step
            step_data=array([
                imph3sprot2_condense(positions, velocities, delT, masses, spring_arr)
                ])

            # Include the first time step
            gat=append(gat, array([step_data[0][0],step_data[0][3]]))
            gbt=append(gbt, array([step_data[0][1],step_data[0][4]]))
            ggt=append(ggt, array([step_data[0][2],step_data[0][5]]))

            # Distance between masses
            dist12=append(dist12,h3dist([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))

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
                    nextpos = array([step_data[0][0:3], step_data[0][3:6]])
                    nextdot = array([step_data[0][6:9], step_data[0][9:]])

                step_data=array([
                    imph3sprot2_condense(nextpos, nextdot, delT, masses, spring_arr)
                    ])

                gat=append(gat, array([step_data[0][0],step_data[0][3]]))
                gbt=append(gbt, array([step_data[0][1],step_data[0][4]]))
                ggt=append(ggt, array([step_data[0][2],step_data[0][5]]))

                # Distance between masses
                dist12=append(dist12,h3dist([sinh(gat[-2])*sin(gbt[-2])*cos(ggt[-2]),sinh(gat[-2])*sin(gbt[-2])*sin(ggt[-2]),sinh(gat[-2])*cos(gbt[-2]),cosh(gat[-2])],[sinh(gat[-1])*sin(gbt[-1])*cos(ggt[-1]),sinh(gat[-1])*sin(gbt[-1])*sin(ggt[-1]),sinh(gat[-1])*cos(gbt[-1]),cosh(gat[-1])]))

                q=q+1

            tot_max_dist_arr=append(tot_max_dist_arr,list([v,l,k,dist12.max()]))
            tot_max_disp_arr=append(tot_max_disp_arr,list([v,l,k,dist12.max()-l]))
            tot_max_elong_arr=append(tot_max_elong_arr,list([v,l,k,(dist12.max()-l)/l*100.]))

np.savetxt("sweep_dist_data.csv",tot_max_dist_arr)
np.savetxt("sweep_disp_data.csv",tot_max_disp_arr)
np.savetxt("sweep_elong_data.csv",tot_max_elong_arr)


# # Transform into Poincare disk model for plotting
# gut=[]
# gvt=[]
# grt=[]


# for b in range(len(gat)):
#     gut=append(gut,sinh(gat[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))
#     gvt=append(gvt,cosh(gat[b])*sinh(gbt[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))
#     grt=append(grt,cosh(gat[b])*cosh(gbt[b])*sinh(ggt[b])/(cosh(gat[b])*cosh(gbt[b])*cosh(ggt[b]) + 1.))	    	     		

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

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])]),particles[0][7])
# part2x,part2y,part2z=hypercirch3(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])]),particles[1][7])

# #draw trajectory
# ax1.plot3D(gut[0::2],gvt[0::2],grt[0::2], label="particle 1")
# ax1.plot3D(gut[1::2],gvt[1::2],grt[1::2], label="particle 2")
# ax1.legend(loc= 'lower left')

# ax1.plot_surface(part1x, part1y, part1z, color="b")
# ax1.plot_surface(part2x, part2y, part2z, color="b")

# plt.show()

# --------------------------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------------------------

# # Plot Trajectory with error
# fig = plt.figure(figsize=(12,4))

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])]),particles[0][7])

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

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[-2]),cosh(gat[-2])*sinh(gbt[-2]),cosh(gat[-2])*cosh(gbt[-2])*sinh(ggt[-2]),cosh(gat[-2])*cosh(gbt[-2])*cosh(ggt[-2])]),particles[0][7])
# part2x,part2y,part2z=hypercirch3(array([sinh(gat[-1]),cosh(gat[-1])*sinh(gbt[-1]),cosh(gat[-1])*cosh(gbt[-1])*sinh(ggt[-1]),cosh(gat[-1])*cosh(gbt[-1])*cosh(ggt[-1])]),particles[1][7])

# #draw trajectory
# ax1.plot3D(gut[0::2],gvt[0::2],grt[0::2], label="particle 1")
# ax1.plot3D(gut[1::2],gvt[1::2],grt[1::2], label="particle 2")
# ax1.legend(loc= 'lower left')

# ax1.plot_surface(part1x, part1y, part1z, color="b")
# ax1.plot_surface(part2x, part2y, part2z, color="b")

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
