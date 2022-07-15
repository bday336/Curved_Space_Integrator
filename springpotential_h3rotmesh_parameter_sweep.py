from symint_bank import imph3sprot3, imph3sprot3_condense_econ, imph3sprot_mesh
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convertpos_hyp2roth3,convertpos_rot2hyph3,initial_con
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
import copy as cp
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi


# These are the parameter sweep loops over initial velocity, equilibrium length, spring stiffness
tot_max_dist_arr=[]     # Array of maximum distance of masses
tot_max_disp_arr=[]     # Array of maximum displacement of masses from equilibrium length of spring
tot_max_elong_arr=[]    # Array of maximum elongation percent
for v in arange(1.,7.,1.): # max for each was set to 11 (1,11,1)    Stable regime (1,7,1)
    for l in arange(1.,7.,1.): # 10 entries (1,11,1)                Stable regime (1,7,1)
        for k in arange(0,7.,1.): # 11 entries (0,11,1)             Stable regime (0,7,1)
            print("On v={}, l={}, and k={}".format(v,l,k))

            # Manually generate mesh data
            mesh=[ # Dumbbell mesh
                [ # Vertex position 
                    [l/2.,np.pi/2.,1.*np.pi/2.],      #particle 1
                    [l/2.,np.pi/2.,3.*np.pi/2.]       #particle 2
                ]
                ,
                [ # Edge connection given by vertex indices
                    [0,1]
                ]
                ,
                [ # Face given by the vertex indices (here for completeness)
                    
                ]
            ]

            # mesh=[ # Equilateral triangle mesh
            #     # [ # Vertex position 
            #     #     [.5,np.pi/2.,0.*2.*np.pi/3.],      #particle 1
            #     #     [.5,np.pi/2.,1.*2.*np.pi/3.],      #particle 2
            #     #     [.5,np.pi/2.,2.*2.*np.pi/3.]       #particle 3
            #     # ]
            #     [ # Vertex position (Analytic check)
            #         [arccosh(cosh(l)/cosh(l/2.)),np.pi/2.,0.],      #particle 1
            #         [l/2.,np.pi/2.,1.*np.pi/2.],                    #particle 2
            #         [l/2.,np.pi/2.,3.*np.pi/2.]                     #particle 3
            #     ]
            #     ,
            #     [ # Edge connection given by vertex indices
            #         [0,1],
            #         [0,2],
            #         [1,2]
            #     ]
            #     ,
            #     [ # Face given by the vertex indices (here for completeness)
            #         [0,1,2]
            #     ]
            # ]

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
                velocities.append(initial_con(e,v,.0).tolist())
                masses.append(1.)

            # Generate parameter array (we will implicitly take all the vertices to have
            # mass of 1 so not included in this array. Can add later if needed). The rest
            # length of the springs for the edges are calculated from the mesh vertices
            # separation values.
            # (spring constant k (all set to 1 here), rest length l_eq)
            sparr=[]
            for f in mesh[1]:
                sparr.append([
                    k,
                    h3dist(convertpos_rot2hyph3(mesh[0][f[0]]),convertpos_rot2hyph3(mesh[0][f[1]]))
                    ])

            # Intialize the time stepping for the integrator.
            delT=.01
            maxT=5+delT
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
                for kp in range(vert_num):
                    gat.append(a_dat[kp])
                    gbt.append(b_dat[kp])
                    ggt.append(g_dat[kp])
                    new_mesh[0][kp]=[a_dat[kp],b_dat[kp],g_dat[kp]]
                    new_velocities[kp]=[ad_dat[kp],bd_dat[kp],gd_dat[kp]]

                # Distance between vertices
                for lp in range(sp_num):
                    dist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][lp][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][lp][1]])))

                # Energy of system
                energy_dat.append(step_data[0][-1])

                q=q+1
            for t in range(sp_num):
                tot_max_dist_arr.append(list([v,l,k,max(dist[t::sp_num])]))

np.savetxt("mesh2sweep_dist_data.csv",tot_max_dist_arr)
