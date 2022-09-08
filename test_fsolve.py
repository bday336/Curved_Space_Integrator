import time
from symint_bank import imph3georot
from scipy import optimize
from symint_bank import imph3sprot3,imph3sprot3_condense_econ, imph3sprot_mesh, imph3sprotdis_mesh
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convertpos_hyp2roth3,convertpos_rot2hyph3,initial_con
import numpy as np
import copy as cp
from mesh_bank import *
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi,exp,identity,append,linalg,add

##########################################
### Geodesic Flow Integrator Time Test ###
##########################################

# Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT
nump=maxT/delT
timearr=np.arange(0,maxT,delT)
h=delT
mydatalist=[]
pydatalist=[]

# initdata=(.5,pi/2.,0.,0.,0.,1.)

# def diffsys(root_vals,*initdata):
#     an,bn,gn,adn,bdn,gdn=initdata
#     an1,bn1,gn1,adn1,bdn1,gdn1=root_vals
#     f=np.zeros(6)
#     f[0]=an1 - an - .5*h*(adn + adn1)
#     f[1]=bn1 - bn - .5*h*(bdn + bdn1)
#     f[2]=gn1 - gn - .5*h*(gdn + gdn1)
#     f[3]=adn1 - adn - .5*h*((bdn*bdn + gdn*gdn*sin(bn)**2.)*sinh(an)*cosh(an) + (bdn1*bdn1 + gdn1*gdn1*sin(bn1)**2.)*sinh(an1)*cosh(an1))
#     f[4]=bdn1 - bdn - .5*h*(gdn*gdn*sin(bn)*cos(bn) - 2.*adn*bdn/tanh(an) + gdn1*gdn1*sin(bn1)*cos(bn1) - 2.*adn1*bdn1/tanh(an1))
#     f[5]=gdn1 - gdn - .5*h*(-2.*adn*gdn/tanh(an) - 2.*bdn*gdn/tan(bn) - 2.*adn1*gdn1/tanh(an1) - 2.*bdn1*gdn1/tan(bn1))
#     return f

# def imph3georot(posn, veln, step):
    
#     def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
#         return an1 - an - .5*h*(adn + adn1)

#     def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
#         return bn1 - bn - .5*h*(bdn + bdn1)

#     def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
#         return gn1 - gn - .5*h*(gdn + gdn1)        

#     def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
#         return adn1 - adn - .5*h*((bdn*bdn + gdn*gdn*sin(bn)**2.)*sinh(an)*cosh(an) + (bdn1*bdn1 + gdn1*gdn1*sin(bn1)**2.)*sinh(an1)*cosh(an1))

#     def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
#         return bdn1 - bdn - .5*h*(gdn*gdn*sin(bn)*cos(bn) - 2.*adn*bdn/tanh(an) + gdn1*gdn1*sin(bn1)*cos(bn1) - 2.*adn1*bdn1/tanh(an1))

#     def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
#         return gdn1 - gdn - .5*h*(-2.*adn*gdn/tanh(an) - 2.*bdn*gdn/tan(bn) - 2.*adn1*gdn1/tanh(an1) - 2.*bdn1*gdn1/tan(bn1))

#     def jacobian(an1, bn1, gn1, adn1, bdn1, gdn1, h):
#         return array([
#                 [1.,0.,0.,-.5*h,0.,0.],
#                 [0.,1.,0.,0.,-.5*h,0.],
#                 [0.,0.,1.,0.,0.,-.5*h],
#                 [-.5*h*(bdn1*bdn1+sin(bn1)*sin(bn1)*gdn1*gdn1)*cosh(2.*an1),-.25*h*sinh(2.*an1)*sin(2.*bn1)*gdn1*gdn1,0.,1.,-.5*h*sinh(2.*an1)*bdn1,-.5*h*sinh(2.*an1)*sin(bn1)*sin(bn1)*gdn1],
#                 [-h*adn1*bdn1/(sinh(an1)*sinh(an1)),-.5*h*cos(2.*bn1)*gdn1*gdn1,0.,h*bdn1/tanh(an1),1.+h*adn1/tanh(an1),-.5*h*sin(2.*bn1)*gdn1],
#                 [-h*adn1*gdn1/(sinh(an1)*sinh(an1)),-h*bdn1*gdn1/(sin(bn1)*sin(bn1)),0.,h*gdn1/tanh(an1),h*gdn1/tan(bn1),1.+h*adn1/tanh(an1)+h*bdn1/tan(bn1)]
#             ])     

#     diff1=linalg.solve(jacobian(posn[0], posn[1], posn[2], veln[0], veln[1], veln[2], step),-array([
#         con1(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
#         con2(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
#         con3(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
#         con4(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
#         con5(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
#         con6(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step)
#     ]))
#     val1 = array([posn[0]+diff1[0], posn[1]+diff1[1], posn[2]+diff1[2], veln[0]+diff1[3], veln[1]+diff1[4], veln[2]+diff1[5]])
#     x = 0
#     while(x < 7):       
#         diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], step),-array([
#             con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
#             con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
#             con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
#             con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
#             con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
#             con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step)
#         ]))
#         val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
#         val1 = val2
#         x=x+1
#     return val1 


# # Testing my solver
# start = time.process_time()
 
# # calls the function
# myroot=imph3georot(list(initdata[0:3]),list(initdata[3:6]),h)
# for a in range(int(nump)):
#     myroot=imph3georot(myroot[0:3],myroot[3:6],h)
 
# # records the time in nanoseceonds at this
# # instant of the program
# end = time.process_time()
 
# # printing the execution time by subtracting
# # the time before the function from
# # the time after the function
# print(end-start)

# # Testing python solver
# start = time.process_time()
 
# # calls the function
# root=fsolve(diffsys,list(initdata),args=initdata)
# for b in range(int(nump)):
#     root=fsolve(diffsys,root,args=tuple(root))
 
# # records the time in nanoseceonds at this
# # instant of the program
# end = time.process_time()
 
# # printing the execution time by subtracting
# # the time before the function from
# # the time after the function
# print(end-start)

# print(myroot)
# print(root)

#######################################
## Spring Mesh Integrator Time Test ###
#######################################

# Parameters for convienence (if uniform throughout mesh)
v, w, k, x = 1., 0., 1., 1.

# Manually generate mesh data
mesh=equilateral_triangle_onaxis_mesh(x)

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
    velocities.append(initial_con(e,v,w).tolist())
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

initdata=np.concatenate((np.array(mesh[0]).flatten(),np.array(velocities).flatten()))
def diffsyssp(root_vals,*initdata):
    mesh_data=np.split(np.array(initdata[0:3*vert_num]),vert_num)
    velocity_data=np.split(np.array(initdata[3*vert_num:2*3*vert_num]),vert_num)
    root_mesh_data=np.split(np.array(root_vals[0:3*vert_num]),vert_num)
    root_velocity_data=np.split(np.array(root_vals[3*vert_num:2*3*vert_num]),vert_num)
    
    # Position update constraint for a coordinate
    def con1(base_pos, base_pos_guess, base_vel, base_vel_guess, h):
        an,bn,gn=base_pos
        an1,bn1,gn1=base_pos_guess
        adn,bdn,gdn=base_vel
        adn1,bdn1,gdn1=base_vel_guess
        return an1 - an - .5*h*(adn + adn1)

    # Position update constraint for b coordinate
    def con2(base_pos, base_pos_guess, base_vel, base_vel_guess, h):
        an,bn,gn=base_pos
        an1,bn1,gn1=base_pos_guess
        adn,bdn,gdn=base_vel
        adn1,bdn1,gdn1=base_vel_guess
        return bn1 - bn - .5*h*(bdn + bdn1)

    # Position update constraint for g coordinate
    def con3(base_pos, base_pos_guess, base_vel, base_vel_guess, h): 
        an,bn,gn=base_pos
        an1,bn1,gn1=base_pos_guess
        adn,bdn,gdn=base_vel
        adn1,bdn1,gdn1=base_vel_guess
        return gn1 - gn - .5*h*(gdn + gdn1)        

    # Velocity update constraint for a coordinate
    def con4(base_pos, base_pos_guess, spokes_conn_list, base_vel, base_vel_guess, m1, h, sp_arr, meshn, meshn1):
        
        # Helper function to generate spring force constribution
        def geo_spring_term_ad(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*1.)*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.))


        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        nval=(bd1n*bd1n + gd1n*gd1n*sin(b1n)**2.)*sinh(a1n)*cosh(a1n)
        n1val=(bd1n1*bd1n1 + gd1n1*gd1n1*sin(b1n1)**2.)*sinh(a1n1)*cosh(a1n1)

        # Cycle over the springs connected to base point to include spring force
        for a in spokes_conn_list:
            axn,bxn,gxn=meshn[0][a[0]]
            nval+=geo_spring_term_ad(a1n, b1n, g1n, axn, bxn, gxn, m1, sp_arr[a[1]])
            axn1,bxn1,gxn1=meshn1[0][a[0]]
            n1val+=geo_spring_term_ad(a1n1, b1n1, g1n1, axn1, bxn1, gxn1, m1, sp_arr[a[1]])

        return (ad1n1 - ad1n - .5*h*(nval+n1val))

    # Velocity update constraint for b coordinate
    def con5(base_pos, base_pos_guess, spokes_conn_list, base_vel, base_vel_guess, m1, h, sp_arr, meshn, meshn1):
        
        # Helper function to generate spring force constribution
        def geo_spring_term_bd(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*sinh(a1)*sinh(a1))*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        nval=gd1n*gd1n*sin(b1n)*cos(b1n) - 2.*ad1n*bd1n/tanh(a1n)
        n1val=gd1n1*gd1n1*sin(b1n1)*cos(b1n1) - 2.*ad1n1*bd1n1/tanh(a1n1)

        # Cycle over the springs connected to base point to include spring force
        for a in spokes_conn_list:
            axn,bxn,gxn=meshn[0][a[0]]
            nval+=geo_spring_term_bd(a1n, b1n, g1n, axn, bxn, gxn, m1, sp_arr[a[1]])
            axn1,bxn1,gxn1=meshn1[0][a[0]]
            n1val+=geo_spring_term_bd(a1n1, b1n1, g1n1, axn1, bxn1, gxn1, m1, sp_arr[a[1]])

        return (bd1n1 - bd1n - .5*h*(nval+n1val))

    # Velocity update constraint for g coordinate
    def con6(base_pos, base_pos_guess, spokes_conn_list, base_vel, base_vel_guess, m1, h, sp_arr, meshn, meshn1):
        
        # Helper function to generate spring force constribution
        def geo_spring_term_gd(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*sinh(a1)*sinh(a1)*sin(b1)*sin(b1))*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        nval=-2.*ad1n*gd1n/tanh(a1n) - 2.*bd1n*gd1n/tan(b1n)
        n1val=-2.*ad1n1*gd1n1/tanh(a1n1) - 2.*bd1n1*gd1n1/tan(b1n1)

        # Cycle over the springs connected to base point to include spring force
        for a in spokes_conn_list:
            axn,bxn,gxn=meshn[0][a[0]]
            nval+=geo_spring_term_gd(a1n, b1n, g1n, axn, bxn, gxn, m1, sp_arr[a[1]])
            axn1,bxn1,gxn1=meshn1[0][a[0]]
            n1val+=geo_spring_term_gd(a1n1, b1n1, g1n1, axn1, bxn1, gxn1, m1, sp_arr[a[1]])

        return (gd1n1 - gd1n - .5*h*(nval+n1val))


    f=np.zeros(2*3*vert_num)
    for a in range(vert_num):
        f[6*a+0]=con1(mesh_data[a], root_mesh_data[a], velocity_data[a], root_velocity_data[a], h)
        f[6*a+1]=con2(mesh_data[a], root_mesh_data[a], velocity_data[a], root_velocity_data[a], h)
        f[6*a+2]=con3(mesh_data[a], root_mesh_data[a], velocity_data[a], root_velocity_data[a], h)
        f[6*a+3]=con4(mesh_data[a], root_mesh_data[a], conn_list[a], velocity_data[a], root_velocity_data[a], masses[a], h, sparr, [mesh_data], [root_mesh_data])
        f[6*a+4]=con5(mesh_data[a], root_mesh_data[a], conn_list[a], velocity_data[a], root_velocity_data[a], masses[a], h, sparr, [mesh_data], [root_mesh_data])
        f[6*a+5]=con6(mesh_data[a], root_mesh_data[a], conn_list[a], velocity_data[a], root_velocity_data[a], masses[a], h, sparr, [mesh_data], [root_mesh_data])

    return f










start = time.process_time()

# Distance between vertices
for j in range(sp_num):
    pydatalist.append(h3dist(convertpos_rot2hyph3(initdata[3*mesh[1][j][0]:3*mesh[1][j][0]+3]),convertpos_rot2hyph3(initdata[3*mesh[1][j][1]:3*mesh[1][j][1]+3])))
 
# calls the function
root=optimize.root(diffsyssp,list(initdata),args=tuple(initdata),method='krylov')
print(root.x)
# Distance between vertices
for j in range(sp_num):
    pydatalist.append(h3dist(convertpos_rot2hyph3((root.x)[3*mesh[1][j][0]:3*mesh[1][j][0]+3]),convertpos_rot2hyph3((root.x)[3*mesh[1][j][1]:3*mesh[1][j][1]+3])))
for b in range(int(nump)-2):
    print(b)
    root=optimize.root(diffsyssp,root.x,args=tuple(root.x),method='krylov')
    # print(root.x)
    # Distance between vertices
    for j in range(sp_num):
        pydatalist.append(h3dist(convertpos_rot2hyph3((root.x)[3*mesh[1][j][0]:3*mesh[1][j][0]+3]),convertpos_rot2hyph3((root.x)[3*mesh[1][j][1]:3*mesh[1][j][1]+3])))
 
# records the time in nanoseceonds at this
# instant of the program
end = time.process_time()
 
# printing the execution time by subtracting
# the time before the function from
# the time after the function
print(end-start)










start = time.process_time()

# calls the function
# Copy of mesh (position of vertices will be changed with the integration)
mesh_copy=cp.deepcopy(mesh)

# Make container for the updating velocities
velocities_copy=cp.deepcopy(velocities)

# Distance between vertices
for j in range(sp_num):
    mydatalist.append(h3dist(convertpos_rot2hyph3(mesh_copy[0][mesh_copy[1][j][0]]),convertpos_rot2hyph3(mesh_copy[0][mesh_copy[1][j][1]])))

step_data=array([imph3sprotdis_mesh(mesh_copy, velocities_copy, delT, masses, sparr, 0., conn_list)])
print(step_data[0])

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
    new_mesh[0][i]=[a_dat[i],b_dat[i],g_dat[i]]
    new_velocities[i]=[ad_dat[i],bd_dat[i],gd_dat[i]]

# Distance between vertices
for j in range(sp_num):
    mydatalist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][1]])))

for a in range(int(nump)-2):
    print(a)
    step_data=array([
        imph3sprotdis_mesh(new_mesh, new_velocities, delT, masses, sparr, 0., conn_list)
        ])
    # print(step_data[0])
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
        new_mesh[0][i]=[a_dat[i],b_dat[i],g_dat[i]]
        new_velocities[i]=[ad_dat[i],bd_dat[i],gd_dat[i]]
    # Distance between vertices
    for j in range(sp_num):
        mydatalist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][1]])))
 
# records the time in nanoseceonds at this
# instant of the program
end = time.process_time()
 
# printing the execution time by subtracting
# the time before the function from
# the time after the function
print(end-start)

print(root)
print(step_data[0])

np.savetxt("mydataroottest.csv",mydatalist)
np.savetxt("pydataroottest.csv",pydatalist)


