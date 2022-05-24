import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append

#########

#Function Bank

#########

# Convert from hyperboloid model to poincare model

def hyper2poinh2(point): 
	return array([point[0]/(point[2] + 1.), point[1]/(point[2] + 1.)])

def hyper2poinh3(point): 
	return array([point[0]/(point[3] + 1.), point[1]/(point[3] + 1.), point[2]/(point[3] + 1.)])

def hyper2poinh2e(point): 
	return array([point[0]/(point[3] + 1.), point[1]/(point[3] + 1.), point[2]])

# Convert from rotational parameterization to translational parameterization (position)
# (Probably should add the inverse at some point)

def convert_rot2transh2(rot_vec):
	return array([arcsinh(sinh(rot_vec[0])*cos(rot_vec[1])), arctanh(tanh(rot_vec[0])*sin(rot_vec[1]))])

def convert_rot2transh3(rot_vec):
	return array([arcsinh(sinh(rot_vec[0])*sin(rot_vec[1])*cos(rot_vec[2])), arcsinh(sinh(rot_vec[0])*sin(rot_vec[1])*sin(rot_vec[2])/cosh(arcsinh(sinh(rot_vec[0])*sin(rot_vec[1])*cos(rot_vec[2])))), arctanh(tanh(rot_vec[0])*cos(rot_vec[1]))])

def convert_rot2transh2e(rot_vec):
   return array([arcsinh(sinh(rot_vec[0])*cos(rot_vec[1])), arctanh(tanh(rot_vec[0])*sin(rot_vec[1])), rot_vec[2]])

# Convert from rotational parameterization to translational parameterization (velocity)
# (Probably should add the inverse at some point)

def convert_rot2transh2vel(rot_pos,rot_vel):
   trans_pos=convert_rot2transh2(rot_pos)
   return array([(rot_vel[0]*cosh(rot_pos[0])*cos(rot_pos[1])-rot_vel[1]*sinh(rot_pos[0])*sin(rot_pos[1]))/(cosh(trans_pos[0])), (rot_vel[0]*(cosh(rot_pos[0])*sin(rot_pos[1])*cosh(trans_pos[1])-sinh(rot_pos[0])*sinh(trans_pos[1]))+rot_vel[1]*sinh(rot_pos[0])*cos(rot_pos[1])*cosh(trans_pos[1]))/(cosh(trans_pos[0]))])

# Convert from hyperboloid model to translational parameterization (position)

def convertpos_hyp2paratransh2(hyp_vec):
	return array([arcsinh(hyp_vec[0]), arctanh(hyp_vec[1]/hyp_vec[2])])

# Convert from hyperboloid model to translational parameterization (velocity)

def convertvel_hyp2paratransh2(hyp_pos,hyp_vel):
   parapos=convertpos_hyp2paratransh2(hyp_pos)
   return array([hyp_vel[0]/cosh(parapos[0]), (hyp_vel[1]*cosh(parapos[1])-hyp_vel[2]*sinh(parapos[1]))/cosh(parapos[0])])


# Distance functions

def h2dist(point1,point2):
   return arccosh(-point1[0]*point2[0]-point1[1]*point2[1]+point1[2]*point2[2])

def h3dist(point1,point2):
   return arccosh(-point1[0]*point2[0]-point1[1]*point2[1]-point1[2]*point2[2]+point1[3]*point2[3])

def h2edist(point1,point2):
   return sqrt(arccosh(-point1[0]*point2[0]-point1[1]*point2[1]+point1[3]*point2[3])**2.+(point2[2]-point1[2])**2.)

# Dot products

def h2dot(point1,point2):
   return point1[0]*point2[0]+point1[1]*point2[1]-point1[2]*point2[2]

def h3dot(point1,point2):
   return point1[0]*point2[0]+point1[1]*point2[1]+point1[2]*point2[2]-point1[3]*point2[3]

def h2edot(point1,point2):
   return point1[0]*point2[0]+point1[1]*point2[1]+point1[2]*point2[2]-point1[3]*point2[3]

# Generate sphere for plotting purposes

def hypercirch2(center,rad):
	theta = np.linspace(0, 2*np.pi, 100)

	xc = sinh(rad)*cos(theta)
	yc = sinh(rad)*sin(theta)
	zc = np.full(len(theta),cosh(rad))

	circ_traj=array([])
	for a in range(len(theta)):
		circ_traj=append(circ_traj,hyper2poinh2(rotzh2(arctan2(center[1],center[0])) @ boostxh2(arccosh(center[2])) @ array([xc[a],yc[a],zc[a]])))
	return circ_traj[0::2],circ_traj[1::2]

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

def hypercirch2e(center,rad):
    u, v = np.mgrid[0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
    x = sinh(rad*cos(u))*cos(v)
    y = sinh(rad*cos(u))*sin(v)
    z = rad*sin(u)
    w = cosh(rad*cos(u))

    for b in range(u.shape[1]):
        for a in range(u.shape[0]):
            # This method currently does not preserve the orientation of the sphere (need to update if we wanted to have some applied texture)
            testarr= rotzh2e(arctan2(center[1],center[0])) @ boostxh2e(arccosh(center[3])) @ (array([x[b,a],y[b,a],z[b,a],w[b,a]]) + transzh2e(center[2]))
            x[b,a],y[b,a],z[b,a],w[b,a]=testarr[0],testarr[1],testarr[2],testarr[3]

    return x/(w+1.),y/(w+1.),z

# SO(n,1) matrices (Formatted to act on vector like (x,y,z) or (x,y,z,w) rather than the typical representation that acts on (z,x,y) or (w,x,y,z))
# Only use boost in x direction
# Need all applicable elements of SO(2/3)

# SO(2,1) -> For H2 isometries (x,y,z)

def boostxh2(u): 
	return array([
	   [cosh(u), 0., sinh(u)],
      [0., 1., 0.],
	   [sinh(u), 0., cosh(u)]
		])

def rotzh2(u): 
	return array([
	   [cos(u), -sin(u), 0.],
	   [sin(u), cos(u), 0.],
      [0., 0., 1.]
		])

# SO(3,1) -> For H3 isometries (x,y,z,w)

def boostxh3(u): 
	return array([
	   [cosh(u), 0., 0., sinh(u)],
	   [0., 1., 0., 0.],
	   [0., 0., 1., 0.],
	   [sinh(u), 0., 0., cosh(u)]
		])

def rotxh3(u): 
	return array([
	   [1., 0., 0., 0.],
	   [0., cos(u), -sin(u), 0.],
	   [0., sin(u), cos(u), 0.],
      [0., 0., 0., 1.]
		])

def rotyh3(u): 
	return array([
	   [cos(u), 0., sin(u), 0.],
	   [0., 1., 0., 0.],
	   [-sin(u), 0., cos(u), 0.],
      [0., 0., 0., 1.]
		])      

def rotzh3(u): 
	return array([
	   [cos(u), -sin(u), 0., 0.],
	   [sin(u), cos(u), 0., 0.],
      [0., 0., 1., 0.],
      [0., 0., 0., 1.]
		])

# SO(2) X E(1) -> For H2E isometries (x,y,z,w)

def boostxh2e(u):
	return array([
	   [cosh(u), 0., 0., sinh(u)],
	   [0., 1., 0., 0.],
	   [0., 0., 1., 0.],
	   [sinh(u), 0., 0., cosh(u)]
		])

def transzh2e(u):
	return array([0.,0.,u,0.])

def rotzh2e(u): 
	return array([
	   [cos(u), -sin(u), 0., 0.],
	   [sin(u), cos(u), 0., 0.],
      [0., 0., 1., 0.],
      [0., 0., 0., 1.]
		])      

# Collision algorithms 

# Potentially can look into make correction for when the distance is less than the combined radii of the spheres
# maybe add a tolerance to help?
# maybe add correction to placement when the midpoint is centered at origin so that the COMs are the same distance from it? <- done
def collisionh2(pos1,pos2,vel1,vel2,mass1,mass2,dist):
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

# Potentially can look into make correction for when the distance is less than the combined radii of the spheres
# maybe add a tolerance to help?
# maybe add correction to placement when the midpoint is centered at origin so that the COMs are the same distance from it? <- done
def collisionh3(pos1,pos2,vel1,vel2,mass1,mass2,dist):
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

# Potentially can look into make correction for when the distance is less than the combined radii of the spheres
# maybe add a tolerance to help?
# maybe add correction to placement when the midpoint is centered at origin so that the COMs are the same distance from it? <- done
def collisionh2e(pos1,pos2,vel1,vel2,mass1,mass2,rad1,rad2):
    pos1hyp=array([
        sinh(pos1[0]),
        cosh(pos1[0])*sinh(pos1[1]),
        pos1[2],
        cosh(pos1[0])*cosh(pos1[1])])
    pos2hyp=array([
        sinh(pos2[0]),
        cosh(pos2[0])*sinh(pos2[1]),
        pos2[2],
        cosh(pos2[0])*cosh(pos2[1])])
    vel1hyp=array([
        vel1[0]*cosh(pos1[0]),
        vel1[0]*sinh(pos1[0])*sinh(pos1[1])+vel1[1]*cosh(pos1[0])*cosh(pos1[1]),
        vel1[2],
        vel1[0]*sinh(pos1[0])*cosh(pos1[1])+vel1[1]*cosh(pos1[0])*sinh(pos1[1])])
    vel2hyp=array([
        vel2[0]*cosh(pos2[0]),
        vel2[0]*sinh(pos2[0])*sinh(pos2[1])+vel2[1]*cosh(pos2[0])*cosh(pos2[1]),
        vel2[2],
        vel2[0]*sinh(pos2[0])*cosh(pos2[1])+vel2[1]*cosh(pos2[0])*sinh(pos2[1])])

    # Had to reformat the translation isometry functions. Might be a more elegant way but this should work for now
    # Can just do any necessary z transations first since the other isometries do not effect z coordinate
    trans12op1= rotzh2e(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2e(-arccosh(pos1hyp[3])) @ rotzh2e(-arctan2(pos1hyp[1],pos1hyp[0])) @ (pos1hyp + transzh2e(-pos1hyp[2]))
    trans12op2= rotzh2e(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2e(-arccosh(pos1hyp[3])) @ rotzh2e(-arctan2(pos1hyp[1],pos1hyp[0])) @ (pos2hyp + transzh2e(-pos1hyp[2]))
    trans12ov1= rotzh2e(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2e(-arccosh(pos1hyp[3])) @ rotzh2e(-arctan2(pos1hyp[1],pos1hyp[0])) @ vel1hyp
    trans12ov2= rotzh2e(arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2e(-arccosh(pos1hyp[3])) @ rotzh2e(-arctan2(pos1hyp[1],pos1hyp[0])) @ vel2hyp

    trans22xp1= rotzh2e(-arctan2(trans12op2[1],trans12op2[0])) @ trans12op1
    trans22xp2= rotzh2e(-arctan2(trans12op2[1],trans12op2[0])) @ trans12op2
    trans22xv1= rotzh2e(-arctan2(trans12op2[1],trans12op2[0])) @ trans12ov1
    trans22xv2= rotzh2e(-arctan2(trans12op2[1],trans12op2[0])) @ trans12ov2

    rad_rat=rad1/(rad1+rad2)

    conpos= array([sinh(rad_rat*arcsinh(trans22xp2[0])),0.,rad_rat*trans22xp2[2],cosh(rad_rat*arcsinh(trans22xp2[0]))])
    contang= array([cosh(rad_rat*arcsinh(trans22xp2[0]))*arcsinh(trans22xp2[0]),0.,trans22xp2[2],sinh(rad_rat*arcsinh(trans22xp2[0]))*arcsinh(trans22xp2[0])])

    transc2op1= boostxh2e(-arcsinh(conpos[0])) @ (trans22xp1 + transzh2e(-conpos[2]))
    transc2op2= boostxh2e(-arcsinh(conpos[0])) @ (trans22xp2 + transzh2e(-conpos[2]))
    transc2ov1= boostxh2e(-arcsinh(conpos[0])) @ trans22xv1
    transc2ov2= boostxh2e(-arcsinh(conpos[0])) @ trans22xv2
    transc2oc= boostxh2e(-arcsinh(conpos[0])) @ (conpos + transzh2e(-conpos[2]))
    transc2octang= boostxh2e(-arcsinh(conpos[0])) @ contang

    transv2ov1= boostxh2e(arccosh(transc2op1[3])) @ (transc2ov1 + transzh2e(-transc2op1[2]))
    transv2ov2= boostxh2e(-arccosh(transc2op2[3])) @ (transc2ov2 + transzh2e(-transc2op2[2]))

    projv1= h2edot(transv2ov1,transc2octang)/h2edot(transc2octang,transc2octang)*transc2octang
    projv2= h2edot(transv2ov2,transc2octang)/h2edot(transc2octang,transc2octang)*transc2octang

    normv1=transv2ov1-projv1
    normv2=transv2ov2-projv2

    postcolv1=projv1+2.*mass1/(mass1+mass2)*(projv2-projv1)+normv1
    postcolv2=projv2+2.*mass2/(mass1+mass2)*(projv1-projv2)+normv2

    newvel1= rotzh2e(-arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2e(arccosh(pos1hyp[3])) @ rotzh2e(arctan2(pos1hyp[1],pos1hyp[0])) @ rotzh2e(arctan2(trans12op2[1],trans12op2[0])) @ boostxh2e(arcsinh(conpos[0])) @ boostxh2e(-arccosh(transc2op1[3])) @ postcolv1
    newvel2= rotzh2e(-arctan2(pos1hyp[1],pos1hyp[0])) @ boostxh2e(arccosh(pos1hyp[3])) @ rotzh2e(arctan2(pos1hyp[1],pos1hyp[0])) @ rotzh2e(arctan2(trans12op2[1],trans12op2[0])) @ boostxh2e(arcsinh(conpos[0])) @ boostxh2e(arccosh(transc2op2[3])) @ postcolv2

    #Editted up to here!
    newvelpara1= array([newvel1[0]/cosh(pos1[0]),(newvel1[1]*cosh(pos1[1])-newvel1[3]*sinh(pos1[1]))/cosh(pos1[0]),newvel1[2]])
    newvelpara2= array([newvel2[0]/cosh(pos2[0]),(newvel2[1]*cosh(pos2[1])-newvel2[3]*sinh(pos2[1]))/cosh(pos2[0]),newvel2[2]])
    return newvelpara1,newvelpara2

    








