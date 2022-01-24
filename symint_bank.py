from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arccos,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,linalg
from function_bank import boostxh2e,h2dist,boostxh2,rotzh2,convertpos_hyp2paratransh2,convertvel_hyp2paratransh2

# This script is the repository of all the various solvers I have constructed
# to run the physics engine of the VR Thurston geometry environments. Currently,
# I have updated the solvers for geodesics in H2, H3, and H2xE. The collision
# algorithm is also functional in these spaces for the case of geometric balls
# in each space. There is also the solvers for the PBH space that we constructed
# for an example of an isotropic and nonhomogenous space for the space.

####################################
### Hyperbolic 2-space Geodesics ###
####################################

# This is the workhorse algorithm for testing the solver in H2. In practice we would not
# use the solver but rather use geodesic flow and the isometry group to generate motion
# since it is more calculationally efficient I would say. This more about testing the
# solver itself so that we have a base line on its effectiveness on generating the trajectories
# we can compare with the exact results that are known. So far it seems that it is rather
# effective at generating the correct geodesics. There does seem to be some resonance 
# behavior in the E3 error compare with the exact trajectory, which I still am not sure
# what the origin of it is.

def imph2geotrans(posn, veln, step):

    # Condition the starting position to be at origin. This is done for computation simplification so as to make it more
    # effective in the Javascript Ray-Marching visualization engine. For this simple case of generating geodesics we do not
    # need this complicated infrastructure, I would say, since we have the isometry group, however I will include it for completeness.
    # The physics engine will be leant on more heavy in the presence of potentials/collisions. 
    # 
    # There is also the issue of parallel transport in the case of doing the numerical integration which is not that important for 
    # geodesic flow using the isometries since it implicitly takes care of it. The repositioning the starting position here to the 
    # origin allows us ignore any complicating factors from parallel transport. If we treat the geodesic to be flowing along the 
    # hyperboloid in the y direction (beta direction in the translational parameterization) the velocity vector remain constant in parameter 
    # space (alpha/beta space) in both the directions thus we can treat it as if it is a connected tangent bundle along beta axis. 
    # No extra parallel transport is needed in this case.

    # I have also updated the jacobian to be completely accurate, however given this repositioning of the system it is an
    # unnecessary calculation since all correction terms are zero when flowing along beta axis (alpha=0). I will keep it
    # for completeness, but could be dropped later for efficiency.

    # I have also updated the parameters this takes so that the first guess in the iteration process is the starting position
    # and velocity for simplicity.

    # UPDATE: So it seems that the introduction of my attempt at handling parallel transport has been unsuccessful. When comparing
    # the error between a geodesic flow trajectory and the solver (with and without the parallel tranport added) the error is less
    # without the transport addition. This is the case for both error in hyperbolic and euclidean distance for each point on the
    # trajectory. It is interesting to notice that the odd resonance behavior I have seem in the solver without the transport
    # does not exist when the transport is added. I suppose at this point we can continue to fudge the parallel transport of 
    # vectors in the solver. I feel that this will need to be addressed at somepoint.
    
    def con1(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return adn1 - adn - .5*h*(bdn*bdn*sinh(an)*cosh(an) + bdn1*bdn1*sinh(an1)*cosh(an1))

    def con4(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn*tanh(an) - 2.*adn1*bdn1*tanh(an1))

    def jacobian(an1, bn1, adn1, bdn1, h):
        return array([
                    [1.,0.,-.5*h,0.],
                    [0.,1.,0.,-.5*h],
                    [-.5*h*bdn1*bdn1*cosh(2*an1),0.,1.,-h*sinh(an1)*cosh(an1)*bdn1],
                    [h*adn1*bdn1/(cosh(an1)*cosh(an1)),0.,h*tanh(an1)*bdn1,1.+h*tanh(an1)*adn1]
                ])

    # # Convert to hyperboloid model for isometry operations

    # poshyp=array([
    #     sinh(posn[0]),
    #     cosh(posn[0])*sinh(posn[1]),
    #     cosh(posn[0])*cosh(posn[1])])
    # velhyp=array([
    #     veln[0]*cosh(posn[0]),
    #     veln[0]*sinh(posn[0])*sinh(posn[1])+veln[1]*cosh(posn[0])*cosh(posn[1]),
    #     veln[0]*sinh(posn[0])*cosh(posn[1])+veln[1]*cosh(posn[0])*sinh(posn[1])])
    # # print("Hyperboloid model position and velocity (Before)")
    # # print(poshyp,velhyp)
    # # print("")
    
    # # Reposition at the origin and reorient so that velocity is along the beta axis

    # pos2op=rotzh2(arctan2(poshyp[1],poshyp[0])) @ boostxh2(-arccosh(poshyp[2])) @ rotzh2(-arctan2(poshyp[1],poshyp[0])) @ poshyp
    # pos2ov=rotzh2(arctan2(poshyp[1],poshyp[0])) @ boostxh2(-arccosh(poshyp[2])) @ rotzh2(-arctan2(poshyp[1],poshyp[0])) @ velhyp
    # # print("Hyperboloid model position and velocity (at origin)")
    # # print(pos2op,pos2ov)
    # # print("")
    # pos2b=rotzh2(arctan2(pos2ov[0],pos2ov[1])) @ pos2op
    # vel2b=rotzh2(arctan2(pos2ov[0],pos2ov[1])) @ pos2ov
    # # print("Hyperboloid model position and velocity (at origin oriented along beta)")
    # # print(pos2b,vel2b)
    # # print("")

    # # Convert to translational parameterization for iterator
    # parapos=convertpos_hyp2paratransh2(pos2b)
    # paravel=convertvel_hyp2paratransh2(pos2b,vel2b)
    # # print("Parameterization position and velocity (Oriented along beta)")
    # # print(parapos,paravel)
    # # print("")

    parapos=posn
    paravel=veln

    # print(jacobian(parapos[0], parapos[1], paravel[0], paravel[1], step))

    
    diff1=linalg.solve(jacobian(parapos[0], parapos[1], paravel[0], paravel[1], step),-array([
        con1(parapos[0], parapos[0], parapos[1], parapos[1], paravel[0], paravel[0], paravel[1], paravel[1], step),
        con2(parapos[0], parapos[0], parapos[1], parapos[1], paravel[0], paravel[0], paravel[1], paravel[1], step),
        con3(parapos[0], parapos[0], parapos[1], parapos[1], paravel[0], paravel[0], paravel[1], paravel[1], step),
        con4(parapos[0], parapos[0], parapos[1], parapos[1], paravel[0], paravel[0], paravel[1], paravel[1], step)
    ]))
    val1 = array([parapos[0]+diff1[0], parapos[1]+diff1[1], paravel[0]+diff1[2], paravel[1]+diff1[3]])
    x = 0
    while(x < 7):      
        # print(jacobian(val1[0], val1[1], val1[2], val1[3], step))
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], step),-array([
            con1(parapos[0], val1[0], parapos[1], val1[1], paravel[0], val1[2], paravel[1], val1[3], step),
            con2(parapos[0], val1[0], parapos[1], val1[1], paravel[0], val1[2], paravel[1], val1[3], step),
            con3(parapos[0], val1[0], parapos[1], val1[1], paravel[0], val1[2], paravel[1], val1[3], step),
            con4(parapos[0], val1[0], parapos[1], val1[1], paravel[0], val1[2], paravel[1], val1[3], step)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3]])      
        val1 = val2
        # print("val1")
        # print(val1)
        # print("")
        x=x+1

    # # Convert to hyperboloid model for isometry operations
    # poshypnew=array([sinh(val1[0]),cosh(val1[0])*sinh(val1[1]),cosh(val1[0])*cosh(val1[1])])
    # velhypnew=array([val1[2]*cosh(val1[0]),val1[2]*sinh(val1[0])*sinh(val1[1])+val1[3]*cosh(val1[0])*cosh(val1[1]),val1[2]*sinh(val1[0])*cosh(val1[1])+val1[3]*cosh(val1[0])*sinh(val1[1])])
    # # print("Hyperboloid model position and velocity (after still oriented along beta)")
    # # print(poshypnew,velhypnew)
    # # print("")

    # # Reposition/reorient to start position
    # nextpos= rotzh2(arctan2(poshyp[1],poshyp[0])) @ boostxh2(arccosh(poshyp[2])) @ rotzh2(-arctan2(poshyp[1],poshyp[0])) @ rotzh2(-arctan2(pos2ov[0],pos2ov[1])) @ poshypnew
    # nextvel= rotzh2(arctan2(poshyp[1],poshyp[0])) @ boostxh2(arccosh(poshyp[2])) @ rotzh2(-arctan2(poshyp[1],poshyp[0])) @ rotzh2(-arctan2(pos2ov[0],pos2ov[1])) @ velhypnew
    # # print("Hyperboloid model position and velocity (back at start position)")
    # # print(nextpos,nextvel)
    # # print("")
    # # print("")

    # # Convert to translational parameterization
    # nextparapos=convertpos_hyp2paratransh2(nextpos)
    # nextparavel=convertvel_hyp2paratransh2(nextpos,nextvel)
    
    # return array([nextparapos[0],nextparapos[1],nextparavel[0],nextparavel[1]]) 
    return val1
 
# I haved updated the rotational parameterization method, however I have not
# explicitly tested it. Updated the jacobian generation so that it is more
# in line with how it should be in the algorithm for the solver. Will be
# usefult to copy and paste when testing with central potential in 2D.

def imph2georot(posn, veln, step):
    
    def con1(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return adn1 - adn - .5*h*(bdn*bdn*cosh(an)*sinh(an) + bdn1*bdn1*cosh(an1)*sinh(an1))

    def con4(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn/tanh(an) - 2.*adn1*bdn1/tanh(an1))

    def jacobian(an1, bn1, adn1, bdn1, h):
        return array([
                    [1.,0.,-.5*h,0.],
                    [0.,1.,0.,-.5*h],
                    [-.5*h*bdn1*bdn1*cosh(2*an1),0.,1.,-h*sinh(an1)*cosh(an1)*bdn1],
                    [-h*adn1*bdn1/(sinh(an1)*sinh(an1)),0.,h*bdn1/tanh(an1),1.+h*adn1/tanh(an1)]
                ])
    

    diff1=linalg.solve(jacobian(posn[0], posn[1], veln[0], veln[1], step),-array([
        con1(posn[0], posn[0], posn[1], posn[1], veln[0], veln[0], veln[1], veln[1], step),
        con2(posn[0], posn[0], posn[1], posn[1], veln[0], veln[0], veln[1], veln[1], step),
        con3(posn[0], posn[0], posn[1], posn[1], veln[0], veln[0], veln[1], veln[1], step),
        con4(posn[0], posn[0], posn[1], posn[1], veln[0], veln[0], veln[1], veln[1], step)
    ]))
    val1 = array([posn[0]+diff1[0], posn[1]+diff1[1], veln[0]+diff1[2], veln[1]+diff1[3]])
    x = 0
    while(x < 7):       
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], step),-array([
            con1(posn[0], val1[0], posn[1], val1[1], veln[0], val1[2], veln[1], val1[3], step),
            con2(posn[0], val1[0], posn[1], val1[1], veln[0], val1[2], veln[1], val1[3], step),
            con3(posn[0], val1[0], posn[1], val1[1], veln[0], val1[2], veln[1], val1[3], step),
            con4(posn[0], val1[0], posn[1], val1[1], veln[0], val1[2], veln[1], val1[3], step)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3]])        
        val1 = val2
        x=x+1
    return val1      

# This method is included here for completeness, but it is NOT appropriate to use for the 
# systems we are trying to model since we need an implicit method to ensure that we can 
# preserve time reversal symmetry of the trajectories. So I will leave this here as a warning
# to my future self in case I forget this feature.

def imprk4h2geotrans(posn, veln, step):
    
    def vfunc1(an, bn, adn, bdn):
        return adn

    def vfunc2(an, bn, adn, bdn): 
        return bdn

    def afunc1(an, bn, adn, bdn): 
        return bdn*bdn*sinh(an)*cosh(an)

    def afunc2(an, bn, adn, bdn): 
        return -2.*adn*bdn*tanh(an)

    k11=step*vfunc1(posn[0],posn[1],veln[0],veln[1])
    k21=step*vfunc2(posn[0],posn[1],veln[0],veln[1])
    k31=step*afunc1(posn[0],posn[1],veln[0],veln[1])
    k41=step*afunc2(posn[0],posn[1],veln[0],veln[1]) 

    k12=step*vfunc1(posn[0]+.5*k11,posn[1]+.5*k21,veln[0]+.5*k31,veln[1]+.5*k41)
    k22=step*vfunc2(posn[0]+.5*k11,posn[1]+.5*k21,veln[0]+.5*k31,veln[1]+.5*k41)
    k32=step*afunc1(posn[0]+.5*k11,posn[1]+.5*k21,veln[0]+.5*k31,veln[1]+.5*k41)
    k42=step*afunc2(posn[0]+.5*k11,posn[1]+.5*k21,veln[0]+.5*k31,veln[1]+.5*k41) 

    k13=step*vfunc1(posn[0]+.5*k12,posn[1]+.5*k22,veln[0]+.5*k32,veln[1]+.5*k42)
    k23=step*vfunc2(posn[0]+.5*k12,posn[1]+.5*k22,veln[0]+.5*k32,veln[1]+.5*k42)
    k33=step*afunc1(posn[0]+.5*k12,posn[1]+.5*k22,veln[0]+.5*k32,veln[1]+.5*k42)
    k43=step*afunc2(posn[0]+.5*k12,posn[1]+.5*k22,veln[0]+.5*k32,veln[1]+.5*k42) 

    k14=step*vfunc1(posn[0]+k13,posn[1]+k23,veln[0]+k33,veln[1]+k43)
    k24=step*vfunc2(posn[0]+k13,posn[1]+k23,veln[0]+k33,veln[1]+k43)
    k34=step*afunc1(posn[0]+k13,posn[1]+k23,veln[0]+k33,veln[1]+k43)
    k44=step*afunc2(posn[0]+k13,posn[1]+k23,veln[0]+k33,veln[1]+k43)

    val1=array([posn[0]+(k11+2.*k12+2.*k13+k14)/6.,posn[1]+(k21+2.*k22+2.*k23+k24)/6.,veln[0]+(k31+2.*k32+2.*k33+k34)/6.,veln[1]+(k41+2.*k42+2.*k43+k44)/6.])

    return val1  

####################################
### Hyperbolic 3-space Geodesics ###
####################################

# This is the workhorse solver for H3. See the label for the H2 version for more
# details.

def imph3geotrans(posn, veln, step):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return adn1 - adn - .5*h*((bdn*bdn + gdn*gdn*cosh(bn)**2.)*sinh(an)*cosh(an) + (bdn1*bdn1 + gdn1*gdn1*cosh(bn1)**2.)*sinh(an1)*cosh(an1))

    def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return bdn1 - bdn - .5*h*(gdn*gdn*sinh(bn)*cosh(bn) - 2.*adn*bdn*tanh(an) + gdn1*gdn1*sinh(bn1)*cosh(bn1) - 2.*adn1*bdn1*tanh(an1))

    def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return gdn1 - gdn - .5*h*(-2.*adn*gdn*tanh(an) - 2.*bdn*gdn*tanh(bn) - 2.*adn1*gdn1*tanh(an1) - 2.*bdn1*gdn1*tanh(bn1))  

    def jacobian(an1, bn1, gn1, adn1, bdn1, gdn1, h):
        return array([
                    [1.,0.,0.,-.5*h,0.,0.],
                    [0.,1.,0.,0.,-.5*h,0.],
                    [0.,0.,1.,0.,0.,-.5*h],
                    [-.5*h*(bdn1*bdn1+cosh(bn1)*cosh(bn1)*gdn1*gdn1)*cosh(2.*an1),-.25*h*sinh(2.*an1)*sinh(2.*bn1)*gdn1*gdn1,0.,1.,-.5*h*sinh(2.*an1)*bdn1,-.5*h*sinh(2.*an1)*cosh(bn1)*cosh(bn1)*gdn1],
                    [h*adn1*bdn1/(cosh(an1)*cosh(an1)),-.5*h*cosh(2.*bn1)*gdn1*gdn1,0.,h*tanh(an1)*bdn1,1.+h*tanh(an1)*adn1,-.5*h*sinh(2.*bn1)*gdn1],
                    [h*adn1*gdn1/(cosh(an1)*cosh(an1)),h*bdn1*gdn1/(cosh(bn1)*cosh(bn1)),0.,h*tanh(an1)*gdn1,h*tanh(bn1)*gdn1,1.+h*tanh(an1)*adn1+h*tanh(bn1)*bdn1]
                ])           

    diff1=linalg.solve(jacobian(posn[0], posn[1], posn[2], veln[0], veln[1], veln[2], step),-array([
        con1(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con2(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con3(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con4(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con5(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con6(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step)
    ]))
    val1 = array([posn[0]+diff1[0], posn[1]+diff1[1], posn[2]+diff1[2], veln[0]+diff1[3], veln[1]+diff1[4], veln[2]+diff1[5]])
    x = 0
    while(x < 7):       
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], step),-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
        val1 = val2
        x=x+1
    return val1     

# I haved updated the rotational parameterization method, however I have not
# explicitly tested it. Updated the jacobian generation so that it is more
# in line with how it should be in the algorithm for the solver. Will be
# usefult to copy and paste when testing with central potential in 3D. Make
# sure to check the jacobian function when using to make sure it is correct
# just to double check.

def imph3georot(posn, veln, step):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return adn1 - adn - .5*h*((bdn*bdn + gdn*gdn*sin(bn)**2.)*sinh(an)*cosh(an) + (bdn1*bdn1 + gdn1*gdn1*sin(bn1)**2.)*sinh(an1)*cosh(an1))

    def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return bdn1 - bdn - .5*h*(gdn*gdn*sin(bn)*cos(bn) - 2.*adn*bdn/tanh(an) + gdn1*gdn1*sin(bn1)*cos(bn1) - 2.*adn1*bdn1/tanh(an1))

    def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return gdn1 - gdn - .5*h*(-2.*adn*gdn/tanh(an) - 2.*bdn*gdn/tan(bn) - 2.*adn1*gdn1/tanh(an1) - 2.*bdn1*gdn1/tan(bn1))

    def jacobian(an1, bn1, gn1, adn1, bdn1, gdn1, h):
        return array([
                [1.,0.,0.,-.5*h,0.,0.],
                [0.,1.,0.,0.,-.5*h,0.],
                [0.,0.,1.,0.,0.,-.5*h],
                [-.5*h*(bdn1*bdn1+sin(bn1)*sin(bn1)*gdn1*gdn1)*cosh(2.*an1),-.25*h*sinh(2.*an1)*sin(2.*bn1)*gdn1*gdn1,0.,1.,-.5*h*sinh(2.*an1)*bdn1,-.5*h*sinh(2.*an1)*sin(bn1)*sin(bn1)*gdn1],
                [-h*adn1*bdn1/(sinh(an1)*sinh(an1)),-.5*h*cos(2.*bn1)*gdn1*gdn1,0.,h*bdn1/tanh(an1),1.+h*adn1/tanh(an1),-.5*h*sin(2.*bn1)*gdn1],
                [-h*adn1*gdn1/(sinh(an1)*sinh(an1)),-h*bdn1*gdn1/(sin(bn1)*sin(bn1)),0.,h*gdn1/tanh(an1),h*gdn1/tan(bn1),1.+h*adn1/tanh(an1)+h*bdn1/tan(bn1)]
            ])     

    diff1=linalg.solve(jacobian(posn[0], posn[1], posn[2], veln[0], veln[1], veln[2], step),-array([
        con1(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con2(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con3(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con4(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con5(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step),
        con6(posn[0], posn[0], posn[1], posn[1], posn[2], posn[2], veln[0], veln[0], veln[1], veln[1], veln[2], veln[2], step)
    ]))
    val1 = array([posn[0]+diff1[0], posn[1]+diff1[1], posn[2]+diff1[2], veln[0]+diff1[3], veln[1]+diff1[4], veln[2]+diff1[5]])
    x = 0
    while(x < 7):       
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], step),-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], step)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
        val1 = val2
        x=x+1
    return val1 

##############################
### H2xE 3-space Geodesics ###
##############################

# This is the workhorse solver for H2xE. See the label for the H2 version for more
# details.

def imph2egeotrans(posn, posn1i, veln, veln1i, step):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return adn1 - adn - .5*h*(bdn*bdn*sinh(an)*cosh(an) + bdn1*bdn1*sinh(an1)*cosh(an1))

    def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn*tanh(an) - 2.*adn1*bdn1*tanh(an1))

    def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return gdn1 - gdn - .5*h*(0. + 0.)         
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,1.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h)
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], veln1i[0]+diff1[3], veln1i[1]+diff1[4], veln1i[2]+diff1[5]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,1.]
        ])        
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
        val1 = val2
        x=x+1
    return val1

def symh2xegeorot(posn, posn1i, veln, veln1i, step):
    
    def con1(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h):
        return xn1 - xn - 2.*mu*(xn1 + xn) - .5*h*(adn*cosh(arccosh(wn))*cos(arctan2(yn,xn)) - bdn*sinh(arccosh(wn))*sin(arctan2(yn,xn)) + adn1*cosh(arccosh(wn1))*cos(arctan2(yn1,xn1)) - bdn1*sinh(arccosh(wn1))*sin(arctan2(yn1,xn1)))

    def con2(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return yn1 - yn - 2.*mu*(yn1 + yn) - .5*h*(adn*cosh(arccosh(wn))*sin(arctan2(yn,xn)) + bdn*sinh(arccosh(wn))*cos(arctan2(yn,xn)) + adn1*cosh(arccosh(wn1))*sin(arctan2(yn1,xn1)) + bdn1*sinh(arccosh(wn1))*cos(arctan2(yn1,xn1)))

    def con3(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return zn1 - zn - .5*h*(gdn+gdn1)

    def con4(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return wn1 - wn + 2.*mu*(wn1 + wn) - .5*h*(adn*sinh(arccosh(wn)) + adn1*sinh(arccosh(wn1)))        

    def con5(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return adn1 - adn - .5*h*(bdn*bdn*sinh(arccosh(wn))*cosh(arccosh(wn)) + bdn1*bdn1*sinh(arccosh(wn1))*cosh(arccosh(wn1)))

    def con6(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn/tanh(arccosh(wn))- 2.*adn1*bdn1/tanh(arccosh(wn1)))

    def con7(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return gdn1 - gdn - .5*h*(0. + 0.)        

    def con8(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return xn1*xn1 + yn1*yn1 - wn1*wn1 + 1.
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,-4.*posn[0]],
        [0.,1.,0.,0.,0.,0.,0.,-4.*posn[1]],
        [0.,0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.,0.,4.*posn[3]],                
        [0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.,0.], 
        [0.,0.,0.,0.,0.,0.,1.,0.],               
        [2.*posn[0],2.*posn[1],0.,-2.*posn[3],0.,0.,0.,0.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con7(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con8(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h)        
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], posn1i[3]+diff1[3], veln1i[0]+diff1[4], veln1i[1]+diff1[5], veln1i[2]+diff1[6], mui+diff1[7]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,-4.*val1[0]],
            [0.,1.,0.,0.,0.,0.,0.,-4.*val1[1]],
            [0.,0.,1.,0.,0.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.,0.,4.*val1[3]],                
            [0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,1.,0.,0.], 
            [0.,0.,0.,0.,0.,0.,1.,0.],               
            [2.*val1[0],2.*val1[1],0.,-2.*val1[3],0.,0.,0.,0.]
        ])
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con7(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con8(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),            
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7]])        
        val1 = val2
        x=x+1
    return[val1]

def imph2xegeorot(posn, posn1i, veln, veln1i, step):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return adn1 - adn - .5*h*(bdn*bdn*sinh(an)*cosh(an) + bdn1*bdn1*sinh(an1)*cosh(an1))

    def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn/tanh(an) - 2.*adn1*bdn1/tanh(an1))

    def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return gdn1 - gdn - .5*h*(0. + 0.)         
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,1.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h)
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], veln1i[0]+diff1[3], veln1i[1]+diff1[4], veln1i[2]+diff1[5]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,1.]
        ])        
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
        val1 = val2
        x=x+1
    return[val1]    

# PBH 3-space Geodesics

def sympbhgeo(posn, posn1i, veln, veln1i, step):
    
    def con1(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h):
        return xn1 - xn - 2.*mu*(xn1 + xn) - .5*h*(adn*exp(wn)*sin(arccos(zn/exp(wn)))*cos(arctan2(yn,xn)) + bdn*exp(wn)*cos(arccos(zn/exp(wn)))*cos(arctan2(yn,xn)) - gdn*exp(wn)*sin(arccos(zn/exp(wn)))*sin(arctan2(yn,xn)) + adn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*cos(arctan2(yn1,xn1)) + bdn1*exp(wn1)*cos(arccos(zn1/exp(wn1)))*cos(arctan2(yn1,xn1)) - gdn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*sin(arctan2(yn1,xn1)))

    def con2(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return yn1 - yn - 2.*mu*(yn1 + yn) - .5*h*(adn*exp(wn)*sin(arccos(zn/exp(wn)))*sin(arctan2(yn,xn)) + bdn*exp(wn)*cos(arccos(zn/exp(wn)))*sin(arctan2(yn,xn)) + gdn*exp(wn)*sin(arccos(zn/exp(wn)))*cos(arctan2(yn,xn)) + adn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*sin(arctan2(yn1,xn1)) + bdn1*exp(wn1)*cos(arccos(zn1/exp(wn1)))*sin(arctan2(yn1,xn1)) + gdn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*cos(arctan2(yn1,xn1)))

    def con3(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return zn1 - zn - 2.*mu*(zn1 + zn) - .5*h*(adn*exp(wn)*cos(arccos(zn/exp(wn))) - bdn*exp(wn)*sin(arccos(zn/exp(wn))) + adn1*exp(wn1)*cos(arccos(zn1/exp(wn1))) - bdn1*exp(wn1)*sin(arccos(zn1/exp(wn1))))

    def con4(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return wn1 - wn + 2.*mu*(exp(2.*wn1) + exp(2.*wn)) - .5*h*(adn + adn1)    

    def con5(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return adn1 - adn - .5*h*(exp(2.*wn)/(1.+exp(2.*wn))*(-adn*adn + bdn*bdn + gdn*gdn*sin(arccos(zn/exp(wn)))*sin(arccos(zn/exp(wn)))) + exp(2.*wn1)/(1.+exp(2.*wn1))*(-adn1*adn1 + bdn1*bdn1 + gdn1*gdn1*sin(arccos(zn1/exp(wn1)))*sin(arccos(zn1/exp(wn1)))))

    def con6(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn + sin(arccos(zn/exp(wn)))*cos(arccos(zn/exp(wn)))*gdn*gdn - 2.*adn1*bdn1 + sin(arccos(zn1/exp(wn1)))*cos(arccos(zn1/exp(wn1)))*gdn1*gdn1)

    def con7(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return gdn1 - gdn - .5*h*(-2.*adn*gdn - 2.*bdn*gdn/tan(arccos(zn/exp(wn))) - 2.*adn1*gdn1 - 2.*bdn1*gdn1/tan(arccos(zn1/exp(wn1))))        

    def con8(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return xn1*xn1 + yn1*yn1 + zn1*zn1 - exp(2.*wn1)
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,-4.*posn[0]],
        [0.,1.,0.,0.,0.,0.,0.,-4.*posn[1]],
        [0.,0.,1.,0.,0.,0.,0.,-4.*posn[2]],
        [0.,0.,0.,1.,0.,0.,0.,4.*exp(2.*posn[3])],                
        [0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.,0.], 
        [0.,0.,0.,0.,0.,0.,1.,0.],               
        [2.*posn[0],2.*posn[1],2.*posn[2],-2.*exp(2.*posn[3]),0.,0.,0.,0.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con7(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con8(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h)        
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], posn1i[3]+diff1[3], veln1i[0]+diff1[4], veln1i[1]+diff1[5], veln1i[2]+diff1[6], mui+diff1[7]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,-4.*val1[0]],
            [0.,1.,0.,0.,0.,0.,0.,-4.*val1[1]],
            [0.,0.,1.,0.,0.,0.,0.,-4.*val1[2]],
            [0.,0.,0.,1.,0.,0.,0.,4.*exp(2.*val1[3])],                
            [0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,1.,0.,0.], 
            [0.,0.,0.,0.,0.,0.,1.,0.],               
            [2.*val1[0],2.*val1[1],2.*val1[2],-2.*exp(2.*val1[3]),0.,0.,0.,0.]
        ])
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con7(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con8(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),            
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7]])        
        val1 = val2
        x=x+1
    return[val1]

def imppbhgeo(posn, posn1i, veln, veln1i, step):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return adn1 - adn - .5*h*(exp(2.*an)/(1.+exp(2.*an))*(-adn*adn + bdn*bdn + gdn*gdn*sin(bn)*sin(bn)) + exp(2.*an1)/(1.+exp(2.*an1))*(-adn1*adn1 + bdn1*bdn1 + gdn1*gdn1*sin(bn1)*sin(bn1)))

    def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn + sin(bn)*cos(bn)*gdn*gdn - 2.*adn1*bdn1 + sin(bn1)*cos(bn1)*gdn1*gdn1)

    def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return gdn1 - gdn - .5*h*(-2.*adn*gdn - 2.*bdn*gdn/tan(bn) - 2.*adn1*gdn1 - 2.*bdn1*gdn1/tan(bn1))         
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,1.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h)
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], veln1i[0]+diff1[3], veln1i[1]+diff1[4], veln1i[2]+diff1[5]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,1.]
        ])        
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
        val1 = val2
        x=x+1
    return[val1]         

# Hyperbolic 3-space Central Potential

def symh3cprot(posn, posn1i, veln, veln1i, step, sourcemass, testmass):
    
    def con1(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h):
        return xn1 - xn - 2.*mu*(xn1 + xn) - .5*h*(adn*cosh(arccosh(wn))*sin(arccos(zn/sinh(arccosh(wn))))*cos(arctan2(yn,xn)) + bdn*cos(arccos(zn/sinh(arccosh(wn))))*cos(arctan2(yn,xn)) - gdn*sin(arctan2(yn,xn)) + adn1*cosh(arccosh(wn1))*sin(arccos(zn1/sinh(arccosh(wn1))))*cos(arctan2(yn1,xn1)) + bdn1*cos(arccos(zn1/sinh(arccosh(wn1))))*cos(arctan2(yn1,xn1)) - gdn1*sin(arctan2(yn1,xn1)))

    def con2(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return yn1 - yn - 2.*mu*(yn1 + yn) - .5*h*(adn*cosh(arccosh(wn))*sin(arccos(zn/sinh(arccosh(wn))))*sin(arctan2(yn,xn)) + bdn*cos(arccos(zn/sinh(arccosh(wn))))*sin(arctan2(yn,xn)) + gdn*cos(arctan2(yn,xn)) + adn1*cosh(arccosh(wn1))*sin(arccos(zn1/sinh(arccosh(wn1))))*sin(arctan2(yn1,xn1)) + bdn1*cos(arccos(zn1/sinh(arccosh(wn1))))*sin(arctan2(yn1,xn1)) + gdn1*cos(arctan2(yn1,xn1)))

    def con3(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return zn1 - zn - 2.*mu*(zn1 + zn) - .5*h*(adn*cosh(arccosh(wn))*cos(arccos(zn/sinh(arccosh(wn)))) - bdn*sin(arccos(zn/sinh(arccosh(wn)))) + adn1*cosh(arccosh(wn1))*cos(arccos(zn1/sinh(arccosh(wn1)))) - bdn1*sin(arccos(zn1/sinh(arccosh(wn1)))))

    def con4(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return wn1 - wn + 2.*mu*(wn1 + wn) - .5*h*(adn*sinh(arccosh(wn)) + adn1*sinh(arccosh(wn1)))    

    def con5(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return adn1 - adn - .5*h*((bdn*bdn + gdn*gdn*sin(arccos(zn/sinh(arccosh(wn))))**2.)*sinh(arccosh(wn))*cosh(arccosh(wn)) - sourcemass*testmass/sinh(arccosh(wn))**2. + (bdn1*bdn1 + gdn1*gdn1*sin(arccos(zn1/sinh(arccosh(wn1))))**2.)*sinh(arccosh(wn1))*cosh(arccosh(wn1)) - sourcemass*testmass/sinh(arccosh(wn1))**2.)

    def con6(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return bdn1 - bdn - .5*h*(gdn*gdn*sin(arccos(zn/sinh(arccosh(wn))))*cos(arccos(zn/sinh(arccosh(wn)))) - 2.*adn*bdn/tanh(arccosh(wn)) + gdn1*gdn1*sin(arccos(zn1/sinh(arccosh(wn1))))*cos(arccos(zn1/sinh(arccosh(wn1)))) - 2.*adn1*bdn1/tanh(arccosh(wn1)))

    def con7(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return gdn1 - gdn - .5*h*(-2.*adn*gdn/tanh(arccosh(wn)) - 2.*bdn*gdn/tan(arccos(zn/sinh(arccosh(wn)))) - 2.*adn1*gdn1/tanh(arccosh(wn1)) - 2.*bdn1*gdn1/tan(arccos(zn1/sinh(arccosh(wn1)))))        

    def con8(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return xn1*xn1 + yn1*yn1 + zn1*zn1 - wn1*wn1 + 1.
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,-4.*posn[0]],
        [0.,1.,0.,0.,0.,0.,0.,-4.*posn[1]],
        [0.,0.,1.,0.,0.,0.,0.,-4.*posn[2]],
        [0.,0.,0.,1.,0.,0.,0.,4.*posn[3]],                
        [0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.,0.], 
        [0.,0.,0.,0.,0.,0.,1.,0.],               
        [2.*posn[0],2.*posn[1],2.*posn[2],-2.*posn[3],0.,0.,0.,0.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con7(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con8(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h)        
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], posn1i[3]+diff1[3], veln1i[0]+diff1[4], veln1i[1]+diff1[5], veln1i[2]+diff1[6], mui+diff1[7]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,-4.*val1[0]],
            [0.,1.,0.,0.,0.,0.,0.,-4.*val1[1]],
            [0.,0.,1.,0.,0.,0.,0.,-4.*val1[2]],
            [0.,0.,0.,1.,0.,0.,0.,4.*val1[3]],                
            [0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,1.,0.,0.], 
            [0.,0.,0.,0.,0.,0.,1.,0.],               
            [2.*val1[0],2.*val1[1],2.*val1[2],-2.*val1[3],0.,0.,0.,0.]
        ])
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con7(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con8(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),            
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7]])        
        val1 = val2
        x=x+1
    return[val1]

def imph3cprot(posn, posn1i, veln, veln1i, step, sourcemass, testmass):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return adn1 - adn - .5*h*((bdn*bdn + gdn*gdn*sin(bn)**2.)*sinh(an)*cosh(an) - sourcemass*testmass/sinh(an)**2. + (bdn1*bdn1 + gdn1*gdn1*sin(bn1)**2.)*sinh(an1)*cosh(an1) - sourcemass*testmass/sinh(an1)**2.)

    def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return bdn1 - bdn - .5*h*(gdn*gdn*sin(bn)*cos(bn) - 2.*adn*bdn/tanh(an) + gdn1*gdn1*sin(bn1)*cos(bn1) - 2.*adn1*bdn1/tanh(an1))

    def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return gdn1 - gdn - .5*h*(-2.*adn*gdn/tanh(an) - 2.*bdn*gdn/tan(bn) - 2.*adn1*gdn1/tanh(an1) - 2.*bdn1*gdn1/tan(bn1))         
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,1.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h)
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], veln1i[0]+diff1[3], veln1i[1]+diff1[4], veln1i[2]+diff1[5]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,1.]
        ])        
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
        val1 = val2
        x=x+1
    return[val1]     

# PBH 3-space Central Potential

def sympbhcp(posn, posn1i, veln, veln1i, step, sourcemass, testmass):
    
    def con1(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h):
        return xn1 - xn - 2.*mu*(xn1 + xn) - .5*h*(adn*exp(wn)*sin(arccos(zn/exp(wn)))*cos(arctan2(yn,xn)) + bdn*exp(wn)*cos(arccos(zn/exp(wn)))*cos(arctan2(yn,xn)) - gdn*exp(wn)*sin(arccos(zn/exp(wn)))*sin(arctan2(yn,xn)) + adn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*cos(arctan2(yn1,xn1)) + bdn1*exp(wn1)*cos(arccos(zn1/exp(wn1)))*cos(arctan2(yn1,xn1)) - gdn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*sin(arctan2(yn1,xn1)))

    def con2(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return yn1 - yn - 2.*mu*(yn1 + yn) - .5*h*(adn*exp(wn)*sin(arccos(zn/exp(wn)))*sin(arctan2(yn,xn)) + bdn*exp(wn)*cos(arccos(zn/exp(wn)))*sin(arctan2(yn,xn)) + gdn*exp(wn)*sin(arccos(zn/exp(wn)))*cos(arctan2(yn,xn)) + adn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*sin(arctan2(yn1,xn1)) + bdn1*exp(wn1)*cos(arccos(zn1/exp(wn1)))*sin(arctan2(yn1,xn1)) + gdn1*exp(wn1)*sin(arccos(zn1/exp(wn1)))*cos(arctan2(yn1,xn1)))

    def con3(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return zn1 - zn - 2.*mu*(zn1 + zn) - .5*h*(adn*exp(wn)*cos(arccos(zn/exp(wn))) - bdn*exp(wn)*sin(arccos(zn/exp(wn))) + adn1*exp(wn1)*cos(arccos(zn1/exp(wn1))) - bdn1*exp(wn1)*sin(arccos(zn1/exp(wn1))))

    def con4(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return wn1 - wn + 2.*mu*(exp(2.*wn1) + exp(2.*wn)) - .5*h*(adn + adn1)    

    def con5(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return adn1 - adn - .5*h*(exp(2.*wn)/(1.+exp(2.*wn))*(-adn*adn + bdn*bdn + gdn*gdn*sin(arccos(zn/exp(wn)))*sin(arccos(zn/exp(wn)))) - sourcemass*testmass*exp(-2.*wn)*sqrt(1.+exp(2.*wn)) + exp(2.*wn1)/(1.+exp(2.*wn1))*(-adn1*adn1 + bdn1*bdn1 + gdn1*gdn1*sin(arccos(zn1/exp(wn1)))*sin(arccos(zn1/exp(wn1)))) - sourcemass*testmass*exp(-2.*wn1)*sqrt(1.+exp(2.*wn1)))

    def con6(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn + sin(arccos(zn/exp(wn)))*cos(arccos(zn/exp(wn)))*gdn*gdn - 2.*adn1*bdn1 + sin(arccos(zn1/exp(wn1)))*cos(arccos(zn1/exp(wn1)))*gdn1*gdn1)

    def con7(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return gdn1 - gdn - .5*h*(-2.*adn*gdn - 2.*bdn*gdn/tan(arccos(zn/exp(wn))) - 2.*adn1*gdn1 - 2.*bdn1*gdn1/tan(arccos(zn1/exp(wn1))))        

    def con8(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return xn1*xn1 + yn1*yn1 + zn1*zn1 - exp(2.*wn1)
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,-4.*posn[0]],
        [0.,1.,0.,0.,0.,0.,0.,-4.*posn[1]],
        [0.,0.,1.,0.,0.,0.,0.,-4.*posn[2]],
        [0.,0.,0.,1.,0.,0.,0.,4.*exp(2.*posn[3])],                
        [0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.,0.], 
        [0.,0.,0.,0.,0.,0.,1.,0.],               
        [2.*posn[0],2.*posn[1],2.*posn[2],-2.*exp(2.*posn[3]),0.,0.,0.,0.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con7(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h),
        con8(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], posn[3], posn1i[3], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], mui, h)        
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], posn1i[3]+diff1[3], veln1i[0]+diff1[4], veln1i[1]+diff1[5], veln1i[2]+diff1[6], mui+diff1[7]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,-4.*val1[0]],
            [0.,1.,0.,0.,0.,0.,0.,-4.*val1[1]],
            [0.,0.,1.,0.,0.,0.,0.,-4.*val1[2]],
            [0.,0.,0.,1.,0.,0.,0.,4.*exp(2.*val1[3])],                
            [0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,1.,0.,0.], 
            [0.,0.,0.,0.,0.,0.,1.,0.],               
            [2.*val1[0],2.*val1[1],2.*val1[2],-2.*exp(2.*val1[3]),0.,0.,0.,0.]
        ])
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con7(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),
            con8(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], posn[3], val1[3], veln[0], val1[4], veln[1], val1[5], veln[2], val1[6], val1[7], h),            
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7]])        
        val1 = val2
        x=x+1
    return[val1]

def imppbhcp(posn, posn1i, veln, veln1i, step, sourcemass, testmass):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return adn1 - adn - .5*h*(exp(2.*an)/(1.+exp(2.*an))*(-adn*adn + bdn*bdn + gdn*gdn*sin(bn)*sin(bn))  - sourcemass*testmass*exp(-2.*an)*sqrt(1.+exp(2.*an)) + exp(2.*an1)/(1.+exp(2.*an1))*(-adn1*adn1 + bdn1*bdn1 + gdn1*gdn1*sin(bn1)*sin(bn1)) - sourcemass*testmass*exp(-2.*an1)*sqrt(1.+exp(2.*an1)))

    def con5(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn + sin(bn)*cos(bn)*gdn*gdn - 2.*adn1*bdn1 + sin(bn1)*cos(bn1)*gdn1*gdn1)

    def con6(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return gdn1 - gdn - .5*h*(-2.*adn*gdn - 2.*bdn*gdn/tan(bn) - 2.*adn1*gdn1 - 2.*bdn1*gdn1/tan(bn1))         
    
    h = step
    mui = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,1.]
    ])
    diff1=linalg.solve(mat,-array([
        con1(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con2(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con3(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con4(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con5(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h),
        con6(posn[0], posn1i[0], posn[1], posn1i[1], posn[2], posn1i[2], veln[0], veln1i[0], veln[1], veln1i[1], veln[2], veln1i[2], h)
    ]))
    val1 = array([posn1i[0]+diff1[0], posn1i[1]+diff1[1], posn1i[2]+diff1[2], veln1i[0]+diff1[3], veln1i[1]+diff1[4], veln1i[2]+diff1[5]])
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,1.]
        ])        
        diff2=linalg.solve(mat,-array([
            con1(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con2(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con3(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con4(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con5(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h),
            con6(posn[0], val1[0], posn[1], val1[1], posn[2], val1[2], veln[0], val1[3], veln[1], val1[4], veln[2], val1[5], h)
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5]])        
        val1 = val2
        x=x+1
    return[val1]         

# Hyperbolic 3-space Spring Potential 

def symh3sptrans(pos1n, pos1n1i, pos2n, pos2n1i, vel1n, vel1n1i, vel2n, vel2n1i, step, massvec, sprcon, eqdist):

    def con1(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h):
        return xn1 - xn - 2.*mu*(xn1 + xn) - .5*h*(adn*cosh(arcsinh(xn)) + adn1*cosh(arcsinh(xn1)))

    def con2(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return yn1 - yn - 2.*mu*(yn1 + yn) - .5*h*(adn*sinh(arcsinh(xn))*sinh(arcsinh(yn/(cosh(arcsinh(xn))))) + bdn*cosh(arcsinh(xn))*cosh(arcsinh(yn/(cosh(arcsinh(xn))))) + adn1*sinh(arcsinh(xn1))*sinh(arcsinh(yn1/(cosh(arcsinh(xn1))))) + bdn1*cosh(arcsinh(xn1))*cosh(arcsinh(yn1/(cosh(arcsinh(xn1))))))

    def con3(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return zn1 - zn - 2.*mu*(zn1 + zn) - .5*h*(adn*sinh(arcsinh(xn))*cosh(arcsinh(yn/(cosh(arcsinh(xn)))))*sinh(arctanh(zn/wn)) + bdn*cosh(arcsinh(xn))*sinh(arcsinh(yn/(cosh(arcsinh(xn)))))*sinh(arctanh(zn/wn)) + gdn*cosh(arcsinh(xn))*cosh(arcsinh(yn/(cosh(arcsinh(xn)))))*cosh(arctanh(zn/wn)) + adn1*sinh(arcsinh(xn1))*cosh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*sinh(arctanh(zn1/wn1)) + bdn1*cosh(arcsinh(xn1))*sinh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*sinh(arctanh(zn1/wn1)) + gdn1*cosh(arcsinh(xn1))*cosh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*cosh(arctanh(zn1/wn1)))

    def con4(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return wn1 - wn + 2.*mu*(wn1 + wn) - .5*h*(adn*sinh(arcsinh(xn))*cosh(arcsinh(yn/(cosh(arcsinh(xn)))))*cosh(arctanh(zn/wn)) + bdn*cosh(arcsinh(xn))*sinh(arcsinh(yn/(cosh(arcsinh(xn)))))*cosh(arctanh(zn/wn)) + gdn*cosh(arcsinh(xn))*cosh(arcsinh(yn/(cosh(arcsinh(xn)))))*sinh(arctanh(zn/wn)) + adn1*sinh(arcsinh(xn1))*cosh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*cosh(arctanh(zn1/wn1)) + bdn1*cosh(arcsinh(xn1))*sinh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*cosh(arctanh(zn1/wn1)) + gdn1*cosh(arcsinh(xn1))*cosh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*sinh(arctanh(zn1/wn1)))    

    def con5(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, m1, h, k, xo): 
        return (ad1n1 - ad1n - .5*h*(-(1./m1)*(-0.5*m1*(2.*bd1n*bd1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n)) + 2.*gd1n*gd1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))**2.) + (k*(-xo + 
            arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))*(cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n))*sinh(arcsinh(x1n)) - cosh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x2n))*sinh(arcsinh(x1n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arcsinh(x1n))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))/(sqrt(-1. + 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n)))*sqrt(1. + 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))) - 

            (1./m1)*(-0.5*m1*(2.*bd1n1*bd1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1)) + 2.*gd1n1*gd1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))**2.) + (k*(-xo + 
            arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))*(cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1))*sinh(arcsinh(x1n1)) - cosh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x2n1))*sinh(arcsinh(x1n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arcsinh(x1n1))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))/(sqrt(-1. + 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))*sqrt(1. + 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))))))

    def con6(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(-(1./m1)*(1./cosh(arcsinh(x1n))**2.)*(2.*m1*ad1n*bd1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n)) - gd1n*gd1n*m1*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(x1n))**2. + (k*(-xo + 
            arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))*(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))/(sqrt(-1. + 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n)))*sqrt(1. + 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))) - 

            (1./m1)*(1./cosh(arcsinh(x1n1))**2.)*(2.*m1*ad1n1*bd1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1)) - gd1n1*gd1n1*m1*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(x1n1))**2. + (k*(-xo + 
            arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))*(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))/(sqrt(-1. + 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))*sqrt(1. + 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))))))

    def con7(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, m1, h, k, xo):
        return (gd1n1 - gd1n - .5*h*(-(1./m1)*(1./cosh(arcsinh(x1n))**2.)*(1./cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))**2.)*(2.*gd1n*m1*ad1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))**2. + 2.*gd1n*m1*bd1n*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(x1n))**.2 + (k*(-xo + 
            arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))*(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z2n/w2n))*sinh(arctanh(z1n/w1n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))/(sqrt(-1. + 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n)))*sqrt(1. + 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))) - 

            (1./m1)*(1./cosh(arcsinh(x1n1))**2.)*(1./cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))**2.)*(2.*gd1n1*m1*ad1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))**2. + 2.*gd1n1*m1*bd1n1*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(x1n1))**.2 + (k*(-xo + 
            arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))*(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z2n1/w2n1))*sinh(arctanh(z1n1/w1n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))/(sqrt(-1. + 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))*sqrt(1. + 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))))))     

    def con8(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return xn1*xn1 + yn1*yn1 + zn1*zn1 - wn1*wn1 + 1.
    
    h = step
    mu1i = 10e-10
    mu2i = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[0],0.],
        [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[1],0.],
        [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[2],0.],
        [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.*pos1n[3],0.],
        [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[0]],
        [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[1]],
        [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[2]],
        [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,4.*pos2n[3]],               
        [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],              
        [2.*pos1n[0],2.*pos1n[1],2.*pos1n[2],-2.*pos1n[3],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,2.*pos2n[0],2.*pos2n[1],2.*pos2n[2],-2.*pos2n[3],0.,0.,0.,0.,0.,0.,0.,0.]
    ])
    # print(mat)
    diff1=linalg.solve(mat,-array([
        con1(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con2(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con3(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con4(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con1(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
        con2(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
        con3(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
        con4(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),

        con5(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),
        con6(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),
        con7(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),

        con5(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),
        con6(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),
        con7(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),

        con8(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con8(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h)        
    ]))
    val1 = array([pos1n1i[0]+diff1[0], pos1n1i[1]+diff1[1], pos1n1i[2]+diff1[2], pos1n1i[3]+diff1[3], pos2n1i[0]+diff1[4], pos2n1i[1]+diff1[5], pos2n1i[2]+diff1[6], pos2n1i[3]+diff1[7], vel1n1i[0]+diff1[8], vel1n1i[1]+diff1[9], vel1n1i[2]+diff1[10], vel2n1i[0]+diff1[11], vel2n1i[1]+diff1[12], vel2n1i[2]+diff1[13], mu1i+diff1[14], mu2i+diff1[15]])
    print(val1)
    x = 0
    # print(x)
    # print("###############################################")
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[0],0.],
            [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[1],0.],
            [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[2],0.],
            [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.*val1[3],0.],
            [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[4]],
            [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[5]],
            [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[6]],
            [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,4.*val1[7]],               
            [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],              
            [2.*val1[0],2.*val1[1],2.*val1[2],-2.*val1[3],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,2.*val1[4],2.*val1[5],2.*val1[6],-2.*val1[7],0.,0.,0.,0.,0.,0.,0.,0.]
        ])
        # print(linalg.det(mat))
        diff2=linalg.solve(mat,-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con1(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
            con2(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
            con3(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
            con4(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),

            con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
            con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
            con7(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
            con5(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),
            con6(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),
            con7(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),

            con8(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con8(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h)        
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11], val1[12]+diff2[12], val1[13]+diff2[13], val1[14]+diff2[14], val1[15]+diff2[15]])        
        val1 = val2
        x=x+1    

    return[val1] 

def imph3sptrans(pos1n, pos1n1i, pos2n, pos2n1i, vel1n, vel1n1i, vel2n, vel2n1i, step, massvec, sprcon, eqdist):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (ad1n1 - ad1n - .5*h*(
            (1./m1)*(.5*m1*(2.*bd1n*bd1n*cosh(a1n)*sinh(a1n) + 2.*gd1n*gd1n*cosh(a1n)*cosh(b1n)*cosh(b1n)*sinh(a1n)) - (k*(-1.*xo + 
            arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))*(cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n)*sinh(a1n) - cosh(a1n)*sinh(a2n) - 
            cosh(a2n)*sinh(a1n)*sinh(b1n)*sinh(b2n) - cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(a1n)*sinh(g1n)*sinh(g2n)))/(sqrt(-1. + 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n))*sqrt(1. + cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - 
            sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))) 

            + 

            (1./m1)*(.5*m1*(2.*bd1n1*bd1n1*cosh(a1n1)*sinh(a1n1) + 2.*gd1n1*gd1n1*cosh(a1n1)*cosh(b1n1)*cosh(b1n1)*sinh(a1n1)) - (k*(-1.*xo + 
            arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1)))*(cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1)*sinh(a1n1) - cosh(a1n1)*sinh(a2n1) - 
            cosh(a2n1)*sinh(a1n1)*sinh(b1n1)*sinh(b2n1) - cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(a1n1)*sinh(g1n1)*sinh(g2n1)))/(sqrt(-1. + 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))*sqrt(1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - 
            sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))))   ))

    def con5(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(
            (1./m1)*(1./(cosh(a1n)*cosh(a1n)))*(-2.*ad1n*bd1n*m1*cosh(a1n)*sinh(a1n) + gd1n*gd1n*m1*cosh(a1n)*cosh(a1n)*cosh(b1n)*sinh(b1n) - (k*(-1.*xo + 
            arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))*(cosh(a1n)*cosh(a2n)*cosh(b2n)*cosh(g1n)*cosh(g2n)*sinh(b1n) - cosh(a1n)*cosh(a2n)*cosh(b1n)*sinh(b2n) - 
            cosh(a1n)*cosh(a2n)*cosh(b2n)*sinh(b1n)*sinh(g1n)*sinh(g2n)))/(sqrt(-1. + cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) -
            sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n))*sqrt(1. + 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))) 

            + 

            (1./m1)*(1./(cosh(a1n1)*cosh(a1n1)))*(-2.*ad1n1*bd1n1*m1*cosh(a1n1)*sinh(a1n1) + gd1n1*gd1n1*m1*cosh(a1n1)*cosh(a1n1)*cosh(b1n1)*sinh(b1n1) - (k*(-1.*xo + 
            arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1)))*(cosh(a1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1)*sinh(b1n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*sinh(b2n1) - 
            cosh(a1n1)*cosh(a2n1)*cosh(b2n1)*sinh(b1n1)*sinh(g1n1)*sinh(g2n1)))/(sqrt(-1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) -
            sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))*sqrt(1. + 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))))   ))

    def con6(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (gd1n1 - gd1n - .5*h*(
            (1./m1)*(1./(cosh(a1n)*cosh(a1n)*cosh(b1n)*cosh(b1n)))*(-2.*ad1n*gd1n*m1*cosh(a1n)*cosh(b1n)*cosh(b1n)*sinh(a1n) - 
            2.*bd1n*gd1n*m1*cosh(a1n)*cosh(a1n)*cosh(b1n)*sinh(b1n) - (k*(-1.*xo + arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
            cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))*(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g2n)*sinh(g1n) - 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*sinh(g2n)))/(sqrt(-1. + cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
            cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n))*sqrt(1. + cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) -
            sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n))))

            + 

            (1./m1)*(1./(cosh(a1n1)*cosh(a1n1)*cosh(b1n1)*cosh(b1n1)))*(-2.*ad1n1*gd1n1*m1*cosh(a1n1)*cosh(b1n1)*cosh(b1n1)*sinh(a1n1) - 
            2.*bd1n1*gd1n1*m1*cosh(a1n1)*cosh(a1n1)*cosh(b1n1)*sinh(b1n1) - (k*(-1.*xo + arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
            cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1)))*(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g2n1)*sinh(g1n1) - 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*sinh(g2n1)))/(sqrt(-1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
            cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))*sqrt(1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) -
            sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))))     ))        
    
    h = step
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
    ])
    diff1=linalg.solve(mat,-array([
        con1(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
        con2(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
        con3(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
        con1(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),
        con2(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),
        con3(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),

        con4(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
        con5(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
        con6(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
        con4(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),
        con5(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),
        con6(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),       
    ]))
    val1 = array([pos1n1i[0]+diff1[0], pos1n1i[1]+diff1[1], pos1n1i[2]+diff1[2], pos2n1i[0]+diff1[3], pos2n1i[1]+diff1[4], pos2n1i[2]+diff1[5], vel1n1i[0]+diff1[6], vel1n1i[1]+diff1[7], vel1n1i[2]+diff1[8], vel2n1i[0]+diff1[9], vel2n1i[1]+diff1[10], vel2n1i[2]+diff1[11]])    
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
        ])

        diff2=linalg.solve(mat,-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
            con1(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),
            con2(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),
            con3(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),

            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
            con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
            con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
            con4(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),
            con5(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),
            con6(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),       
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]])       
        val1 = val2
        x=x+1
    return[val1]

# H2xE 3-space Spring Potential 

def symh2esptrans(pos1n, pos1n1i, pos2n, pos2n1i, vel1n, vel1n1i, vel2n, vel2n1i, step, massvec, sprcon, eqdist):

    def con1(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h):
        return xn1 - xn - 2.*mu*(xn1 + xn) - .5*h*(
            adn*cosh(arcsinh(xn)) 
            + 
            adn1*cosh(arcsinh(xn1))     )

    def con2(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return yn1 - yn - 2.*mu*(yn1 + yn) - .5*h*(
            adn*sinh(arcsinh(xn))*sinh(arctanh(yn/wn)) + bdn*cosh(arcsinh(xn))*cosh(arctanh(yn/wn)) 
            + 
            adn1*sinh(arcsinh(xn1))*sinh(arctanh(yn1/wn1)) + bdn1*cosh(arcsinh(xn1))*cosh(arctanh(yn1/wn1))    )

    def con3(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return zn1 - zn - .5*h*(
            gdn
            +
            gdn1    )

    def con4(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return wn1 - wn + 2.*mu*(wn1 + wn) - .5*h*(
            adn*sinh(arcsinh(xn))*cosh(arctanh(yn/wn)) + bdn*cosh(arcsinh(xn))*sinh(arctanh(yn/wn)) 
            + 
            adn1*sinh(arcsinh(xn1))*cosh(arctanh(yn1/wn1)) + bdn1*cosh(arcsinh(xn1))*sinh(arctanh(yn1/wn1))  )

    def con5(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, m1, h, k, xo):
        return (ad1n1 - ad1n - .5*h*(
            (1./m1)*(bd1n*bd1n*m1*cosh(arcsinh(x1n))*sinh(arcsinh(x1n)) - (k*arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))*(-xo + sqrt((-z1n + z2n)**2. + arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - 
            sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))**2.))*(cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n))*sinh(arcsinh(x1n)) - 
            cosh(arcsinh(x1n))*sinh(arcsinh(x2n)) - cosh(arcsinh(x2n))*sinh(arcsinh(x1n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n))))/(sqrt((-z1n + z2n)**2. + arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - 
            sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))**2.)*sqrt(-1. + cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - 
            sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))*sqrt(1. + cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))))

            + 

            (1./m1)*(bd1n1*bd1n1*m1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1)) - (k*arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))*(-xo + sqrt((-z1n1 + z2n1)**2. + arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - 
            sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))**2.))*(cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1))*sinh(arcsinh(x1n1)) - 
            cosh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - cosh(arcsinh(x2n1))*sinh(arcsinh(x1n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1))))/(sqrt((-z1n1 + z2n1)**2. + arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - 
            sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))**2.)*sqrt(-1. + cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - 
            sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))*sqrt(1. + cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))))   ))

    def con6(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(
            (1./m1)*(1./cosh(arcsinh(x1n))**2.)*(-2.*ad1n*bd1n*m1*cosh(arcsinh(x1n))*sinh(arcsinh(x1n)) - (k*arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - 
            sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))*(-xo + sqrt((-z1n + z2n)**2. + arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - 
            sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))**2.))*(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y2n/w2n))*sinh(arctanh(y1n/w1n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n))))/(sqrt((-z1n + z2n)**2. + arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))**2.)*sqrt(-1. + cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))*sqrt(1. + cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))))

            + 

            (1./m1)*(1./cosh(arcsinh(x1n1))**2.)*(-2.*ad1n1*bd1n1*m1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1)) - (k*arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - 
            sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))*(-xo + sqrt((-z1n1 + z2n1)**2. + arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - 
            sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))**2.))*(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y2n1/w2n1))*sinh(arctanh(y1n1/w1n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1))))/(sqrt((-z1n1 + z2n1)**2. + arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))**2.)*sqrt(-1. + cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))*sqrt(1. + cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))))   ))

    def con7(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, m1, h, k, xo):
        return (gd1n1 - gd1n - .5*h*(
            ((-z1n + z2n)*k*(-xo + sqrt((-z1n + z2n)**2. + arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
            cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))**2.)))/(m1*sqrt((-z1n + z2n)**2. + arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arctanh(y1n/w1n))*cosh(arctanh(y2n/w2n)) - 
            sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arctanh(y1n/w1n))*sinh(arctanh(y2n/w2n)))**2.))

            + 

            ((-z1n1 + z2n1)*k*(-xo + sqrt((-z1n1 + z2n1)**2. + arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
            cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))**2.)))/(m1*sqrt((-z1n1 + z2n1)**2. + arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arctanh(y1n1/w1n1))*cosh(arctanh(y2n1/w2n1)) - 
            sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arctanh(y1n1/w1n1))*sinh(arctanh(y2n1/w2n1)))**2.))    ))        
    
    def con8(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
        return xn1*xn1 + yn1*yn1 - wn1*wn1 + 1.
    
    h = step
    mu1i = 10e-10
    mu2i = 10e-10
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[0],0.],
        [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[1],0.],
        [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.*pos1n[3],0.],
        [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[0]],
        [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[1]],
        [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,4.*pos2n[3]],               
        [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],              
        [2.*pos1n[0],2.*pos1n[1],0.,-2.*pos1n[3],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,2.*pos2n[0],2.*pos2n[1],0.,-2.*pos2n[3],0.,0.,0.,0.,0.,0.,0.,0.]
    ])
    # print(mat)
    diff1=linalg.solve(mat,-array([
        con1(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con2(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con3(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con4(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con1(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
        con2(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
        con3(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
        con4(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),

        con5(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),
        con6(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),
        con7(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),

        con5(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),
        con6(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),
        con7(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),

        con8(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
        con8(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h)        
    ]))
    val1 = array([pos1n1i[0]+diff1[0], pos1n1i[1]+diff1[1], pos1n1i[2]+diff1[2], pos1n1i[3]+diff1[3], pos2n1i[0]+diff1[4], pos2n1i[1]+diff1[5], pos2n1i[2]+diff1[6], pos2n1i[3]+diff1[7], vel1n1i[0]+diff1[8], vel1n1i[1]+diff1[9], vel1n1i[2]+diff1[10], vel2n1i[0]+diff1[11], vel2n1i[1]+diff1[12], vel2n1i[2]+diff1[13], mu1i+diff1[14], mu2i+diff1[15]])
    # print(val1)
    x = 0
    # print(x)
    # print("###############################################")
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[0],0.],
            [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[1],0.],
            [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.*val1[3],0.],
            [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[4]],
            [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[5]],
            [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,4.*val1[7]],               
            [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],              
            [2.*val1[0],2.*val1[1],0.,-2.*val1[3],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,2.*val1[4],2.*val1[5],0.,-2.*val1[7],0.,0.,0.,0.,0.,0.,0.,0.]
        ])
        # print(linalg.det(mat))
        diff2=linalg.solve(mat,-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con1(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
            con2(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
            con3(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
            con4(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),

            con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
            con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
            con7(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
            con5(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),
            con6(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),
            con7(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),

            con8(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
            con8(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h)        
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11], val1[12]+diff2[12], val1[13]+diff2[13], val1[14]+diff2[14], val1[15]+diff2[15]])        
        val1 = val2
        x=x+1    

    return[val1] 

def imph2esptrans(pos1n, pos1n1i, pos2n, pos2n1i, vel1n, vel1n1i, vel2n, vel2n1i, step, massvec, sprcon, eqdist):
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (ad1n1 - ad1n - .5*h*(
            (1./m1)*(bd1n*bd1n*m1*cosh(a1n)*sinh(a1n) - (k*arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))*(-xo + 
            sqrt((-g1n + g2n)**2. + arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))**2.))*(cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(a1n) - 
            cosh(a1n)*sinh(a2n) - cosh(a2n)*sinh(a1n)*sinh(b1n)*sinh(b2n)))/(sqrt((-g1n + g2n)**2. + arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - 
            cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))**2.)*sqrt(-1. + cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))*sqrt(1. + 
            cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))))

            + 

            (1./m1)*(bd1n1*bd1n1*m1*cosh(a1n1)*sinh(a1n1) - (k*arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))*(-xo + 
            sqrt((-g1n1 + g2n1)**2. + arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))**2.))*(cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(a1n1) - 
            cosh(a1n1)*sinh(a2n1) - cosh(a2n1)*sinh(a1n1)*sinh(b1n1)*sinh(b2n1)))/(sqrt((-g1n1 + g2n1)**2. + arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - 
            cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))**2.)*sqrt(-1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))*sqrt(1. + 
            cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))))   ))

    def con5(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(
            (1./m1)*(1./cosh(a1n)**2.)*(-2.*ad1n*bd1n*m1*cosh(a1n)*sinh(a1n) - (k*arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) -
            cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))*(-xo + sqrt((-g1n + g2n)**2. + arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - 
            cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))**2.))*(cosh(a1n)*cosh(a2n)*cosh(b2n)*sinh(b1n) - cosh(a1n)*cosh(a2n)*cosh(b1n)*sinh(b2n)))/(sqrt((-g1n + g2n)**2. + 
            arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))**2.)*sqrt(-1. + cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - 
            sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))*sqrt(1. + cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))))

            + 

            (1./m1)*(1./cosh(a1n1)**2.)*(-2.*ad1n1*bd1n1*m1*cosh(a1n1)*sinh(a1n1) - (k*arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) -
            cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))*(-xo + sqrt((-g1n1 + g2n1)**2. + arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - 
            cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))**2.))*(cosh(a1n1)*cosh(a2n1)*cosh(b2n1)*sinh(b1n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*sinh(b2n1)))/(sqrt((-g1n1 + g2n1)**2. + 
            arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))**2.)*sqrt(-1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - 
            sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))*sqrt(1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))))   ))

    def con6(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (gd1n1 - gd1n - .5*h*(
            ((-g1n + g2n)*k*(-xo + sqrt((-g1n + g2n)**2. + arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))**2.)))/(m1*sqrt((-g1n + g2n)**2. + 
            arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n) - sinh(a1n)*sinh(a2n) - cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n))**2.))

            + 

            ((-g1n1 + g2n1)*k*(-xo + sqrt((-g1n1 + g2n1)**2. + arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))**2.)))/(m1*sqrt((-g1n1 + g2n1)**2. + 
            arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1))**2.))    ))        
    
    h = step
    mat=array([
        [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
    ])
    diff1=linalg.solve(mat,-array([
        con1(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
        con2(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
        con3(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
        con1(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),
        con2(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),
        con3(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),

        con4(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
        con5(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
        con6(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
        con4(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),
        con5(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),
        con6(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),       
    ]))
    val1 = array([pos1n1i[0]+diff1[0], pos1n1i[1]+diff1[1], pos1n1i[2]+diff1[2], pos2n1i[0]+diff1[3], pos2n1i[1]+diff1[4], pos2n1i[2]+diff1[5], vel1n1i[0]+diff1[6], vel1n1i[1]+diff1[7], vel1n1i[2]+diff1[8], vel2n1i[0]+diff1[9], vel2n1i[1]+diff1[10], vel2n1i[2]+diff1[11]])    
    x = 0
    while(x < 7):
        mat=array([
            [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
        ])

        diff2=linalg.solve(mat,-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
            con1(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),
            con2(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),
            con3(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),

            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
            con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
            con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
            con4(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),
            con5(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),
            con6(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),       
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]])       
        val1 = val2
        x=x+1
    return[val1]








