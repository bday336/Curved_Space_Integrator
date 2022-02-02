from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arccos,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,linalg
from function_bank import boostxh2e,h2dist,boostxh2,rotzh2,convertpos_hyp2paratransh2,convertvel_hyp2paratransh2

# This script is the repository of all the various solvers I have constructed
# to run the physics engine of the VR Thurston geometry environments. Currently,
# I have updated the solvers for geodesics in H2, H3, and H2xE. The collision
# algorithm is also functional in these spaces for the case of geometric balls
# in each space. There is also the solvers for the PBH space that we constructed
# for an example of an isotropic and nonhomogenous space for the space.

#################
#################
### GEODESICS ###
#################
#################

# -----------------------------------------------------------------------------------------

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
# what the origin of it is. I have also simplified the solver in that the initial guess
# for the iterator automatically takes the values from the given current position and velocity.

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

def imph2egeotrans(posn, veln, step):
    
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

    def jacobian(an1, bn1, gn1, adn1, bdn1, gdn1, h):
        return array([
                    [1.,0.,0.,-.5*h,0.,0.],
                    [0.,1.,0.,0.,-.5*h,0.],
                    [0.,0.,1.,0.,0.,-.5*h],
                    [-.5*h*bdn1*bdn1*cosh(2*an1),0.,0.,1.,-h*sinh(an1)*cosh(an1)*bdn1,0.],
                    [h*adn1*bdn1/(cosh(an1)*cosh(an1)),0.,0.,h*tanh(an1)*bdn1,1.+h*tanh(an1)*adn1,0.],
                    [0.,0.,0.,0.,0.,1.]
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

# I have not made any updates to this solver. Since we are still not able to get the
# form of the central potential at the moment, I do not really need both parameterizations
# of h2e. Come back to this for completeness of the update and/or if you are able to 
# derive a closed form of the central potential in this geometry.

# WARNING -- HAS NOT BEEN UPDATED TO THE CURRENT FORMAT

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

##########################################################
### Pseudo-Black Hole (PBH) geometry 3-space Geodesics ###
##########################################################

# This is the geodesic solver for the constructed PBH space we were going to use for the 
# paper that represents an isotropic and nonhomogenous geometry. Does not correspond to
# the Thurston geometries. We do not have an analytical expression for the exact geodesics
# in this space so cannot do the same comparison as the others. Also, do not have a
# distance function for this space. Need to consider doing a lookup table, but has not been
# a main priority so far. I will leave this solver in the un-updated format until speaking
# with Sabetta to determine the best use of my time.

# WARNING -- HAS NOT BEEN UPDATED TO THE CURRENT FORMAT

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

#########################
#########################
### CENTRAL POTENTIAL ###
#########################
#########################

# -----------------------------------------------------------------------------------------

############################################
### Hyperbolic 3-space Central Potential ###
############################################

# This is the main central potential script for H3. However for more general functionality
# I want to try and get it working in the translational parameterization to prevent any
# issues that might occur if something gets close to the origin. Right now, it will only
# function as a single source at the origin. The more general equations are much more
# complicated so I will save them for down the line. Would be interested to have for
# general gravitational systems. Also currently have G=1 for simplicity.

# WARNING -- THE ONLY GRAVITATIONAL INTERACTION IS BETWEEN THE SOURCE MASS AND THE
# ORBITING TEST MASSES. THE TEST MASSES FEEL NO GRAVITATIONAL INTERACTION BETWEEN
# EACH OTHER. THE SOLVER IS ONLY FOR THE TEST PARTICLES. THE SOURCE PARTICLE IS 
# CONSIDERED TO BE STATIONARY AT THE ORIGIN.

def imph3cprot(posn, veln, step, sourcemass, testmass):
    
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
    
    def jacobian(an1, bn1, gn1, adn1, bdn1, gdn1, h):
        return array([
                [1.,0.,0.,-.5*h,0.,0.],
                [0.,1.,0.,0.,-.5*h,0.],
                [0.,0.,1.,0.,0.,-.5*h],
                [-.5*h*(bdn1*bdn1+sin(bn1)*sin(bn1)*gdn1*gdn1)*cosh(2.*an1)-h*sourcemass*testmass/(sinh(an1)*sinh(an1)*tanh(an1)),-.25*h*sinh(2.*an1)*sin(2.*bn1)*gdn1*gdn1,0.,1.,-.5*h*sinh(2.*an1)*bdn1,-.5*h*sinh(2.*an1)*sin(bn1)*sin(bn1)*gdn1],
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

##################################################################
### Pseudo-Black Hole (PBH) geometry 3-space Central Potential ###
##################################################################

# This is the central potential solver for the constructed PBH space we were going to use for the 
# paper that represents an isotropic and nonhomogenous geometry. Does not correspond to
# the Thurston geometries. We do not have an analytical expression for the exact geodesics
# in this space so cannot do the same comparison as the others. Also, do not have a
# distance function for this space. Need to consider doing a lookup table, but has not been
# a main priority so far. I will leave this solver in the un-updated format until speaking
# with Sabetta to determine the best use of my time.

# WARNING -- HAS NOT BEEN UPDATED TO THE CURRENT FORMAT

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

########################
########################
### SPRING POTENTIAL ###
########################
########################

# -----------------------------------------------------------------------------------------

###########################################
### Hyperbolic 3-space Spring Potential ###
###########################################

# This took a lot more of work than I anticipated to make sure that the Newton solver
# jacobian was correct, but it is finally updated and actually correct now. Currently,
# it only supports the interaction of a two mass spring system, but I am hopeful that
# the functions I have constructed should make increasing the complexity of the system
# more streamlined. However, it does take a hot minute to run even now, so it will take
# even longer with more complicated spring systems.

def imph3sptrans(pos1n, pos2n, vel1n, vel2n, step, m1, m2, sprcon, eqdist):

    # This seems more complicated, but I am constructing these so that the jacobian elements are not super long.
    # I have found that I do not need to make functions for both particles since I can just flip the arguments
    # and it should work fine since they are symmetric.

    # This is the argument inside the arccosh of the distance function. It is the same for both particles
    # due to the symmetric of the equation between particle 1 and 2.
    def D12(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sinh(a2) - cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only three functions are needed instead of six.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, g2, a1, b1, g1)
    # db2D12 = db1D12(a2, b2, g2, a1, b1, g1)
    # dg2D12 = dg1D12(a2, b2, g2, a1, b1, g1)
    def da1D12(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sinh(a2) - sinh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + sinh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def db1D12(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2) 

    def dg1D12(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)      
    

    # These are the second derivatives of the D12 function with respective to combinations of a1, b1, g1, a2, b2, g2. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only second
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only twelve functions are needed instead of thirty-six. Due to the symmetry of the partial
    # derivatives the square matrix of thirty-six values can be reduced to the upper triangular metrix. Of the twenty-one values in
    # upper triangular matrix symmetry of the particles allows for the number of functions to be further reduced to twelve
    
    # For the remaining nine functions of the upper triangular matrix use:
    # da2D12b1 = db2D12a1(a2, b2, g2, a1, b1, g1)

    # da2D12g1 = dg2D12a1(a2, b2, g2, a1, b1, g1)
    # db2D12g1 = dg2D12b1(a2, b2, g2, a1, b1, g1)

    # da2D12a2 = da1D12a1(a2, b2, g2, a1, b1, g1)
    # db2D12a2 = db1D12a1(a2, b2, g2, a1, b1, g1)
    # dg2D12a2 = dg1D12a1(a2, b2, g2, a1, b1, g1)

    # db2D12b2 = db1D12b1(a2, b2, g2, a1, b1, g1)
    # dg2D12b2 = dg1D12b1(a2, b2, g2, a1, b1, g1)

    # dg2D12g2 = dg1D12g1(a2, b2, g2, a1, b1, g1)

    # For the remaining lower portion of the total square matrix of terms (fifteen values) simply interchange the indices of 
    # the partial derivatives.
    def da1D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sinh(a2) - cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def db1D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cosh(b1)*cosh(a2)*sinh(b2) + sinh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def dg1D12a1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def da2D12a1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(a2) - sinh(a1)*sinh(b1)*sinh(a2)*sinh(b2) + sinh(a1)*cosh(b1)*sinh(a2)*cosh(b2)*cosh(g1 - g2)

    def db2D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sinh(b1)*cosh(a2)*cosh(b2) + sinh(a1)*cosh(b1)*cosh(a2)*sinh(b2)*cosh(g1 - g2)

    def dg2D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def db1D12b1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def dg1D12b1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def db2D12b1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2) + cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2)*cosh(g1 - g2)

    def dg2D12b1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def dg1D12g1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def dg2D12g1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(a1, b1, g1, a2, b2, g2, m1, m2, k, xo):
        # D function
        d12=D12(a1, b1, g1, a2, b2, g2)
        # First derivatives of D function
        da1d12=da1D12(a1, b1, g1, a2, b2, g2)
        db1d12=db1D12(a1, b1, g1, a2, b2, g2)
        dg1d12=dg1D12(a1, b1, g1, a2, b2, g2)
        da2d12=da1D12(a2, b2, g2, a1, b1, g1)
        db2d12=db1D12(a2, b2, g2, a1, b1, g1)
        dg2d12=dg1D12(a2, b2, g2, a1, b1, g1)
        # Second derivatives of D function
        da1d12a1=da1D12a1(a1, b1, g1, a2, b2, g2)
        db1d12a1=db1D12a1(a1, b1, g1, a2, b2, g2)
        dg1d12a1=dg1D12a1(a1, b1, g1, a2, b2, g2)
        da2d12a1=da2D12a1(a1, b1, g1, a2, b2, g2)
        db2d12a1=db2D12a1(a1, b1, g1, a2, b2, g2)
        dg2d12a1=dg2D12a1(a1, b1, g1, a2, b2, g2)
        
        da1d12b1=db1d12a1
        db1d12b1=db1D12b1(a1, b1, g1, a2, b2, g2)
        dg1d12b1=dg1D12b1(a1, b1, g1, a2, b2, g2)
        da2d12b1 = db2D12a1(a2, b2, g2, a1, b1, g1)
        db2d12b1=db2D12b1(a1, b1, g1, a2, b2, g2)
        dg2d12b1=dg2D12b1(a1, b1, g1, a2, b2, g2)

        da1d12g1=dg1d12a1
        db1d12g1=dg1d12b1
        dg1d12g1=dg1D12g1(a1, b1, g1, a2, b2, g2)
        da2d12g1 = dg2D12a1(a2, b2, g2, a1, b1, g1)
        db2d12g1 = dg2D12b1(a2, b2, g2, a1, b1, g1)
        dg2d12g1=dg2D12g1(a1, b1, g1, a2, b2, g2)

        da1d12a2=da2d12a1
        db1d12a2=da2d12b1
        dg1d12a2=da2d12g1
        da2d12a2 = da1D12a1(a2, b2, g2, a1, b1, g1)
        db2d12a2 = db1D12a1(a2, b2, g2, a1, b1, g1)
        dg2d12a2 = dg1D12a1(a2, b2, g2, a1, b1, g1)

        da1d12b2=db2d12a1
        db1d12b2=db2d12b1
        dg1d12b2=db2d12g1
        da2d12b2=db2d12a2
        db2d12b2 = db1D12b1(a2, b2, g2, a1, b1, g1)
        dg2d12b2 = dg1D12b1(a2, b2, g2, a1, b1, g1)

        da1d12g2=dg2d12a1
        db1d12g2=dg2d12b1
        dg1d12g2=dg2d12g1
        da2d12g2=dg2d12a2
        db2d12g2=dg2d12b2
        dg2d12g2 = dg1D12g1(a2, b2, g2, a1, b1, g1)

        return array([
            [-k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*da1d12/(d12**2. - 1.) - da1d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*db1d12/(d12**2. - 1.) - db1d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*dg1d12/(d12**2. - 1.) - dg1d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*da2d12/(d12**2. - 1.) - da2d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*db2d12/(d12**2. - 1.) - db2d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*dg2d12/(d12**2. - 1.) - dg2d12a1) )],

            [-k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*sinh(2.*a1)/(cosh(a1)*cosh(a1)) + d12*db1d12*da1d12/(d12**2. - 1.) - da1d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db1d12/(d12**2. - 1.) - db1d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*dg1d12/(d12**2. - 1.) - dg1d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*da2d12/(d12**2. - 1.) - da2d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db2d12/(d12**2. - 1.) - db2d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*dg2d12/(d12**2. - 1.) - dg2d12b1) )],

            [-k/(m1*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg1d12*sinh(2.*a1)*cosh(b1)*cosh(b1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*da1d12/(d12**2. - 1.) - da1d12g1) ),
            -k/(m1*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg1d12*sinh(2.*b1)*cosh(a1)*cosh(a1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*db1d12/(d12**2. - 1.) - db1d12g1) ),
            -k/(m1*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*dg1d12/(d12**2. - 1.) - dg1d12g1) ),
            -k/(m1*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*da2d12/(d12**2. - 1.) - da2d12g1) ),
            -k/(m1*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*db2d12/(d12**2. - 1.) - db2d12g1) ),
            -k/(m1*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*dg2d12/(d12**2. - 1.) - dg2d12g1) )],

            [-k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*da1d12/(d12**2. - 1.) - da1d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*db1d12/(d12**2. - 1.) - db1d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*dg1d12/(d12**2. - 1.) - dg1d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*da2d12/(d12**2. - 1.) - da2d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*db2d12/(d12**2. - 1.) - db2d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*dg2d12/(d12**2. - 1.) - dg2d12a2) )],

            [-k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*da1d12/(d12**2. - 1.) - da1d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db1d12/(d12**2. - 1.) - db1d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*dg1d12/(d12**2. - 1.) - dg1d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*sinh(2.*a2)/(cosh(a2)*cosh(a2)) + d12*db2d12*da2d12/(d12**2. - 1.) - da2d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db2d12/(d12**2. - 1.) - db2d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*dg2d12/(d12**2. - 1.) - dg2d12b2) )],

            [-k/(m2*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*da1d12/(d12**2. - 1.) - da1d12g2) ),
            -k/(m2*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*db1d12/(d12**2. - 1.) - db1d12g2) ),
            -k/(m2*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*dg1d12/(d12**2. - 1.) - dg1d12g2) ),
            -k/(m2*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg2d12*sinh(2.*a2)*cosh(b2)*cosh(b2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*da2d12/(d12**2. - 1.) - da2d12g2) ),
            -k/(m2*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg2d12*sinh(2.*b2)*cosh(a2)*cosh(a2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*db2d12/(d12**2. - 1.) - db2d12g2) ),
            -k/(m2*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*dg2d12/(d12**2. - 1.) - dg2d12g2) )]

        ])
   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*cosh(b1n)**2.)*sinh(a1n)*cosh(a1n) - k/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (-cosh(a1n)*sinh(a2n) - sinh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + sinh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*cosh(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) - k/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (-cosh(a1n1)*sinh(a2n1) - sinh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + sinh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) 
            ))

    def con5(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sinh(b1n)*cosh(b1n) - 2.*ad1n*bd1n*tanh(a1n) - k/(m1*cosh(a1n)*cosh(a1n))*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (-cosh(a1n)*cosh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*sinh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            gd1n1*gd1n1*sinh(b1n1)*cosh(b1n1) - 2.*ad1n1*bd1n1*tanh(a1n1) - k/(m1*cosh(a1n1)*cosh(a1n1))*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.)  
            ))

    def con6(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n*tanh(a1n) - 2.*bd1n*gd1n*tanh(b1n) - k/(m1*cosh(a1n)*cosh(a1n)*cosh(b1n)*cosh(b1n))*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*sinh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            -2.*ad1n1*gd1n1*tanh(a1n1) - 2.*bd1n1*gd1n1*tanh(b1n1) - k/(m1*cosh(a1n1)*cosh(a1n1)*cosh(b1n1)*cosh(b1n1))*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*sinh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.)
            )) 
    
    def jacobian(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, ad1n1, bd1n1, gd1n1, ad2n1, bd2n1, gd2n1, m1, m2, h, k, xo):
        spring_terms=jacobi_sp_terms(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, m2, k, xo)
        return array([
            [1.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1+cosh(b1n1)*cosh(b1n1)*gd1n1*gd1n1)*cosh(2.*a1n1) + spring_terms[0,0],
            -.25*h*sinh(2.*a1n1)*sinh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[0,1],
            0. + spring_terms[0,2], 
            
            spring_terms[0,3],
            spring_terms[0,4],
            spring_terms[0,5],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            -.5*h*sinh(2.*a1n1)*cosh(b1n1)*cosh(b1n1)*gd1n1,

            0.,
            0.,
            0.],

            # bd1 update
            [h*ad1n1*bd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[1,0],
            -.5*h*cosh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[1,1],
            0. + spring_terms[1,2], 
            
            spring_terms[1,3],
            spring_terms[1,4],
            spring_terms[1,5],
            
            h*tanh(a1n1)*bd1n1,
            1.+h*tanh(a1n1)*ad1n1,
            -.5*h*sinh(2.*b1n1)*gd1n1,

            0.,
            0.,
            0.],

            # gd1 update
            [h*ad1n1*gd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[2,0],
            h*bd1n1*gd1n1/(cosh(b1n1)*cosh(b1n1)) + spring_terms[2,1],
            0. + spring_terms[2,2],
             
            spring_terms[2,3],
            spring_terms[2,4],
            spring_terms[2,5],

            h*tanh(a1n1)*gd1n1,
            h*tanh(b1n1)*gd1n1,
            1.+h*tanh(a1n1)*ad1n1+h*tanh(b1n1)*bd1n1,

            0.,
            0.,
            0.],

            # ad2 update
            [spring_terms[3,0],
            spring_terms[3,1],
            spring_terms[3,2],

            -.5*h*(bd2n1*bd2n1+cosh(b2n1)*cosh(b2n1)*gd2n1*gd2n1)*cosh(2.*a2n1) + spring_terms[3,3],
            -.25*h*sinh(2.*a2n1)*sinh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[3,4],
            0. + spring_terms[3,5],

            0.,
            0.,
            0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            -.5*h*sinh(2.*a2n1)*cosh(b2n1)*cosh(b2n1)*gd2n1],

            # bd2 update
            [spring_terms[4,0],
            spring_terms[4,1],
            spring_terms[4,2],

            h*ad2n1*bd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[4,3],
            -.5*h*cosh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[4,4],
            0. + spring_terms[4,5],

            0.,
            0.,
            0.,

            h*tanh(a2n1)*bd2n1,
            1.+h*tanh(a2n1)*ad2n1,
            -.5*h*sinh(2.*b2n1)*gd2n1],

            # gd2 update
            [spring_terms[5,0],
            spring_terms[5,1],
            spring_terms[5,2],

            h*ad2n1*gd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[5,3],
            h*bd2n1*gd2n1/(cosh(b2n1)*cosh(b2n1)) + spring_terms[5,4],
            0. + spring_terms[5,5],

            0.,
            0.,
            0.,

            h*tanh(a2n1)*gd2n1,
            h*tanh(b2n1)*gd2n1,
            1.+h*tanh(a2n1)*ad2n1+h*tanh(b2n1)*bd2n1]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist),-array([
        con1(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con2(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con3(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con1(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),
        con2(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),
        con3(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),

        con4(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con5(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con6(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con4(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),
        con5(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),
        con6(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),       
    ]))
    val1 = array([pos1n[0]+diff1[0], pos1n[1]+diff1[1], pos1n[2]+diff1[2], pos2n[0]+diff1[3], pos2n[1]+diff1[4], pos2n[2]+diff1[5], vel1n[0]+diff1[6], vel1n[1]+diff1[7], vel1n[2]+diff1[8], vel2n[0]+diff1[9], vel2n[1]+diff1[10], vel2n[2]+diff1[11]])    
    x = 0
    while(x < 7):
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], val1[6], val1[7], val1[8], val1[9], val1[10], val1[11], m1, m2, step, sprcon, eqdist),-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con1(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),
            con2(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),
            con3(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),

            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con4(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),
            con5(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),
            con6(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),       
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]])       
        val1 = val2
        x=x+1
    # print(val1)
    return val1

#####################################
### H2xE 3-space Spring Potential ###
#####################################

def imph2esptrans(pos1n, pos2n, vel1n, vel2n, step, m1, m2, sprcon, eqdist):

    # This seems more complicated, but I am constructing these so that the jacobian elements are not super long.
    # I have found that I do not need to make functions for both particles since I can just flip the arguments
    # and it should work fine since they are symmetric. These functions are only needed for the hyperbolic portion
    # of the distance function. The euclidean portion is simple enough to not been special functions.

    # This is the argument inside the arccosh of the distance function. It is the same for both particles
    # due to the symmetric of the equation between particle 1 and 2.
    def D12(a1, b1, a2, b2):
        return -sinh(a1)*sinh(a2) - cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)

    # These are the first derivatives of the D12 function with respective to a1 and b1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only two functions are needed instead of four.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, a1, b1)
    # db2D12 = db1D12(a2, b2, a1, b1)
    def da1D12(a1, b1, a2, b2):
        return -cosh(a1)*sinh(a2) - sinh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + sinh(a1)*cosh(b1)*cosh(a2)*cosh(b2)

    def db1D12(a1, b1, a2, b2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*sinh(b1)*cosh(a2)*cosh(b2)      

    # These are the second derivatives of the D12 function with respective to combinations of a1, b1, a2, and b2. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only second
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only six functions are needed instead of sixteen. Due to the symmetry of the partial
    # derivatives the square matrix of sixteen values can be reduced to the upper triangular metrix. Of the ten values in
    # upper triangular matrix symmetry of the particles allows for the number of functions to be further reduced to six
    
    # For the remaining four functions of the upper triangular matrix use:
    # da2D12b1 = db2D12a1(a2, b2, a1, b1)

    # da2D12a2 = da1D12a1(a2, b2, a1, b1)
    # db2D12a2 = db1D12a1(a2, b2, a1, b1)

    # db2D12b2 = db1D12b1(a2, b2, a1, b1)

    # For the remaining lower portion of the total square matrix of terms (six values) simply interchange the indices of 
    # the partial derivatives.
    def da1D12a1(a1, b1, a2, b2):
        return -sinh(a1)*sinh(a2) - cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)

    def db1D12a1(a1, b1, a2, b2):
        return -sinh(a1)*cosh(b1)*cosh(a2)*sinh(b2) + sinh(a1)*sinh(b1)*cosh(a2)*cosh(b2)

    def da2D12a1(a1, b1, a2, b2):
        return -cosh(a1)*cosh(a2) - sinh(a1)*sinh(b1)*sinh(a2)*sinh(b2) + sinh(a1)*cosh(b1)*sinh(a2)*cosh(b2)

    def db2D12a1(a1, b1, a2, b2):
        return -sinh(a1)*sinh(b1)*cosh(a2)*cosh(b2) + sinh(a1)*cosh(b1)*cosh(a2)*sinh(b2)

    def db1D12b1(a1, b1, a2, b2):
        return -cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)

    def db2D12b1(a1, b1, a2, b2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2) + cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # The following function is used to just simplify the calculation of the jacobian matrix being returned
    def simfuncA(d12, g1, g2, xo):
        return ( sqrt( arccosh(d12)**2. + (g2-g1)**2. ) - xo )/sqrt( arccosh(d12)**2. + (g2-g1)**2. )

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(a1, b1, g1, a2, b2, g2, m1, m2, k, xo):
        # D function
        d12=D12(a1, b1, a2, b2)
        # First derivatives of D function
        da1d12=da1D12(a1, b1, a2, b2)
        db1d12=db1D12(a1, b1, a2, b2)
        da2d12=da1D12(a2, b2, a1, b1)
        db2d12=db1D12(a2, b2, a1, b1)
        # Second derivatives of D function
        da1d12a1=da1D12a1(a1, b1, a2, b2)
        db1d12a1=db1D12a1(a1, b1, a2, b2)
        da2d12a1=da2D12a1(a1, b1, a2, b2)
        db2d12a1=db2D12a1(a1, b1, a2, b2)
        
        da1d12b1=db1d12a1
        db1d12b1=db1D12b1(a1, b1, a2, b2)
        da2d12b1 = db2D12a1(a2, b2, a1, b1)
        db2d12b1=db2D12b1(a1, b1, a2, b2)

        da1d12a2=da2d12a1
        db1d12a2=da2d12b1
        da2d12a2 = da1D12a1(a2, b2, a1, b1)
        db2d12a2 = db1D12a1(a2, b2, a1, b1)

        da1d12b2=db2d12a1
        db1d12b2=db2d12b1
        da2d12b2=db2d12a2
        db2d12b2 = db1D12b1(a2, b2, a1, b1)
        # Simplifying function A
        A=simfuncA(d12, g1, g2, xo)

        return array([
            [-k/(m1*1.)*( ( (arccosh(d12)*da1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da1d12*da1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da1d12a1 - d12*da1d12*da1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*da1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da1d12*db1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db1d12a1 - d12*da1d12*db1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*da1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) ) - A*0./1. ) + A*( da1d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*da1d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(0. - 1.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*da1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da1d12*da2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da2d12a1 - d12*da1d12*da2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*da1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da1d12*db2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db2d12a1 - d12*da1d12*db2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*da1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) ) - A*0./1. ) + A*( da1d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*da1d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(1. - 0.) + (g2 - g1)*(0. - 0.) ) )],

            [-k/(m1*cosh(a1)*cosh(a1))*( ( (arccosh(d12)*db1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*sinh(2.*a1)/(cosh(a1)*cosh(a1)) ) + A*( db1d12*da1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da1d12b1 - d12*db1d12*da1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*cosh(a1)*cosh(a1))*( ( (arccosh(d12)*db1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0.         /(cosh(a1)*cosh(a1)) ) + A*( db1d12*db1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db1d12b1 - d12*db1d12*db1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*cosh(a1)*cosh(a1))*( ( (arccosh(d12)*db1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) ) - A*0.         /(cosh(a1)*cosh(a1)) ) + A*( db1d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*db1d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(0. - 1.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*cosh(a1)*cosh(a1))*( ( (arccosh(d12)*db1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0.         /(cosh(a1)*cosh(a1)) ) + A*( db1d12*da2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da2d12b1 - d12*db1d12*da2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*cosh(a1)*cosh(a1))*( ( (arccosh(d12)*db1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0.         /(cosh(a1)*cosh(a1)) ) + A*( db1d12*db2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db2d12b1 - d12*db1d12*db2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*cosh(a1)*cosh(a1))*( ( (arccosh(d12)*db1d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) ) - A*0.         /(cosh(a1)*cosh(a1)) ) + A*( db1d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*db1d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(1. - 0.) + (g2 - g1)*(0. - 0.) ) )],

            [-k/(m1*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*da1d12/(d12**2. - 1.)) + (0. - 1.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*db1d12/(d12**2. - 1.)) + (0. - 1.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*0.    /(d12**2. - 1.)) + (0. - 1.)*(0. - 1.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*da2d12/(d12**2. - 1.)) + (0. - 1.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*db2d12/(d12**2. - 1.)) + (0. - 1.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m1*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*0.    /(d12**2. - 1.)) + (0. - 1.)*(1. - 0.) + (g2 - g1)*(0. - 0.) ) )],

            [-k/(m2*1.)*( ( (arccosh(d12)*da2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da2d12*da1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da1d12a2 - d12*da2d12*da1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*da2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da2d12*db1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db1d12a2 - d12*da2d12*db1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*da2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) ) - A*0./1. ) + A*( da2d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*da2d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(0. - 1.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*da2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da2d12*da2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da2d12a2 - d12*da2d12*da2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*da2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( da2d12*db2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db2d12a2 - d12*da2d12*db2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*da2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) ) - A*0./1. ) + A*( da2d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*da2d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(1. - 0.) + (g2 - g1)*(0. - 0.) ) )],

            [-k/(m2*cosh(a2)*cosh(a2))*( ( (arccosh(d12)*db2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0.         /(cosh(a2)*cosh(a2)) ) + A*( db2d12*da1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da1d12b2 - d12*db2d12*da1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*cosh(a2)*cosh(a2))*( ( (arccosh(d12)*db2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db1d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0.         /(cosh(a2)*cosh(a2)) ) + A*( db2d12*db1d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db1d12b2 - d12*db2d12*db1d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*cosh(a2)*cosh(a2))*( ( (arccosh(d12)*db2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) ) - A*0.         /(cosh(a2)*cosh(a2)) ) + A*( db2d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*db2d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(0. - 1.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*cosh(a2)*cosh(a2))*( ( (arccosh(d12)*db2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*da2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*sinh(2.*a2)/(cosh(a2)*cosh(a2)) ) + A*( db2d12*da2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(da2d12b2 - d12*db2d12*da2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*cosh(a2)*cosh(a2))*( ( (arccosh(d12)*db2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*db2d12)/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0.         /(cosh(a2)*cosh(a2)) ) + A*( db2d12*db2d12/(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(db2d12b2 - d12*db2d12*db2d12/(d12**2. - 1.)) + (0. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*cosh(a2)*cosh(a2))*( ( (arccosh(d12)*db2d12 )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0.    )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) ) - A*0.         /(cosh(a2)*cosh(a2)) ) + A*( db2d12*0.    /(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0.       - d12*db2d12*0.    /(d12**2. - 1.)) + (0. - 0.)*(1. - 0.) + (g2 - g1)*(0. - 0.) ) )],

            [-k/(m2*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*da1d12/(d12**2. - 1.)) + (1. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*db1d12/(d12**2. - 1.)) + (1. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 1.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*0.    /(d12**2. - 1.)) + (1. - 0.)*(0. - 1.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*da2d12/(d12**2. - 1.)) + (1. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(0. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*db2d12/(d12**2. - 1.)) + (1. - 0.)*(0. - 0.) + (g2 - g1)*(0. - 0.) ) ),
            -k/(m2*1.)*( ( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) )*( (1. - A)/( arccosh(d12)**2. + (g2 - g1)**2. )*( (arccosh(d12)*0. )/sqrt(d12**2. - 1.) + (g2 - g1)*(1. - 0.) ) - A*0./1. ) + A*( 0.*0./(d12**2. - 1.) + arccosh(d12)/sqrt(d12**2. - 1)*(0. - d12*0.*0.    /(d12**2. - 1.)) + (1. - 0.)*(1. - 0.) + (g2 - g1)*(0. - 0.) ) )]
        ])
    
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (ad1n1 - ad1n - .5*h*(
            bd1n*bd1n*cosh(a1n)*sinh(a1n) - k/(m1*1.)*( 
            sqrt(arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. + (g2n - g1n)**2.) - xo )/sqrt(
            arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. + (g2n - g1n)**2.)*( 
            arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))*(
            -cosh(a1n)*sinh(a2n) - sinh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + sinh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))/sqrt(
            (-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. - 1.) + (g2n - g1n)*(0. - 0.) )
            + 
            bd1n1*bd1n1*cosh(a1n1)*sinh(a1n1) - k/(m1*1.)*( 
            sqrt(arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. + (g2n1 - g1n1)**2.) - xo )/sqrt(
            arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. + (g2n1 - g1n1)**2.)*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))*(
            -cosh(a1n1)*sinh(a2n1) - sinh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + sinh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))/sqrt(
            (-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. - 1.) + (g2n1 - g1n1)*(0. - 0.) )
           ))

    def con5(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(
            -2.*ad1n*bd1n*tanh(a1n) - k/(m1*cosh(a1n)*cosh(a1n))*( 
            sqrt(arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. + (g2n - g1n)**2.) - xo )/sqrt(
            arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. + (g2n - g1n)**2.)*( 
            arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))*(
            -cosh(a1n)*cosh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*sinh(b1n)*cosh(a2n)*cosh(b2n))/sqrt(
            (-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. - 1.) + (g2n - g1n)*(0. - 0.) )
            + 
            -2.*ad1n1*bd1n1*tanh(a1n1) - k/(m1*cosh(a1n1)*cosh(a1n1))*( 
            sqrt(arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. + (g2n1 - g1n1)**2.) - xo )/sqrt(
            arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. + (g2n1 - g1n1)**2.)*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))*(
            -cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*cosh(b2n1))/sqrt(
            (-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. - 1.) + (g2n1 - g1n1)*(0. - 0.) )  
            ))

    def con6(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (gd1n1 - gd1n - .5*h*(
            0. - k/m1*( 
            sqrt(arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. + (g2n - g1n)**2.) - xo )/sqrt(
            arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. + (g2n - g1n)**2.)*( 
            arccosh(-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))*(
            0.)/sqrt(
            (-sinh(a1n)*sinh(a2n) - cosh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))**2. - 1.) + (g2n - g1n)*(0. - 1.) )
            + 
            0. - k/m1*( 
            sqrt(arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. + (g2n1 - g1n1)**2.) - xo )/sqrt(
            arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. + (g2n1 - g1n1)**2.)*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))*(
            0.)/sqrt(
            (-sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))**2. - 1.) + (g2n1 - g1n1)*(0. - 1.) )    
            ))

    def jacobian(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, ad1n1, bd1n1, gd1n1, ad2n1, bd2n1, gd2n1, m1, m2, h, k, xo):
        spring_terms=jacobi_sp_terms(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, m2, k, xo)
        return array([
            [1.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*bd1n1*bd1n1*cosh(2.*a1n1) + spring_terms[0,0],
            0. + spring_terms[0,1],
            0. + spring_terms[0,2], 
            
            spring_terms[0,3],
            spring_terms[0,4],
            spring_terms[0,5],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            0.,

            0.,
            0.,
            0.],

            # bd1 update
            [h*ad1n1*bd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[1,0],
            0. + spring_terms[1,1],
            0. + spring_terms[1,2], 
            
            spring_terms[1,3],
            spring_terms[1,4],
            spring_terms[1,5],
            
            h*tanh(a1n1)*bd1n1,
            1.+h*tanh(a1n1)*ad1n1,
            0.,

            0.,
            0.,
            0.],

            # gd1 update
            [0. + spring_terms[2,0],
            0. + spring_terms[2,1],
            0. + spring_terms[2,2],
             
            spring_terms[2,3],
            spring_terms[2,4],
            spring_terms[2,5],

            0.,
            0.,
            1.,

            0.,
            0.,
            0.],

            # ad2 update
            [spring_terms[3,0],
            spring_terms[3,1],
            spring_terms[3,2],

            -.5*h*bd2n1*bd2n1*cosh(2.*a2n1) + spring_terms[3,3],
            0. + spring_terms[3,4],
            0. + spring_terms[3,5],

            0.,
            0.,
            0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            0.],

            # bd2 update
            [spring_terms[4,0],
            spring_terms[4,1],
            spring_terms[4,2],

            h*ad2n1*bd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[4,3],
            0. + spring_terms[4,4],
            0. + spring_terms[4,5],

            0.,
            0.,
            0.,

            h*tanh(a2n1)*bd2n1,
            1.+h*tanh(a2n1)*ad2n1,
            0.],

            # gd2 update
            [spring_terms[5,0],
            spring_terms[5,1],
            spring_terms[5,2],

            0. + spring_terms[5,3],
            0. + spring_terms[5,4],
            0. + spring_terms[5,5],

            0.,
            0.,
            0.,

            0.,
            0.,
            1.]
        ])       
    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist),-array([
        con1(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con2(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con3(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con1(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),
        con2(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),
        con3(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),

        con4(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con5(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con6(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con4(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),
        con5(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),
        con6(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),       
    ]))
    val1 = array([pos1n[0]+diff1[0], pos1n[1]+diff1[1], pos1n[2]+diff1[2], pos2n[0]+diff1[3], pos2n[1]+diff1[4], pos2n[2]+diff1[5], vel1n[0]+diff1[6], vel1n[1]+diff1[7], vel1n[2]+diff1[8], vel2n[0]+diff1[9], vel2n[1]+diff1[10], vel2n[2]+diff1[11]])    
    x = 0
    while(x < 7):
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], val1[6], val1[7], val1[8], val1[9], val1[10], val1[11], m1, m2, step, sprcon, eqdist),-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con1(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),
            con2(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),
            con3(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),

            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con4(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),
            con5(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),
            con6(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),       
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]])       
        val1 = val2
        x=x+1
    # print(val1)
    return val1

##################
##################
### SPRING BOX ###
##################
##################

# -----------------------------------------------------------------------------------------

def imph3boxtrans(posn_arr, veln_arr, step, mass_arr, spring_arr):

    # This seems more complicated, but I am constructing these so that the jacobian elements are not super long.
    # I have found that I do not need to make functions for both particles since I can just flip the arguments
    # and it should work fine since they are symmetric. In principle, I think that I do not need to generate
    # any new functions for this system. I just have to be careful with the arguments I use in the functions.
    # In spring terms are more numerous given that each particle is connected to three other particles. When 
    # using the below functions the more general notation of Dmn just needs to be considered with 1=m and 2=n.
    # This is so I do not introduce any unintended errors by changing the variables.

    # This is the argument inside the arccosh of the distance function. It is the same for both particles
    # due to the symmetric of the equation between particle 1 and 2. This can be used as a general function
    # I just have to be careful about the arguments for the various particle pairs.
    def D12(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sinh(a2) - cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only three functions are needed instead of six.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, g2, a1, b1, g1)
    # db2D12 = db1D12(a2, b2, g2, a1, b1, g1)
    # dg2D12 = dg1D12(a2, b2, g2, a1, b1, g1)
    def da1D12(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sinh(a2) - sinh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + sinh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def db1D12(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2) 

    def dg1D12(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)      
    

    # These are the second derivatives of the D12 function with respective to combinations of a1, b1, g1, a2, b2, g2. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only second
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only twelve functions are needed instead of thirty-six. Due to the symmetry of the partial
    # derivatives the square matrix of thirty-six values can be reduced to the upper triangular metrix. Of the twenty-one values in
    # upper triangular matrix symmetry of the particles allows for the number of functions to be further reduced to twelve
    
    # For the remaining nine functions of the upper triangular matrix use:
    # da2D12b1 = db2D12a1(a2, b2, g2, a1, b1, g1)

    # da2D12g1 = dg2D12a1(a2, b2, g2, a1, b1, g1)
    # db2D12g1 = dg2D12b1(a2, b2, g2, a1, b1, g1)

    # da2D12a2 = da1D12a1(a2, b2, g2, a1, b1, g1)
    # db2D12a2 = db1D12a1(a2, b2, g2, a1, b1, g1)
    # dg2D12a2 = dg1D12a1(a2, b2, g2, a1, b1, g1)

    # db2D12b2 = db1D12b1(a2, b2, g2, a1, b1, g1)
    # dg2D12b2 = dg1D12b1(a2, b2, g2, a1, b1, g1)

    # dg2D12g2 = dg1D12g1(a2, b2, g2, a1, b1, g1)

    # For the remaining lower portion of the total square matrix of terms (fifteen values) simply interchange the indices of 
    # the partial derivatives.
    def da1D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sinh(a2) - cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def db1D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cosh(b1)*cosh(a2)*sinh(b2) + sinh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def dg1D12a1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def da2D12a1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(a2) - sinh(a1)*sinh(b1)*sinh(a2)*sinh(b2) + sinh(a1)*cosh(b1)*sinh(a2)*cosh(b2)*cosh(g1 - g2)

    def db2D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sinh(b1)*cosh(a2)*cosh(b2) + sinh(a1)*cosh(b1)*cosh(a2)*sinh(b2)*cosh(g1 - g2)

    def dg2D12a1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def db1D12b1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2) + cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def dg1D12b1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def db2D12b1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2) + cosh(a1)*sinh(b1)*cosh(a2)*sinh(b2)*cosh(g1 - g2)

    def dg2D12b1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sinh(b1)*cosh(a2)*cosh(b2)*sinh(g1 - g2)

    def dg1D12g1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    def dg2D12g1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*cosh(b1)*cosh(a2)*cosh(b2)*cosh(g1 - g2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(a1, b1, g1, a2, b2, g2, a3, b3, g3, a4, b4, g4, a5, b5, g5, a6, b6, g6, a7, b7, g7, a8, b8, g8, mass_arr, spring_arr):
        ### Spring 1-2 spring_arr[0]###
        # D function
        d12=D12(a1, b1, g1, a2, b2, g2)
        # First derivatives of D function
        da1d12=da1D12(a1, b1, g1, a2, b2, g2)
        db1d12=db1D12(a1, b1, g1, a2, b2, g2)
        dg1d12=dg1D12(a1, b1, g1, a2, b2, g2)
        da2d12=da1D12(a2, b2, g2, a1, b1, g1)
        db2d12=db1D12(a2, b2, g2, a1, b1, g1)
        dg2d12=dg1D12(a2, b2, g2, a1, b1, g1)
        # Second derivatives of D function
        da1d12a1=da1D12a1(a1, b1, g1, a2, b2, g2)
        db1d12a1=db1D12a1(a1, b1, g1, a2, b2, g2)
        dg1d12a1=dg1D12a1(a1, b1, g1, a2, b2, g2)
        da2d12a1=da2D12a1(a1, b1, g1, a2, b2, g2)
        db2d12a1=db2D12a1(a1, b1, g1, a2, b2, g2)
        dg2d12a1=dg2D12a1(a1, b1, g1, a2, b2, g2)
        
        da1d12b1=db1d12a1
        db1d12b1=db1D12b1(a1, b1, g1, a2, b2, g2)
        dg1d12b1=dg1D12b1(a1, b1, g1, a2, b2, g2)
        da2d12b1 = db2D12a1(a2, b2, g2, a1, b1, g1)
        db2d12b1=db2D12b1(a1, b1, g1, a2, b2, g2)
        dg2d12b1=dg2D12b1(a1, b1, g1, a2, b2, g2)

        da1d12g1=dg1d12a1
        db1d12g1=dg1d12b1
        dg1d12g1=dg1D12g1(a1, b1, g1, a2, b2, g2)
        da2d12g1 = dg2D12a1(a2, b2, g2, a1, b1, g1)
        db2d12g1 = dg2D12b1(a2, b2, g2, a1, b1, g1)
        dg2d12g1=dg2D12g1(a1, b1, g1, a2, b2, g2)

        da1d12a2=da2d12a1
        db1d12a2=da2d12b1
        dg1d12a2=da2d12g1
        da2d12a2 = da1D12a1(a2, b2, g2, a1, b1, g1)
        db2d12a2 = db1D12a1(a2, b2, g2, a1, b1, g1)
        dg2d12a2 = dg1D12a1(a2, b2, g2, a1, b1, g1)

        da1d12b2=db2d12a1
        db1d12b2=db2d12b1
        dg1d12b2=db2d12g1
        da2d12b2=db2d12a2
        db2d12b2 = db1D12b1(a2, b2, g2, a1, b1, g1)
        dg2d12b2 = dg1D12b1(a2, b2, g2, a1, b1, g1)

        da1d12g2=dg2d12a1
        db1d12g2=dg2d12b1
        dg1d12g2=dg2d12g1
        da2d12g2=dg2d12a2
        db2d12g2=dg2d12b2
        dg2d12g2 = dg1D12g1(a2, b2, g2, a1, b1, g1)


        ### Spring 1-3 spring_arr[1]###
        # D function
        d13=D12(a1, b1, g1, a3, b3, g3)
        # First derivatives of D function
        da1d13=da1D12(a1, b1, g1, a3, b3, g3)
        db1d13=db1D12(a1, b1, g1, a3, b3, g3)
        dg1d13=dg1D12(a1, b1, g1, a3, b3, g3)
        da3d13=da1D12(a3, b3, g3, a1, b1, g1)
        db3d13=db1D12(a3, b3, g3, a1, b1, g1)
        dg3d13=dg1D12(a3, b3, g3, a1, b1, g1)
        # Second derivatives of D function
        da1d13a1=da1D12a1(a1, b1, g1, a3, b3, g3)
        db1d13a1=db1D12a1(a1, b1, g1, a3, b3, g3)
        dg1d13a1=dg1D12a1(a1, b1, g1, a3, b3, g3)
        da3d13a1=da2D12a1(a1, b1, g1, a3, b3, g3)
        db3d13a1=db2D12a1(a1, b1, g1, a3, b3, g3)
        dg3d13a1=dg2D12a1(a1, b1, g1, a3, b3, g3)
        
        da1d13b1=db1d13a1
        db1d13b1=db1D12b1(a1, b1, g1, a3, b3, g3)
        dg1d13b1=dg1D12b1(a1, b1, g1, a3, b3, g3)
        da3d13b1 = db2D12a1(a3, b3, g3, a1, b1, g1)
        db3d13b1=db2D12b1(a1, b1, g1, a3, b3, g3)
        dg3d13b1=dg2D12b1(a1, b1, g1, a3, b3, g3)

        da1d13g1=dg1d13a1
        db1d13g1=dg1d13b1
        dg1d13g1=dg1D12g1(a1, b1, g1, a3, b3, g3)
        da3d13g1 = dg2D12a1(a3, b3, g3, a1, b1, g1)
        db3d13g1 = dg2D12b1(a3, b3, g3, a1, b1, g1)
        dg3d13g1=dg2D12g1(a1, b1, g1, a3, b3, g3)

        da1d13a3=da3d13a1
        db1d13a3=da3d13b1
        dg1d13a3=da3d13g1
        da3d13a3 = da1D12a1(a3, b3, g3, a1, b1, g1)
        db3d13a3 = db1D12a1(a3, b3, g3, a1, b1, g1)
        dg3d13a3 = dg1D12a1(a3, b3, g3, a1, b1, g1)

        da1d13b3=db3d13a1
        db1d13b3=db3d13b1
        dg1d13b3=db3d13g1
        da3d13b3=db3d13a3
        db3d13b3 = db1D12b1(a3, b3, g3, a1, b1, g1)
        dg3d13b3 = dg1D12b1(a3, b3, g3, a1, b1, g1)

        da1d13g3=dg3d13a1
        db1d13g3=dg3d13b1
        dg1d13g3=dg3d13g1
        da3d13g3=dg3d13a3
        db3d13g3=dg3d13b3
        dg3d13g3 = dg1D12g1(a3, b3, g3, a1, b1, g1)


        ### Spring 1-5 spring_arr[2]###
        # D function
        d15=D12(a1, b1, g1, a5, b5, g5)
        # First derivatives of D function
        da1d15=da1D12(a1, b1, g1, a5, b5, g5)
        db1d15=db1D12(a1, b1, g1, a5, b5, g5)
        dg1d15=dg1D12(a1, b1, g1, a5, b5, g5)
        da5d15=da1D12(a5, b5, g5, a1, b1, g1)
        db5d15=db1D12(a5, b5, g5, a1, b1, g1)
        dg5d15=dg1D12(a5, b5, g5, a1, b1, g1)
        # Second derivatives of D function
        da1d15a1=da1D12a1(a1, b1, g1, a5, b5, g5)
        db1d15a1=db1D12a1(a1, b1, g1, a5, b5, g5)
        dg1d15a1=dg1D12a1(a1, b1, g1, a5, b5, g5)
        da5d15a1=da2D12a1(a1, b1, g1, a5, b5, g5)
        db5d15a1=db2D12a1(a1, b1, g1, a5, b5, g5)
        dg5d15a1=dg2D12a1(a1, b1, g1, a5, b5, g5)
        
        da1d15b1=db1d15a1
        db1d15b1=db1D12b1(a1, b1, g1, a5, b5, g5)
        dg1d15b1=dg1D12b1(a1, b1, g1, a5, b5, g5)
        da5d15b1 = db2D12a1(a5, b5, g5, a1, b1, g1)
        db5d15b1=db2D12b1(a1, b1, g1, a5, b5, g5)
        dg5d15b1=dg2D12b1(a1, b1, g1, a5, b5, g5)

        da1d15g1=dg1d15a1
        db1d15g1=dg1d15b1
        dg1d15g1=dg1D12g1(a1, b1, g1, a5, b5, g5)
        da5d15g1 = dg2D12a1(a5, b5, g5, a1, b1, g1)
        db5d15g1 = dg2D12b1(a5, b5, g5, a1, b1, g1)
        dg5d15g1=dg2D12g1(a1, b1, g1, a5, b5, g5)

        da1d15a5=da5d15a1
        db1d15a5=da5d15b1
        dg1d15a5=da5d15g1
        da5d15a5 = da1D12a1(a5, b5, g5, a1, b1, g1)
        db5d15a5 = db1D12a1(a5, b5, g5, a1, b1, g1)
        dg5d15a5 = dg1D12a1(a5, b5, g5, a1, b1, g1)

        da1d15b5=db5d15a1
        db1d15b5=db5d15b1
        dg1d15b5=db5d15g1
        da5d15b5=db5d15a5
        db5d15b5 = db1D12b1(a5, b5, g5, a1, b1, g1)
        dg5d15b5 = dg1D12b1(a5, b5, g5, a1, b1, g1)

        da1d15g5=dg5d15a1
        db1d15g5=dg5d15b1
        dg1d15g5=dg5d15g1
        da5d15g5=dg5d15a5
        db5d15g5=dg5d15b5
        dg5d15g5 = dg1D12g1(a5, b5, g5, a1, b1, g1)


        ### Spring 2-4 spring_arr[3]###
        # D function
        d24=D12(a2, b2, g2, a4, b4, g4)
        # First derivatives of D function
        da2d24=da1D12(a2, b2, g2, a4, b4, g4)
        db2d24=db1D12(a2, b2, g2, a4, b4, g4)
        dg2d24=dg1D12(a2, b2, g2, a4, b4, g4)
        da4d24=da1D12(a4, b4, g4, a2, b2, g2)
        db4d24=db1D12(a4, b4, g4, a2, b2, g2)
        dg4d24=dg1D12(a4, b4, g4, a2, b2, g2)
        # Second derivatives of D function
        da2d24a2=da1D12a1(a2, b2, g2, a4, b4, g4)
        db2d24a2=db1D12a1(a2, b2, g2, a4, b4, g4)
        dg2d24a2=dg1D12a1(a2, b2, g2, a4, b4, g4)
        da4d24a2=da2D12a1(a2, b2, g2, a4, b4, g4)
        db4d24a2=db2D12a1(a2, b2, g2, a4, b4, g4)
        dg4d24a2=dg2D12a1(a2, b2, g2, a4, b4, g4)
        
        da2d24b2=db2d24a2
        db2d24b2=db1D12b1(a2, b2, g2, a4, b4, g4)
        dg2d24b2=dg1D12b1(a2, b2, g2, a4, b4, g4)
        da4d24b2 = db2D12a1(a4, b4, g4, a2, b2, g2)
        db4d24b2=db2D12b1(a2, b2, g2, a4, b4, g4)
        dg4d24b2=dg2D12b1(a2, b2, g2, a4, b4, g4)

        da2d24g2=dg2d24a2
        db2d24g2=dg2d24b2
        dg2d24g2=dg1D12g1(a2, b2, g2, a4, b4, g4)
        da4d24g2 = dg2D12a1(a4, b4, g4, a2, b2, g2)
        db4d24g2 = dg2D12b1(a4, b4, g4, a2, b2, g2)
        dg4d24g2=dg2D12g1(a2, b2, g2, a4, b4, g4)

        da2d24a4=da4d24a2
        db2d24a4=da4d24b2
        dg2d24a4=da4d24g2
        da4d24a4 = da1D12a1(a4, b4, g4, a2, b2, g2)
        db4d24a4 = db1D12a1(a4, b4, g4, a2, b2, g2)
        dg4d24a4 = dg1D12a1(a4, b4, g4, a2, b2, g2)

        da2d24b4=db4d24a2
        db2d24b4=db4d24b2
        dg2d24b4=db4d24g2
        da4d24b4=db4d24a4
        db4d24b4 = db1D12b1(a4, b4, g4, a2, b2, g2)
        dg4d24b4 = dg1D12b1(a4, b4, g4, a2, b2, g2)

        da2d24g4=dg4d24a2
        db2d24g4=dg4d24b2
        dg2d24g4=dg4d24g2
        da4d24g4=dg4d24a4
        db4d24g4=dg4d24b4
        dg4d24g4 = dg1D12g1(a4, b4, g4, a2, b2, g2)


        ### Spring 2-6 spring_arr[4]###
        # D function
        d26=D12(a2, b2, g2, a6, b6, g6)
        # First derivatives of D function
        da2d26=da1D12(a2, b2, g2, a6, b6, g6)
        db2d26=db1D12(a2, b2, g2, a6, b6, g6)
        dg2d26=dg1D12(a2, b2, g2, a6, b6, g6)
        da6d26=da1D12(a6, b6, g6, a2, b2, g2)
        db6d26=db1D12(a6, b6, g6, a2, b2, g2)
        dg6d26=dg1D12(a6, b6, g6, a2, b2, g2)
        # Second derivatives of D function
        da2d26a2=da1D12a1(a2, b2, g2, a6, b6, g6)
        db2d26a2=db1D12a1(a2, b2, g2, a6, b6, g6)
        dg2d26a2=dg1D12a1(a2, b2, g2, a6, b6, g6)
        da6d26a2=da2D12a1(a2, b2, g2, a6, b6, g6)
        db6d26a2=db2D12a1(a2, b2, g2, a6, b6, g6)
        dg6d26a2=dg2D12a1(a2, b2, g2, a6, b6, g6)
        
        da2d26b2=db2d26a2
        db2d26b2=db1D12b1(a2, b2, g2, a6, b6, g6)
        dg2d26b2=dg1D12b1(a2, b2, g2, a6, b6, g6)
        da6d26b2 = db2D12a1(a6, b6, g6, a2, b2, g2)
        db6d26b2=db2D12b1(a2, b2, g2, a6, b6, g6)
        dg6d26b2=dg2D12b1(a2, b2, g2, a6, b6, g6)

        da2d26g2=dg2d26a2
        db2d26g2=dg2d26b2
        dg2d26g2=dg1D12g1(a2, b2, g2, a6, b6, g6)
        da6d26g2 = dg2D12a1(a6, b6, g6, a2, b2, g2)
        db6d26g2 = dg2D12b1(a6, b6, g6, a2, b2, g2)
        dg6d26g2=dg2D12g1(a2, b2, g2, a6, b6, g6)

        da2d26a6=da6d26a2
        db2d26a6=da6d26b2
        dg2d26a6=da6d26g2
        da6d26a6 = da1D12a1(a6, b6, g6, a2, b2, g2)
        db6d26a6 = db1D12a1(a6, b6, g6, a2, b2, g2)
        dg6d26a6 = dg1D12a1(a6, b6, g6, a2, b2, g2)

        da2d26b6=db6d26a2
        db2d26b6=db6d26b2
        dg2d26b6=db6d26g2
        da6d26b6=db6d26a6
        db6d26b6 = db1D12b1(a6, b6, g6, a2, b2, g2)
        dg6d26b6 = dg1D12b1(a6, b6, g6, a2, b2, g2)

        da2d26g6=dg6d26a2
        db2d26g6=dg6d26b2
        dg2d26g6=dg6d26g2
        da6d26g6=dg6d26a6
        db6d26g6=dg6d26b6
        dg6d26g6 = dg1D12g1(a6, b6, g6, a2, b2, g2)


        ### Spring 3-4 spring_arr[5]###
        # D function
        d34=D12(a3, b3, g3, a4, b4, g4)
        # First derivatives of D function
        da3d34=da1D12(a3, b3, g3, a4, b4, g4)
        db3d34=db1D12(a3, b3, g3, a4, b4, g4)
        dg3d34=dg1D12(a3, b3, g3, a4, b4, g4)
        da4d34=da1D12(a4, b4, g4, a3, b3, g3)
        db4d34=db1D12(a4, b4, g4, a3, b3, g3)
        dg4d34=dg1D12(a4, b4, g4, a3, b3, g3)
        # Second derivatives of D function
        da3d34a3=da1D12a1(a3, b3, g3, a4, b4, g4)
        db3d34a3=db1D12a1(a3, b3, g3, a4, b4, g4)
        dg3d34a3=dg1D12a1(a3, b3, g3, a4, b4, g4)
        da4d34a3=da2D12a1(a3, b3, g3, a4, b4, g4)
        db4d34a3=db2D12a1(a3, b3, g3, a4, b4, g4)
        dg4d34a3=dg2D12a1(a3, b3, g3, a4, b4, g4)
        
        da3d34b3=db3d34a3
        db3d34b3=db1D12b1(a3, b3, g3, a4, b4, g4)
        dg3d34b3=dg1D12b1(a3, b3, g3, a4, b4, g4)
        da4d34b3 = db2D12a1(a4, b4, g4, a3, b3, g3)
        db4d34b3=db2D12b1(a3, b3, g3, a4, b4, g4)
        dg4d34b3=dg2D12b1(a3, b3, g3, a4, b4, g4)

        da3d34g3=dg3d34a3
        db3d34g3=dg3d34b3
        dg3d34g3=dg1D12g1(a3, b3, g3, a4, b4, g4)
        da4d34g3 = dg2D12a1(a4, b4, g4, a3, b3, g3)
        db4d34g3 = dg2D12b1(a4, b4, g4, a3, b3, g3)
        dg4d34g3=dg2D12g1(a3, b3, g3, a4, b4, g4)

        da3d34a4=da4d34a3
        db3d34a4=da4d34b3
        dg3d34a4=da4d34g3
        da4d34a4 = da1D12a1(a4, b4, g4, a3, b3, g3)
        db4d34a4 = db1D12a1(a4, b4, g4, a3, b3, g3)
        dg4d34a4 = dg1D12a1(a4, b4, g4, a3, b3, g3)

        da3d34b4=db4d34a3
        db3d34b4=db4d34b3
        dg3d34b4=db4d34g3
        da4d34b4=db4d34a4
        db4d34b4 = db1D12b1(a4, b4, g4, a3, b3, g3)
        dg4d34b4 = dg1D12b1(a4, b4, g4, a3, b3, g3)

        da3d34g4=dg4d34a3
        db3d34g4=dg4d34b3
        dg3d34g4=dg4d34g3
        da4d34g4=dg4d34a4
        db4d34g4=dg4d34b4
        dg4d34g4 = dg1D12g1(a4, b4, g4, a3, b3, g3)


        ### Spring 3-7 spring_arr[6]###
        # D function
        d37=D12(a3, b3, g3, a7, b7, g7)
        # First derivatives of D function
        da3d37=da1D12(a3, b3, g3, a7, b7, g7)
        db3d37=db1D12(a3, b3, g3, a7, b7, g7)
        dg3d37=dg1D12(a3, b3, g3, a7, b7, g7)
        da7d37=da1D12(a7, b7, g7, a3, b3, g3)
        db7d37=db1D12(a7, b7, g7, a3, b3, g3)
        dg7d37=dg1D12(a7, b7, g7, a3, b3, g3)
        # Second derivatives of D function
        da3d37a3=da1D12a1(a3, b3, g3, a7, b7, g7)
        db3d37a3=db1D12a1(a3, b3, g3, a7, b7, g7)
        dg3d37a3=dg1D12a1(a3, b3, g3, a7, b7, g7)
        da7d37a3=da2D12a1(a3, b3, g3, a7, b7, g7)
        db7d37a3=db2D12a1(a3, b3, g3, a7, b7, g7)
        dg7d37a3=dg2D12a1(a3, b3, g3, a7, b7, g7)
        
        da3d37b3=db3d37a3
        db3d37b3=db1D12b1(a3, b3, g3, a7, b7, g7)
        dg3d37b3=dg1D12b1(a3, b3, g3, a7, b7, g7)
        da7d37b3 = db2D12a1(a7, b7, g7, a3, b3, g3)
        db7d37b3=db2D12b1(a3, b3, g3, a7, b7, g7)
        dg7d37b3=dg2D12b1(a3, b3, g3, a7, b7, g7)

        da3d37g3=dg3d37a3
        db3d37g3=dg3d37b3
        dg3d37g3=dg1D12g1(a3, b3, g3, a7, b7, g7)
        da7d37g3 = dg2D12a1(a7, b7, g7, a3, b3, g3)
        db7d37g3 = dg2D12b1(a7, b7, g7, a3, b3, g3)
        dg7d37g3=dg2D12g1(a3, b3, g3, a7, b7, g7)

        da3d37a7=da7d37a3
        db3d37a7=da7d37b3
        dg3d37a7=da7d37g3
        da7d37a7 = da1D12a1(a7, b7, g7, a3, b3, g3)
        db7d37a7 = db1D12a1(a7, b7, g7, a3, b3, g3)
        dg7d37a7 = dg1D12a1(a7, b7, g7, a3, b3, g3)

        da3d37b7=db7d37a3
        db3d37b7=db7d37b3
        dg3d37b7=db7d37g3
        da7d37b7=db7d37a7
        db7d37b7 = db1D12b1(a7, b7, g7, a3, b3, g3)
        dg7d37b7 = dg1D12b1(a7, b7, g7, a3, b3, g3)

        da3d37g7=dg7d37a3
        db3d37g7=dg7d37b3
        dg3d37g7=dg7d37g3
        da7d37g7=dg7d37a7
        db7d37g7=dg7d37b7
        dg7d37g7 = dg1D12g1(a7, b7, g7, a3, b3, g3)


        ### Spring 4-8 spring_arr[7]###
        # D function
        d48=D12(a4, b4, g4, a8, b8, g8)
        # First derivatives of D function
        da4d48=da1D12(a4, b4, g4, a8, b8, g8)
        db4d48=db1D12(a4, b4, g4, a8, b8, g8)
        dg4d48=dg1D12(a4, b4, g4, a8, b8, g8)
        da8d48=da1D12(a8, b8, g8, a4, b4, g4)
        db8d48=db1D12(a8, b8, g8, a4, b4, g4)
        dg8d48=dg1D12(a8, b8, g8, a4, b4, g4)
        # Second derivatives of D function
        da4d48a4=da1D12a1(a4, b4, g4, a8, b8, g8)
        db4d48a4=db1D12a1(a4, b4, g4, a8, b8, g8)
        dg4d48a4=dg1D12a1(a4, b4, g4, a8, b8, g8)
        da8d48a4=da2D12a1(a4, b4, g4, a8, b8, g8)
        db8d48a4=db2D12a1(a4, b4, g4, a8, b8, g8)
        dg8d48a4=dg2D12a1(a4, b4, g4, a8, b8, g8)
        
        da4d48b4=db4d48a4
        db4d48b4=db1D12b1(a4, b4, g4, a8, b8, g8)
        dg4d48b4=dg1D12b1(a4, b4, g4, a8, b8, g8)
        da8d48b4 = db2D12a1(a8, b8, g8, a4, b4, g4)
        db8d48b4=db2D12b1(a4, b4, g4, a8, b8, g8)
        dg8d48b4=dg2D12b1(a4, b4, g4, a8, b8, g8)

        da4d48g4=dg4d48a4
        db4d48g4=dg4d48b4
        dg4d48g4=dg1D12g1(a4, b4, g4, a8, b8, g8)
        da8d48g4 = dg2D12a1(a8, b8, g8, a4, b4, g4)
        db8d48g4 = dg2D12b1(a8, b8, g8, a4, b4, g4)
        dg8d48g4=dg2D12g1(a4, b4, g4, a8, b8, g8)

        da4d48a8=da8d48a4
        db4d48a8=da8d48b4
        dg4d48a8=da8d48g4
        da8d48a8 = da1D12a1(a8, b8, g8, a4, b4, g4)
        db8d48a8 = db1D12a1(a8, b8, g8, a4, b4, g4)
        dg8d48a8 = dg1D12a1(a8, b8, g8, a4, b4, g4)

        da4d48b8=db8d48a4
        db4d48b8=db8d48b4
        dg4d48b8=db8d48g4
        da8d48b8=db8d48a8
        db8d48b8 = db1D12b1(a8, b8, g8, a4, b4, g4)
        dg8d48b8 = dg1D12b1(a8, b8, g8, a4, b4, g4)

        da4d48g8=dg8d48a4
        db4d48g8=dg8d48b4
        dg4d48g8=dg8d48g4
        da8d48g8=dg8d48a8
        db8d48g8=dg8d48b8
        dg8d48g8 = dg1D12g1(a8, b8, g8, a4, b4, g4)


        ### Spring 5-6 spring_arr[8]###
        # D function
        d56=D12(a5, b5, g5, a6, b6, g6)
        # First derivatives of D function
        da5d56=da1D12(a5, b5, g5, a6, b6, g6)
        db5d56=db1D12(a5, b5, g5, a6, b6, g6)
        dg5d56=dg1D12(a5, b5, g5, a6, b6, g6)
        da6d56=da1D12(a6, b6, g6, a5, b5, g5)
        db6d56=db1D12(a6, b6, g6, a5, b5, g5)
        dg6d56=dg1D12(a6, b6, g6, a5, b5, g5)
        # Second derivatives of D function
        da5d56a5=da1D12a1(a5, b5, g5, a6, b6, g6)
        db5d56a5=db1D12a1(a5, b5, g5, a6, b6, g6)
        dg5d56a5=dg1D12a1(a5, b5, g5, a6, b6, g6)
        da6d56a5=da2D12a1(a5, b5, g5, a6, b6, g6)
        db6d56a5=db2D12a1(a5, b5, g5, a6, b6, g6)
        dg6d56a5=dg2D12a1(a5, b5, g5, a6, b6, g6)
        
        da5d56b5=db5d56a5
        db5d56b5=db1D12b1(a5, b5, g5, a6, b6, g6)
        dg5d56b5=dg1D12b1(a5, b5, g5, a6, b6, g6)
        da6d56b5 = db2D12a1(a6, b6, g6, a5, b5, g5)
        db6d56b5=db2D12b1(a5, b5, g5, a6, b6, g6)
        dg6d56b5=dg2D12b1(a5, b5, g5, a6, b6, g6)

        da5d56g5=dg5d56a5
        db5d56g5=dg5d56b5
        dg5d56g5=dg1D12g1(a5, b5, g5, a6, b6, g6)
        da6d56g5 = dg2D12a1(a6, b6, g6, a5, b5, g5)
        db6d56g5 = dg2D12b1(a6, b6, g6, a5, b5, g5)
        dg6d56g5=dg2D12g1(a5, b5, g5, a6, b6, g6)

        da5d56a6=da6d56a5
        db5d56a6=da6d56b5
        dg5d56a6=da6d56g5
        da6d56a6 = da1D12a1(a6, b6, g6, a5, b5, g5)
        db6d56a6 = db1D12a1(a6, b6, g6, a5, b5, g5)
        dg6d56a6 = dg1D12a1(a6, b6, g6, a5, b5, g5)

        da5d56b6=db6d56a5
        db5d56b6=db6d56b5
        dg5d56b6=db6d56g5
        da6d56b6=db6d56a6
        db6d56b6 = db1D12b1(a6, b6, g6, a5, b5, g5)
        dg6d56b6 = dg1D12b1(a6, b6, g6, a5, b5, g5)

        da5d56g6=dg6d56a5
        db5d56g6=dg6d56b5
        dg5d56g6=dg6d56g5
        da6d56g6=dg6d56a6
        db6d56g6=dg6d56b6
        dg6d56g6 = dg1D12g1(a6, b6, g6, a5, b5, g5)


        ### Spring 5-7 spring_arr[9]###
        # D function
        d57=D12(a5, b5, g5, a7, b7, g7)
        # First derivatives of D function
        da5d57=da1D12(a5, b5, g5, a7, b7, g7)
        db5d57=db1D12(a5, b5, g5, a7, b7, g7)
        dg5d57=dg1D12(a5, b5, g5, a7, b7, g7)
        da7d57=da1D12(a7, b7, g7, a5, b5, g5)
        db7d57=db1D12(a7, b7, g7, a5, b5, g5)
        dg7d57=dg1D12(a7, b7, g7, a5, b5, g5)
        # Second derivatives of D function
        da5d57a5=da1D12a1(a5, b5, g5, a7, b7, g7)
        db5d57a5=db1D12a1(a5, b5, g5, a7, b7, g7)
        dg5d57a5=dg1D12a1(a5, b5, g5, a7, b7, g7)
        da7d57a5=da2D12a1(a5, b5, g5, a7, b7, g7)
        db7d57a5=db2D12a1(a5, b5, g5, a7, b7, g7)
        dg7d57a5=dg2D12a1(a5, b5, g5, a7, b7, g7)
        
        da5d57b5=db5d57a5
        db5d57b5=db1D12b1(a5, b5, g5, a7, b7, g7)
        dg5d57b5=dg1D12b1(a5, b5, g5, a7, b7, g7)
        da7d57b5 = db2D12a1(a7, b7, g7, a5, b5, g5)
        db7d57b5=db2D12b1(a5, b5, g5, a7, b7, g7)
        dg7d57b5=dg2D12b1(a5, b5, g5, a7, b7, g7)

        da5d57g5=dg5d57a5
        db5d57g5=dg5d57b5
        dg5d57g5=dg1D12g1(a5, b5, g5, a7, b7, g7)
        da7d57g5 = dg2D12a1(a7, b7, g7, a5, b5, g5)
        db7d57g5 = dg2D12b1(a7, b7, g7, a5, b5, g5)
        dg7d57g5=dg2D12g1(a5, b5, g5, a7, b7, g7)

        da5d57a7=da7d57a5
        db5d57a7=da7d57b5
        dg5d57a7=da7d57g5
        da7d57a7 = da1D12a1(a7, b7, g7, a5, b5, g5)
        db7d57a7 = db1D12a1(a7, b7, g7, a5, b5, g5)
        dg7d57a7 = dg1D12a1(a7, b7, g7, a5, b5, g5)

        da5d57b7=db7d57a5
        db5d57b7=db7d57b5
        dg5d57b7=db7d57g5
        da7d57b7=db7d57a7
        db7d57b7 = db1D12b1(a7, b7, g7, a5, b5, g5)
        dg7d57b7 = dg1D12b1(a7, b7, g7, a5, b5, g5)

        da5d57g7=dg7d57a5
        db5d57g7=dg7d57b5
        dg5d57g7=dg7d57g5
        da7d57g7=dg7d57a7
        db7d57g7=dg7d57b7
        dg7d57g7 = dg1D12g1(a7, b7, g7, a5, b5, g5)


        ### Spring 6-8 spring_arr[10]###
        # D function
        d68=D12(a6, b6, g6, a8, b8, g8)
        # First derivatives of D function
        da6d68=da1D12(a6, b6, g6, a8, b8, g8)
        db6d68=db1D12(a6, b6, g6, a8, b8, g8)
        dg6d68=dg1D12(a6, b6, g6, a8, b8, g8)
        da8d68=da1D12(a8, b8, g8, a6, b6, g6)
        db8d68=db1D12(a8, b8, g8, a6, b6, g6)
        dg8d68=dg1D12(a8, b8, g8, a6, b6, g6)
        # Second derivatives of D function
        da6d68a6=da1D12a1(a6, b6, g6, a8, b8, g8)
        db6d68a6=db1D12a1(a6, b6, g6, a8, b8, g8)
        dg6d68a6=dg1D12a1(a6, b6, g6, a8, b8, g8)
        da8d68a6=da2D12a1(a6, b6, g6, a8, b8, g8)
        db8d68a6=db2D12a1(a6, b6, g6, a8, b8, g8)
        dg8d68a6=dg2D12a1(a6, b6, g6, a8, b8, g8)
        
        da6d68b6=db6d68a6
        db6d68b6=db1D12b1(a6, b6, g6, a8, b8, g8)
        dg6d68b6=dg1D12b1(a6, b6, g6, a8, b8, g8)
        da8d68b6 = db2D12a1(a8, b8, g8, a6, b6, g6)
        db8d68b6=db2D12b1(a6, b6, g6, a8, b8, g8)
        dg8d68b6=dg2D12b1(a6, b6, g6, a8, b8, g8)

        da6d68g6=dg6d68a6
        db6d68g6=dg6d68b6
        dg6d68g6=dg1D12g1(a6, b6, g6, a8, b8, g8)
        da8d68g6 = dg2D12a1(a8, b8, g8, a6, b6, g6)
        db8d68g6 = dg2D12b1(a8, b8, g8, a6, b6, g6)
        dg8d68g6=dg2D12g1(a6, b6, g6, a8, b8, g8)

        da6d68a8=da8d68a6
        db6d68a8=da8d68b6
        dg6d68a8=da8d68g6
        da8d68a8 = da1D12a1(a8, b8, g8, a6, b6, g6)
        db8d68a8 = db1D12a1(a8, b8, g8, a6, b6, g6)
        dg8d68a8 = dg1D12a1(a8, b8, g8, a6, b6, g6)

        da6d68b8=db8d68a6
        db6d68b8=db8d68b6
        dg6d68b8=db8d68g6
        da8d68b8=db8d68a8
        db8d68b8 = db1D12b1(a8, b8, g8, a6, b6, g6)
        dg8d68b8 = dg1D12b1(a8, b8, g8, a6, b6, g6)

        da6d68g8=dg8d68a6
        db6d68g8=dg8d68b6
        dg6d68g8=dg8d68g6
        da8d68g8=dg8d68a8
        db8d68g8=dg8d68b8
        dg8d68g8 = dg1D12g1(a8, b8, g8, a6, b6, g6)


        ### Spring 7-8 spring_arr[11]###
        # D function
        d78=D12(a7, b7, g7, a8, b8, g8)
        # First derivatives of D function
        da7d78=da1D12(a7, b7, g7, a8, b8, g8)
        db7d78=db1D12(a7, b7, g7, a8, b8, g8)
        dg7d78=dg1D12(a7, b7, g7, a8, b8, g8)
        da8d78=da1D12(a8, b8, g8, a7, b7, g7)
        db8d78=db1D12(a8, b8, g8, a7, b7, g7)
        dg8d78=dg1D12(a8, b8, g8, a7, b7, g7)
        # Second derivatives of D function
        da7d78a7=da1D12a1(a6, b6, g6, a8, b8, g8)
        db7d78a7=db1D12a1(a6, b6, g6, a8, b8, g8)
        dg7d78a7=dg1D12a1(a6, b6, g6, a8, b8, g8)
        da8d78a7=da2D12a1(a6, b6, g6, a8, b8, g8)
        db8d78a7=db2D12a1(a6, b6, g6, a8, b8, g8)
        dg8d78a7=dg2D12a1(a6, b6, g6, a8, b8, g8)
        
        da7d78b7=db7d78a7
        db7d78b7=db1D12b1(a6, b6, g6, a8, b8, g8)
        dg7d78b7=dg1D12b1(a6, b6, g6, a8, b8, g8)
        da8d78b7 = db2D12a1(a8, b8, g8, a6, b6, g6)
        db8d78b7=db2D12b1(a6, b6, g6, a8, b8, g8)
        dg8d78b7=dg2D12b1(a6, b6, g6, a8, b8, g8)

        da7d78g7=dg7d78a7
        db7d78g7=dg7d78b7
        dg7d78g7=dg1D12g1(a6, b6, g6, a8, b8, g8)
        da8d78g7 = dg2D12a1(a8, b8, g8, a6, b6, g6)
        db8d78g7 = dg2D12b1(a8, b8, g8, a6, b6, g6)
        dg8d78g7=dg2D12g1(a6, b6, g6, a8, b8, g8)

        da7d78a8=da8d78a7
        db7d78a8=da8d78b7
        dg7d78a8=da8d78g7
        da8d78a8 = da1D12a1(a8, b8, g8, a6, b6, g6)
        db8d78a8 = db1D12a1(a8, b8, g8, a6, b6, g6)
        dg8d78a8 = dg1D12a1(a8, b8, g8, a6, b6, g6)

        da7d78b8=db8d78a7
        db7d78b8=db8d78b7
        dg7d78b8=db8d78g7
        da8d78b8=db8d78a8
        db8d78b8 = db1D12b1(a8, b8, g8, a6, b6, g6)
        dg8d78b8 = dg1D12b1(a8, b8, g8, a6, b6, g6)

        da7d78g8=dg8d78a7
        db7d78g8=dg8d78b7
        dg7d78g8=dg8d78g7
        da8d78g8=dg8d78a8
        db8d78g8=dg8d78b8
        dg8d78g8 = dg1D12g1(a8, b8, g8, a6, b6, g6)
        

        return array([

            # ---------- #
            #     V1     #
            # ---------- #
            
            # ad1 a1
            [-spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da1d12/(d12**2. - 1.) - da1d12a1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da1d13/(d13**2. - 1.) - da1d13a1) ) +
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d15**2. - 1. ))*( (da1d15*da1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da1d15*0./1. + d15*da1d15*da1d15/(d15**2. - 1.) - da1d15a1) ),

            # ad1 b1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db1d12/(d12**2. - 1.) - db1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db1d13/(d13**2. - 1.) - db1d13a1) ) +
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d15**2. - 1. ))*( (da1d15*db1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da1d15*0./1. + d15*da1d15*db1d15/(d15**2. - 1.) - db1d15a1) ),

            # ad1 g1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg1d12/(d12**2. - 1.) - dg1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg1d13/(d13**2. - 1.) - dg1d13a1) ) +
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d15**2. - 1. ))*( (da1d15*dg1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da1d15*0./1. + d15*da1d15*dg1d15/(d15**2. - 1.) - dg1d15a1) ),

            # ad1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da2d12/(d12**2. - 1.) - da2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db2d12/(d12**2. - 1.) - db2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg2d12/(d12**2. - 1.) - dg2d12a1) ),

            # ad1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da3d13/(d13**2. - 1.) - da3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db3d13/(d13**2. - 1.) - db3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg3d13/(d13**2. - 1.) - dg3d13a1) ),

            # ad1 a4,b4,g4
            0.,0.,0.,

            # ad1 a5,b5,g5
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d15**2. - 1. ))*( (da1d15*da5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da1d15*0./1. + d15*da1d15*da5d15/(d15**2. - 1.) - da5d15a1) ),
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d15**2. - 1. ))*( (da1d15*db5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da1d15*0./1. + d15*da1d15*db5d15/(d15**2. - 1.) - db5d15a1) ),
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d15**2. - 1. ))*( (da1d15*dg5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da1d15*0./1. + d15*da1d15*dg5d15/(d15**2. - 1.) - dg5d15a1) ),

            # ad1 a6,b6,g6
            0.,0.,0.,

            # ad1 a7,b7,g7
            0.,0.,0.,

            # ad1 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # bd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*sinh(2.*a1)/(cosh(a1)*cosh(a1)) + d12*db1d12*da1d12/(d12**2. - 1.) - da1d12b1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*sinh(2.*a1)/(cosh(a1)*cosh(a1)) + d13*db1d13*da1d13/(d13**2. - 1.) - da1d13b1) ) +
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d15**2. - 1. ))*( (db1d15*da1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db1d15*sinh(2.*a1)/(cosh(a1)*cosh(a1)) + d15*db1d15*da1d15/(d15**2. - 1.) - da1d15b1) ),

            # bd1 b1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db1d12/(d12**2. - 1.) - db1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*db1d13/(d13**2. - 1.) - db1d13b1) ) +
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d15**2. - 1. ))*( (db1d15*db1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db1d15*0./(cosh(a1)*cosh(a1)) + d15*db1d15*db1d15/(d15**2. - 1.) - db1d15b1) ),

            # bd1 g1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*dg1d12/(d12**2. - 1.) - dg1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*dg1d13/(d13**2. - 1.) - dg1d13b1) ) +
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d15**2. - 1. ))*( (db1d15*dg1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db1d15*0./(cosh(a1)*cosh(a1)) + d15*db1d15*dg1d15/(d15**2. - 1.) - dg1d15b1) ),

            # bd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*da2d12/(d12**2. - 1.) - da2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db2d12/(d12**2. - 1.) - db2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*dg2d12/(d12**2. - 1.) - dg2d12b1) ),

            # bd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*da3d13/(d13**2. - 1.) - da3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*db3d13/(d13**2. - 1.) - db3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*dg3d13/(d13**2. - 1.) - dg3d13b1) ),

            # bd1 a4,b4,g4
            0.,0.,0.,

            # bd1 a5,b5,g5
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d15**2. - 1. ))*( (db1d15*da5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db1d15*0./(cosh(a1)*cosh(a1)) + d15*db1d15*da5d15/(d15**2. - 1.) - da5d15b1) ),
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d15**2. - 1. ))*( (db1d15*db5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db1d15*0./(cosh(a1)*cosh(a1)) + d15*db1d15*db5d15/(d15**2. - 1.) - db5d15b1) ),
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d15**2. - 1. ))*( (db1d15*dg5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db1d15*0./(cosh(a1)*cosh(a1)) + d15*db1d15*dg5d15/(d15**2. - 1.) - dg5d15b1) ),

            # bd1 a6,b6,g6
            0.,0.,0.,

            # bd1 a7,b7,g7
            0.,0.,0.,

            # bd1 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # gd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sinh(2.*a1)*cosh(b1)*cosh(b1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*da1d12/(d12**2. - 1.) - da1d12g1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sinh(2.*a1)*cosh(b1)*cosh(b1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*da1d13/(d13**2. - 1.) - da1d13g1) ) +
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d15**2. - 1. ))*( (dg1d15*da1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg1d15*sinh(2.*a1)*cosh(b1)*cosh(b1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d15*dg1d15*da1d15/(d15**2. - 1.) - da1d15g1) ),

            # gd1 b1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sinh(2.*b1)*cosh(a1)*cosh(a1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*db1d12/(d12**2. - 1.) - db1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sinh(2.*b1)*cosh(a1)*cosh(a1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*db1d13/(d13**2. - 1.) - db1d13g1) ) +
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d15**2. - 1. ))*( (dg1d15*db1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg1d15*sinh(2.*b1)*cosh(a1)*cosh(a1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d15*dg1d15*db1d15/(d15**2. - 1.) - db1d15g1) ),

            # gd1 g1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*dg1d12/(d12**2. - 1.) - dg1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*dg1d13/(d13**2. - 1.) - dg1d13g1) ) +
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d15**2. - 1. ))*( (dg1d15*dg1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg1d15*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d15*dg1d15*dg1d15/(d15**2. - 1.) - dg1d15g1) ),

            # gd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*da2d12/(d12**2. - 1.) - da2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*db2d12/(d12**2. - 1.) - db2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*dg2d12/(d12**2. - 1.) - dg2d12g1) ),

            # gd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*da3d13/(d13**2. - 1.) - da3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*db3d13/(d13**2. - 1.) - db3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*dg3d13/(d13**2. - 1.) - dg3d13g1) ),

            # gd1 a4,b4,g4
            0.,0.,0.,

            # gd1 a5,b5,g5
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d15**2. - 1. ))*( (dg1d15*da5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg1d15*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d15*dg1d15*da5d15/(d15**2. - 1.) - da5d15g1) ),
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d15**2. - 1. ))*( (dg1d15*db5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg1d15*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d15*dg1d15*db5d15/(d15**2. - 1.) - db5d15g1) ),
            -spring_arr[2][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d15**2. - 1. ))*( (dg1d15*dg5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg1d15*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d15*dg1d15*dg5d15/(d15**2. - 1.) - dg5d15g1) ),

            # gd1 a6,b6,g6
            0.,0.,0.,

            # gd1 a7,b7,g7
            0.,0.,0.,

            # gd1 a8,b8,g8
            0.,0.,0.],

            # ---------- #
            #     V2     #
            # ---------- #

            # ad2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da1d12/(d12**2. - 1.) - da1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db1d12/(d12**2. - 1.) - db1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg1d12/(d12**2. - 1.) - dg1d12a2) ),

            # ad2 a2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da2d12/(d12**2. - 1.) - da2d12a2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da2d24*0./1. + d24*da2d24*da2d24/(d24**2. - 1.) - da2d24a2) ) +
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d26**2. - 1. ))*( (da2d26*da2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da2d26*0./1. + d26*da2d26*da2d26/(d26**2. - 1.) - da2d26a2) ),

            # ad2 b2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db2d12/(d12**2. - 1.) - db2d12a2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da2d24*0./1. + d24*da2d24*db2d24/(d24**2. - 1.) - db2d24a2) ) +
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d26**2. - 1. ))*( (da2d26*db2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da2d26*0./1. + d26*da2d26*db2d26/(d26**2. - 1.) - db2d26a2) ),

            # ad2 g2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg2d12/(d12**2. - 1.) - dg2d12a2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da2d24*0./1. + d24*da2d24*dg2d24/(d24**2. - 1.) - dg2d24a2) ) +
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d26**2. - 1. ))*( (da2d26*dg2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da2d26*0./1. + d26*da2d26*dg2d26/(d26**2. - 1.) - dg2d26a2) ),

            # ad2 a3,b3,g3
            0.,0.,0.,

            # ad2 a4,b4,g4
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da2d24*0./1. + d24*da2d24*da4d24/(d24**2. - 1.) - da4d24a2) ),
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da2d24*0./1. + d24*da2d24*db4d24/(d24**2. - 1.) - db4d24a2) ), 
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da2d24*0./1. + d24*da2d24*dg4d24/(d24**2. - 1.) - dg4d24a2) ),

            # ad2 a5,b5,g5
            0.,0.,0.,

            # ad2 a6,b6,g6
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d26**2. - 1. ))*( (da2d26*da6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da2d26*0./1. + d26*da2d26*da6d26/(d26**2. - 1.) - da6d26a2) ),
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d26**2. - 1. ))*( (da2d26*db6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da2d26*0./1. + d26*da2d26*db6d26/(d26**2. - 1.) - db6d26a2) ),
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d26**2. - 1. ))*( (da2d26*dg6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da2d26*0./1. + d26*da2d26*dg6d26/(d26**2. - 1.) - dg6d26a2) ),

            # ad2 a7,b7,g7
            0.,0.,0.,

            # ad2 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # bd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*da1d12/(d12**2. - 1.) - da1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db1d12/(d12**2. - 1.) - db1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*dg1d12/(d12**2. - 1.) - dg1d12b2) ),

            # bd2 a2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*sinh(2.*a2)/(cosh(a2)*cosh(a2)) + d12*db2d12*da2d12/(d12**2. - 1.) - da2d12b2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db2d24*sinh(2.*a2)/(cosh(a2)*cosh(a2)) + d24*db2d24*da2d24/(d24**2. - 1.) - da2d24b2) ) +
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d26**2. - 1. ))*( (db2d26*da2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db2d26*sinh(2.*a2)/(cosh(a2)*cosh(a2)) + d26*db2d26*da2d26/(d26**2. - 1.) - da2d26b2) ),

            # bd2 b2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db2d12/(d12**2. - 1.) - db2d12b2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db2d24*0./(cosh(a2)*cosh(a2)) + d24*db2d24*db2d24/(d24**2. - 1.) - db2d24b2) ) +
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d26**2. - 1. ))*( (db2d26*db2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db2d26*0./(cosh(a2)*cosh(a2)) + d26*db2d26*db2d26/(d26**2. - 1.) - db2d26b2) ),

            # bd2 g2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*dg2d12/(d12**2. - 1.) - dg2d12b2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db2d24*0./(cosh(a2)*cosh(a2)) + d24*db2d24*dg2d24/(d24**2. - 1.) - dg2d24b2) ) +
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d26**2. - 1. ))*( (db2d26*dg2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db2d26*0./(cosh(a2)*cosh(a2)) + d26*db2d26*dg2d26/(d26**2. - 1.) - dg2d26b2) ),

            # bd2 a3,b3,g3
            0.,0.,0.,

            # bd2 a4,b4,g4
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db2d24*0./(cosh(a2)*cosh(a2)) + d24*db2d24*da4d24/(d24**2. - 1.) - da4d24b2) ), 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db2d24*0./(cosh(a2)*cosh(a2)) + d24*db2d24*db4d24/(d24**2. - 1.) - db4d24b2) ),
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db2d24*0./(cosh(a2)*cosh(a2)) + d24*db2d24*dg4d24/(d24**2. - 1.) - dg4d24b2) ),

            # bd2 a5,b5,g5
            0.,0.,0.,

            # bd2 a6
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d26**2. - 1. ))*( (db2d26*da6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db2d26*0./(cosh(a2)*cosh(a2)) + d26*db2d26*da6d26/(d26**2. - 1.) - da6d26b2) ),
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d26**2. - 1. ))*( (db2d26*db6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db2d26*0./(cosh(a2)*cosh(a2)) + d26*db2d26*db6d26/(d26**2. - 1.) - db6d26b2) ),
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d26**2. - 1. ))*( (db2d26*dg6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db2d26*0./(cosh(a2)*cosh(a2)) + d26*db2d26*dg6d26/(d26**2. - 1.) - dg6d26b2) ),

            # bd2 a7,b7,g7
            0.,0.,0.,

            # bd2 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # gd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*da1d12/(d12**2. - 1.) - da1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*db1d12/(d12**2. - 1.) - db1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*dg1d12/(d12**2. - 1.) - dg1d12g2) ),

            # gd2 a2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sinh(2.*a2)*cosh(b2)*cosh(b2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*da2d12/(d12**2. - 1.) - da2d12g2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg2d24*sinh(2.*a2)*cosh(b2)*cosh(b2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d24*dg2d24*da2d24/(d24**2. - 1.) - da2d24g2) ) +
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d26**2. - 1. ))*( (dg2d26*da2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg2d26*sinh(2.*a2)*cosh(b2)*cosh(b2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d26*dg2d26*da2d26/(d26**2. - 1.) - da2d26g2) ),

            # gd2 b2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sinh(2.*b2)*cosh(a2)*cosh(a2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*db2d12/(d12**2. - 1.) - db2d12g2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg2d24*sinh(2.*b2)*cosh(a2)*cosh(a2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d24*dg2d24*db2d24/(d24**2. - 1.) - db2d24g2) ) +
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d26**2. - 1. ))*( (dg2d26*db2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg2d26*sinh(2.*b2)*cosh(a2)*cosh(a2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d26*dg2d26*db2d26/(d26**2. - 1.) - db2d26g2) ),

            # gd2 g2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*dg2d12/(d12**2. - 1.) - dg2d12g2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg2d24*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d24*dg2d24*dg2d24/(d24**2. - 1.) - dg2d24g2) ) +
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d26**2. - 1. ))*( (dg2d26*dg2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg2d26*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d26*dg2d26*dg2d26/(d26**2. - 1.) - dg2d26g2) ),

            # gd2 a3,b3,g3
            0.,0.,0.,

            # gd2 a4,b4,g4 
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg2d24*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d24*dg2d24*da4d24/(d24**2. - 1.) - da4d24g2) ),
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg2d24*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d24*dg2d24*db4d24/(d24**2. - 1.) - db4d24g2) ),
            -spring_arr[3][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg2d24*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d24*dg2d24*dg4d24/(d24**2. - 1.) - dg4d24g2) ),

            # gd2 a5,b5,g5
            0.,0.,0.,

            # gd2 a6,b6,g6
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d26**2. - 1. ))*( (dg2d26*da6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg2d26*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d26*dg2d26*da6d26/(d26**2. - 1.) - da6d26g2) ),
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d26**2. - 1. ))*( (dg2d26*db6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg2d26*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d26*dg2d26*db6d26/(d26**2. - 1.) - db6d26g2) ),
            -spring_arr[4][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d26**2. - 1. ))*( (dg2d26*dg6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg2d26*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d26*dg2d26*dg6d26/(d26**2. - 1.) - dg6d26g2) ),

            # gd2 a7,b7,g7
            0.,0.,0.,

            # gd2 a8,b8,g8
            0.,0.,0.],

            # ---------- #
            #     V3     #
            # ---------- #

            # ad3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da1d13/(d13**2. - 1.) - da1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db1d13/(d13**2. - 1.) - db1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg1d13/(d13**2. - 1.) - dg1d13a3) ),

            # ad3 a2,b2,g2
            0.,0.,0.,

            # ad3 a3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da3d13/(d13**2. - 1.) - da3d13a3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*da3d34/(d34**2. - 1.) - da3d34a3) ) +
            -spring_arr[6][0]/(mass_arr[2]*1.*sqrt( d37**2. - 1. ))*( (da3d37*da3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da3d37*0./1. + d37*da3d37*da3d37/(d37**2. - 1.) - da3d37a3) ),

            # ad3 b3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db3d13/(d13**2. - 1.) - db3d13a3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*db3d34/(d34**2. - 1.) - db3d34a3) ) +
            -spring_arr[6][0]/(mass_arr[2]*1.*sqrt( d37**2. - 1. ))*( (da3d37*db3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da3d37*0./1. + d37*da3d37*db3d37/(d37**2. - 1.) - db3d37a3) ),

            # ad3 g3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg3d13/(d13**2. - 1.) - dg3d13a3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*dg3d34/(d34**2. - 1.) - dg3d34a3) ) +
            -spring_arr[6][0]/(mass_arr[2]*1.*sqrt( d37**2. - 1. ))*( (da3d37*dg3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da3d37*0./1. + d37*da3d37*dg3d37/(d37**2. - 1.) - dg3d37a3) ),

            # ad3 a4,b4,g4
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*da4d34/(d34**2. - 1.) - da4d34a3) ),
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*db4d34/(d34**2. - 1.) - db4d34a3) ),
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*dg4d34/(d34**2. - 1.) - dg4d34a3) ),

            # ad3 a5,b5,g5
            0.,0.,0.,

            # ad3 a6,b6,g6
            0.,0.,0.,

            # ad3 a7,b7,g7
            -spring_arr[6][0]/(mass_arr[2]*1.*sqrt( d37**2. - 1. ))*( (da3d37*da7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da3d37*0./1. + d37*da3d37*da7d37/(d37**2. - 1.) - da7d37a3) ),
            -spring_arr[6][0]/(mass_arr[2]*1.*sqrt( d37**2. - 1. ))*( (da3d37*db7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da3d37*0./1. + d37*da3d37*db7d37/(d37**2. - 1.) - db7d37a3) ),
            -spring_arr[6][0]/(mass_arr[2]*1.*sqrt( d37**2. - 1. ))*( (da3d37*dg7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da3d37*0./1. + d37*da3d37*dg7d37/(d37**2. - 1.) - dg7d37a3) ),

            # ad3 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # bd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*da1d13/(d13**2. - 1.) - da1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*db1d13/(d13**2. - 1.) - db1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*dg1d13/(d13**2. - 1.) - dg1d13b3) ),

            # bd3 a2,b2,g2
            0.,0.,0.,

            # bd3 a3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*sinh(2.*a3)/(cosh(a3)*cosh(a3)) + d13*db3d13*da3d13/(d13**2. - 1.) - da3d13b3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*sinh(2.*a3)/(cosh(a3)*cosh(a3)) + d34*db3d34*da3d34/(d34**2. - 1.) - da3d34b3) ) +
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d37**2. - 1. ))*( (db3d37*da3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db3d37*sinh(2.*a3)/(cosh(a3)*cosh(a3)) + d37*db3d37*da3d37/(d37**2. - 1.) - da3d37b3) ),

            # bd3 b3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*db3d13/(d13**2. - 1.) - db3d13b3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(cosh(a3)*cosh(a3)) + d34*db3d34*db3d34/(d34**2. - 1.) - db3d34b3) ) +
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d37**2. - 1. ))*( (db3d37*db3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db3d37*0./(cosh(a3)*cosh(a3)) + d37*db3d37*db3d37/(d37**2. - 1.) - db3d37b3) ),

            # bd3 g3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*dg3d13/(d13**2. - 1.) - dg3d13b3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(cosh(a3)*cosh(a3)) + d34*db3d34*dg3d34/(d34**2. - 1.) - dg3d34b3) ) +
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d37**2. - 1. ))*( (db3d37*dg3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db3d37*0./(cosh(a3)*cosh(a3)) + d37*db3d37*dg3d37/(d37**2. - 1.) - dg3d37b3) ),

            # bd3 a4,b4,g4
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(cosh(a3)*cosh(a3)) + d34*db3d34*da4d34/(d34**2. - 1.) - da4d34b3) ),
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(cosh(a3)*cosh(a3)) + d34*db3d34*db4d34/(d34**2. - 1.) - db4d34b3) ),
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(cosh(a3)*cosh(a3)) + d34*db3d34*dg4d34/(d34**2. - 1.) - dg4d34b3) ),

            # bd3 a5,b5,g5
            0.,0.,0.,

            # bd3 a6
            0.,0.,0.,

            # bd3 a7,b7,g7
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d37**2. - 1. ))*( (db3d37*da7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db3d37*0./(cosh(a3)*cosh(a3)) + d37*db3d37*da7d37/(d37**2. - 1.) - da7d37b3) ),
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d37**2. - 1. ))*( (db3d37*db7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db3d37*0./(cosh(a3)*cosh(a3)) + d37*db3d37*db7d37/(d37**2. - 1.) - db7d37b3) ),
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d37**2. - 1. ))*( (db3d37*dg7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db3d37*0./(cosh(a3)*cosh(a3)) + d37*db3d37*dg7d37/(d37**2. - 1.) - dg7d37b3) ),

            # bd3 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # gd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*da1d13/(d13**2. - 1.) - da1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*db1d13/(d13**2. - 1.) - db1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*dg1d13/(d13**2. - 1.) - dg1d13g3) ),

            # gd3 a2,b2,g2
            0.,0.,0.,

            # gd3 a3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sinh(2.*a3)*cosh(b3)*cosh(b3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*da3d13/(d13**2. - 1.) - da3d13g3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*sinh(2.*a3)*cosh(b3)*cosh(b3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d34*dg3d34*da3d34/(d34**2. - 1.) - da3d34g3) ) +
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d37**2. - 1. ))*( (dg3d37*da3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg3d37*sinh(2.*a3)*cosh(b3)*cosh(b3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d37*dg3d37*da3d37/(d37**2. - 1.) - da3d37g3) ),

            # gd3 b3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sinh(2.*b3)*cosh(a3)*cosh(a3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*db3d13/(d13**2. - 1.) - db3d13g3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*sinh(2.*b3)*cosh(a3)*cosh(a3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d34*dg3d34*db3d34/(d34**2. - 1.) - db3d34g3) ) +
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d37**2. - 1. ))*( (dg3d37*db3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg3d37*sinh(2.*b3)*cosh(a3)*cosh(a3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d37*dg3d37*db3d37/(d37**2. - 1.) - db3d37g3) ),

            # gd3 g3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*dg3d13/(d13**2. - 1.) - dg3d13g3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d34*dg3d34*dg3d34/(d34**2. - 1.) - dg3d34g3) ) +
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d37**2. - 1. ))*( (dg3d37*dg3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg3d37*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d37*dg3d37*dg3d37/(d37**2. - 1.) - dg3d37g3) ),

            # gd3 a4,b4,g4
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d34*dg3d34*da4d34/(d34**2. - 1.) - da4d34g3) ),
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d34*dg3d34*db4d34/(d34**2. - 1.) - db4d34g3) ),
            -spring_arr[5][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d34*dg3d34*dg4d34/(d34**2. - 1.) - dg4d34g3) ),

            # gd3 a5,b5,g5
            0.,0.,0.,

            # gd3 a6,b6,g6
            0.,0.,0.,

            # gd3 a7,b7,g7
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d37**2. - 1. ))*( (dg3d37*da7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg3d37*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d37*dg3d37*da7d37/(d37**2. - 1.) - da7d37g3) ),
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d37**2. - 1. ))*( (dg3d37*db7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg3d37*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d37*dg3d37*db7d37/(d37**2. - 1.) - db7d37g3) ),
            -spring_arr[6][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d37**2. - 1. ))*( (dg3d37*dg7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg3d37*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d37*dg3d37*dg7d37/(d37**2. - 1.) - dg7d37g3) ),

            # gd3 a8,b8,g8
            0.,0.,0.],

            # ---------- #
            #     V4     #
            # ---------- #

            # ad4 a1,b1,g1
            [0.,0.,0.,

            # ad4 a2,b2,g2
            -spring_arr[3][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da4d24*0./1. + d24*da4d24*da2d24/(d24**2. - 1.) - da2d24a4) ),
            -spring_arr[3][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da4d24*0./1. + d24*da4d24*db2d24/(d24**2. - 1.) - db2d24a4) ),
            -spring_arr[3][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da4d24*0./1. + d24*da4d24*dg2d24/(d24**2. - 1.) - dg2d24a4) ),

            # ad4 a3,b3,g3
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*da3d34/(d34**2. - 1.) - da3d34a4) ),
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*db3d34/(d34**2. - 1.) - db3d34a4) ),
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*dg3d34/(d34**2. - 1.) - dg3d34a4) ),

            # ad4 a4
            -spring_arr[3][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da4d24*0./1. + d24*da4d24*da4d24/(d24**2. - 1.) - da4d24a4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*da4d34/(d34**2. - 1.) - da4d34a4) ) +
            -spring_arr[7][0]/(mass_arr[3]*1.*sqrt( d48**2. - 1. ))*( (da4d48*da4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da4d48*0./1. + d48*da4d48*da4d48/(d48**2. - 1.) - da4d48a4) ),

            # ad4 b4
            -spring_arr[3][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da4d24*0./1. + d24*da4d24*db4d24/(d24**2. - 1.) - db4d24a4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*db4d34/(d34**2. - 1.) - db4d34a4) ) +
            -spring_arr[7][0]/(mass_arr[3]*1.*sqrt( d48**2. - 1. ))*( (da4d48*db4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da4d48*0./1. + d48*da4d48*db4d48/(d48**2. - 1.) - db4d48a4) ),

            # ad4 g4
            -spring_arr[3][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(da4d24*0./1. + d24*da4d24*dg4d24/(d24**2. - 1.) - dg4d24a4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*dg4d34/(d34**2. - 1.) - dg4d34a4) ) +
            -spring_arr[7][0]/(mass_arr[3]*1.*sqrt( d48**2. - 1. ))*( (da4d48*dg4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da4d48*0./1. + d48*da4d48*dg4d48/(d48**2. - 1.) - dg4d48a4) ),

            # ad4 a5,b5,g5
            0.,0.,0.,

            # ad4 a6,b6,g6
            0.,0.,0.,

            # ad4 a7,b7,g7
            0.,0.,0.,

            # ad4 a8,b8,g8
            -spring_arr[7][0]/(mass_arr[3]*1.*sqrt( d48**2. - 1. ))*( (da4d48*da8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da4d48*0./1. + d48*da4d48*da8d48/(d48**2. - 1.) - da8d48a4) ),
            -spring_arr[7][0]/(mass_arr[3]*1.*sqrt( d48**2. - 1. ))*( (da4d48*db8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da4d48*0./1. + d48*da4d48*db8d48/(d48**2. - 1.) - db8d48a4) ),
            -spring_arr[7][0]/(mass_arr[3]*1.*sqrt( d48**2. - 1. ))*( (da4d48*dg8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da4d48*0./1. + d48*da4d48*dg8d48/(d48**2. - 1.) - dg8d48a4) )],

            # ---------- #

            # bd4 a1,b1,g1
            [0.,0.,0.,

            # bd4 a2,b2,g2
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db4d24*0./(cosh(a4)*cosh(a4)) + d24*db4d24*da2d24/(d24**2. - 1.) - da2d24b4) ),
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db4d24*0./(cosh(a4)*cosh(a4)) + d24*db4d24*db2d24/(d24**2. - 1.) - db2d24b4) ),
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db4d24*0./(cosh(a4)*cosh(a4)) + d24*db4d24*dg2d24/(d24**2. - 1.) - dg2d24b4) ),

            # bd4 a3,b3,g3
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(cosh(a4)*cosh(a4)) + d34*db4d34*da3d34/(d34**2. - 1.) - da3d34b4) ),
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(cosh(a4)*cosh(a4)) + d34*db4d34*db3d34/(d34**2. - 1.) - db3d34b4) ),
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(cosh(a4)*cosh(a4)) + d34*db4d34*dg3d34/(d34**2. - 1.) - dg3d34b4) ),

            # bd4 a4
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db4d24*sinh(2.*a4)/(cosh(a4)*cosh(a4)) + d24*db4d24*da4d24/(d24**2. - 1.) - da4d24b4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*sinh(2.*a4)/(cosh(a4)*cosh(a4)) + d34*db4d34*da4d34/(d34**2. - 1.) - da4d34b4) ) +
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d48**2. - 1. ))*( (db4d48*da4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db4d48*sinh(2.*a4)/(cosh(a4)*cosh(a4)) + d48*db4d48*da4d48/(d48**2. - 1.) - da4d48b4) ),

            # bd4 b4
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db4d24*0./(cosh(a4)*cosh(a4)) + d24*db4d24*db4d24/(d24**2. - 1.) - db4d24b4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(cosh(a4)*cosh(a4)) + d34*db4d34*db4d34/(d34**2. - 1.) - db4d34b4) ) +
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d48**2. - 1. ))*( (db4d48*db4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db4d48*0./(cosh(a4)*cosh(a4)) + d48*db4d48*db4d48/(d48**2. - 1.) - db4d48b4) ),

            # bd4 g4
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(db4d24*0./(cosh(a4)*cosh(a4)) + d24*db4d24*dg4d24/(d24**2. - 1.) - dg4d24b4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(cosh(a4)*cosh(a4)) + d34*db4d34*dg4d34/(d34**2. - 1.) - dg4d34b4) ) +
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d48**2. - 1. ))*( (db4d48*dg4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db4d48*0./(cosh(a4)*cosh(a4)) + d48*db4d48*dg4d48/(d48**2. - 1.) - dg4d48b4) ),

            # bd4 a5,b5,g5
            0.,0.,0.,

            # bd4 a6,b6,g6
            0.,0.,0.,

            # bd4 a7,b7,g7
            0.,0.,0.,

            # bd4 a8,b8,g8
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d48**2. - 1. ))*( (db4d48*da8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db4d48*0./(cosh(a4)*cosh(a4)) + d48*db4d48*da8d48/(d48**2. - 1.) - da8d48b4) ),
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d48**2. - 1. ))*( (db4d48*db8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db4d48*0./(cosh(a4)*cosh(a4)) + d48*db4d48*db8d48/(d48**2. - 1.) - db8d48b4) ),
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*sqrt( d48**2. - 1. ))*( (db4d48*dg8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db4d48*0./(cosh(a4)*cosh(a4)) + d48*db4d48*dg8d48/(d48**2. - 1.) - dg8d48b4) )],

            # ---------- #

            # gd4 a1,b1,g1
            [0.,0.,0.,

            # gd4 a2,b2,g2
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg4d24*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d24*dg4d24*da2d24/(d24**2. - 1.) - da2d24g4) ),
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg4d24*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d24*dg4d24*db2d24/(d24**2. - 1.) - db2d24g4) ),
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg4d24*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d24*dg4d24*dg2d24/(d24**2. - 1.) - dg2d24g4) ),

            # gd4 a3,b3,g3
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d34*dg4d34*da3d34/(d34**2. - 1.) - da3d34g4) ),
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d34*dg4d34*db3d34/(d34**2. - 1.) - db3d34g4) ),
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d34*dg4d34*dg3d34/(d34**2. - 1.) - dg3d34g4) ),

            # gd4 a4
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg4d24*sinh(2.*a4)*cosh(b4)*cosh(b4)/(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d24*dg4d24*da4d24/(d24**2. - 1.) - da4d24g4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*sinh(2.*a4)*cosh(b4)*cosh(b4)/(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d34*dg4d34*da4d34/(d34**2. - 1.) - da4d34g4) ) +
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d48**2. - 1. ))*( (dg4d48*da4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg4d48*sinh(2.*a4)*cosh(b4)*cosh(b4)/(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d48*dg4d48*da4d48/(d48**2. - 1.) - da4d48g4) ),

            # gd4 b4
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg4d24*sinh(2.*b4)*cosh(a4)*cosh(a4)/(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d24*dg4d24*db4d24/(d24**2. - 1.) - db4d24g4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*sinh(2.*b4)*cosh(a4)*cosh(a4)/(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d34*dg4d34*db4d34/(d34**2. - 1.) - db4d34g4) ) +
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d48**2. - 1. ))*( (dg4d48*db4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg4d48*sinh(2.*b4)*cosh(a4)*cosh(a4)/(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d48*dg4d48*db4d48/(d48**2. - 1.) - db4d48g4) ),

            # gd4 g4
            -spring_arr[3][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[3][1] )*(dg4d24*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d24*dg4d24*dg4d24/(d24**2. - 1.) - dg4d24g4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d34*dg4d34*dg4d34/(d34**2. - 1.) - dg4d34g4) ) +
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d48**2. - 1. ))*( (dg4d48*dg4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg4d48*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d48*dg4d48*dg4d48/(d48**2. - 1.) - dg4d48g4) ),

            # gd4 a5,b5,g5
            0.,0.,0.,

            # gd4 a6,b6,g6
            0.,0.,0.,

            # gd4 a7,b7,g7
            0.,0.,0.,

            # gd4 a8,b8,g8
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d48**2. - 1. ))*( (dg4d48*da8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg4d48*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d48*dg4d48*da8d48/(d48**2. - 1.) - da8d48g4) ),
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d48**2. - 1. ))*( (dg4d48*db8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg4d48*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d48*dg4d48*db8d48/(d48**2. - 1.) - db8d48g4) ),
            -spring_arr[7][0]/(mass_arr[3]*cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)*sqrt( d48**2. - 1. ))*( (dg4d48*dg8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg4d48*0./(cosh(a4)*cosh(a4)*cosh(b4)*cosh(b4)) + d48*dg4d48*dg8d48/(d48**2. - 1.) - dg8d48g4) )],

            # ---------- #
            #     V5     #
            # ---------- #

            # ad5 a1,b1,g1
            [-spring_arr[2][0]/(mass_arr[4]*1.*sqrt( d15**2. - 1. ))*( (da5d15*da1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da5d15*0./1. + d15*da5d15*da1d15/(d15**2. - 1.) - da1d15a5) ), 
            -spring_arr[2][0]/(mass_arr[4]*1.*sqrt( d15**2. - 1. ))*( (da5d15*db1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da5d15*0./1. + d15*da5d15*db1d15/(d15**2. - 1.) - db1d15a5) ),
            -spring_arr[2][0]/(mass_arr[4]*1.*sqrt( d15**2. - 1. ))*( (da5d15*dg1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da5d15*0./1. + d15*da5d15*dg1d15/(d15**2. - 1.) - dg1d15a5) ),

            # ad5 a2,b2,g2
            0.,0.,0.,

            # ad5 a3,b3,g3
            0.,0.,0.,

            # ad5 a4,b4,g4
            0.,0.,0.,

            # ad5 a5
            -spring_arr[2][0]/(mass_arr[4]*1.*sqrt( d15**2. - 1. ))*( (da5d15*da5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da5d15*0./1. + d15*da5d15*da5d15/(d15**2. - 1.) - da5d15a5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*1.*sqrt( d56**2. - 1. ))*( (da5d56*da5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da5d56*0./1. + d56*da5d56*da5d56/(d56**2. - 1.) - da5d56a5) ) +
            -spring_arr[9][0]/(mass_arr[4]*1.*sqrt( d57**2. - 1. ))*( (da5d57*da5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da5d57*0./1. + d57*da5d57*da5d57/(d57**2. - 1.) - da5d57a5) ),

            # ad5 b5
            -spring_arr[2][0]/(mass_arr[4]*1.*sqrt( d15**2. - 1. ))*( (da5d15*db5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da5d15*0./1. + d15*da5d15*db5d15/(d15**2. - 1.) - db5d15a5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*1.*sqrt( d56**2. - 1. ))*( (da5d56*db5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da5d56*0./1. + d56*da5d56*db5d56/(d56**2. - 1.) - db5d56a5) ) +
            -spring_arr[9][0]/(mass_arr[4]*1.*sqrt( d57**2. - 1. ))*( (da5d57*db5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da5d57*0./1. + d57*da5d57*db5d57/(d57**2. - 1.) - db5d57a5) ),

            # ad5 g5
            -spring_arr[2][0]/(mass_arr[4]*1.*sqrt( d15**2. - 1. ))*( (da5d15*dg5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(da5d15*0./1. + d15*da5d15*dg5d15/(d15**2. - 1.) - dg5d15a5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*1.*sqrt( d56**2. - 1. ))*( (da5d56*dg5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da5d56*0./1. + d56*da5d56*dg5d56/(d56**2. - 1.) - dg5d56a5) ) +
            -spring_arr[9][0]/(mass_arr[4]*1.*sqrt( d57**2. - 1. ))*( (da5d57*dg5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da5d57*0./1. + d57*da5d57*dg5d57/(d57**2. - 1.) - dg5d57a5) ),

            # ad5 a6,b6,g6
            -spring_arr[8][0]/(mass_arr[4]*1.*sqrt( d56**2. - 1. ))*( (da5d56*da6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da5d56*0./1. + d56*da5d56*da6d56/(d56**2. - 1.) - da6d56a5) ), 
            -spring_arr[8][0]/(mass_arr[4]*1.*sqrt( d56**2. - 1. ))*( (da5d56*db6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da5d56*0./1. + d56*da5d56*db6d56/(d56**2. - 1.) - db6d56a5) ),
            -spring_arr[8][0]/(mass_arr[4]*1.*sqrt( d56**2. - 1. ))*( (da5d56*dg6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da5d56*0./1. + d56*da5d56*dg6d56/(d56**2. - 1.) - dg6d56a5) ),

            # ad5 a7,b7,g7
            -spring_arr[9][0]/(mass_arr[4]*1.*sqrt( d57**2. - 1. ))*( (da5d57*da7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da5d57*0./1. + d57*da5d57*da7d57/(d57**2. - 1.) - da7d57a5) ), 
            -spring_arr[9][0]/(mass_arr[4]*1.*sqrt( d57**2. - 1. ))*( (da5d57*db7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da5d57*0./1. + d57*da5d57*db7d57/(d57**2. - 1.) - db7d57a5) ),
            -spring_arr[9][0]/(mass_arr[4]*1.*sqrt( d57**2. - 1. ))*( (da5d57*dg7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da5d57*0./1. + d57*da5d57*dg7d57/(d57**2. - 1.) - dg7d57a5) ),

            # ad5 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # bd5 a1,b1,g1
            [-spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d15**2. - 1. ))*( (db5d15*da1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db5d15*0./(cosh(a5)*cosh(a5)) + d15*db5d15*da1d15/(d15**2. - 1.) - da1d15b5) ), 
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d15**2. - 1. ))*( (db5d15*db1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db5d15*0./(cosh(a5)*cosh(a5)) + d15*db5d15*db1d15/(d15**2. - 1.) - db1d15b5) ),
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d15**2. - 1. ))*( (db5d15*dg1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db5d15*0./(cosh(a5)*cosh(a5)) + d15*db5d15*dg1d15/(d15**2. - 1.) - dg1d15b5) ),

            # bd5 a2,b2,g2
            0.,0.,0.,

            # bd5 a3,b3,g3
            0.,0.,0.,

            # bd5 a4,b4,g4
            0.,0.,0.,

            # bd5 a5
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d15**2. - 1. ))*( (db5d15*da5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db5d15*sinh(2.*a5)/(cosh(a5)*cosh(a5)) + d15*db5d15*da5d15/(d15**2. - 1.) - da5d15b5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d56**2. - 1. ))*( (db5d56*da5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db5d56*sinh(2.*a5)/(cosh(a5)*cosh(a5)) + d56*db5d56*da5d56/(d56**2. - 1.) - da5d56b5) ) +
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d57**2. - 1. ))*( (db5d57*da5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db5d57*sinh(2.*a5)/(cosh(a5)*cosh(a5)) + d57*db5d57*da5d57/(d57**2. - 1.) - da5d57b5) ),

            # bd5 b5
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d15**2. - 1. ))*( (db5d15*db5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db5d15*0./(cosh(a5)*cosh(a5)) + d15*db5d15*db5d15/(d15**2. - 1.) - db5d15b5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d56**2. - 1. ))*( (db5d56*db5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db5d56*0./(cosh(a5)*cosh(a5)) + d56*db5d56*db5d56/(d56**2. - 1.) - db5d56b5) ) +
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d57**2. - 1. ))*( (db5d57*db5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db5d57*0./(cosh(a5)*cosh(a5)) + d57*db5d57*db5d57/(d57**2. - 1.) - db5d57b5) ),

            # bd5 g5
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d15**2. - 1. ))*( (db5d15*dg5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(db5d15*0./(cosh(a5)*cosh(a5)) + d15*db5d15*dg5d15/(d15**2. - 1.) - dg5d15b5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d56**2. - 1. ))*( (db5d56*dg5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db5d56*0./(cosh(a5)*cosh(a5)) + d56*db5d56*dg5d56/(d56**2. - 1.) - dg5d56b5) ) +
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d57**2. - 1. ))*( (db5d57*dg5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db5d57*0./(cosh(a5)*cosh(a5)) + d57*db5d57*dg5d57/(d57**2. - 1.) - dg5d57b5) ),

            # bd5 a6,b6,g6
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d56**2. - 1. ))*( (db5d56*da6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db5d56*0./(cosh(a5)*cosh(a5)) + d56*db5d56*da6d56/(d56**2. - 1.) - da6d56b5) ), 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d56**2. - 1. ))*( (db5d56*db6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db5d56*0./(cosh(a5)*cosh(a5)) + d56*db5d56*db6d56/(d56**2. - 1.) - db6d56b5) ),
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d56**2. - 1. ))*( (db5d56*dg6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db5d56*0./(cosh(a5)*cosh(a5)) + d56*db5d56*dg6d56/(d56**2. - 1.) - dg6d56b5) ),

            # bd5 a7,b7,g7
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d57**2. - 1. ))*( (db5d57*da7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db5d57*0./(cosh(a5)*cosh(a5)) + d57*db5d57*da7d57/(d57**2. - 1.) - da7d57b5) ), 
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d57**2. - 1. ))*( (db5d57*db7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db5d57*0./(cosh(a5)*cosh(a5)) + d57*db5d57*db7d57/(d57**2. - 1.) - db7d57b5) ),
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*sqrt( d57**2. - 1. ))*( (db5d57*dg7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db5d57*0./(cosh(a5)*cosh(a5)) + d57*db5d57*dg7d57/(d57**2. - 1.) - dg7d57b5) ),

            # bd5 a8,b8,g8
            0.,0.,0.],

            # ---------- #

            # gd5 a1,b1,g1
            [-spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d15**2. - 1. ))*( (dg5d15*da1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg5d15*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d15*dg5d15*da1d15/(d15**2. - 1.) - da1d15g5) ), 
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d15**2. - 1. ))*( (dg5d15*db1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg5d15*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d15*dg5d15*db1d15/(d15**2. - 1.) - db1d15g5) ),
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d15**2. - 1. ))*( (dg5d15*dg1d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg5d15*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d15*dg5d15*dg1d15/(d15**2. - 1.) - dg1d15g5) ),

            # gd5 a2,b2,g2
            0.,0.,0.,

            # gd5 a3,b3,g3
            0.,0.,0.,

            # gd5 a4,b4,g4
            0.,0.,0.,

            # gd5 a5
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d15**2. - 1. ))*( (dg5d15*da5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg5d15*sinh(2.*a5)*cosh(b5)*cosh(b5)/(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d15*dg5d15*da5d15/(d15**2. - 1.) - da5d15g5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d56**2. - 1. ))*( (dg5d56*da5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg5d56*sinh(2.*a5)*cosh(b5)*cosh(b5)/(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d56*dg5d56*da5d56/(d56**2. - 1.) - da5d56g5) ) +
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d57**2. - 1. ))*( (dg5d57*da5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg5d57*sinh(2.*a5)*cosh(b5)*cosh(b5)/(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d57*dg5d57*da5d57/(d57**2. - 1.) - da5d57g5) ),

            # gd5 b5
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d15**2. - 1. ))*( (dg5d15*db5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg5d15*sinh(2.*b5)*cosh(a5)*cosh(a5)/(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d15*dg5d15*db5d15/(d15**2. - 1.) - db5d15g5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d56**2. - 1. ))*( (dg5d56*db5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg5d56*sinh(2.*b5)*cosh(a5)*cosh(a5)/(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d56*dg5d56*db5d56/(d56**2. - 1.) - db5d56g5) ) +
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d57**2. - 1. ))*( (dg5d57*db5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg5d57*sinh(2.*b5)*cosh(a5)*cosh(a5)/(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d57*dg5d57*db5d57/(d57**2. - 1.) - db5d57g5) ),

            # gd5 g5
            -spring_arr[2][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d15**2. - 1. ))*( (dg5d15*dg5d15)/sqrt( d15**2. - 1.) - ( arccosh(d15) - spring_arr[2][1] )*(dg5d15*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d15*dg5d15*dg5d15/(d15**2. - 1.) - dg5d15g5) ) + 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d56**2. - 1. ))*( (dg5d56*dg5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg5d56*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d56*dg5d56*dg5d56/(d56**2. - 1.) - dg5d56g5) ) +
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d57**2. - 1. ))*( (dg5d57*dg5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg5d57*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d57*dg5d57*dg5d57/(d57**2. - 1.) - dg5d57g5) ),

            # gd5 a6,b6,g6
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d56**2. - 1. ))*( (dg5d56*da6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg5d56*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d56*dg5d56*da6d56/(d56**2. - 1.) - da6d56g5) ), 
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d56**2. - 1. ))*( (dg5d56*db6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg5d56*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d56*dg5d56*db6d56/(d56**2. - 1.) - db6d56g5) ),
            -spring_arr[8][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d56**2. - 1. ))*( (dg5d56*dg6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg5d56*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d56*dg5d56*dg6d56/(d56**2. - 1.) - dg6d56g5) ),

            # gd5 a7,b7,g7
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d57**2. - 1. ))*( (dg5d57*da7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg5d57*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d57*dg5d57*da7d57/(d57**2. - 1.) - da7d57g5) ), 
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d57**2. - 1. ))*( (dg5d57*db7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg5d57*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d57*dg5d57*db7d57/(d57**2. - 1.) - db7d57g5) ),
            -spring_arr[9][0]/(mass_arr[4]*cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)*sqrt( d57**2. - 1. ))*( (dg5d57*dg7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg5d57*0./(cosh(a5)*cosh(a5)*cosh(b5)*cosh(b5)) + d57*dg5d57*dg7d57/(d57**2. - 1.) - dg7d57g5) ),

            # gd5 a8,b8,g8
            0.,0.,0.],

            # ---------- #
            #     V6     #
            # ---------- #

            # ad6 a1,b1,g1
            [0.,0.,0.,

            # ad6 a2,b2,g2
            -spring_arr[4][0]/(mass_arr[5]*1.*sqrt( d26**2. - 1. ))*( (da6d26*da2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da6d26*0./1. + d26*da6d26*da2d26/(d26**2. - 1.) - da2d26a6) ), 
            -spring_arr[4][0]/(mass_arr[5]*1.*sqrt( d26**2. - 1. ))*( (da6d26*db2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da6d26*0./1. + d26*da6d26*db2d26/(d26**2. - 1.) - db2d26a6) ),
            -spring_arr[4][0]/(mass_arr[5]*1.*sqrt( d26**2. - 1. ))*( (da6d26*dg2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da6d26*0./1. + d26*da6d26*dg2d26/(d26**2. - 1.) - dg2d26a6) ),

            # ad6 a3,b3,g3
            0.,0.,0.,

            # ad6 a4,b4,g4
            0.,0.,0.,

            # ad6 a5,b5,g5
            -spring_arr[8][0]/(mass_arr[5]*1.*sqrt( d56**2. - 1. ))*( (da6d56*da5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(da6d56*0./1. + d56*da6d56*da5d56/(d56**2. - 1.) - da5d56a6) ), 
            -spring_arr[8][0]/(mass_arr[5]*1.*sqrt( d56**2. - 1. ))*( (da6d56*db5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(da6d56*0./1. + d56*da6d56*db5d56/(d56**2. - 1.) - db5d56a6) ),
            -spring_arr[8][0]/(mass_arr[5]*1.*sqrt( d56**2. - 1. ))*( (da6d56*dg5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(da6d56*0./1. + d56*da6d56*dg5d56/(d56**2. - 1.) - dg5d56a6) ),

            # ad6 a6
            -spring_arr[4][0]/(mass_arr[5]*1.*sqrt( d26**2. - 1. ))*( (da6d26*da6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da6d26*0./1. + d26*da6d26*da6d26/(d26**2. - 1.) - da6d26a6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*1.*sqrt( d56**2. - 1. ))*( (da6d56*da6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da6d56*0./1. + d56*da6d56*da6d56/(d56**2. - 1.) - da6d56a6) ) +
            -spring_arr[10][0]/(mass_arr[5]*1.*sqrt( d68**2. - 1. ))*( (da6d68*da6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da6d68*0./1. + d68*da6d68*da6d68/(d68**2. - 1.) - da6d68a6) ),

            # ad6 b6
            -spring_arr[4][0]/(mass_arr[5]*1.*sqrt( d26**2. - 1. ))*( (da6d26*db6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da6d26*0./1. + d26*da6d26*db6d26/(d26**2. - 1.) - db6d26a6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*1.*sqrt( d56**2. - 1. ))*( (da6d56*db6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da6d56*0./1. + d56*da6d56*db6d56/(d56**2. - 1.) - db6d56a6) ) +
            -spring_arr[10][0]/(mass_arr[5]*1.*sqrt( d68**2. - 1. ))*( (da6d68*db6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da6d68*0./1. + d68*da6d68*db6d68/(d68**2. - 1.) - db6d68a6) ),

            # ad6 g6
            -spring_arr[4][0]/(mass_arr[5]*1.*sqrt( d26**2. - 1. ))*( (da6d26*dg6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(da6d26*0./1. + d26*da6d26*dg6d26/(d26**2. - 1.) - dg6d26a6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*1.*sqrt( d56**2. - 1. ))*( (da6d56*dg6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(da6d56*0./1. + d56*da6d56*dg6d56/(d56**2. - 1.) - dg6d56a6) ) +
            -spring_arr[10][0]/(mass_arr[5]*1.*sqrt( d68**2. - 1. ))*( (da6d68*dg6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da6d68*0./1. + d68*da6d68*dg6d68/(d68**2. - 1.) - dg6d68a6) ),

            # ad6 a6,b6,g6
            0.,0.,0.,

            # ad6 a7,b7,g7
            0.,0.,0.

            # ad6 a8,b8,g8
            -spring_arr[10][0]/(mass_arr[5]*1.*sqrt( d68**2. - 1. ))*( (da6d68*da8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da6d68*0./1. + d68*da6d68*da8d68/(d68**2. - 1.) - da8d68a6) ), 
            -spring_arr[10][0]/(mass_arr[5]*1.*sqrt( d68**2. - 1. ))*( (da6d68*db8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da6d68*0./1. + d68*da6d68*db8d68/(d68**2. - 1.) - db8d68a6) ),
            -spring_arr[10][0]/(mass_arr[5]*1.*sqrt( d68**2. - 1. ))*( (da6d68*dg8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da6d68*0./1. + d68*da6d68*dg8d68/(d68**2. - 1.) - dg8d68a6) )],

            # ---------- #

            # bd6 a1,b1,g1
            [0.,0.,0.,

            # bd6 a2,b2,g2
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d26**2. - 1. ))*( (db6d26*da2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db6d26*0./(cosh(a6)*cosh(a6)) + d26*db6d26*da2d26/(d26**2. - 1.) - da2d26b6) ), 
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d26**2. - 1. ))*( (db6d26*db2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db6d26*0./(cosh(a6)*cosh(a6)) + d26*db6d26*db2d26/(d26**2. - 1.) - db2d26b6) ),
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d26**2. - 1. ))*( (db6d26*dg2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db6d26*0./(cosh(a6)*cosh(a6)) + d26*db6d26*dg2d26/(d26**2. - 1.) - dg2d26b6) ),

            # bd6 a3,b3,g3
            0.,0.,0.,

            # bd6 a4,b4,g4
            0.,0.,0.,

            # bd6 a5,b5,g5
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d56**2. - 1. ))*( (db6d56*da5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(db6d56*0./(cosh(a6)*cosh(a6)) + d56*db6d56*da5d56/(d56**2. - 1.) - da5d56b6) ), 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d56**2. - 1. ))*( (db6d56*db5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(db6d56*0./(cosh(a6)*cosh(a6)) + d56*db6d56*db5d56/(d56**2. - 1.) - db5d56b6) ),
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d56**2. - 1. ))*( (db6d56*dg5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(db6d56*0./(cosh(a6)*cosh(a6)) + d56*db6d56*dg5d56/(d56**2. - 1.) - dg5d56b6) ),

            # bd6 a6
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d26**2. - 1. ))*( (db6d26*da6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db6d26*sinh(2.*a6)/(cosh(a6)*cosh(a6)) + d26*db6d26*da6d26/(d26**2. - 1.) - da6d26b6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d56**2. - 1. ))*( (db6d56*da6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db6d56*sinh(2.*a6)/(cosh(a6)*cosh(a6)) + d56*db6d56*da6d56/(d56**2. - 1.) - da6d56b6) ) +
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d68**2. - 1. ))*( (db6d68*da6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db6d68*sinh(2.*a6)/(cosh(a6)*cosh(a6)) + d68*db6d68*da6d68/(d68**2. - 1.) - da6d68b6) ),

            # bd6 b6
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d26**2. - 1. ))*( (db6d26*db6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db6d26*0./(cosh(a6)*cosh(a6)) + d26*db6d26*db6d26/(d26**2. - 1.) - db6d26b6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d56**2. - 1. ))*( (db6d56*db6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db6d56*0./(cosh(a6)*cosh(a6)) + d56*db6d56*db6d56/(d56**2. - 1.) - db6d56b6) ) +
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d68**2. - 1. ))*( (db6d68*db6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db6d68*0./(cosh(a6)*cosh(a6)) + d68*db6d68*db6d68/(d68**2. - 1.) - db6d68b6) ),

            # bd6 g6
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d26**2. - 1. ))*( (db6d26*dg6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(db6d26*0./(cosh(a6)*cosh(a6)) + d26*db6d26*dg6d26/(d26**2. - 1.) - dg6d26b6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d56**2. - 1. ))*( (db6d56*dg6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(db6d56*0./(cosh(a6)*cosh(a6)) + d56*db6d56*dg6d56/(d56**2. - 1.) - dg6d56b6) ) +
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d68**2. - 1. ))*( (db6d68*dg6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db6d68*0./(cosh(a6)*cosh(a6)) + d68*db6d68*dg6d68/(d68**2. - 1.) - dg6d68b6) ),

            # bd6 a6,b6,g6
            0.,0.,0.,

            # bd6 a7,b7,g7
            0.,0.,0.

            # bd6 a8,b8,g8
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d68**2. - 1. ))*( (db6d68*da8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db6d68*0./(cosh(a6)*cosh(a6)) + d68*db6d68*da8d68/(d68**2. - 1.) - da8d68b6) ), 
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d68**2. - 1. ))*( (db6d68*db8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db6d68*0./(cosh(a6)*cosh(a6)) + d68*db6d68*db8d68/(d68**2. - 1.) - db8d68b6) ),
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*sqrt( d68**2. - 1. ))*( (db6d68*dg8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db6d68*0./(cosh(a6)*cosh(a6)) + d68*db6d68*dg8d68/(d68**2. - 1.) - dg8d68b6) )],

            # ---------- #

            # gd6 a1,b1,g1
            [0.,0.,0.,

            # gd6 a2,b2,g2
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d26**2. - 1. ))*( (dg6d26*da2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg6d26*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d26*dg6d26*da2d26/(d26**2. - 1.) - da2d26g6) ), 
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d26**2. - 1. ))*( (dg6d26*db2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg6d26*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d26*dg6d26*db2d26/(d26**2. - 1.) - db2d26g6) ),
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d26**2. - 1. ))*( (dg6d26*dg2d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg6d26*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d26*dg6d26*dg2d26/(d26**2. - 1.) - dg2d26g6) ),

            # gd6 a3,b3,g3
            0.,0.,0.,

            # gd6 a4,b4,g4
            0.,0.,0.,

            # gd6 a5,b5,g5
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d56**2. - 1. ))*( (dg6d56*da5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(dg6d56*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d56*dg6d56*da5d56/(d56**2. - 1.) - da5d56g6) ), 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d56**2. - 1. ))*( (dg6d56*db5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(dg6d56*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d56*dg6d56*db5d56/(d56**2. - 1.) - db5d56g6) ),
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d56**2. - 1. ))*( (dg6d56*dg5d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[4][1] )*(dg6d56*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d56*dg6d56*dg5d56/(d56**2. - 1.) - dg5d56g6) ),

            # gd6 a6
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d26**2. - 1. ))*( (dg6d26*da6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg6d26*sinh(2.*a6)*cosh(b6)*cosh(b6)/(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d26*dg6d26*da6d26/(d26**2. - 1.) - da6d26g6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d56**2. - 1. ))*( (dg6d56*da6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg6d56*sinh(2.*a6)*cosh(b6)*cosh(b6)/(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d56*dg6d56*da6d56/(d56**2. - 1.) - da6d56g6) ) +
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d68**2. - 1. ))*( (dg6d68*da6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg6d68*sinh(2.*a6)*cosh(b6)*cosh(b6)/(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d68*dg6d68*da6d68/(d68**2. - 1.) - da6d68g6) ),

            # gd6 b6
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d26**2. - 1. ))*( (dg6d26*db6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg6d26*sinh(2.*b6)*cosh(a6)*cosh(a6)/(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d26*dg6d26*db6d26/(d26**2. - 1.) - db6d26g6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d56**2. - 1. ))*( (dg6d56*db6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg6d56*sinh(2.*b6)*cosh(a6)*cosh(a6)/(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d56*dg6d56*db6d56/(d56**2. - 1.) - db6d56g6) ) +
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d68**2. - 1. ))*( (dg6d68*db6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg6d68*sinh(2.*b6)*cosh(a6)*cosh(a6)/(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d68*dg6d68*db6d68/(d68**2. - 1.) - db6d68g6) ),

            # gd6 g6
            -spring_arr[4][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d26**2. - 1. ))*( (dg6d26*dg6d26)/sqrt( d26**2. - 1.) - ( arccosh(d26) - spring_arr[4][1] )*(dg6d26*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d26*dg6d26*dg6d26/(d26**2. - 1.) - dg6d26g6) ) + 
            -spring_arr[8][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d56**2. - 1. ))*( (dg6d56*dg6d56)/sqrt( d56**2. - 1.) - ( arccosh(d56) - spring_arr[8][1] )*(dg6d56*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d56*dg6d56*dg6d56/(d56**2. - 1.) - dg6d56g6) ) +
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d68**2. - 1. ))*( (dg6d68*dg6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg6d68*0./(cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)) + d68*dg6d68*dg6d68/(d68**2. - 1.) - dg6d68g6) ),

            # gd6 a6,b6,g6
            0.,0.,0.,

            # gd6 a7,b7,g7
            0.,0.,0.

            # gd6 a8,b8,g8
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d68**2. - 1. ))*( (dg6d68*da8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg6d68*0./(cosh(a6)*cosh(a6)) + d68*dg6d68*da8d68/(d68**2. - 1.) - da8d68g6) ), 
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d68**2. - 1. ))*( (dg6d68*db8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg6d68*0./(cosh(a6)*cosh(a6)) + d68*dg6d68*db8d68/(d68**2. - 1.) - db8d68g6) ),
            -spring_arr[10][0]/(mass_arr[5]*cosh(a6)*cosh(a6)*cosh(b6)*cosh(b6)*sqrt( d68**2. - 1. ))*( (dg6d68*dg8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg6d68*0./(cosh(a6)*cosh(a6)) + d68*dg6d68*dg8d68/(d68**2. - 1.) - dg8d68g6) )],

            # ---------- #
            #     V7     #
            # ---------- #

            # ad7 a1,b1,g1
            [0.,0.,0.,

            # ad7 a2,b2,g2
            0.,0.,0.,

            # ad7 a3,b3,g3
            -spring_arr[6][0]/(mass_arr[6]*1.*sqrt( d37**2. - 1. ))*( (da7d37*da3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da7d37*0./1. + d37*da7d37*da3d37/(d37**2. - 1.) - da3d37a7) ), 
            -spring_arr[6][0]/(mass_arr[6]*1.*sqrt( d37**2. - 1. ))*( (da7d37*db3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da7d37*0./1. + d37*da7d37*db3d37/(d37**2. - 1.) - db3d37a7) ),
            -spring_arr[6][0]/(mass_arr[6]*1.*sqrt( d37**2. - 1. ))*( (da7d37*dg3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da7d37*0./1. + d37*da7d37*dg3d37/(d37**2. - 1.) - dg3d37a7) ),

            # ad7 a4,b4,g4
            0.,0.,0.,

            # ad7 a5,b5,g5
            -spring_arr[9][0]/(mass_arr[6]*1.*sqrt( d57**2. - 1. ))*( (da7d57*da5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da7d57*0./1. + d57*da7d57*da5d57/(d57**2. - 1.) - da5d57a7) ), 
            -spring_arr[9][0]/(mass_arr[6]*1.*sqrt( d57**2. - 1. ))*( (da7d57*db5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da7d57*0./1. + d57*da7d57*db5d57/(d57**2. - 1.) - db5d57a7) ),
            -spring_arr[9][0]/(mass_arr[6]*1.*sqrt( d57**2. - 1. ))*( (da7d57*dg5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da7d57*0./1. + d57*da7d57*dg5d57/(d57**2. - 1.) - dg5d57a7) ),

            # ad7 a6,b6,g6
            0.,0.,0.,

            # ad7 a7
            -spring_arr[6][0]/(mass_arr[6]*1.*sqrt( d37**2. - 1. ))*( (da7d37*da7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da7d37*0./1. + d37*da7d37*da7d37/(d37**2. - 1.) - da7d37a7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*1.*sqrt( d57**2. - 1. ))*( (da7d57*da7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da7d57*0./1. + d57*da7d57*da7d57/(d57**2. - 1.) - da7d57a7) ) +
            -spring_arr[11][0]/(mass_arr[6]*1.*sqrt( d78**2. - 1. ))*( (da7d78*da7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da7d78*0./1. + d78*da7d78*da7d78/(d78**2. - 1.) - da7d78a7) ),

            # ad7 b7
            -spring_arr[6][0]/(mass_arr[6]*1.*sqrt( d37**2. - 1. ))*( (da7d37*db7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da7d37*0./1. + d37*da7d37*db7d37/(d37**2. - 1.) - db7d37a7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*1.*sqrt( d57**2. - 1. ))*( (da7d57*db7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da7d57*0./1. + d57*da7d57*db7d57/(d57**2. - 1.) - db7d57a7) ) +
            -spring_arr[11][0]/(mass_arr[6]*1.*sqrt( d78**2. - 1. ))*( (da7d78*db7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da7d78*0./1. + d78*da7d78*db7d78/(d78**2. - 1.) - db7d78a7) ),

            # ad7 g7
            -spring_arr[6][0]/(mass_arr[6]*1.*sqrt( d37**2. - 1. ))*( (da7d37*dg7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(da7d37*0./1. + d37*da7d37*dg7d37/(d37**2. - 1.) - dg7d37a7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*1.*sqrt( d57**2. - 1. ))*( (da7d57*dg7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(da7d57*0./1. + d57*da7d57*dg7d57/(d57**2. - 1.) - dg7d57a7) ) +
            -spring_arr[11][0]/(mass_arr[6]*1.*sqrt( d78**2. - 1. ))*( (da7d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da7d78*0./1. + d78*da7d78*dg7d78/(d78**2. - 1.) - dg7d78a7) ),

            # ad7 a8,b8,g8
            -spring_arr[11][0]/(mass_arr[6]*1.*sqrt( d78**2. - 1. ))*( (da7d78*da8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da7d78*0./1. + d78*da7d78*da8d78/(d78**2. - 1.) - da8d78a7) ), 
            -spring_arr[11][0]/(mass_arr[6]*1.*sqrt( d78**2. - 1. ))*( (da7d78*db8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da7d78*0./1. + d78*da7d78*db8d78/(d78**2. - 1.) - db8d78a7) ),
            -spring_arr[11][0]/(mass_arr[6]*1.*sqrt( d78**2. - 1. ))*( (da7d78*dg8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da7d78*0./1. + d78*da7d78*dg8d78/(d78**2. - 1.) - dg8d78a7) )],

            # ---------- #

            # bd7 a1,b1,g1
            [0.,0.,0.,

            # bd7 a2,b2,g2
            0.,0.,0.,

            # bd7 a3,b3,g3
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d37**2. - 1. ))*( (db7d37*da3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db7d37*0./(cosh(a7)*cosh(a7)) + d37*db7d37*da3d37/(d37**2. - 1.) - da3d37b7) ), 
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d37**2. - 1. ))*( (db7d37*db3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db7d37*0./(cosh(a7)*cosh(a7)) + d37*db7d37*db3d37/(d37**2. - 1.) - db3d37b7) ),
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d37**2. - 1. ))*( (db7d37*dg3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db7d37*0./(cosh(a7)*cosh(a7)) + d37*db7d37*dg3d37/(d37**2. - 1.) - dg3d37b7) ),

            # bd7 a4,b4,g4
            0.,0.,0.,

            # bd7 a5,b5,g5
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d57**2. - 1. ))*( (db7d57*da5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db7d57*0./(cosh(a7)*cosh(a7)) + d57*db7d57*da5d57/(d57**2. - 1.) - da5d57b7) ), 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d57**2. - 1. ))*( (db7d57*db5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db7d57*0./(cosh(a7)*cosh(a7)) + d57*db7d57*db5d57/(d57**2. - 1.) - db5d57b7) ),
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d57**2. - 1. ))*( (db7d57*dg5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db7d57*0./(cosh(a7)*cosh(a7)) + d57*db7d57*dg5d57/(d57**2. - 1.) - dg5d57b7) ),

            # bd7 a6,b6,g6
            0.,0.,0.,

            # bd7 a7
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d37**2. - 1. ))*( (db7d37*da7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db7d37*sinh(2.*a7)/(cosh(a7)*cosh(a7)) + d37*db7d37*da7d37/(d37**2. - 1.) - da7d37b7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d57**2. - 1. ))*( (db7d57*da7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db7d57*sinh(2.*a7)/(cosh(a7)*cosh(a7)) + d57*db7d57*da7d57/(d57**2. - 1.) - da7d57b7) ) +
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d78**2. - 1. ))*( (db7d78*da7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db7d78*sinh(2.*a7)/(cosh(a7)*cosh(a7)) + d78*db7d78*da7d78/(d78**2. - 1.) - da7d78b7) ),

            # bd7 b7
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d37**2. - 1. ))*( (db7d37*db7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db7d37*0./(cosh(a7)*cosh(a7)) + d37*db7d37*db7d37/(d37**2. - 1.) - db7d37b7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d57**2. - 1. ))*( (db7d57*db7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db7d57*0./(cosh(a7)*cosh(a7)) + d57*db7d57*db7d57/(d57**2. - 1.) - db7d57b7) ) +
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d78**2. - 1. ))*( (db7d78*db7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db7d78*0./(cosh(a7)*cosh(a7)) + d78*db7d78*db7d78/(d78**2. - 1.) - db7d78b7) ),

            # bd7 g7
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d37**2. - 1. ))*( (db7d37*dg7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(db7d37*0./(cosh(a7)*cosh(a7)) + d37*db7d37*dg7d37/(d37**2. - 1.) - dg7d37b7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d57**2. - 1. ))*( (db7d57*dg7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(db7d57*0./(cosh(a7)*cosh(a7)) + d57*db7d57*dg7d57/(d57**2. - 1.) - dg7d57b7) ) +
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d78**2. - 1. ))*( (db7d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db7d78*0./(cosh(a7)*cosh(a7)) + d78*db7d78*dg7d78/(d78**2. - 1.) - dg7d78b7) ),

            # bd7 a8,b8,g8
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d78**2. - 1. ))*( (db7d78*da8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db7d78*0./(cosh(a7)*cosh(a7)) + d78*db7d78*da8d78/(d78**2. - 1.) - da8d78b7) ), 
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d78**2. - 1. ))*( (db7d78*db8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db7d78*0./(cosh(a7)*cosh(a7)) + d78*db7d78*db8d78/(d78**2. - 1.) - db8d78b7) ),
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*sqrt( d78**2. - 1. ))*( (db7d78*dg8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db7d78*0./(cosh(a7)*cosh(a7)) + d78*db7d78*dg8d78/(d78**2. - 1.) - dg8d78b7) )],

            # ---------- #

            # gd7 a1,b1,g1
            [0.,0.,0.,

            # gd7 a2,b2,g2
            0.,0.,0.,

            # gd7 a3,b3,g3
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d37**2. - 1. ))*( (dg7d37*da3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg7d37*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d37*dg7d37*da3d37/(d37**2. - 1.) - da3d37g7) ), 
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d37**2. - 1. ))*( (dg7d37*db3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg7d37*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d37*dg7d37*db3d37/(d37**2. - 1.) - db3d37g7) ),
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d37**2. - 1. ))*( (dg7d37*dg3d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg7d37*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d37*dg7d37*dg3d37/(d37**2. - 1.) - dg3d37g7) ),

            # gd7 a4,b4,g4
            0.,0.,0.,

            # gd7 a5,b5,g5
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d57**2. - 1. ))*( (dg7d57*da5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg7d57*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d57*dg7d57*da5d57/(d57**2. - 1.) - da5d57g7) ), 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d57**2. - 1. ))*( (dg7d57*db5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg7d57*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d57*dg7d57*db5d57/(d57**2. - 1.) - db5d57g7) ),
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d57**2. - 1. ))*( (dg7d57*dg5d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg7d57*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d57*dg7d57*dg5d57/(d57**2. - 1.) - dg5d57g7) ),

            # gd7 a6,b6,g6
            0.,0.,0.,

            # gd7 a7
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d37**2. - 1. ))*( (dg7d37*da7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg7d37*sinh(2.*a7)*cosh(b7)*cosh(b7)/(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d37*dg7d37*da7d37/(d37**2. - 1.) - da7d37g7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d57**2. - 1. ))*( (dg7d57*da7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg7d57*sinh(2.*a7)*cosh(b7)*cosh(b7)/(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d57*dg7d57*da7d57/(d57**2. - 1.) - da7d57g7) ) +
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d78**2. - 1. ))*( (dg7d78*da7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg7d78*sinh(2.*a7)*cosh(b7)*cosh(b7)/(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d78*dg7d78*da7d78/(d78**2. - 1.) - da7d78g7) ),

            # gd7 b7
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d37**2. - 1. ))*( (dg7d37*db7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg7d37*sinh(2.*b7)*cosh(a7)*cosh(a7)/(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d37*dg7d37*db7d37/(d37**2. - 1.) - db7d37g7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d57**2. - 1. ))*( (dg7d57*db7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg7d57*sinh(2.*b7)*cosh(a7)*cosh(a7)/(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d57*dg7d57*db7d57/(d57**2. - 1.) - db7d57g7) ) +
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d78**2. - 1. ))*( (dg7d78*db7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg7d78*sinh(2.*b7)*cosh(a7)*cosh(a7)/(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d78*dg7d78*db7d78/(d78**2. - 1.) - db7d78g7) ),

            # gd7 g7
            -spring_arr[6][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d37**2. - 1. ))*( (dg7d37*dg7d37)/sqrt( d37**2. - 1.) - ( arccosh(d37) - spring_arr[6][1] )*(dg7d37*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d37*dg7d37*dg7d37/(d37**2. - 1.) - dg7d37g7) ) + 
            -spring_arr[9][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d57**2. - 1. ))*( (dg7d57*dg7d57)/sqrt( d57**2. - 1.) - ( arccosh(d57) - spring_arr[9][1] )*(dg7d57*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d57*dg7d57*dg7d57/(d57**2. - 1.) - dg7d57g7) ) +
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d78**2. - 1. ))*( (dg7d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg7d78*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d78*dg7d78*dg7d78/(d78**2. - 1.) - dg7d78g7) ),

            # gd7 a8,b8,g8
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d78**2. - 1. ))*( (dg7d78*da8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg7d78*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d78*dg7d78*da8d78/(d78**2. - 1.) - da8d78g7) ), 
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d78**2. - 1. ))*( (dg7d78*db8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg7d78*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d78*dg7d78*db8d78/(d78**2. - 1.) - db8d78g7) ),
            -spring_arr[11][0]/(mass_arr[6]*cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)*sqrt( d78**2. - 1. ))*( (dg7d78*dg8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg7d78*0./(cosh(a7)*cosh(a7)*cosh(b7)*cosh(b7)) + d78*dg7d78*dg8d78/(d78**2. - 1.) - dg8d78g7) )],

            # ---------- #
            #     V8     #
            # ---------- #

            # ad8 a1,b1,g1
            [0.,0.,0.,

            # ad8 a2,b2,g2
            0.,0.,0.,

            # ad8 a3,b3,g3
            0.,0.,0.,

            # ad8 a4,b4,g4
            -spring_arr[7][0]/(mass_arr[7]*1.*sqrt( d48**2. - 1. ))*( (da8d48*da4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da8d48*0./1. + d48*da8d48*da4d48/(d48**2. - 1.) - da4d48a8) ), 
            -spring_arr[7][0]/(mass_arr[7]*1.*sqrt( d48**2. - 1. ))*( (da8d48*db4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da8d48*0./1. + d48*da8d48*db4d48/(d48**2. - 1.) - db4d48a8) ),
            -spring_arr[7][0]/(mass_arr[7]*1.*sqrt( d48**2. - 1. ))*( (da8d48*dg4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da8d48*0./1. + d48*da8d48*dg4d48/(d48**2. - 1.) - dg4d48a8) ),

            # ad8 a5,b5,g5
            0.,0.,0.,

            # ad8 a6,b6,g6
            -spring_arr[10][0]/(mass_arr[7]*1.*sqrt( d68**2. - 1. ))*( (da8d68*da6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da8d68*0./1. + d68*da8d68*da6d68/(d68**2. - 1.) - da6d68a8) ), 
            -spring_arr[10][0]/(mass_arr[7]*1.*sqrt( d68**2. - 1. ))*( (da8d68*db6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da8d68*0./1. + d68*da8d68*db6d68/(d68**2. - 1.) - db6d68a8) ),
            -spring_arr[10][0]/(mass_arr[7]*1.*sqrt( d68**2. - 1. ))*( (da8d68*dg6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da8d68*0./1. + d68*da8d68*dg6d68/(d68**2. - 1.) - dg6d68a8) ),

            # ad8 a7,b7,g7
            -spring_arr[11][0]/(mass_arr[7]*1.*sqrt( d78**2. - 1. ))*( (da8d78*da7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da8d78*0./1. + d78*da8d78*da7d78/(d78**2. - 1.) - da7d78a8) ), 
            -spring_arr[11][0]/(mass_arr[7]*1.*sqrt( d78**2. - 1. ))*( (da8d78*db7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da8d78*0./1. + d78*da8d78*db7d78/(d78**2. - 1.) - db7d78a8) ),
            -spring_arr[11][0]/(mass_arr[7]*1.*sqrt( d78**2. - 1. ))*( (da8d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da8d78*0./1. + d78*da8d78*dg7d78/(d78**2. - 1.) - dg7d78a8) )

            # ad8 a8
            -spring_arr[7][0]/(mass_arr[7]*1.*sqrt( d48**2. - 1. ))*( (da8d48*da8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da8d48*0./1. + d48*da8d48*da8d48/(d48**2. - 1.) - da8d48a8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*1.*sqrt( d68**2. - 1. ))*( (da8d68*da8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da8d68*0./1. + d68*da8d68*da8d68/(d68**2. - 1.) - da8d68a8) ) +
            -spring_arr[11][0]/(mass_arr[7]*1.*sqrt( d78**2. - 1. ))*( (da8d78*da8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da8d78*0./1. + d78*da8d78*da8d78/(d78**2. - 1.) - da8d78a8) ),

            # ad8 b8
            -spring_arr[7][0]/(mass_arr[7]*1.*sqrt( d48**2. - 1. ))*( (da8d48*db8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da8d48*0./1. + d48*da8d48*db8d48/(d48**2. - 1.) - db8d48a8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*1.*sqrt( d68**2. - 1. ))*( (da8d68*db8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da8d68*0./1. + d68*da8d68*db8d68/(d68**2. - 1.) - db8d68a8) ) +
            -spring_arr[11][0]/(mass_arr[7]*1.*sqrt( d78**2. - 1. ))*( (da8d78*db8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da8d78*0./1. + d78*da8d78*db8d78/(d78**2. - 1.) - db8d78a8) ),

            # ad8 g8
            -spring_arr[7][0]/(mass_arr[7]*1.*sqrt( d48**2. - 1. ))*( (da8d48*dg8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(da8d48*0./1. + d48*da8d48*dg8d48/(d48**2. - 1.) - dg8d48a8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*1.*sqrt( d68**2. - 1. ))*( (da8d68*dg8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(da8d68*0./1. + d68*da8d68*dg8d68/(d68**2. - 1.) - dg8d68a8) ) +
            -spring_arr[11][0]/(mass_arr[7]*1.*sqrt( d78**2. - 1. ))*( (da8d78*dg8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da8d78*0./1. + d78*da8d78*dg8d78/(d78**2. - 1.) - dg8d78a8) )],

            # ---------- #

            # bd8 a1,b1,g1
            [0.,0.,0.,

            # bd8 a2,b2,g2
            0.,0.,0.,

            # bd8 a3,b3,g3
            0.,0.,0.,

            # bd8 a4,b4,g4
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d48**2. - 1. ))*( (db8d48*da4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db8d48*0./(cosh(a8)*cosh(a8)) + d48*db8d48*da4d48/(d48**2. - 1.) - da4d48b8) ), 
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d48**2. - 1. ))*( (db8d48*db4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db8d48*0./(cosh(a8)*cosh(a8)) + d48*db8d48*db4d48/(d48**2. - 1.) - db4d48b8) ),
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d48**2. - 1. ))*( (db8d48*dg4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db8d48*0./(cosh(a8)*cosh(a8)) + d48*db8d48*dg4d48/(d48**2. - 1.) - dg4d48b8) ),

            # bd8 a5,b5,g5
            0.,0.,0.,

            # bd8 a6,b6,g6
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d68**2. - 1. ))*( (db8d68*da6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db8d68*0./(cosh(a8)*cosh(a8)) + d68*db8d68*da6d68/(d68**2. - 1.) - da6d68b8) ), 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d68**2. - 1. ))*( (db8d68*db6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db8d68*0./(cosh(a8)*cosh(a8)) + d68*db8d68*db6d68/(d68**2. - 1.) - db6d68b8) ),
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d68**2. - 1. ))*( (db8d68*dg6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db8d68*0./(cosh(a8)*cosh(a8)) + d68*db8d68*dg6d68/(d68**2. - 1.) - dg6d68b8) ),

            # bd8 a7,b7,g7
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d78**2. - 1. ))*( (db8d78*da7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db8d78*0./(cosh(a8)*cosh(a8)) + d78*db8d78*da7d78/(d78**2. - 1.) - da7d78b8) ), 
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d78**2. - 1. ))*( (db8d78*db7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db8d78*0./(cosh(a8)*cosh(a8)) + d78*db8d78*db7d78/(d78**2. - 1.) - db7d78b8) ),
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d78**2. - 1. ))*( (db8d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db8d78*0./(cosh(a8)*cosh(a8)) + d78*db8d78*dg7d78/(d78**2. - 1.) - dg7d78b8) )

            # bd8 a8
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d48**2. - 1. ))*( (db8d48*da8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db8d48*sinh(2.*a8)/(cosh(a8)*cosh(a8)) + d48*db8d48*da8d48/(d48**2. - 1.) - da8d48b8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d68**2. - 1. ))*( (db8d68*da8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db8d68*sinh(2.*a8)/(cosh(a8)*cosh(a8)) + d68*db8d68*da8d68/(d68**2. - 1.) - da8d68b8) ) +
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d78**2. - 1. ))*( (db8d78*da8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db8d78*sinh(2.*a8)/(cosh(a8)*cosh(a8)) + d78*db8d78*da8d78/(d78**2. - 1.) - da8d78b8) ),

            # bd8 b8
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d48**2. - 1. ))*( (db8d48*db8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db8d48*0./(cosh(a8)*cosh(a8)) + d48*db8d48*db8d48/(d48**2. - 1.) - db8d48b8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d68**2. - 1. ))*( (db8d68*db8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db8d68*0./(cosh(a8)*cosh(a8)) + d68*db8d68*db8d68/(d68**2. - 1.) - db8d68b8) ) +
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d78**2. - 1. ))*( (db8d78*db8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db8d78*0./(cosh(a8)*cosh(a8)) + d78*db8d78*db8d78/(d78**2. - 1.) - db8d78b8) ),

            # bd8 g8
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d48**2. - 1. ))*( (db8d48*dg8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(db8d48*0./(cosh(a8)*cosh(a8)) + d48*db8d48*dg8d48/(d48**2. - 1.) - dg8d48b8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d68**2. - 1. ))*( (db8d68*dg8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(db8d68*0./(cosh(a8)*cosh(a8)) + d68*db8d68*dg8d68/(d68**2. - 1.) - dg8d68b8) ) +
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d78**2. - 1. ))*( (db8d78*dg8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db8d78*0./(cosh(a8)*cosh(a8)) + d78*db8d78*dg8d78/(d78**2. - 1.) - dg8d78b8) )],

            # ---------- #

            # gd8 a1,b1,g1
            [0.,0.,0.,

            # gd8 a2,b2,g2
            0.,0.,0.,

            # gd8 a3,b3,g3
            0.,0.,0.,

            # gd8 a4,b4,g4
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d48**2. - 1. ))*( (dg8d48*da4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg8d48*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d48*dg8d48*da4d48/(d48**2. - 1.) - da4d48g8) ), 
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d48**2. - 1. ))*( (dg8d48*db4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg8d48*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d48*dg8d48*db4d48/(d48**2. - 1.) - db4d48g8) ),
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d48**2. - 1. ))*( (dg8d48*dg4d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg8d48*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d48*dg8d48*dg4d48/(d48**2. - 1.) - dg4d48g8) ),

            # gd8 a5,b5,g5
            0.,0.,0.,

            # gd8 a6,b6,g6
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d68**2. - 1. ))*( (dg8d68*da6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg8d68*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d68*dg8d68*da6d68/(d68**2. - 1.) - da6d68g8) ), 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d68**2. - 1. ))*( (dg8d68*db6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg8d68*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d68*dg8d68*db6d68/(d68**2. - 1.) - db6d68g8) ),
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d68**2. - 1. ))*( (dg8d68*dg6d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg8d68*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d68*dg8d68*dg6d68/(d68**2. - 1.) - dg6d68g8) ),

            # gd8 a7,b7,g7
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d78**2. - 1. ))*( (dg8d78*da7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg8d78*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d78*dg8d78*da7d78/(d78**2. - 1.) - da7d78g8) ), 
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d78**2. - 1. ))*( (dg8d78*db7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg8d78*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d78*dg8d78*db7d78/(d78**2. - 1.) - db7d78g8) ),
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d78**2. - 1. ))*( (dg8d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg8d78*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d78*dg8d78*dg7d78/(d78**2. - 1.) - dg7d78g8) )

            # gd8 a8
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d48**2. - 1. ))*( (dg8d48*da8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg8d48*sinh(2.*a8)*cosh(b8)*cosh(b8)/(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d48*dg8d48*da8d48/(d48**2. - 1.) - da8d48g8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d68**2. - 1. ))*( (dg8d68*da8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg8d68*sinh(2.*a8)*cosh(b8)*cosh(b8)/(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d68*dg8d68*da8d68/(d68**2. - 1.) - da8d68g8) ) +
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d78**2. - 1. ))*( (dg8d78*da8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg8d78*sinh(2.*a8)*cosh(b8)*cosh(b8)/(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d78*dg8d78*da8d78/(d78**2. - 1.) - da8d78g8) ),

            # gd8 b8
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d48**2. - 1. ))*( (dg8d48*db8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg8d48*sinh(2.*b8)*cosh(a8)*cosh(a8)/(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d48*dg8d48*db8d48/(d48**2. - 1.) - db8d48g8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d68**2. - 1. ))*( (dg8d68*db8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg8d68*sinh(2.*b8)*cosh(a8)*cosh(a8)/(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d68*dg8d68*db8d68/(d68**2. - 1.) - db8d68g8) ) +
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d78**2. - 1. ))*( (dg8d78*db8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg8d78*sinh(2.*b8)*cosh(a8)*cosh(a8)/(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d78*dg8d78*db8d78/(d78**2. - 1.) - db8d78g8) ),

            # gd8 g8
            -spring_arr[7][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d48**2. - 1. ))*( (dg8d48*dg8d48)/sqrt( d48**2. - 1.) - ( arccosh(d48) - spring_arr[7][1] )*(dg8d48*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d48*dg8d48*dg8d48/(d48**2. - 1.) - dg8d48g8) ) + 
            -spring_arr[10][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d68**2. - 1. ))*( (dg8d68*dg8d68)/sqrt( d68**2. - 1.) - ( arccosh(d68) - spring_arr[10][1] )*(dg8d68*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d68*dg8d68*dg8d68/(d68**2. - 1.) - dg8d68g8) ) +
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d78**2. - 1. ))*( (dg8d78*dg8d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg8d78*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d78*dg8d78*dg8d78/(d78**2. - 1.) - dg8d78g8) )]

        ])
   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*cosh(b1n)**2.)*sinh(a1n)*cosh(a1n) - k/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (-cosh(a1n)*sinh(a2n) - sinh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + sinh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*cosh(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) - k/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (-cosh(a1n1)*sinh(a2n1) - sinh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + sinh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) 
            ))

    def con5(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sinh(b1n)*cosh(b1n) - 2.*ad1n*bd1n*tanh(a1n) - k/(m1*cosh(a1n)*cosh(a1n))*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (-cosh(a1n)*cosh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*sinh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            gd1n1*gd1n1*sinh(b1n1)*cosh(b1n1) - 2.*ad1n1*bd1n1*tanh(a1n1) - k/(m1*cosh(a1n1)*cosh(a1n1))*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.)  
            ))

    def con6(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1, ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, m1, h, k, xo):
        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n*tanh(a1n) - 2.*bd1n*gd1n*tanh(b1n) - k/(m1*cosh(a1n)*cosh(a1n)*cosh(b1n)*cosh(b1n))*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*sinh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            -2.*ad1n1*gd1n1*tanh(a1n1) - 2.*bd1n1*gd1n1*tanh(b1n1) - k/(m1*cosh(a1n1)*cosh(a1n1)*cosh(b1n1)*cosh(b1n1))*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*sinh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.)
            )) 
    
    def jacobian(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, ad1n1, bd1n1, gd1n1, ad2n1, bd2n1, gd2n1, m1, m2, h, k, xo):
        spring_terms=jacobi_sp_terms(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, m2, k, xo)
        return array([
            [1.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1+cosh(b1n1)*cosh(b1n1)*gd1n1*gd1n1)*cosh(2.*a1n1) + spring_terms[0,0],
            -.25*h*sinh(2.*a1n1)*sinh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[0,1],
            0. + spring_terms[0,2], 
            
            spring_terms[0,3],
            spring_terms[0,4],
            spring_terms[0,5],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            -.5*h*sinh(2.*a1n1)*cosh(b1n1)*cosh(b1n1)*gd1n1,

            0.,
            0.,
            0.],

            # bd1 update
            [h*ad1n1*bd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[1,0],
            -.5*h*cosh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[1,1],
            0. + spring_terms[1,2], 
            
            spring_terms[1,3],
            spring_terms[1,4],
            spring_terms[1,5],
            
            h*tanh(a1n1)*bd1n1,
            1.+h*tanh(a1n1)*ad1n1,
            -.5*h*sinh(2.*b1n1)*gd1n1,

            0.,
            0.,
            0.],

            # gd1 update
            [h*ad1n1*gd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[2,0],
            h*bd1n1*gd1n1/(cosh(b1n1)*cosh(b1n1)) + spring_terms[2,1],
            0. + spring_terms[2,2],
             
            spring_terms[2,3],
            spring_terms[2,4],
            spring_terms[2,5],

            h*tanh(a1n1)*gd1n1,
            h*tanh(b1n1)*gd1n1,
            1.+h*tanh(a1n1)*ad1n1+h*tanh(b1n1)*bd1n1,

            0.,
            0.,
            0.],

            # ad2 update
            [spring_terms[3,0],
            spring_terms[3,1],
            spring_terms[3,2],

            -.5*h*(bd2n1*bd2n1+cosh(b2n1)*cosh(b2n1)*gd2n1*gd2n1)*cosh(2.*a2n1) + spring_terms[3,3],
            -.25*h*sinh(2.*a2n1)*sinh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[3,4],
            0. + spring_terms[3,5],

            0.,
            0.,
            0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            -.5*h*sinh(2.*a2n1)*cosh(b2n1)*cosh(b2n1)*gd2n1],

            # bd2 update
            [spring_terms[4,0],
            spring_terms[4,1],
            spring_terms[4,2],

            h*ad2n1*bd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[4,3],
            -.5*h*cosh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[4,4],
            0. + spring_terms[4,5],

            0.,
            0.,
            0.,

            h*tanh(a2n1)*bd2n1,
            1.+h*tanh(a2n1)*ad2n1,
            -.5*h*sinh(2.*b2n1)*gd2n1],

            # gd2 update
            [spring_terms[5,0],
            spring_terms[5,1],
            spring_terms[5,2],

            h*ad2n1*gd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[5,3],
            h*bd2n1*gd2n1/(cosh(b2n1)*cosh(b2n1)) + spring_terms[5,4],
            0. + spring_terms[5,5],

            0.,
            0.,
            0.,

            h*tanh(a2n1)*gd2n1,
            h*tanh(b2n1)*gd2n1,
            1.+h*tanh(a2n1)*ad2n1+h*tanh(b2n1)*bd2n1]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist),-array([
        con1(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con2(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con3(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], step),
        con1(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),
        con2(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),
        con3(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], step),

        con4(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con5(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con6(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], m1, step, sprcon, eqdist),
        con4(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),
        con5(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),
        con6(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos2n[2], pos2n[2], pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos1n[2], pos1n[2], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel2n[2], vel2n[2], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel1n[2], vel1n[2], m2, step, sprcon, eqdist),       
    ]))
    val1 = array([pos1n[0]+diff1[0], pos1n[1]+diff1[1], pos1n[2]+diff1[2], pos2n[0]+diff1[3], pos2n[1]+diff1[4], pos2n[2]+diff1[5], vel1n[0]+diff1[6], vel1n[1]+diff1[7], vel1n[2]+diff1[8], vel2n[0]+diff1[9], vel2n[1]+diff1[10], vel2n[2]+diff1[11]])    
    x = 0
    while(x < 7):
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], val1[6], val1[7], val1[8], val1[9], val1[10], val1[11], m1, m2, step, sprcon, eqdist),-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], step),
            con1(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),
            con2(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),
            con3(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], step),

            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], m1, step, sprcon, eqdist),
            con4(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),
            con5(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),
            con6(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], m2, step, sprcon, eqdist),       
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]])       
        val1 = val2
        x=x+1
    # print(val1)
    return val1






