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

#######################################################
### Hyperbolic 2-space Spring Potential (Dumb-Bell) ###
#######################################################

# This is a pared down version of the 3D version below for running in H2.

def imph2sptrans2(pos1n, pos2n, vel1n, vel2n, step, m1, m2, sprcon, eqdist):

    # This seems more complicated, but I am constructing these so that the jacobian elements are not super long.
    # I have found that I do not need to make functions for both particles since I can just flip the arguments
    # and it should work fine since they are symmetric.

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
    # with the arguments flipped. Thus only ten functions are needed instead of sixteen. Due to the symmetry of the partial
    # derivatives the square matrix of sixteen values can be reduced to the upper triangular metrix. Of the ten values in
    # upper triangular matrix symmetry of the particles allows for the number of functions to be further reduced to six
    
    # For the remaining nine functions of the upper triangular matrix use:
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

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(a1, b1, a2, b2, m1, m2, k, xo):
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

        return array([
            [-k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*da1d12/(d12**2. - 1.) - da1d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*db1d12/(d12**2. - 1.) - db1d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*da2d12/(d12**2. - 1.) - da2d12a1) ),
            -k/(m1*1.*sqrt( d12**2. - 1. ))*( (da1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da1d12*0./1. + d12*da1d12*db2d12/(d12**2. - 1.) - db2d12a1) )],

            [-k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*sinh(2.*a1)/(cosh(a1)*cosh(a1)) + d12*db1d12*da1d12/(d12**2. - 1.) - da1d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db1d12/(d12**2. - 1.) - db1d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*da2d12/(d12**2. - 1.) - da2d12b1) ),
            -k/(m1*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db2d12/(d12**2. - 1.) - db2d12b1) )],

            [-k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*da1d12/(d12**2. - 1.) - da1d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*db1d12/(d12**2. - 1.) - db1d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*da2d12/(d12**2. - 1.) - da2d12a2) ),
            -k/(m2*1.*sqrt( d12**2. - 1. ))*( (da2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(da2d12*0./1. + d12*da2d12*db2d12/(d12**2. - 1.) - db2d12a2) )],

            [-k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*da1d12/(d12**2. - 1.) - da1d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db1d12/(d12**2. - 1.) - db1d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*sinh(2.*a2)/(cosh(a2)*cosh(a2)) + d12*db2d12*da2d12/(d12**2. - 1.) - da2d12b2) ),
            -k/(m2*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - xo )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db2d12/(d12**2. - 1.) - db2d12b2) )]

        ])
   
    def con1(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)       

    def con3(a1n, a1n1, b1n, b1n1, a2n, a2n1, b2n, b2n1, ad1n, ad1n1, bd1n, bd1n1, ad2n, ad2n1, bd2n, bd2n1, m1, h, k, xo):
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n)*sinh(a1n)*cosh(a1n) - k/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (-cosh(a1n)*sinh(a2n) - sinh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + sinh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            (bd1n1*bd1n1)*sinh(a1n1)*cosh(a1n1) - k/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (-cosh(a1n1)*sinh(a2n1) - sinh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + sinh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1) - sinh(b1n1)*sinh(b2n1)))**2.) 
            ))

    def con4(a1n, a1n1, b1n, b1n1, a2n, a2n1, b2n, b2n1, ad1n, ad1n1, bd1n, bd1n1, ad2n, ad2n1, bd2n, bd2n1, m1, h, k, xo):
        return (bd1n1 - bd1n - .5*h*(
            -2.*ad1n*bd1n*tanh(a1n) - k/(m1*cosh(a1n)*cosh(a1n))*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n) - sinh(b1n)*sinh(b2n))) - xo)*
            (-cosh(a1n)*cosh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*sinh(b1n)*cosh(a2n)*cosh(b2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n) - sinh(b1n)*sinh(b2n)))**2.)
            + 
            -2.*ad1n1*bd1n1*tanh(a1n1) - k/(m1*cosh(a1n1)*cosh(a1n1))*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1) - sinh(b1n1)*sinh(b2n1))) - xo)*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*cosh(b2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1) - sinh(b1n1)*sinh(b2n1)))**2.)  
            ))
    
    def jacobian(a1n1, b1n1, a2n1, b2n1, ad1n1, bd1n1, ad2n1, bd2n1, m1, m2, h, k, xo):
        spring_terms=jacobi_sp_terms(a1n1, b1n1, a2n1, b2n1, m1, m2, k, xo)
        return array([
            [1.,0., 0.,0., -.5*h,0., 0.,0.],
            [0.,1., 0.,0., 0.,-.5*h, 0.,0.],

            [0.,0., 1.,0., 0.,0., -.5*h,0.],
            [0.,0., 0.,1., 0.,0., 0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1)*cosh(2.*a1n1) + spring_terms[0,0],
            0. + spring_terms[0,1], 
            
            spring_terms[0,2],
            spring_terms[0,3],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,

            0.,
            0.],

            # bd1 update
            [h*ad1n1*bd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[1,0],
            0. + spring_terms[1,1],
            
            spring_terms[1,2],
            spring_terms[1,3],
            
            h*tanh(a1n1)*bd1n1,
            1.+h*tanh(a1n1)*ad1n1,

            0.,
            0.],

            # ad2 update
            [spring_terms[2,0],
            spring_terms[2,1],

            -.5*h*(bd2n1*bd2n1)*cosh(2.*a2n1) + spring_terms[2,2],
            0. + spring_terms[2,3],

            0.,
            0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1],

            # bd2 update
            [spring_terms[3,0],
            spring_terms[3,1],

            h*ad2n1*bd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[3,2],
            0. + spring_terms[3,3],

            0.,
            0.,

            h*tanh(a2n1)*bd2n1,
            1.+h*tanh(a2n1)*ad2n1]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(pos1n[0], pos1n[1], pos2n[0], pos2n[1], vel1n[0], vel1n[1], vel2n[0], vel2n[1], m1, m2, step, sprcon, eqdist),-array([
        con1(pos1n[0], pos1n[0], pos1n[1], pos1n[1], vel1n[0], vel1n[0], vel1n[1], vel1n[1], step),
        con2(pos1n[0], pos1n[0], pos1n[1], pos1n[1], vel1n[0], vel1n[0], vel1n[1], vel1n[1], step),
        con1(pos2n[0], pos2n[0], pos2n[1], pos2n[1], vel2n[0], vel2n[0], vel2n[1], vel2n[1], step),
        con2(pos2n[0], pos2n[0], pos2n[1], pos2n[1], vel2n[0], vel2n[0], vel2n[1], vel2n[1], step),

        con3(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos2n[0], pos2n[0], pos2n[1], pos2n[1], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel2n[0], vel2n[0], vel2n[1], vel2n[1], m1, step, sprcon, eqdist),
        con4(pos1n[0], pos1n[0], pos1n[1], pos1n[1], pos2n[0], pos2n[0], pos2n[1], pos2n[1], vel1n[0], vel1n[0], vel1n[1], vel1n[1], vel2n[0], vel2n[0], vel2n[1], vel2n[1], m1, step, sprcon, eqdist),
        con3(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos1n[0], pos1n[0], pos1n[1], pos1n[1], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel1n[0], vel1n[0], vel1n[1], vel1n[1], m2, step, sprcon, eqdist),
        con4(pos2n[0], pos2n[0], pos2n[1], pos2n[1], pos1n[0], pos1n[0], pos1n[1], pos1n[1], vel2n[0], vel2n[0], vel2n[1], vel2n[1], vel1n[0], vel1n[0], vel1n[1], vel1n[1], m2, step, sprcon, eqdist),      
    ]))
    val1 = array([pos1n[0]+diff1[0], pos1n[1]+diff1[1], pos2n[0]+diff1[2], pos2n[1]+diff1[3], vel1n[0]+diff1[4], vel1n[1]+diff1[5], vel2n[0]+diff1[6], vel2n[1]+diff1[7]])    
    x = 0
    while(x < 7):
        diff2=linalg.solve(jacobian(val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], val1[6], val1[7], m1, m2, step, sprcon, eqdist),-array([
            con1(pos1n[0], val1[0], pos1n[1], val1[1], vel1n[0], val1[4], vel1n[1], val1[5], step),
            con2(pos1n[0], val1[0], pos1n[1], val1[1], vel1n[0], val1[4], vel1n[1], val1[5], step),
            con1(pos2n[0], val1[2], pos2n[1], val1[3], vel2n[0], val1[6], vel2n[1], val1[7], step),
            con2(pos2n[0], val1[2], pos2n[1], val1[3], vel2n[0], val1[6], vel2n[1], val1[7], step),

            con3(pos1n[0], val1[0], pos1n[1], val1[1], pos2n[0], val1[2], pos2n[1], val1[3], vel1n[0], val1[4], vel1n[1], val1[5], vel2n[0], val1[6], vel2n[1], val1[7], m1, step, sprcon, eqdist),
            con4(pos1n[0], val1[0], pos1n[1], val1[1], pos2n[0], val1[2], pos2n[1], val1[3], vel1n[0], val1[4], vel1n[1], val1[5], vel2n[0], val1[6], vel2n[1], val1[7], m1, step, sprcon, eqdist),
            con3(pos2n[0], val1[2], pos2n[1], val1[3], pos1n[0], val1[0], pos1n[1], val1[1], vel2n[0], val1[6], vel2n[1], val1[7], vel1n[0], val1[4], vel1n[1], val1[5], m2, step, sprcon, eqdist),
            con4(pos2n[0], val1[2], pos2n[1], val1[3], pos1n[0], val1[0], pos1n[1], val1[1], vel2n[0], val1[6], vel2n[1], val1[7], vel1n[0], val1[4], vel1n[1], val1[5], m2, step, sprcon, eqdist),      
        ]))
        val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7]])       
        val1 = val2
        x=x+1
    # print(val1)
    return val1


#######################################################
### Hyperbolic 3-space Spring Potential (Dumb-Bell) ###
#######################################################

# This took a lot more of work than I anticipated to make sure that the Newton solver
# jacobian was correct, but it is finally updated and actually correct now. Currently,
# it only supports the interaction of a two mass spring system, but I am hopeful that
# the functions I have constructed should make increasing the complexity of the system
# more streamlined. However, it does take a hot minute to run even now, so it will take
# even longer with more complicated spring systems.
# DO NOT USE TRANSLATIONAL PARAMETERIZATION FOR STRUCTURES WITH ROTATIONAL
# SYMMETRY OF ODD DEGREE.

def imph3sptrans2(pos1n, pos2n, vel1n, vel2n, step, m1, m2, sprcon, eqdist):

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

# This is an extension of the dumb-bell system using the structure for the cube
# solver. The configuration is a triangle of three masses and three springs. 
# I found that for the case of shapes with odd number rotational symmetry
# that the translational parameterization is not appropriate due to the 
# metric factors being unequal for point with degenerate mirror symmetry 
# compared to the remaining points that have a mirror point. Since it is a 
# coordinate issue, we need to use the rotational coordinate system to remove
# this artifical directional bias. (I have included this for completeness)

def imph3sprot2(posn_arr, veln_arr, step, mass_arr, spring_arr):

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only three functions are needed instead of six.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, g2, a1, b1, g1)
    # db2D12 = db1D12(a2, b2, g2, a1, b1, g1)
    # dg2D12 = dg1D12(a2, b2, g2, a1, b1, g1)
    def da1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2) 

    def dg1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)      
    

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*cos(b2) - cosh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def da2D12a1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sinh(a2) - cosh(a1)*cos(b1)*cosh(a2)*cos(b2) - cosh(a1)*sin(b1)*cosh(a2)*sin(b2)*cos(g1 - g2)

    def db2D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cos(b1)*sinh(a2)*sin(b2) - cosh(a1)*sin(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12a1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*cos(b2) + sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def dg1D12g1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg2D12g1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(positions, mass_arr, spring_arr):
        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        
        
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
        

        return array([

            # ---------- #
            #     V1     #
            # ---------- #
            
            # ad1 a1
            [-spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da1d12/(d12**2. - 1.) - da1d12a1) ),

            # ad1 b1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db1d12/(d12**2. - 1.) - db1d12a1) ),

            # ad1 g1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg1d12/(d12**2. - 1.) - dg1d12a1) ),

            # ad1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da2d12/(d12**2. - 1.) - da2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db2d12/(d12**2. - 1.) - db2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg2d12/(d12**2. - 1.) - dg2d12a1) )
            ],

            # ---------- #

            # bd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*sinh(2.*a1)/(sinh(a1)*sinh(a1)) + d12*db1d12*da1d12/(d12**2. - 1.) - da1d12b1) ),

            # bd1 b1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*db1d12/(d12**2. - 1.) - db1d12b1) ),

            # bd1 g1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*dg1d12/(d12**2. - 1.) - dg1d12b1) ),

            # bd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*da2d12/(d12**2. - 1.) - da2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*db2d12/(d12**2. - 1.) - db2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*dg2d12/(d12**2. - 1.) - dg2d12b1) )
            ],

            # ---------- #

            # gd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sinh(2.*a1)*sin(b1)*sin(b1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*da1d12/(d12**2. - 1.) - da1d12g1) ),

            # gd1 b1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sin(2.*b1)*sinh(a1)*sinh(a1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*db1d12/(d12**2. - 1.) - db1d12g1) ),

            # gd1 g1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*dg1d12/(d12**2. - 1.) - dg1d12g1) ),

            # gd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*da2d12/(d12**2. - 1.) - da2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*db2d12/(d12**2. - 1.) - db2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*dg2d12/(d12**2. - 1.) - dg2d12g1) )
            ],

            # ---------- #
            #     V2     #
            # ---------- #

            # ad2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da1d12/(d12**2. - 1.) - da1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db1d12/(d12**2. - 1.) - db1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg1d12/(d12**2. - 1.) - dg1d12a2) ),

            # ad2 a2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da2d12/(d12**2. - 1.) - da2d12a2) ),

            # ad2 b2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db2d12/(d12**2. - 1.) - db2d12a2) ),

            # ad2 g2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg2d12/(d12**2. - 1.) - dg2d12a2) )
            ],

            # ---------- #

            # bd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*da1d12/(d12**2. - 1.) - da1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*db1d12/(d12**2. - 1.) - db1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*dg1d12/(d12**2. - 1.) - dg1d12b2) ),

            # bd2 a2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*sinh(2.*a2)/(sinh(a2)*sinh(a2)) + d12*db2d12*da2d12/(d12**2. - 1.) - da2d12b2) ),

            # bd2 b2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*db2d12/(d12**2. - 1.) - db2d12b2) ),

            # bd2 g2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*dg2d12/(d12**2. - 1.) - dg2d12b2) )
            ],

            # ---------- #

            # gd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*da1d12/(d12**2. - 1.) - da1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*db1d12/(d12**2. - 1.) - db1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*dg1d12/(d12**2. - 1.) - dg1d12g2) ),

            # gd2 a2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sinh(2.*a2)*sin(b2)*sin(b2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*da2d12/(d12**2. - 1.) - da2d12g2) ),

            # gd2 b2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sin(2.*b2)*sinh(a2)*sinh(a2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*db2d12/(d12**2. - 1.) - db2d12g2) ),

            # gd2 g2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*dg2d12/(d12**2. - 1.) - dg2d12g2) )
            ]

        ])
   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*sin(b1n)**2.)*sinh(a1n)*cosh(a1n) - 
            sp12[0]/m1*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*cosh(a2n) - cosh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - cosh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*sin(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) - 
            sp12[0]/m1*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*cosh(a2n1) - cosh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - cosh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.)
            ))

    def con5(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sin(b1n)*cos(b1n) - 2.*ad1n*bd1n/tanh(a1n) - 
            sp12[0]/(m1*sinh(a1n)*sinh(a1n))*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*sin(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.)
            + 
            gd1n1*gd1n1*sin(b1n1)*cos(b1n1) - 2.*ad1n1*bd1n1/tanh(a1n1) - 
            sp12[0]/(m1*sinh(a1n1)*sinh(a1n1))*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.)
            ))

    def con6(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess

        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n/tanh(a1n) - 2.*bd1n*gd1n/tan(b1n) - 
            sp12[0]/(m1*sinh(a1n)*sinh(a1n)*sin(b1n)*sin(b1n))*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*sin(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.)
            + 
            -2.*ad1n1*gd1n1/tanh(a1n1) - 2.*bd1n1*gd1n1/tan(b1n1) - 
            sp12[0]/(m1*sinh(a1n1)*sinh(a1n1)*sin(b1n1)*sin(b1n1))*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*sin(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.)
            )) 
    
    def jacobian(positions, velocities, mass_arr, h, spring_arr):
        a1n1,b1n1,g1n1=positions[0]
        a2n1,b2n1,g2n1=positions[1]
        ad1n1,bd1n1,gd1n1=velocities[0]
        ad2n1,bd2n1,gd2n1=velocities[1]
        spring_terms=jacobi_sp_terms(positions, mass_arr, spring_arr)
        #print(spring_terms)
        return array([
            [1.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1+sin(b1n1)*sin(b1n1)*gd1n1*gd1n1)*cosh(2.*a1n1) + spring_terms[0][0],
            -.25*h*sinh(2.*a1n1)*sin(2.*b1n1)*gd1n1*gd1n1 + spring_terms[0][1],
            0. + spring_terms[0][2], 
            
            spring_terms[0][3],
            spring_terms[0][4],
            spring_terms[0][5],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            -.5*h*sinh(2.*a1n1)*sin(b1n1)*sin(b1n1)*gd1n1,

            0.,0.,0.
            ],

            # bd1 update
            [-h*ad1n1*bd1n1/(sinh(a1n1)*sinh(a1n1)) + spring_terms[1][0],
            -.5*h*cos(2.*b1n1)*gd1n1*gd1n1 + spring_terms[1][1],
            0. + spring_terms[1][2], 
            
            spring_terms[1][3],
            spring_terms[1][4],
            spring_terms[1][5],
            
            h/tanh(a1n1)*bd1n1,
            1.+h/tanh(a1n1)*ad1n1,
            -.5*h*sin(2.*b1n1)*gd1n1,

            0.,0.,0.
            ],

            # gd1 update
            [-h*ad1n1*gd1n1/(sinh(a1n1)*sinh(a1n1)) + spring_terms[2][0],
            -h*bd1n1*gd1n1/(sin(b1n1)*sin(b1n1)) + spring_terms[2][1],
            0. + spring_terms[2][2],
             
            spring_terms[2][3],
            spring_terms[2][4],
            spring_terms[2][5],

            h/tanh(a1n1)*gd1n1,
            h/tan(b1n1)*gd1n1,
            1.+h/tanh(a1n1)*ad1n1+h/tan(b1n1)*bd1n1,

            0.,0.,0.
            ],

            # ad2 update
            [spring_terms[3][0],
            spring_terms[3][1],
            spring_terms[3][2],

            -.5*h*(bd2n1*bd2n1+sin(b2n1)*sin(b2n1)*gd2n1*gd2n1)*cosh(2.*a2n1) + spring_terms[3][3],
            -.25*h*sinh(2.*a2n1)*sin(2.*b2n1)*gd2n1*gd2n1 + spring_terms[3][4],
            0. + spring_terms[3][5],

            0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            -.5*h*sinh(2.*a2n1)*sin(b2n1)*sin(b2n1)*gd2n1
            ],

            # bd2 update
            [spring_terms[4][0],
            spring_terms[4][1],
            spring_terms[4][2],

            -h*ad2n1*bd2n1/(sinh(a2n1)*sinh(a2n1)) + spring_terms[4][3],
            -.5*h*cos(2.*b2n1)*gd2n1*gd2n1 + spring_terms[4][4],
            0. + spring_terms[4][5],

            0.,0.,0.,

            h/tanh(a2n1)*bd2n1,
            1.+h/tanh(a2n1)*ad2n1,
            -.5*h*sin(2.*b2n1)*gd2n1
            ],

            # gd2 update
            [spring_terms[5][0],
            spring_terms[5][1],
            spring_terms[5][2],

            -h*ad2n1*gd2n1/(sinh(a2n1)*sinh(a2n1)) + spring_terms[5][3],
            -h*bd2n1*gd2n1/(sin(b2n1)*sin(b2n1)) + spring_terms[5][4],
            0. + spring_terms[5][5],

            0.,0.,0.,

            h/tanh(a2n1)*gd2n1,
            h/tan(b2n1)*gd2n1,
            1.+h/tanh(a2n1)*ad2n1+h/tan(b2n1)*bd2n1
            ]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr),-array([
        #p1
        con1(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con2(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con3(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        #p2
        con1(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con2(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con3(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),

        #v1
        con4(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        con5(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        con6(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        #v2
        con4(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0]),
        con5(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0]),
        con6(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0])     
    ]))
    val1 = array([
            posn_arr[0][0]+diff1[0], posn_arr[0][1]+diff1[1], posn_arr[0][2]+diff1[2], 
            posn_arr[1][0]+diff1[3], posn_arr[1][1]+diff1[4], posn_arr[1][2]+diff1[5],

            veln_arr[0][0]+diff1[6], veln_arr[0][1]+diff1[7], veln_arr[0][2]+diff1[8], 
            veln_arr[1][0]+diff1[9], veln_arr[1][1]+diff1[10], veln_arr[1][2]+diff1[11]
            ])    
    x = 0
    while(x < 7):
        new_pos_arr=array([val1[0:3],val1[3:6]])
        new_vel_arr=array([val1[6:9],val1[9:12]])
        diff2=linalg.solve(jacobian(new_pos_arr, new_vel_arr, mass_arr, step, spring_arr),-array([
            #p1
            con1(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con2(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con3(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            #p2
            con1(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con2(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con3(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),

            #v1
            con4(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            con5(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            con6(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            #v2
            con4(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0]),
            con5(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0]),
            con6(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0])  
        ]))      
        val2 = array([
            val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], 
            val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5],

            val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8],
            val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]
            ])
        val1 = val2
        x=x+1
    #print(val1[9:17])
    return val1

# This is the dumb-bell solver using the condensed structure for setting up
# the geodesic and spring potential terms for the solver. This is designed to
# make integration of more complex meshs more streamlined. I have checked to make
# sure that it is the same output as the old solver, so I will try to use this as
# the default solver structure moving forward. I need to update the solvers for
# the triangle and tetrahedron to follow this same structure.

def imph3sprot2_condense(posn_arr, veln_arr, step, mass_arr, spring_arr):

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only three functions are needed instead of six.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, g2, a1, b1, g1)
    # db2D12 = db1D12(a2, b2, g2, a1, b1, g1)
    # dg2D12 = dg1D12(a2, b2, g2, a1, b1, g1)
    def da1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2) 

    def dg1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)      
    

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*cos(b2) - cosh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def da2D12a1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sinh(a2) - cosh(a1)*cos(b1)*cosh(a2)*cos(b2) - cosh(a1)*sin(b1)*cosh(a2)*sin(b2)*cos(g1 - g2)

    def db2D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cos(b1)*sinh(a2)*sin(b2) - cosh(a1)*sin(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12a1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*cos(b2) + sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def dg1D12g1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg2D12g1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(positions, mass_arr, spring_arr):

        def da2da1V12(m, f, k, l, d12, da1d12, da2d12, df, da2d12da1):
            return -k/(m*f*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) + ( arccosh(d12) - l )*( da2d12da1 - da1d12*(df/f + d12*da2d12/(d12**2. - 1.)) ) )

        def derivative_terms(a1, b1, g1, a2, b2, g2):
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
            return [
                [ # D function
                    d12
                ],
                [ # First derivatives of D function
                    da1d12, 
                    db1d12, 
                    dg1d12, 
                    da2d12, 
                    db2d12, 
                    dg2d12],
                [ # Second derivatives of D function
                    [
                        da1d12a1,
                        db1d12a1,
                        dg1d12a1,
                        da2d12a1,
                        db2d12a1,
                        dg2d12a1
                    ],
                    [
                        da1d12b1,
                        db1d12b1,
                        dg1d12b1,
                        da2d12b1,
                        db2d12b1,
                        dg2d12b1
                    ],
                    [
                        da1d12g1,
                        db1d12g1,
                        dg1d12g1,
                        da2d12g1,
                        db2d12g1,
                        dg2d12g1
                    ],
                    [
                        da1d12a2,
                        db1d12a2,
                        dg1d12a2,
                        da2d12a2,
                        db2d12a2,
                        dg2d12a2
                    ],
                    [
                        da1d12b2,
                        db1d12b2,
                        dg1d12b2,
                        da2d12b2,
                        db2d12b2,
                        dg2d12b2
                    ],
                    [
                        da1d12g2,
                        db1d12g2,
                        dg1d12g2,
                        da2d12g2,
                        db2d12g2,
                        dg2d12g2
                    ],
                ]
            ]

        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        ### Spring 1-2 spring_arr[0]###
        sp12_darr=derivative_terms(a1, b1, g1, a2, b2, g2)

        

        return array([

            # ---------- #
            #     V1     #
            # ---------- #

            [
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][0], 0., sp12_darr[2][0][0]),  # ad1 a1
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][1], 0., sp12_darr[2][0][1]),  # ad1 b1
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][2], 0., sp12_darr[2][0][2]),  # ad1 g1
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][3], 0., sp12_darr[2][0][3]),  # ad1 a2
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][4], 0., sp12_darr[2][0][4]),  # ad1 b2
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][5], 0., sp12_darr[2][0][5])   # ad1 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][0], sinh(2.*a1), sp12_darr[2][1][0]),  # bd1 a1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][1], 0.,          sp12_darr[2][1][1]),  # bd1 b1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][2], 0.,          sp12_darr[2][1][2]),  # bd1 g1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][3], 0.,          sp12_darr[2][1][3]),  # bd1 a2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][4], 0.,          sp12_darr[2][1][4]),  # bd1 b2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][5], 0.,          sp12_darr[2][1][5])   # bd1 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][0], sinh(2.*a1)*sin(b1)*sin(b1),  sp12_darr[2][2][0]),  # gd1 a1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][1], sin(2.*b1)*sinh(a1)*sinh(a1), sp12_darr[2][2][1]),  # gd1 b1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][2], 0.,                           sp12_darr[2][2][2]),  # gd1 g1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][3], 0.,                           sp12_darr[2][2][3]),  # gd1 a2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][4], 0.,                           sp12_darr[2][2][4]),  # gd1 b2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][5], 0.,                           sp12_darr[2][2][5])   # gd1 g2
            ],

            # ---------- #
            #     V2     #
            # ---------- #

            [
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][0], 0., sp12_darr[2][3][0]),  # ad2 a1
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][1], 0., sp12_darr[2][3][1]),  # ad2 b1
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][2], 0., sp12_darr[2][3][2]),  # ad2 g1
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][3], 0., sp12_darr[2][3][3]),  # ad2 a2
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][4], 0., sp12_darr[2][3][4]),  # ad2 b2
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][5], 0., sp12_darr[2][3][5])   # ad2 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][0], sinh(2.*a2), sp12_darr[2][4][0]),  # bd2 a1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][1], 0.,          sp12_darr[2][4][1]),  # bd2 b1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][2], 0.,          sp12_darr[2][4][2]),  # bd2 g1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][3], 0.,          sp12_darr[2][4][3]),  # bd2 a2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][4], 0.,          sp12_darr[2][4][4]),  # bd2 b2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][5], 0.,          sp12_darr[2][4][5])   # bd2 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][0], sinh(2.*a2)*sin(b2)*sin(b2),  sp12_darr[2][5][0]),  # gd2 a1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][1], sin(2.*b2)*sinh(a2)*sinh(a2), sp12_darr[2][5][1]),  # gd2 b1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][2], 0.,                           sp12_darr[2][5][2]),  # gd2 g1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][3], 0.,                           sp12_darr[2][5][3]),  # gd2 a2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][4], 0.,                           sp12_darr[2][5][4]),  # gd2 b2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][5], 0.,                           sp12_darr[2][5][5])   # gd2 g2
            ]

        ])
   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        def geo_spring_term_ad(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*1.)*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.))


        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*sin(b1n)**2.)*sinh(a1n)*cosh(a1n) + geo_spring_term_ad(a1n, b1n, g1n, a2n, b2n, g2n, m1, sp12)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*sin(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) + geo_spring_term_ad(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, sp12)
            ))

    def con5(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        def geo_spring_term_bd(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*sinh(a1)*sinh(a1))*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sin(b1n)*cos(b1n) - 2.*ad1n*bd1n/tanh(a1n) + geo_spring_term_bd(a1n, b1n, g1n, a2n, b2n, g2n, m1, sp12)
            + 
            gd1n1*gd1n1*sin(b1n1)*cos(b1n1) - 2.*ad1n1*bd1n1/tanh(a1n1) + geo_spring_term_bd(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, sp12)
            ))

    def con6(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        def geo_spring_term_gd(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*sinh(a1)*sinh(a1)*sin(b1)*sin(b1))*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess

        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n/tanh(a1n) - 2.*bd1n*gd1n/tan(b1n) + geo_spring_term_gd(a1n, b1n, g1n, a2n, b2n, g2n, m1, sp12)
            + 
            -2.*ad1n1*gd1n1/tanh(a1n1) - 2.*bd1n1*gd1n1/tan(b1n1) + geo_spring_term_gd(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, sp12)
            )) 
    
    def jacobian(positions, velocities, mass_arr, h, spring_arr):
        def geo_term_arr(a1, b1, g1, ad1, bd1, gd1, h):
            return [
                [   # a1, b1, g1 derivatives of adn1 update constraint
                    -.5*h*(bd1*bd1+sin(b1)*sin(b1)*gd1*gd1)*cosh(2.*a1),
                    -.25*h*sinh(2.*a1)*sin(2.*b1)*gd1*gd1,
                    0,
                ],
                [   # a1, b1, g1 derivatives of bdn1 update constraint
                    -h*ad1*bd1/(sinh(a1)*sinh(a1)),
                    -.5*h*cos(2.*b1)*gd1*gd1,
                    0.
                ],
                [   # a1, b1, g1 derivatives of gdn1 update constraint
                    -h*ad1*gd1/(sinh(a1)*sinh(a1)),
                    -h*bd1*gd1/(sin(b1)*sin(b1)),
                    0.
                ],
                [   # ad1, bd1, gd1 derivatives of adn1 update constraint
                    1.,
                    -.5*h*sinh(2.*a1)*bd1,
                    -.5*h*sinh(2.*a1)*sin(b1)*sin(b1)*gd1
                ],
                [   # ad1, bd1, gd1 derivatives of bdn1 update constraint
                    h/tanh(a1)*bd1,
                    1.+h/tanh(a1)*ad1,
                    -.5*h*sin(2.*b1)*gd1
                ],
                [   # ad1, bd1, gd1 derivatives of gdn1 update constraint
                    h/tanh(a1)*gd1,
                    h/tan(b1)*gd1,
                    1.+h/tanh(a1)*ad1+h/tan(b1)*bd1
                ]
            ]

        a1n1,b1n1,g1n1=positions[0]
        a2n1,b2n1,g2n1=positions[1]
        ad1n1,bd1n1,gd1n1=velocities[0]
        ad2n1,bd2n1,gd2n1=velocities[1]
        spring_terms=jacobi_sp_terms(positions, mass_arr, spring_arr)
        geo_term_p1=geo_term_arr(a1n1, b1n1, g1n1, ad1n1, bd1n1, gd1n1, h)
        geo_term_p2=geo_term_arr(a2n1, b2n1, g2n1, ad2n1, bd2n1, gd2n1, h)
        #print(spring_terms)
        return array([
            [1.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [
                geo_term_p1[0][0] + spring_terms[0][0],
                geo_term_p1[0][1] + spring_terms[0][1],
                geo_term_p1[0][2] + spring_terms[0][2], 
            
                spring_terms[0][3],
                spring_terms[0][4],
                spring_terms[0][5],

                geo_term_p1[3][0],
                geo_term_p1[3][1],
                geo_term_p1[3][2],

                0.,0.,0.
            ],

            # bd1 update
            [
                geo_term_p1[1][0] + spring_terms[1][0],
                geo_term_p1[1][1] + spring_terms[1][1],
                geo_term_p1[1][2] + spring_terms[1][2], 

                spring_terms[1][3],
                spring_terms[1][4],
                spring_terms[1][5],

                geo_term_p1[4][0],
                geo_term_p1[4][1],
                geo_term_p1[4][2],

                0.,0.,0.
            ],

            # gd1 update
            [
                geo_term_p1[2][0] + spring_terms[2][0],
                geo_term_p1[2][1] + spring_terms[2][1],
                geo_term_p1[2][2] + spring_terms[2][2],

                spring_terms[2][3],
                spring_terms[2][4],
                spring_terms[2][5],

                geo_term_p1[5][0],
                geo_term_p1[5][1],
                geo_term_p1[5][2],

                0.,0.,0.
            ],

            # ad2 update
            [
                spring_terms[3][0],
                spring_terms[3][1],
                spring_terms[3][2],

                geo_term_p2[0][0] + spring_terms[3][3],
                geo_term_p2[0][1] + spring_terms[3][4],
                geo_term_p2[0][2] + spring_terms[3][5],

                0.,0.,0.,

                geo_term_p2[3][0],
                geo_term_p2[3][1],
                geo_term_p2[3][2]
            ],

            # bd2 update
            [
                spring_terms[4][0],
                spring_terms[4][1],
                spring_terms[4][2],

                geo_term_p2[1][0] + spring_terms[4][3],
                geo_term_p2[1][1] + spring_terms[4][4],
                geo_term_p2[1][2] + spring_terms[4][5],

                0.,0.,0.,

                geo_term_p2[4][0],
                geo_term_p2[4][1],
                geo_term_p2[4][2]
            ],

            # gd2 update
            [
                spring_terms[5][0],
                spring_terms[5][1],
                spring_terms[5][2],

                geo_term_p2[2][0] + spring_terms[5][3],
                geo_term_p2[2][1] + spring_terms[5][4],
                geo_term_p2[2][2] + spring_terms[5][5],

                0.,0.,0.,

                geo_term_p2[5][0],
                geo_term_p2[5][1],
                geo_term_p2[5][2]
            ]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr),-array([
        #p1
        con1(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con2(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con3(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        #p2
        con1(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con2(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con3(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),

        #v1
        con4(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        con5(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        con6(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        #v2
        con4(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0]),
        con5(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0]),
        con6(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0])     
    ]))
    val1 = array([
            posn_arr[0][0]+diff1[0], posn_arr[0][1]+diff1[1], posn_arr[0][2]+diff1[2], 
            posn_arr[1][0]+diff1[3], posn_arr[1][1]+diff1[4], posn_arr[1][2]+diff1[5],

            veln_arr[0][0]+diff1[6], veln_arr[0][1]+diff1[7], veln_arr[0][2]+diff1[8], 
            veln_arr[1][0]+diff1[9], veln_arr[1][1]+diff1[10], veln_arr[1][2]+diff1[11]
            ])    
    x = 0
    while(x < 7):
        new_pos_arr=array([val1[0:3],val1[3:6]])
        new_vel_arr=array([val1[6:9],val1[9:12]])
        diff2=linalg.solve(jacobian(new_pos_arr, new_vel_arr, mass_arr, step, spring_arr),-array([
            #p1
            con1(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con2(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con3(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            #p2
            con1(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con2(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con3(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),

            #v1
            con4(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            con5(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            con6(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            #v2
            con4(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0]),
            con5(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0]),
            con6(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0])  
        ]))      
        val2 = array([
            val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], 
            val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5],

            val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8],
            val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]
            ])
        val1 = val2
        x=x+1
    #print(val1[9:17])
    return val1

def imph3sprot2_condense_econ(posn_arr, veln_arr, step, mass_arr, spring_arr, energy):

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only three functions are needed instead of six.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, g2, a1, b1, g1)
    # db2D12 = db1D12(a2, b2, g2, a1, b1, g1)
    # dg2D12 = dg1D12(a2, b2, g2, a1, b1, g1)
    def da1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2) 

    def dg1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)      
    

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*cos(b2) - cosh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def da2D12a1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sinh(a2) - cosh(a1)*cos(b1)*cosh(a2)*cos(b2) - cosh(a1)*sin(b1)*cosh(a2)*sin(b2)*cos(g1 - g2)

    def db2D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cos(b1)*sinh(a2)*sin(b2) - cosh(a1)*sin(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12a1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*cos(b2) + sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def dg1D12g1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg2D12g1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(positions, mass_arr, spring_arr):

        def da2da1V12(m, f, k, l, d12, da1d12, da2d12, df, da2d12da1):
            return -k/(m*f*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) + ( arccosh(d12) - l )*( da2d12da1 - da1d12*(df/f + d12*da2d12/(d12**2. - 1.)) ) )

        def derivative_terms(a1, b1, g1, a2, b2, g2):
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
            return [
                [ # D function
                    d12
                ],
                [ # First derivatives of D function
                    da1d12, 
                    db1d12, 
                    dg1d12, 
                    da2d12, 
                    db2d12, 
                    dg2d12],
                [ # Second derivatives of D function
                    [
                        da1d12a1,
                        db1d12a1,
                        dg1d12a1,
                        da2d12a1,
                        db2d12a1,
                        dg2d12a1
                    ],
                    [
                        da1d12b1,
                        db1d12b1,
                        dg1d12b1,
                        da2d12b1,
                        db2d12b1,
                        dg2d12b1
                    ],
                    [
                        da1d12g1,
                        db1d12g1,
                        dg1d12g1,
                        da2d12g1,
                        db2d12g1,
                        dg2d12g1
                    ],
                    [
                        da1d12a2,
                        db1d12a2,
                        dg1d12a2,
                        da2d12a2,
                        db2d12a2,
                        dg2d12a2
                    ],
                    [
                        da1d12b2,
                        db1d12b2,
                        dg1d12b2,
                        da2d12b2,
                        db2d12b2,
                        dg2d12b2
                    ],
                    [
                        da1d12g2,
                        db1d12g2,
                        dg1d12g2,
                        da2d12g2,
                        db2d12g2,
                        dg2d12g2
                    ],
                ]
            ]

        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        ### Spring 1-2 spring_arr[0]###
        sp12_darr=derivative_terms(a1, b1, g1, a2, b2, g2)

        

        return array([

            # ---------- #
            #     V1     #
            # ---------- #

            [
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][0], 0., sp12_darr[2][0][0]),  # ad1 a1
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][1], 0., sp12_darr[2][0][1]),  # ad1 b1
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][2], 0., sp12_darr[2][0][2]),  # ad1 g1
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][3], 0., sp12_darr[2][0][3]),  # ad1 a2
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][4], 0., sp12_darr[2][0][4]),  # ad1 b2
                da2da1V12(mass_arr[0], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][0], sp12_darr[1][5], 0., sp12_darr[2][0][5])   # ad1 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][0], sinh(2.*a1), sp12_darr[2][1][0]),  # bd1 a1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][1], 0.,          sp12_darr[2][1][1]),  # bd1 b1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][2], 0.,          sp12_darr[2][1][2]),  # bd1 g1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][3], 0.,          sp12_darr[2][1][3]),  # bd1 a2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][4], 0.,          sp12_darr[2][1][4]),  # bd1 b2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][1], sp12_darr[1][5], 0.,          sp12_darr[2][1][5])   # bd1 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][0], sinh(2.*a1)*sin(b1)*sin(b1),  sp12_darr[2][2][0]),  # gd1 a1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][1], sin(2.*b1)*sinh(a1)*sinh(a1), sp12_darr[2][2][1]),  # gd1 b1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][2], 0.,                           sp12_darr[2][2][2]),  # gd1 g1
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][3], 0.,                           sp12_darr[2][2][3]),  # gd1 a2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][4], 0.,                           sp12_darr[2][2][4]),  # gd1 b2
                da2da1V12(mass_arr[0], sinh(a1)*sinh(a1)*sin(b1)*sin(b1), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][2], sp12_darr[1][5], 0.,                           sp12_darr[2][2][5])   # gd1 g2
            ],

            # ---------- #
            #     V2     #
            # ---------- #

            [
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][0], 0., sp12_darr[2][3][0]),  # ad2 a1
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][1], 0., sp12_darr[2][3][1]),  # ad2 b1
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][2], 0., sp12_darr[2][3][2]),  # ad2 g1
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][3], 0., sp12_darr[2][3][3]),  # ad2 a2
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][4], 0., sp12_darr[2][3][4]),  # ad2 b2
                da2da1V12(mass_arr[1], 1., spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][3], sp12_darr[1][5], 0., sp12_darr[2][3][5])   # ad2 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][0], sinh(2.*a2), sp12_darr[2][4][0]),  # bd2 a1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][1], 0.,          sp12_darr[2][4][1]),  # bd2 b1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][2], 0.,          sp12_darr[2][4][2]),  # bd2 g1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][3], 0.,          sp12_darr[2][4][3]),  # bd2 a2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][4], 0.,          sp12_darr[2][4][4]),  # bd2 b2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][4], sp12_darr[1][5], 0.,          sp12_darr[2][4][5])   # bd2 g2
            ],

            # ---------- #

            [
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][0], sinh(2.*a2)*sin(b2)*sin(b2),  sp12_darr[2][5][0]),  # gd2 a1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][1], sin(2.*b2)*sinh(a2)*sinh(a2), sp12_darr[2][5][1]),  # gd2 b1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][2], 0.,                           sp12_darr[2][5][2]),  # gd2 g1
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][3], 0.,                           sp12_darr[2][5][3]),  # gd2 a2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][4], 0.,                           sp12_darr[2][5][4]),  # gd2 b2
                da2da1V12(mass_arr[1], sinh(a2)*sinh(a2)*sin(b2)*sin(b2), spring_arr[0][0], spring_arr[0][1], sp12_darr[0][0], sp12_darr[1][5], sp12_darr[1][5], 0.,                           sp12_darr[2][5][5])   # gd2 g2
            ]

        ])

    # This function is to simplify the following function that generates the square matrix of energy terms for the jacobian

    def jacobi_energy_terms(positions, velocities, mass_arr, spring_arr):

        def derivative_terms(a1, b1, g1, a2, b2, g2):
            # D function
            d12=D12(a1, b1, g1, a2, b2, g2)
            # First derivatives of D function
            da1d12=da1D12(a1, b1, g1, a2, b2, g2)
            db1d12=db1D12(a1, b1, g1, a2, b2, g2)
            dg1d12=dg1D12(a1, b1, g1, a2, b2, g2)
            da2d12=da1D12(a2, b2, g2, a1, b1, g1)
            db2d12=db1D12(a2, b2, g2, a1, b1, g1)
            dg2d12=dg1D12(a2, b2, g2, a1, b1, g1)
            return [
                [ # D function
                    d12
                ],
                [ # First derivatives of D function
                    da1d12, 
                    db1d12, 
                    dg1d12, 
                    da2d12, 
                    db2d12, 
                    dg2d12
                ]
            ]

        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        ad1,bd1,gd1=velocities[0]
        ad2,bd2,gd2=velocities[1]
        ### Spring 1-2 spring_arr[0]###
        sp12_darr=derivative_terms(a1, b1, g1, a2, b2, g2)

        return array([
            -mass_arr[0]*sinh(a1)*cosh(a1)*( bd1*bd1 + sin(b1)*sin(b1)*gd1*gd1 ) - spring_arr[0][0]*( arccosh(sp12_darr[0][0]) - spring_arr[0][1] )*( sp12_darr[1][0] / sqrt(sp12_darr[0][0]*sp12_darr[0][0] - 1.) ),
            -mass_arr[0]*sin(b1)*cos(b1)*gd1*gd1                                 - spring_arr[0][0]*( arccosh(sp12_darr[0][0]) - spring_arr[0][1] )*( sp12_darr[1][1] / sqrt(sp12_darr[0][0]*sp12_darr[0][0] - 1.) ),
            -mass_arr[0]*0.                                                      - spring_arr[0][0]*( arccosh(sp12_darr[0][0]) - spring_arr[0][1] )*( sp12_darr[1][2] / sqrt(sp12_darr[0][0]*sp12_darr[0][0] - 1.) ),
            -mass_arr[1]*sinh(a2)*cosh(a2)*( bd2*bd2 + sin(b2)*sin(b2)*gd2*gd2 ) - spring_arr[0][0]*( arccosh(sp12_darr[0][0]) - spring_arr[0][1] )*( sp12_darr[1][3] / sqrt(sp12_darr[0][0]*sp12_darr[0][0] - 1.) ),
            -mass_arr[1]*sin(b2)*cos(b2)*gd2*gd2                                 - spring_arr[0][0]*( arccosh(sp12_darr[0][0]) - spring_arr[0][1] )*( sp12_darr[1][4] / sqrt(sp12_darr[0][0]*sp12_darr[0][0] - 1.) ),
            -mass_arr[1]*0.                                                      - spring_arr[0][0]*( arccosh(sp12_darr[0][0]) - spring_arr[0][1] )*( sp12_darr[1][5] / sqrt(sp12_darr[0][0]*sp12_darr[0][0] - 1.) ),
            -mass_arr[0]*ad1,
            -mass_arr[0]*sinh(a1)*sinh(a1)*bd1,
            -mass_arr[0]*sin(b1)*sin(b1)*gd1,
            -mass_arr[1]*ad2,
            -mass_arr[1]*sinh(a2)*sinh(a2)*bd2,
            -mass_arr[1]*sin(b2)*sin(b2)*gd2
        ])


   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        def geo_spring_term_ad(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*1.)*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.))


        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*sin(b1n)**2.)*sinh(a1n)*cosh(a1n) + geo_spring_term_ad(a1n, b1n, g1n, a2n, b2n, g2n, m1, sp12)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*sin(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) + geo_spring_term_ad(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, sp12)
            ))

    def con5(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        def geo_spring_term_bd(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*sinh(a1)*sinh(a1))*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sin(b1n)*cos(b1n) - 2.*ad1n*bd1n/tanh(a1n) + geo_spring_term_bd(a1n, b1n, g1n, a2n, b2n, g2n, m1, sp12)
            + 
            gd1n1*gd1n1*sin(b1n1)*cos(b1n1) - 2.*ad1n1*bd1n1/tanh(a1n1) + geo_spring_term_bd(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, sp12)
            ))

    def con6(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, base_vel, base_vel_guess, m1, h, sp12):
        def geo_spring_term_gd(a1, b1, g1, a2, b2, g2, m, sp12):
            return (-sp12[0]/(m*sinh(a1)*sinh(a1)*sin(b1)*sin(b1))*( 
            arccosh(cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)) - sp12[1])*
            (sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess

        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n/tanh(a1n) - 2.*bd1n*gd1n/tan(b1n) + geo_spring_term_gd(a1n, b1n, g1n, a2n, b2n, g2n, m1, sp12)
            + 
            -2.*ad1n1*gd1n1/tanh(a1n1) - 2.*bd1n1*gd1n1/tan(b1n1) + geo_spring_term_gd(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1, m1, sp12)
            ))
    
    def con7(positions, velocities, mass_arr, spring_arr, energy):
        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        ad1,bd1,gd1=velocities[0]
        ad2,bd2,gd2=velocities[1]
        d12=cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

        return (energy -
            .5*mass_arr[0]*( ad1*ad1 + sinh(a1)*sinh(a1)*bd1*bd1 + sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*gd1*gd1 ) - 
            .5*mass_arr[1]*( ad2*ad2 + sinh(a2)*sinh(a2)*bd2*bd2 + sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*gd2*gd2 ) -
            .5*spring_arr[0][0]*( arccosh(d12) - spring_arr[0][1] )**2.
            )


    
    def jacobian(positions, velocities, mass_arr, h, spring_arr):
        def geo_term_arr(a1, b1, g1, ad1, bd1, gd1, h):
            return [
                [   # a1, b1, g1 derivatives of adn1 update constraint
                    -.5*h*(bd1*bd1+sin(b1)*sin(b1)*gd1*gd1)*cosh(2.*a1),
                    -.25*h*sinh(2.*a1)*sin(2.*b1)*gd1*gd1,
                    0,
                ],
                [   # a1, b1, g1 derivatives of bdn1 update constraint
                    -h*ad1*bd1/(sinh(a1)*sinh(a1)),
                    -.5*h*cos(2.*b1)*gd1*gd1,
                    0.
                ],
                [   # a1, b1, g1 derivatives of gdn1 update constraint
                    -h*ad1*gd1/(sinh(a1)*sinh(a1)),
                    -h*bd1*gd1/(sin(b1)*sin(b1)),
                    0.
                ],
                [   # ad1, bd1, gd1 derivatives of adn1 update constraint
                    1.,
                    -.5*h*sinh(2.*a1)*bd1,
                    -.5*h*sinh(2.*a1)*sin(b1)*sin(b1)*gd1
                ],
                [   # ad1, bd1, gd1 derivatives of bdn1 update constraint
                    h/tanh(a1)*bd1,
                    1.+h/tanh(a1)*ad1,
                    -.5*h*sin(2.*b1)*gd1
                ],
                [   # ad1, bd1, gd1 derivatives of gdn1 update constraint
                    h/tanh(a1)*gd1,
                    h/tan(b1)*gd1,
                    1.+h/tanh(a1)*ad1+h/tan(b1)*bd1
                ]
            ]

        a1n1,b1n1,g1n1=positions[0]
        a2n1,b2n1,g2n1=positions[1]
        ad1n1,bd1n1,gd1n1=velocities[0]
        ad2n1,bd2n1,gd2n1=velocities[1]
        spring_terms=jacobi_sp_terms(positions, mass_arr, spring_arr)
        energy_terms=jacobi_energy_terms(positions, velocities, mass_arr, spring_arr)
        geo_term_p1=geo_term_arr(a1n1, b1n1, g1n1, ad1n1, bd1n1, gd1n1, h)
        geo_term_p2=geo_term_arr(a2n1, b2n1, g2n1, ad2n1, bd2n1, gd2n1, h)
        #print(spring_terms)
        return array([
            [1.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.],
            [0.,1.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,-.5*h,0., 0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,-.5*h, 0.],

            # ad1 update
            [
                geo_term_p1[0][0] + spring_terms[0][0],
                geo_term_p1[0][1] + spring_terms[0][1],
                geo_term_p1[0][2] + spring_terms[0][2], 
            
                spring_terms[0][3],
                spring_terms[0][4],
                spring_terms[0][5],

                geo_term_p1[3][0],
                geo_term_p1[3][1],
                geo_term_p1[3][2],

                0.,0.,0., 0.
            ],

            # bd1 update
            [
                geo_term_p1[1][0] + spring_terms[1][0],
                geo_term_p1[1][1] + spring_terms[1][1],
                geo_term_p1[1][2] + spring_terms[1][2], 

                spring_terms[1][3],
                spring_terms[1][4],
                spring_terms[1][5],

                geo_term_p1[4][0],
                geo_term_p1[4][1],
                geo_term_p1[4][2],

                0.,0.,0., 0.
            ],

            # gd1 update
            [
                geo_term_p1[2][0] + spring_terms[2][0],
                geo_term_p1[2][1] + spring_terms[2][1],
                geo_term_p1[2][2] + spring_terms[2][2],

                spring_terms[2][3],
                spring_terms[2][4],
                spring_terms[2][5],

                geo_term_p1[5][0],
                geo_term_p1[5][1],
                geo_term_p1[5][2],

                0.,0.,0., 0.
            ],

            # ad2 update
            [
                spring_terms[3][0],
                spring_terms[3][1],
                spring_terms[3][2],

                geo_term_p2[0][0] + spring_terms[3][3],
                geo_term_p2[0][1] + spring_terms[3][4],
                geo_term_p2[0][2] + spring_terms[3][5],

                0.,0.,0.,

                geo_term_p2[3][0],
                geo_term_p2[3][1],
                geo_term_p2[3][2], 0.
            ],

            # bd2 update
            [
                spring_terms[4][0],
                spring_terms[4][1],
                spring_terms[4][2],

                geo_term_p2[1][0] + spring_terms[4][3],
                geo_term_p2[1][1] + spring_terms[4][4],
                geo_term_p2[1][2] + spring_terms[4][5],

                0.,0.,0.,

                geo_term_p2[4][0],
                geo_term_p2[4][1],
                geo_term_p2[4][2], 0.
            ],

            # gd2 update
            [
                spring_terms[5][0],
                spring_terms[5][1],
                spring_terms[5][2],

                geo_term_p2[2][0] + spring_terms[5][3],
                geo_term_p2[2][1] + spring_terms[5][4],
                geo_term_p2[2][2] + spring_terms[5][5],

                0.,0.,0.,

                geo_term_p2[5][0],
                geo_term_p2[5][1],
                geo_term_p2[5][2], 0.
            ],

            # energy conservation
            [
                energy_terms[0],
                energy_terms[1],
                energy_terms[2],
                energy_terms[3],
                energy_terms[4],
                energy_terms[5],
                energy_terms[6],
                energy_terms[7],
                energy_terms[8],
                energy_terms[9],
                energy_terms[10],
                energy_terms[11], 1.
            ]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr),-array([
        #p1
        con1(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con2(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con3(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        #p2
        con1(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con2(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con3(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),

        #v1
        con4(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        con5(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        con6(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0]),
        #v2
        con4(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0]),
        con5(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0]),
        con6(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0]),

        #energy
        con7(posn_arr, veln_arr, mass_arr, spring_arr, energy)    
    ]))
    val1 = array([
            posn_arr[0][0]+diff1[0], posn_arr[0][1]+diff1[1], posn_arr[0][2]+diff1[2], 
            posn_arr[1][0]+diff1[3], posn_arr[1][1]+diff1[4], posn_arr[1][2]+diff1[5],

            veln_arr[0][0]+diff1[6], veln_arr[0][1]+diff1[7], veln_arr[0][2]+diff1[8], 
            veln_arr[1][0]+diff1[9], veln_arr[1][1]+diff1[10], veln_arr[1][2]+diff1[11],

            energy+diff1[12]
            ])    
    x = 0
    while(x < 7):
        new_pos_arr=array([val1[0:3],val1[3:6]])
        new_vel_arr=array([val1[6:9],val1[9:12]])
        new_energy=val1[12]
        diff2=linalg.solve(jacobian(new_pos_arr, new_vel_arr, mass_arr, step, spring_arr),-array([
            #p1
            con1(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con2(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con3(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            #p2
            con1(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con2(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con3(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),

            #v1
            con4(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            con5(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            con6(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0]),
            #v2
            con4(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0]),
            con5(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0]),
            con6(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0]),

            #energy 
            con7(new_pos_arr, new_vel_arr, mass_arr, spring_arr, new_energy)
        ]))      
        val2 = array([
            val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], 
            val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5],

            val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8],
            val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11],

            val1[12]+diff2[12]
            ])
        val1 = val2
        x=x+1
    #print(val1[9:17])
    return val1

######################################################
### Hyperbolic 3-space Spring Potential (Triangle) ###
######################################################

# This is an extension of the dumb-bell system using the structure for the cube
# solver. The configuration is a triangle of three masses and three springs. 
# There seems to be some issue with this setup. While the solver works for 
# the case of three masses even though the cube solver works fine.
# DO NOT USE TRANSLATIONAL PARAMETERIZATION FOR STRUCTURES WITH ROTATIONAL
# SYMMETRY OF ODD DEGREE.

def imph3sptrans3(posn_arr, veln_arr, step, mass_arr, spring_arr):

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

    def jacobi_sp_terms(positions, mass_arr, spring_arr):
        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        a3,b3,g3=positions[2]
        
        
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


        ### Spring 2-3 spring_arr[2]###
        # D function
        d23=D12(a2, b2, g2, a3, b3, g3)
        # First derivatives of D function
        da2d23=da1D12(a2, b2, g2, a3, b3, g3)
        db2d23=db1D12(a2, b2, g2, a3, b3, g3)
        dg2d23=dg1D12(a2, b2, g2, a3, b3, g3)
        da3d23=da1D12(a3, b3, g3, a2, b2, g2)
        db3d23=db1D12(a3, b3, g3, a2, b2, g2)
        dg3d23=dg1D12(a3, b3, g3, a2, b2, g2)
        # Second derivatives of D function
        da2d23a2=da1D12a1(a2, b2, g2, a3, b3, g3)
        db2d23a2=db1D12a1(a2, b2, g2, a3, b3, g3)
        dg2d23a2=dg1D12a1(a2, b2, g2, a3, b3, g3)
        da3d23a2=da2D12a1(a2, b2, g2, a3, b3, g3)
        db3d23a2=db2D12a1(a2, b2, g2, a3, b3, g3)
        dg3d23a2=dg2D12a1(a2, b2, g2, a3, b3, g3)
        
        da2d23b2=db2d23a2
        db2d23b2=db1D12b1(a2, b2, g2, a3, b3, g3)
        dg2d23b2=dg1D12b1(a2, b2, g2, a3, b3, g3)
        da3d23b2 = db2D12a1(a3, b3, g3, a2, b2, g2)
        db3d23b2=db2D12b1(a2, b2, g2, a3, b3, g3)
        dg3d23b2=dg2D12b1(a2, b2, g2, a3, b3, g3)

        da2d23g2=dg2d23a2
        db2d23g2=dg2d23b2
        dg2d23g2=dg1D12g1(a2, b2, g2, a3, b3, g3)
        da3d23g2 = dg2D12a1(a3, b3, g3, a2, b2, g2)
        db3d23g2 = dg2D12b1(a3, b3, g3, a2, b2, g2)
        dg3d23g2=dg2D12g1(a2, b2, g2, a3, b3, g3)

        da2d23a3=da3d23a2
        db2d23a3=da3d23b2
        dg2d23a3=da3d23g2
        da3d23a3 = da1D12a1(a3, b3, g3, a2, b2, g2)
        db3d23a3 = db1D12a1(a3, b3, g3, a2, b2, g2)
        dg3d23a3 = dg1D12a1(a3, b3, g3, a2, b2, g2)

        da2d23b3=db3d23a2
        db2d23b3=db3d23b2
        dg2d23b3=db3d23g2
        da3d23b3=db3d23a3
        db3d23b3 = db1D12b1(a3, b3, g3, a2, b2, g2)
        dg3d23b3 = dg1D12b1(a3, b3, g3, a2, b2, g2)

        da2d23g3=dg3d23a2
        db2d23g3=dg3d23b2
        dg2d23g3=dg3d23g2
        da3d23g3=dg3d23a3
        db3d23g3=dg3d23b3
        dg3d23g3 = dg1D12g1(a3, b3, g3, a2, b2, g2)
        

        return array([

            # ---------- #
            #     V1     #
            # ---------- #
            
            # ad1 a1
            [-spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da1d12/(d12**2. - 1.) - da1d12a1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da1d13/(d13**2. - 1.) - da1d13a1) ),

            # ad1 b1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db1d12/(d12**2. - 1.) - db1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db1d13/(d13**2. - 1.) - db1d13a1) ),

            # ad1 g1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg1d12/(d12**2. - 1.) - dg1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg1d13/(d13**2. - 1.) - dg1d13a1) ),

            # ad1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da2d12/(d12**2. - 1.) - da2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db2d12/(d12**2. - 1.) - db2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg2d12/(d12**2. - 1.) - dg2d12a1) ),

            # ad1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da3d13/(d13**2. - 1.) - da3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db3d13/(d13**2. - 1.) - db3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg3d13/(d13**2. - 1.) - dg3d13a1) )
            ],

            # ---------- #

            # bd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*sinh(2.*a1)/(cosh(a1)*cosh(a1)) + d12*db1d12*da1d12/(d12**2. - 1.) - da1d12b1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*sinh(2.*a1)/(cosh(a1)*cosh(a1)) + d13*db1d13*da1d13/(d13**2. - 1.) - da1d13b1) ),

            # bd1 b1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db1d12/(d12**2. - 1.) - db1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*db1d13/(d13**2. - 1.) - db1d13b1) ),

            # bd1 g1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*dg1d12/(d12**2. - 1.) - dg1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*dg1d13/(d13**2. - 1.) - dg1d13b1) ),

            # bd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*da2d12/(d12**2. - 1.) - da2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*db2d12/(d12**2. - 1.) - db2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(cosh(a1)*cosh(a1)) + d12*db1d12*dg2d12/(d12**2. - 1.) - dg2d12b1) ),

            # bd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*da3d13/(d13**2. - 1.) - da3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*db3d13/(d13**2. - 1.) - db3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(cosh(a1)*cosh(a1)) + d13*db1d13*dg3d13/(d13**2. - 1.) - dg3d13b1) ),
            ],

            # ---------- #

            # gd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sinh(2.*a1)*cosh(b1)*cosh(b1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*da1d12/(d12**2. - 1.) - da1d12g1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sinh(2.*a1)*cosh(b1)*cosh(b1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*da1d13/(d13**2. - 1.) - da1d13g1) ),

            # gd1 b1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sinh(2.*b1)*cosh(a1)*cosh(a1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*db1d12/(d12**2. - 1.) - db1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sinh(2.*b1)*cosh(a1)*cosh(a1)/(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*db1d13/(d13**2. - 1.) - db1d13g1) ),

            # gd1 g1
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*dg1d12/(d12**2. - 1.) - dg1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*dg1d13/(d13**2. - 1.) - dg1d13g1) ),

            # gd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*da2d12/(d12**2. - 1.) - da2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*db2d12/(d12**2. - 1.) - db2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d12*dg1d12*dg2d12/(d12**2. - 1.) - dg2d12g1) ),

            # gd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*da3d13/(d13**2. - 1.) - da3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*db3d13/(d13**2. - 1.) - db3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(cosh(a1)*cosh(a1)*cosh(b1)*cosh(b1)) + d13*dg1d13*dg3d13/(d13**2. - 1.) - dg3d13g1) ),
            ],

            # ---------- #
            #     V2     #
            # ---------- #

            # ad2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da1d12/(d12**2. - 1.) - da1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db1d12/(d12**2. - 1.) - db1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg1d12/(d12**2. - 1.) - dg1d12a2) ),

            # ad2 a2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da2d12/(d12**2. - 1.) - da2d12a2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*da2d23/(d23**2. - 1.) - da2d23a2) ),

            # ad2 b2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db2d12/(d12**2. - 1.) - db2d12a2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*db2d23/(d23**2. - 1.) - db2d23a2) ),

            # ad2 g2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg2d12/(d12**2. - 1.) - dg2d12a2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*dg2d23/(d23**2. - 1.) - dg2d23a2) ),

            # ad2 a3,b3,g3
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*da3d23/(d23**2. - 1.) - da3d23a2) ),
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*db3d23/(d23**2. - 1.) - db3d23a2) ),
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*dg3d23/(d23**2. - 1.) - dg3d23a2) ),
            ],

            # ---------- #

            # bd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*da1d12/(d12**2. - 1.) - da1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db1d12/(d12**2. - 1.) - db1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*dg1d12/(d12**2. - 1.) - dg1d12b2) ),

            # bd2 a2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*sinh(2.*a2)/(cosh(a2)*cosh(a2)) + d12*db2d12*da2d12/(d12**2. - 1.) - da2d12b2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*sinh(2.*a2)/(cosh(a2)*cosh(a2)) + d23*db2d23*da2d23/(d23**2. - 1.) - da2d23b2) ),

            # bd2 b2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*db2d12/(d12**2. - 1.) - db2d12b2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(cosh(a2)*cosh(a2)) + d23*db2d23*db2d23/(d23**2. - 1.) - db2d23b2) ),

            # bd2 g2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(cosh(a2)*cosh(a2)) + d12*db2d12*dg2d12/(d12**2. - 1.) - dg2d12b2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(cosh(a2)*cosh(a2)) + d23*db2d23*dg2d23/(d23**2. - 1.) - dg2d23b2) ),

            # bd2 a3,b3,g3
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(cosh(a2)*cosh(a2)) + d23*db2d23*da3d23/(d23**2. - 1.) - da3d23b2) ),
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(cosh(a2)*cosh(a2)) + d23*db2d23*db3d23/(d23**2. - 1.) - db3d23b2) ),
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(cosh(a2)*cosh(a2)) + d23*db2d23*dg3d23/(d23**2. - 1.) - dg3d23b2) ),
            ],

            # ---------- #

            # gd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*da1d12/(d12**2. - 1.) - da1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*db1d12/(d12**2. - 1.) - db1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*dg1d12/(d12**2. - 1.) - dg1d12g2) ),

            # gd2 a2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sinh(2.*a2)*cosh(b2)*cosh(b2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*da2d12/(d12**2. - 1.) - da2d12g2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*sinh(2.*a2)*cosh(b2)*cosh(b2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d23*dg2d23*da2d23/(d23**2. - 1.) - da2d23g2) ),

            # gd2 b2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sinh(2.*b2)*cosh(a2)*cosh(a2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*db2d12/(d12**2. - 1.) - db2d12g2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*sinh(2.*b2)*cosh(a2)*cosh(a2)/(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d23*dg2d23*db2d23/(d23**2. - 1.) - db2d23g2) ),

            # gd2 g2
            -spring_arr[0][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d12*dg2d12*dg2d12/(d12**2. - 1.) - dg2d12g2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d23*dg2d23*dg2d23/(d23**2. - 1.) - dg2d23g2) ),

            # gd2 a3,b3,g3
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d23*dg2d23*da3d23/(d23**2. - 1.) - da3d23g2) ),
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d23*dg2d23*db3d23/(d23**2. - 1.) - db3d23g2) ),
            -spring_arr[2][0]/(mass_arr[1]*cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(cosh(a2)*cosh(a2)*cosh(b2)*cosh(b2)) + d23*dg2d23*dg3d23/(d23**2. - 1.) - dg3d23g2) ),
            ],

            # ---------- #
            #     V3     #
            # ---------- #

            # ad3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da1d13/(d13**2. - 1.) - da1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db1d13/(d13**2. - 1.) - db1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg1d13/(d13**2. - 1.) - dg1d13a3) ),

            # ad3 a2,b2,g2
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*da2d23/(d23**2. - 1.) - da2d23a3) ),
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*db2d23/(d23**2. - 1.) - db2d23a3) ),
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*dg2d23/(d23**2. - 1.) - dg2d23a3) ),

            # ad3 a3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da3d13/(d13**2. - 1.) - da3d13a3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*da3d23/(d23**2. - 1.) - da3d23a3) ),

            # ad3 b3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db3d13/(d13**2. - 1.) - db3d13a3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*db3d23/(d23**2. - 1.) - db3d23a3) ),

            # ad3 g3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg3d13/(d13**2. - 1.) - dg3d13a3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*dg3d23/(d23**2. - 1.) - dg3d23a3) ),
            ],

            # ---------- #

            # bd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*da1d13/(d13**2. - 1.) - da1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*db1d13/(d13**2. - 1.) - db1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*dg1d13/(d13**2. - 1.) - dg1d13b3) ),

            # bd3 a2,b2,g2
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*da2d23/(d23**2. - 1.) - da2d23b3) ),
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*db2d23/(d23**2. - 1.) - db2d23b3) ),
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*dg2d23/(d23**2. - 1.) - dg2d23b3) ),

            # bd3 a3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*sinh(2.*a3)/(cosh(a3)*cosh(a3)) + d13*db3d13*da3d13/(d13**2. - 1.) - da3d13b3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*sinh(2.*a3)/(cosh(a3)*cosh(a3)) + d23*db3d23*da3d23/(d23**2. - 1.) - da3d23b3) ),

            # bd3 b3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*db3d13/(d13**2. - 1.) - db3d13b3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*db3d23/(d23**2. - 1.) - db3d23b3) ),

            # bd3 g3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(cosh(a3)*cosh(a3)) + d13*db3d13*dg3d13/(d13**2. - 1.) - dg3d13b3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*dg3d23/(d23**2. - 1.) - dg3d23b3) ),
            ],

            # ---------- #

            # gd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*da1d13/(d13**2. - 1.) - da1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*db1d13/(d13**2. - 1.) - db1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*dg1d13/(d13**2. - 1.) - dg1d13g3) ),

            # gd3 a2,b2,g2
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d23*dg3d23*da2d23/(d23**2. - 1.) - da2d23g3) ),
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d23*dg3d23*db2d23/(d23**2. - 1.) - db2d23g3) ),
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d23*dg3d23*dg2d23/(d23**2. - 1.) - dg2d23g3) ),

            # gd3 a3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sinh(2.*a3)*cosh(b3)*cosh(b3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*da3d13/(d13**2. - 1.) - da3d13g3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*sinh(2.*a3)*cosh(b3)*cosh(b3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d23*dg3d23*da3d23/(d23**2. - 1.) - da3d23g3) ),

            # gd3 b3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sinh(2.*b3)*cosh(a3)*cosh(a3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*db3d13/(d13**2. - 1.) - db3d13g3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*sinh(2.*b3)*cosh(a3)*cosh(a3)/(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d23*dg3d23*db3d23/(d23**2. - 1.) - db3d23g3) ),

            # gd3 g3
            -spring_arr[1][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d13*dg3d13*dg3d13/(d13**2. - 1.) - dg3d13g3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(cosh(a3)*cosh(a3)*cosh(b3)*cosh(b3)) + d23*dg3d23*dg3d23/(d23**2. - 1.) - dg3d23g3) ),
            ]

        ])
   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*cosh(b1n)**2.)*sinh(a1n)*cosh(a1n) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - sp12[1])*
            (-cosh(a1n)*sinh(a2n) - sinh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + sinh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n))) - sp13[1])*
            (-cosh(a1n)*sinh(a3n) - sinh(a1n)*sinh(b1n)*cosh(a3n)*sinh(b3n) + sinh(a1n)*cosh(b1n)*cosh(a3n)*cosh(b3n)*cosh(g1n - g3n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n)))**2.)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*cosh(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - sp12[1])*
            (-cosh(a1n1)*sinh(a2n1) - sinh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + sinh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1))) - sp13[1])*
            (-cosh(a1n1)*sinh(a3n1) - sinh(a1n1)*sinh(b1n1)*cosh(a3n1)*sinh(b3n1) + sinh(a1n1)*cosh(b1n1)*cosh(a3n1)*cosh(b3n1)*cosh(g1n1 - g3n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1)))**2.)
            ))

    def con5(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sinh(b1n)*cosh(b1n) - 2.*ad1n*bd1n*tanh(a1n) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - sp12[1])*
            (-cosh(a1n)*cosh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*sinh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n))) - sp13[1])*
            (-cosh(a1n)*cosh(b1n)*cosh(a3n)*sinh(b3n) + cosh(a1n)*sinh(b1n)*cosh(a3n)*cosh(b3n)*cosh(g1n - g3n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n)))**2.)
            + 
            gd1n1*gd1n1*sinh(b1n1)*cosh(b1n1) - 2.*ad1n1*bd1n1*tanh(a1n1) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - sp12[1])*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1))) - sp13[1])*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a3n1)*sinh(b3n1) + cosh(a1n1)*sinh(b1n1)*cosh(a3n1)*cosh(b3n1)*cosh(g1n1 - g3n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1)))**2.)
            ))

    def con6(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess

        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n*tanh(a1n) - 2.*bd1n*gd1n*tanh(b1n) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - sp12[1])*
            (cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*sinh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n))) - sp13[1])*
            (cosh(a1n)*cosh(b1n)*cosh(a3n)*cosh(b3n)*sinh(g1n - g3n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n)))**2.)
            + 
            -2.*ad1n1*gd1n1*tanh(a1n1) - 2.*bd1n1*gd1n1*tanh(b1n1) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - sp12[1])*
            (cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*sinh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1))) - sp13[1])*
            (cosh(a1n1)*cosh(b1n1)*cosh(a3n1)*cosh(b3n1)*sinh(g1n1 - g3n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1)))**2.)
            )) 
    
    def jacobian(positions, velocities, mass_arr, h, spring_arr):
        a1n1,b1n1,g1n1=positions[0]
        a2n1,b2n1,g2n1=positions[1]
        a3n1,b3n1,g3n1=positions[2]
        ad1n1,bd1n1,gd1n1=velocities[0]
        ad2n1,bd2n1,gd2n1=velocities[1]
        ad3n1,bd3n1,gd3n1=velocities[2]
        spring_terms=jacobi_sp_terms(positions, mass_arr, spring_arr)
        #print(spring_terms)
        return array([
            [1.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1+cosh(b1n1)*cosh(b1n1)*gd1n1*gd1n1)*cosh(2.*a1n1) + spring_terms[0][0],
            -.25*h*sinh(2.*a1n1)*sinh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[0][1],
            0. + spring_terms[0][2], 
            
            spring_terms[0][3],
            spring_terms[0][4],
            spring_terms[0][5],

            spring_terms[0][6],
            spring_terms[0][7],
            spring_terms[0][8],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            -.5*h*sinh(2.*a1n1)*cosh(b1n1)*cosh(b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0.
            ],

            # bd1 update
            [h*ad1n1*bd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[1][0],
            -.5*h*cosh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[1][1],
            0. + spring_terms[1][2], 
            
            spring_terms[1][3],
            spring_terms[1][4],
            spring_terms[1][5],

            spring_terms[1][6],
            spring_terms[1][7],
            spring_terms[1][8],
            
            h*tanh(a1n1)*bd1n1,
            1.+h*tanh(a1n1)*ad1n1,
            -.5*h*sinh(2.*b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0.
            ],

            # gd1 update
            [h*ad1n1*gd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[2][0],
            h*bd1n1*gd1n1/(cosh(b1n1)*cosh(b1n1)) + spring_terms[2][1],
            0. + spring_terms[2][2],
             
            spring_terms[2][3],
            spring_terms[2][4],
            spring_terms[2][5],

            spring_terms[2][6],
            spring_terms[2][7],
            spring_terms[2][8],

            h*tanh(a1n1)*gd1n1,
            h*tanh(b1n1)*gd1n1,
            1.+h*tanh(a1n1)*ad1n1+h*tanh(b1n1)*bd1n1,

            0.,0.,0., 0.,0.,0.
            ],

            # ad2 update
            [spring_terms[3][0],
            spring_terms[3][1],
            spring_terms[3][2],

            -.5*h*(bd2n1*bd2n1+cosh(b2n1)*cosh(b2n1)*gd2n1*gd2n1)*cosh(2.*a2n1) + spring_terms[3][3],
            -.25*h*sinh(2.*a2n1)*sinh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[3][4],
            0. + spring_terms[3][5],

            spring_terms[3][6],
            spring_terms[3][7],
            spring_terms[3][8],

            0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            -.5*h*sinh(2.*a2n1)*cosh(b2n1)*cosh(b2n1)*gd2n1,

            0.,0.,0.
            ],

            # bd2 update
            [spring_terms[4][0],
            spring_terms[4][1],
            spring_terms[4][2],

            h*ad2n1*bd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[4][3],
            -.5*h*cosh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[4][4],
            0. + spring_terms[4][5],

            spring_terms[4][6],
            spring_terms[4][7],
            spring_terms[4][8],

            0.,0.,0.,

            h*tanh(a2n1)*bd2n1,
            1.+h*tanh(a2n1)*ad2n1,
            -.5*h*sinh(2.*b2n1)*gd2n1,
            
            0.,0.,0.
            ],

            # gd2 update
            [spring_terms[5][0],
            spring_terms[5][1],
            spring_terms[5][2],

            h*ad2n1*gd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[5][3],
            h*bd2n1*gd2n1/(cosh(b2n1)*cosh(b2n1)) + spring_terms[5][4],
            0. + spring_terms[5][5],

            spring_terms[5][6],
            spring_terms[5][7],
            spring_terms[5][8],

            0.,0.,0.,

            h*tanh(a2n1)*gd2n1,
            h*tanh(b2n1)*gd2n1,
            1.+h*tanh(a2n1)*ad2n1+h*tanh(b2n1)*bd2n1,
            
            0.,0.,0.
            ],

            # ad3 update
            [spring_terms[6][0],
            spring_terms[6][1],
            spring_terms[6][2],

            spring_terms[6][3],
            spring_terms[6][4],
            spring_terms[6][5],

            -.5*h*(bd3n1*bd3n1+cosh(b3n1)*cosh(b3n1)*gd3n1*gd3n1)*cosh(2.*a3n1) + spring_terms[6][6],
            -.25*h*sinh(2.*a3n1)*sinh(2.*b3n1)*gd3n1*gd3n1 + spring_terms[6][7],
            0. + spring_terms[6][8],

            0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a3n1)*bd3n1,
            -.5*h*sinh(2.*a3n1)*cosh(b3n1)*cosh(b3n1)*gd3n1
            ],

            # bd3 update
            [spring_terms[7][0],
            spring_terms[7][1],
            spring_terms[7][2],

            spring_terms[7][3],
            spring_terms[7][4],
            spring_terms[7][5],

            h*ad3n1*bd3n1/(cosh(a3n1)*cosh(a3n1)) + spring_terms[7][6],
            -.5*h*cosh(2.*b3n1)*gd3n1*gd3n1 + spring_terms[7][7],
            0. + spring_terms[7][8],

            0.,0.,0., 0.,0.,0.,

            h*tanh(a3n1)*bd3n1,
            1.+h*tanh(a3n1)*ad3n1,
            -.5*h*sinh(2.*b3n1)*gd3n1
            ],

            # gd3 update
            [spring_terms[8][0],
            spring_terms[8][1],
            spring_terms[8][2],

            spring_terms[8][3],
            spring_terms[8][4],
            spring_terms[8][5],

            h*ad3n1*gd3n1/(cosh(a3n1)*cosh(a3n1)) + spring_terms[8][6],
            h*bd3n1*gd3n1/(cosh(b3n1)*cosh(b3n1)) + spring_terms[8][7],
            0. + spring_terms[8][8],

            0.,0.,0., 0.,0.,0.,

            h*tanh(a3n1)*gd3n1,
            h*tanh(b3n1)*gd3n1,
            1.+h*tanh(a3n1)*ad3n1+h*tanh(b3n1)*bd3n1
            ]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr),-array([
        #p1
        con1(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con2(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con3(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        #p2
        con1(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con2(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con3(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        #p3
        con1(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con2(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con3(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),

        #v1
        con4(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
        con5(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
        con6(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
        #v2
        con4(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
        con5(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
        con6(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
        #v3
        con4(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
        con5(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
        con6(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2])      
    ]))
    val1 = array([
            posn_arr[0][0]+diff1[0], posn_arr[0][1]+diff1[1], posn_arr[0][2]+diff1[2], 
            posn_arr[1][0]+diff1[3], posn_arr[1][1]+diff1[4], posn_arr[1][2]+diff1[5],
            posn_arr[2][0]+diff1[6], posn_arr[2][1]+diff1[7], posn_arr[2][2]+diff1[8],

            veln_arr[0][0]+diff1[9], veln_arr[0][1]+diff1[10], veln_arr[0][2]+diff1[11], 
            veln_arr[1][0]+diff1[12], veln_arr[1][1]+diff1[13], veln_arr[1][2]+diff1[14],
            veln_arr[2][0]+diff1[15], veln_arr[2][1]+diff1[16], veln_arr[2][2]+diff1[17]
            ])    
    x = 0
    while(x < 7):
        new_pos_arr=array([val1[0:3],val1[3:6],val1[6:9]])
        new_vel_arr=array([val1[9:12],val1[12:15],val1[15:18]])
        diff2=linalg.solve(jacobian(new_pos_arr, new_vel_arr, mass_arr, step, spring_arr),-array([
            #p1
            con1(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con2(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con3(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            #p2
            con1(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con2(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con3(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            #p3
            con1(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con2(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con3(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),

            #v1
            con4(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
            con5(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
            con6(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
            #v2
            con4(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
            con5(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
            con6(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
            #v3
            con4(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
            con5(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
            con6(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2])    
        ]))      
        val2 = array([
            val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], 
            val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5],
            val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8],

            val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11], 
            val1[12]+diff2[12], val1[13]+diff2[13], val1[14]+diff2[14],
            val1[15]+diff2[15], val1[16]+diff2[16], val1[17]+diff2[17]
            ])
        val1 = val2
        x=x+1
    #print(val1[9:17])
    return val1

# This is an extension of the dumb-bell system using the structure for the cube
# solver. The configuration is a triangle of three masses and three springs. 
# I found that for the case of shapes with odd number rotational symmetry
# that the translational parameterization is not appropriate due to the 
# metric factors being unequal for point with degenerate mirror symmetry 
# compared to the remaining points that have a mirror point. Since it is a 
# coordinate issue, we need to use the rotational coordinate system to remove
# this artifical directional bias. Use rotational parameterization as default.

def imph3sprot3(posn_arr, veln_arr, step, mass_arr, spring_arr):

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only three functions are needed instead of six.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, g2, a1, b1, g1)
    # db2D12 = db1D12(a2, b2, g2, a1, b1, g1)
    # dg2D12 = dg1D12(a2, b2, g2, a1, b1, g1)
    def da1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2) 

    def dg1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)      
    

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*cos(b2) - cosh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def da2D12a1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sinh(a2) - cosh(a1)*cos(b1)*cosh(a2)*cos(b2) - cosh(a1)*sin(b1)*cosh(a2)*sin(b2)*cos(g1 - g2)

    def db2D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cos(b1)*sinh(a2)*sin(b2) - cosh(a1)*sin(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12a1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*cos(b2) + sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def dg1D12g1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg2D12g1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(positions, mass_arr, spring_arr):
        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        a3,b3,g3=positions[2]
        
        
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


        ### Spring 2-3 spring_arr[2]###
        # D function
        d23=D12(a2, b2, g2, a3, b3, g3)
        # First derivatives of D function
        da2d23=da1D12(a2, b2, g2, a3, b3, g3)
        db2d23=db1D12(a2, b2, g2, a3, b3, g3)
        dg2d23=dg1D12(a2, b2, g2, a3, b3, g3)
        da3d23=da1D12(a3, b3, g3, a2, b2, g2)
        db3d23=db1D12(a3, b3, g3, a2, b2, g2)
        dg3d23=dg1D12(a3, b3, g3, a2, b2, g2)
        # Second derivatives of D function
        da2d23a2=da1D12a1(a2, b2, g2, a3, b3, g3)
        db2d23a2=db1D12a1(a2, b2, g2, a3, b3, g3)
        dg2d23a2=dg1D12a1(a2, b2, g2, a3, b3, g3)
        da3d23a2=da2D12a1(a2, b2, g2, a3, b3, g3)
        db3d23a2=db2D12a1(a2, b2, g2, a3, b3, g3)
        dg3d23a2=dg2D12a1(a2, b2, g2, a3, b3, g3)
        
        da2d23b2=db2d23a2
        db2d23b2=db1D12b1(a2, b2, g2, a3, b3, g3)
        dg2d23b2=dg1D12b1(a2, b2, g2, a3, b3, g3)
        da3d23b2 = db2D12a1(a3, b3, g3, a2, b2, g2)
        db3d23b2=db2D12b1(a2, b2, g2, a3, b3, g3)
        dg3d23b2=dg2D12b1(a2, b2, g2, a3, b3, g3)

        da2d23g2=dg2d23a2
        db2d23g2=dg2d23b2
        dg2d23g2=dg1D12g1(a2, b2, g2, a3, b3, g3)
        da3d23g2 = dg2D12a1(a3, b3, g3, a2, b2, g2)
        db3d23g2 = dg2D12b1(a3, b3, g3, a2, b2, g2)
        dg3d23g2=dg2D12g1(a2, b2, g2, a3, b3, g3)

        da2d23a3=da3d23a2
        db2d23a3=da3d23b2
        dg2d23a3=da3d23g2
        da3d23a3 = da1D12a1(a3, b3, g3, a2, b2, g2)
        db3d23a3 = db1D12a1(a3, b3, g3, a2, b2, g2)
        dg3d23a3 = dg1D12a1(a3, b3, g3, a2, b2, g2)

        da2d23b3=db3d23a2
        db2d23b3=db3d23b2
        dg2d23b3=db3d23g2
        da3d23b3=db3d23a3
        db3d23b3 = db1D12b1(a3, b3, g3, a2, b2, g2)
        dg3d23b3 = dg1D12b1(a3, b3, g3, a2, b2, g2)

        da2d23g3=dg3d23a2
        db2d23g3=dg3d23b2
        dg2d23g3=dg3d23g2
        da3d23g3=dg3d23a3
        db3d23g3=dg3d23b3
        dg3d23g3 = dg1D12g1(a3, b3, g3, a2, b2, g2)
        

        return array([

            # ---------- #
            #     V1     #
            # ---------- #
            
            # ad1 a1
            [-spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da1d12/(d12**2. - 1.) - da1d12a1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da1d13/(d13**2. - 1.) - da1d13a1) ),

            # ad1 b1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db1d12/(d12**2. - 1.) - db1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db1d13/(d13**2. - 1.) - db1d13a1) ),

            # ad1 g1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg1d12/(d12**2. - 1.) - dg1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg1d13/(d13**2. - 1.) - dg1d13a1) ),

            # ad1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da2d12/(d12**2. - 1.) - da2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db2d12/(d12**2. - 1.) - db2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg2d12/(d12**2. - 1.) - dg2d12a1) ),

            # ad1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da3d13/(d13**2. - 1.) - da3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db3d13/(d13**2. - 1.) - db3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg3d13/(d13**2. - 1.) - dg3d13a1) )
            ],

            # ---------- #

            # bd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*sinh(2.*a1)/(sinh(a1)*sinh(a1)) + d12*db1d12*da1d12/(d12**2. - 1.) - da1d12b1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*sinh(2.*a1)/(sinh(a1)*sinh(a1)) + d13*db1d13*da1d13/(d13**2. - 1.) - da1d13b1) ),

            # bd1 b1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*db1d12/(d12**2. - 1.) - db1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*db1d13/(d13**2. - 1.) - db1d13b1) ),

            # bd1 g1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*dg1d12/(d12**2. - 1.) - dg1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*dg1d13/(d13**2. - 1.) - dg1d13b1) ),

            # bd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*da2d12/(d12**2. - 1.) - da2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*db2d12/(d12**2. - 1.) - db2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*dg2d12/(d12**2. - 1.) - dg2d12b1) ),

            # bd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*da3d13/(d13**2. - 1.) - da3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*db3d13/(d13**2. - 1.) - db3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*dg3d13/(d13**2. - 1.) - dg3d13b1) ),
            ],

            # ---------- #

            # gd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sinh(2.*a1)*sin(b1)*sin(b1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*da1d12/(d12**2. - 1.) - da1d12g1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sinh(2.*a1)*sin(b1)*sin(b1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*da1d13/(d13**2. - 1.) - da1d13g1) ),

            # gd1 b1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sin(2.*b1)*sinh(a1)*sinh(a1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*db1d12/(d12**2. - 1.) - db1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sin(2.*b1)*sinh(a1)*sinh(a1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*db1d13/(d13**2. - 1.) - db1d13g1) ),

            # gd1 g1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*dg1d12/(d12**2. - 1.) - dg1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*dg1d13/(d13**2. - 1.) - dg1d13g1) ),

            # gd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*da2d12/(d12**2. - 1.) - da2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*db2d12/(d12**2. - 1.) - db2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*dg2d12/(d12**2. - 1.) - dg2d12g1) ),

            # gd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*da3d13/(d13**2. - 1.) - da3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*db3d13/(d13**2. - 1.) - db3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*dg3d13/(d13**2. - 1.) - dg3d13g1) ),
            ],

            # ---------- #
            #     V2     #
            # ---------- #

            # ad2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da1d12/(d12**2. - 1.) - da1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db1d12/(d12**2. - 1.) - db1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg1d12/(d12**2. - 1.) - dg1d12a2) ),

            # ad2 a2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da2d12/(d12**2. - 1.) - da2d12a2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*da2d23/(d23**2. - 1.) - da2d23a2) ),

            # ad2 b2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db2d12/(d12**2. - 1.) - db2d12a2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*db2d23/(d23**2. - 1.) - db2d23a2) ),

            # ad2 g2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg2d12/(d12**2. - 1.) - dg2d12a2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*dg2d23/(d23**2. - 1.) - dg2d23a2) ),

            # ad2 a3,b3,g3
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*da3d23/(d23**2. - 1.) - da3d23a2) ),
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*db3d23/(d23**2. - 1.) - db3d23a2) ),
            -spring_arr[2][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da2d23*0./1. + d23*da2d23*dg3d23/(d23**2. - 1.) - dg3d23a2) ),
            ],

            # ---------- #

            # bd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*da1d12/(d12**2. - 1.) - da1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*db1d12/(d12**2. - 1.) - db1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*dg1d12/(d12**2. - 1.) - dg1d12b2) ),

            # bd2 a2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*sinh(2.*a2)/(sinh(a2)*sinh(a2)) + d12*db2d12*da2d12/(d12**2. - 1.) - da2d12b2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*sinh(2.*a2)/(sinh(a2)*sinh(a2)) + d23*db2d23*da2d23/(d23**2. - 1.) - da2d23b2) ),

            # bd2 b2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*db2d12/(d12**2. - 1.) - db2d12b2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*db2d23/(d23**2. - 1.) - db2d23b2) ),

            # bd2 g2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*dg2d12/(d12**2. - 1.) - dg2d12b2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*dg2d23/(d23**2. - 1.) - dg2d23b2) ),

            # bd2 a3,b3,g3
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*da3d23/(d23**2. - 1.) - da3d23b2) ),
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*db3d23/(d23**2. - 1.) - db3d23b2) ),
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*dg3d23/(d23**2. - 1.) - dg3d23b2) ),
            ],

            # ---------- #

            # gd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*da1d12/(d12**2. - 1.) - da1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*db1d12/(d12**2. - 1.) - db1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*dg1d12/(d12**2. - 1.) - dg1d12g2) ),

            # gd2 a2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sinh(2.*a2)*sin(b2)*sin(b2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*da2d12/(d12**2. - 1.) - da2d12g2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*sinh(2.*a2)*sin(b2)*sin(b2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*da2d23/(d23**2. - 1.) - da2d23g2) ),

            # gd2 b2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sin(2.*b2)*sinh(a2)*sinh(a2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*db2d12/(d12**2. - 1.) - db2d12g2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*sin(2.*b2)*sinh(a2)*sinh(a2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*db2d23/(d23**2. - 1.) - db2d23g2) ),

            # gd2 g2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*dg2d12/(d12**2. - 1.) - dg2d12g2) ) + 
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*dg2d23/(d23**2. - 1.) - dg2d23g2) ),

            # gd2 a3,b3,g3
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*da3d23/(d23**2. - 1.) - da3d23g2) ),
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*db3d23/(d23**2. - 1.) - db3d23g2) ),
            -spring_arr[2][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*dg3d23/(d23**2. - 1.) - dg3d23g2) ),
            ],

            # ---------- #
            #     V3     #
            # ---------- #

            # ad3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da1d13/(d13**2. - 1.) - da1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db1d13/(d13**2. - 1.) - db1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg1d13/(d13**2. - 1.) - dg1d13a3) ),

            # ad3 a2,b2,g2
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*da2d23/(d23**2. - 1.) - da2d23a3) ),
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*db2d23/(d23**2. - 1.) - db2d23a3) ),
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*dg2d23/(d23**2. - 1.) - dg2d23a3) ),

            # ad3 a3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da3d13/(d13**2. - 1.) - da3d13a3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*da3d23/(d23**2. - 1.) - da3d23a3) ),

            # ad3 b3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db3d13/(d13**2. - 1.) - db3d13a3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*db3d23/(d23**2. - 1.) - db3d23a3) ),

            # ad3 g3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg3d13/(d13**2. - 1.) - dg3d13a3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(da3d23*0./1. + d23*da3d23*dg3d23/(d23**2. - 1.) - dg3d23a3) ),
            ],

            # ---------- #

            # bd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*da1d13/(d13**2. - 1.) - da1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*db1d13/(d13**2. - 1.) - db1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*dg1d13/(d13**2. - 1.) - dg1d13b3) ),

            # bd3 a2,b2,g2
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*da2d23/(d23**2. - 1.) - da2d23b3) ),
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*db2d23/(d23**2. - 1.) - db2d23b3) ),
            -spring_arr[2][0]/(mass_arr[2]*cosh(a3)*cosh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(cosh(a3)*cosh(a3)) + d23*db3d23*dg2d23/(d23**2. - 1.) - dg2d23b3) ),

            # bd3 a3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*sinh(2.*a3)/(sinh(a3)*sinh(a3)) + d13*db3d13*da3d13/(d13**2. - 1.) - da3d13b3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*sinh(2.*a3)/(sinh(a3)*sinh(a3)) + d23*db3d23*da3d23/(d23**2. - 1.) - da3d23b3) ),

            # bd3 b3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*db3d13/(d13**2. - 1.) - db3d13b3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(sinh(a3)*sinh(a3)) + d23*db3d23*db3d23/(d23**2. - 1.) - db3d23b3) ),

            # bd3 g3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*dg3d13/(d13**2. - 1.) - dg3d13b3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(db3d23*0./(sinh(a3)*sinh(a3)) + d23*db3d23*dg3d23/(d23**2. - 1.) - dg3d23b3) ),
            ],

            # ---------- #

            # gd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*da1d13/(d13**2. - 1.) - da1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*db1d13/(d13**2. - 1.) - db1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*dg1d13/(d13**2. - 1.) - dg1d13g3) ),

            # gd3 a2,b2,g2
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*da2d23/(d23**2. - 1.) - da2d23g3) ),
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*db2d23/(d23**2. - 1.) - db2d23g3) ),
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*dg2d23/(d23**2. - 1.) - dg2d23g3) ),

            # gd3 a3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sinh(2.*a3)*sin(b3)*sin(b3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*da3d13/(d13**2. - 1.) - da3d13g3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*sinh(2.*a3)*sin(b3)*sin(b3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*da3d23/(d23**2. - 1.) - da3d23g3) ),

            # gd3 b3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sin(2.*b3)*sinh(a3)*sinh(a3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*db3d13/(d13**2. - 1.) - db3d13g3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*sin(2.*b3)*sinh(a3)*sinh(a3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*db3d23/(d23**2. - 1.) - db3d23g3) ),

            # gd3 g3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*dg3d13/(d13**2. - 1.) - dg3d13g3) ) + 
            -spring_arr[2][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[2][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*dg3d23/(d23**2. - 1.) - dg3d23g3) ),
            ]

        ])
   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*sin(b1n)**2.)*sinh(a1n)*cosh(a1n) - 
            sp12[0]/m1*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*cosh(a2n) - cosh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - cosh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.) -
            sp13[0]/m1*( 
            arccosh(cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n)) - sp13[1])*
            (sinh(a1n)*cosh(a3n) - cosh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - cosh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))**2.)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*sin(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) - 
            sp12[0]/m1*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*cosh(a2n1) - cosh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - cosh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.) -
            sp13[0]/m1*( 
            arccosh(cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1)) - sp13[1])*
            (sinh(a1n1)*cosh(a3n1) - cosh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - cosh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))**2.)
            ))

    def con5(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sin(b1n)*cos(b1n) - 2.*ad1n*bd1n/tanh(a1n) - 
            sp12[0]/(m1*sinh(a1n)*sinh(a1n))*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*sin(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.) -
            sp13[0]/(m1*sinh(a1n)*sinh(a1n))*( 
            arccosh(cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n)) - sp13[1])*
            (sinh(a1n)*sin(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))**2.)
            + 
            gd1n1*gd1n1*sin(b1n1)*cos(b1n1) - 2.*ad1n1*bd1n1/tanh(a1n1) - 
            sp12[0]/(m1*sinh(a1n1)*sinh(a1n1))*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.) -
            sp13[0]/(m1*sinh(a1n1)*sinh(a1n1))*( 
            arccosh(cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1)) - sp13[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))**2.)
            ))

    def con6(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess

        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n/tanh(a1n) - 2.*bd1n*gd1n/tan(b1n) - 
            sp12[0]/(m1*sinh(a1n)*sinh(a1n)*sin(b1n)*sin(b1n))*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*sin(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.) -
            sp13[0]/(m1*sinh(a1n)*sinh(a1n)*sin(b1n)*sin(b1n))*( 
            arccosh(cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n)) - sp13[1])*
            (sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*sin(g1n - g3n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))**2.)
            + 
            -2.*ad1n1*gd1n1/tanh(a1n1) - 2.*bd1n1*gd1n1/tan(b1n1) - 
            sp12[0]/(m1*sinh(a1n1)*sinh(a1n1)*sin(b1n1)*sin(b1n1))*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*sin(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.) -
            sp13[0]/(m1*sinh(a1n1)*sinh(a1n1)*sin(b1n1)*sin(b1n1))*( 
            arccosh(cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1)) - sp13[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*sin(g1n1 - g3n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))**2.)
            )) 
    
    def jacobian(positions, velocities, mass_arr, h, spring_arr):
        a1n1,b1n1,g1n1=positions[0]
        a2n1,b2n1,g2n1=positions[1]
        a3n1,b3n1,g3n1=positions[2]
        ad1n1,bd1n1,gd1n1=velocities[0]
        ad2n1,bd2n1,gd2n1=velocities[1]
        ad3n1,bd3n1,gd3n1=velocities[2]
        spring_terms=jacobi_sp_terms(positions, mass_arr, spring_arr)
        #print(spring_terms)
        return array([
            [1.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1+sin(b1n1)*sin(b1n1)*gd1n1*gd1n1)*cosh(2.*a1n1) + spring_terms[0][0],
            -.25*h*sinh(2.*a1n1)*sin(2.*b1n1)*gd1n1*gd1n1 + spring_terms[0][1],
            0. + spring_terms[0][2], 
            
            spring_terms[0][3],
            spring_terms[0][4],
            spring_terms[0][5],

            spring_terms[0][6],
            spring_terms[0][7],
            spring_terms[0][8],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            -.5*h*sinh(2.*a1n1)*sin(b1n1)*sin(b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0.
            ],

            # bd1 update
            [-h*ad1n1*bd1n1/(sinh(a1n1)*sinh(a1n1)) + spring_terms[1][0],
            -.5*h*cos(2.*b1n1)*gd1n1*gd1n1 + spring_terms[1][1],
            0. + spring_terms[1][2], 
            
            spring_terms[1][3],
            spring_terms[1][4],
            spring_terms[1][5],

            spring_terms[1][6],
            spring_terms[1][7],
            spring_terms[1][8],
            
            h/tanh(a1n1)*bd1n1,
            1.+h/tanh(a1n1)*ad1n1,
            -.5*h*sin(2.*b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0.
            ],

            # gd1 update
            [-h*ad1n1*gd1n1/(sinh(a1n1)*sinh(a1n1)) + spring_terms[2][0],
            -h*bd1n1*gd1n1/(sin(b1n1)*sin(b1n1)) + spring_terms[2][1],
            0. + spring_terms[2][2],
             
            spring_terms[2][3],
            spring_terms[2][4],
            spring_terms[2][5],

            spring_terms[2][6],
            spring_terms[2][7],
            spring_terms[2][8],

            h/tanh(a1n1)*gd1n1,
            h/tan(b1n1)*gd1n1,
            1.+h/tanh(a1n1)*ad1n1+h/tan(b1n1)*bd1n1,

            0.,0.,0., 0.,0.,0.
            ],

            # ad2 update
            [spring_terms[3][0],
            spring_terms[3][1],
            spring_terms[3][2],

            -.5*h*(bd2n1*bd2n1+sin(b2n1)*sin(b2n1)*gd2n1*gd2n1)*cosh(2.*a2n1) + spring_terms[3][3],
            -.25*h*sinh(2.*a2n1)*sin(2.*b2n1)*gd2n1*gd2n1 + spring_terms[3][4],
            0. + spring_terms[3][5],

            spring_terms[3][6],
            spring_terms[3][7],
            spring_terms[3][8],

            0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            -.5*h*sinh(2.*a2n1)*sin(b2n1)*sin(b2n1)*gd2n1,

            0.,0.,0.
            ],

            # bd2 update
            [spring_terms[4][0],
            spring_terms[4][1],
            spring_terms[4][2],

            -h*ad2n1*bd2n1/(sinh(a2n1)*sinh(a2n1)) + spring_terms[4][3],
            -.5*h*cos(2.*b2n1)*gd2n1*gd2n1 + spring_terms[4][4],
            0. + spring_terms[4][5],

            spring_terms[4][6],
            spring_terms[4][7],
            spring_terms[4][8],

            0.,0.,0.,

            h/tanh(a2n1)*bd2n1,
            1.+h/tanh(a2n1)*ad2n1,
            -.5*h*sin(2.*b2n1)*gd2n1,
            
            0.,0.,0.
            ],

            # gd2 update
            [spring_terms[5][0],
            spring_terms[5][1],
            spring_terms[5][2],

            -h*ad2n1*gd2n1/(sinh(a2n1)*sinh(a2n1)) + spring_terms[5][3],
            -h*bd2n1*gd2n1/(sin(b2n1)*sin(b2n1)) + spring_terms[5][4],
            0. + spring_terms[5][5],

            spring_terms[5][6],
            spring_terms[5][7],
            spring_terms[5][8],

            0.,0.,0.,

            h/tanh(a2n1)*gd2n1,
            h/tan(b2n1)*gd2n1,
            1.+h/tanh(a2n1)*ad2n1+h/tan(b2n1)*bd2n1,
            
            0.,0.,0.
            ],

            # ad3 update
            [spring_terms[6][0],
            spring_terms[6][1],
            spring_terms[6][2],

            spring_terms[6][3],
            spring_terms[6][4],
            spring_terms[6][5],

            -.5*h*(bd3n1*bd3n1+sin(b3n1)*sin(b3n1)*gd3n1*gd3n1)*cosh(2.*a3n1) + spring_terms[6][6],
            -.25*h*sinh(2.*a3n1)*sin(2.*b3n1)*gd3n1*gd3n1 + spring_terms[6][7],
            0. + spring_terms[6][8],

            0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a3n1)*bd3n1,
            -.5*h*sinh(2.*a3n1)*sin(b3n1)*sin(b3n1)*gd3n1
            ],

            # bd3 update
            [spring_terms[7][0],
            spring_terms[7][1],
            spring_terms[7][2],

            spring_terms[7][3],
            spring_terms[7][4],
            spring_terms[7][5],

            -h*ad3n1*bd3n1/(sinh(a3n1)*sinh(a3n1)) + spring_terms[7][6],
            -.5*h*cos(2.*b3n1)*gd3n1*gd3n1 + spring_terms[7][7],
            0. + spring_terms[7][8],

            0.,0.,0., 0.,0.,0.,

            h/tanh(a3n1)*bd3n1,
            1.+h/tanh(a3n1)*ad3n1,
            -.5*h*sin(2.*b3n1)*gd3n1
            ],

            # gd3 update
            [spring_terms[8][0],
            spring_terms[8][1],
            spring_terms[8][2],

            spring_terms[8][3],
            spring_terms[8][4],
            spring_terms[8][5],

            -h*ad3n1*gd3n1/(sinh(a3n1)*sinh(a3n1)) + spring_terms[8][6],
            -h*bd3n1*gd3n1/(sin(b3n1)*sin(b3n1)) + spring_terms[8][7],
            0. + spring_terms[8][8],

            0.,0.,0., 0.,0.,0.,

            h/tanh(a3n1)*gd3n1,
            h/tan(b3n1)*gd3n1,
            1.+h/tanh(a3n1)*ad3n1+h/tan(b3n1)*bd3n1
            ]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr),-array([
        #p1
        con1(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con2(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con3(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        #p2
        con1(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con2(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con3(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        #p3
        con1(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con2(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con3(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),

        #v1
        con4(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
        con5(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
        con6(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
        #v2
        con4(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
        con5(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
        con6(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
        #v3
        con4(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
        con5(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
        con6(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2])      
    ]))
    val1 = array([
            posn_arr[0][0]+diff1[0], posn_arr[0][1]+diff1[1], posn_arr[0][2]+diff1[2], 
            posn_arr[1][0]+diff1[3], posn_arr[1][1]+diff1[4], posn_arr[1][2]+diff1[5],
            posn_arr[2][0]+diff1[6], posn_arr[2][1]+diff1[7], posn_arr[2][2]+diff1[8],

            veln_arr[0][0]+diff1[9], veln_arr[0][1]+diff1[10], veln_arr[0][2]+diff1[11], 
            veln_arr[1][0]+diff1[12], veln_arr[1][1]+diff1[13], veln_arr[1][2]+diff1[14],
            veln_arr[2][0]+diff1[15], veln_arr[2][1]+diff1[16], veln_arr[2][2]+diff1[17]
            ])    
    x = 0
    while(x < 7):
        new_pos_arr=array([val1[0:3],val1[3:6],val1[6:9]])
        new_vel_arr=array([val1[9:12],val1[12:15],val1[15:18]])
        diff2=linalg.solve(jacobian(new_pos_arr, new_vel_arr, mass_arr, step, spring_arr),-array([
            #p1
            con1(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con2(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con3(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            #p2
            con1(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con2(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con3(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            #p3
            con1(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con2(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con3(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),

            #v1
            con4(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
            con5(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
            con6(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1]),
            #v2
            con4(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
            con5(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
            con6(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[2]),
            #v3
            con4(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
            con5(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2]),
            con6(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[2])    
        ]))      
        val2 = array([
            val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], 
            val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5],
            val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8],

            val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11], 
            val1[12]+diff2[12], val1[13]+diff2[13], val1[14]+diff2[14],
            val1[15]+diff2[15], val1[16]+diff2[16], val1[17]+diff2[17]
            ])
        val1 = val2
        x=x+1
    #print(val1[9:17])
    return val1

#########################################################
### Hyperbolic 3-space Spring Potential (Tetrahedron) ###
#########################################################

# This is an extension of the dumb-bell system using the structure for the cube
# solver. The configuration is a square of four masses and four springs. 
# I found that for the case of shapes with odd number rotational symmetry
# that the translational parameterization is not appropriate due to the 
# metric factors being unequal for point with degenerate mirror symmetry 
# compared to the remaining points that have a mirror point. Since it is a 
# coordinate issue, we need to use the rotational coordinate system to remove
# this artifical directional bias. Use rotational parameterization as default.

# There seems to be a bug. Explode when you perturb a single mass need to check
# the spring terms I think.

def imph3sprot4(posn_arr, veln_arr, step, mass_arr, spring_arr):

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetric
    # of the particle coordinates between particle 1 and 2, I have verified that these are the only first
    # order derivative functions needed. This is because the expressions for the coordinates of particle 2 use the same functions
    # with the arguments flipped. Thus only three functions are needed instead of six.
    
    # For the remaining three functions use:
    # da2D12 = da1D12(a2, b2, g2, a1, b1, g1)
    # db2D12 = db1D12(a2, b2, g2, a1, b1, g1)
    # dg2D12 = dg1D12(a2, b2, g2, a1, b1, g1)
    def da1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2) 

    def dg1D12(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)      
    

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
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def db1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*cos(b2) - cosh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def da2D12a1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sinh(a2) - cosh(a1)*cos(b1)*cosh(a2)*cos(b2) - cosh(a1)*sin(b1)*cosh(a2)*sin(b2)*cos(g1 - g2)

    def db2D12a1(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cos(b1)*sinh(a2)*sin(b2) - cosh(a1)*sin(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12a1(a1, b1, g1, a2, b2, g2):
        return -cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*cos(b2) + sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg1D12b1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def db2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2)*cos(g1 - g2)

    def dg2D12b1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*sin(g1 - g2)

    def dg1D12g1(a1, b1, g1, a2, b2, g2):
        return sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    def dg2D12g1(a1, b1, g1, a2, b2, g2):
        return -sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # This function is to simplify the following function that generates the square matrix of spring potential terms (maybe?)

    # Now that the necessary functions have been defined they can now be used to generate the spring potential terms
    # found the in jacobian matrix used to solve the system of equations to determine the position and velocity at the
    # next point in the trajectory of each particle. This function construct a square matrix of values that will be 
    # included in the bottom left block of the complete jacobian.

    def jacobi_sp_terms(positions, mass_arr, spring_arr):
        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        a3,b3,g3=positions[2]
        a4,b4,g4=positions[3]
        
        
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


        ### Spring 1-4 spring_arr[2]###
        # D function
        d14=D12(a1, b1, g1, a4, b4, g4)
        # First derivatives of D function
        da1d14=da1D12(a1, b1, g1, a4, b4, g4)
        db1d14=db1D12(a1, b1, g1, a4, b4, g4)
        dg1d14=dg1D12(a1, b1, g1, a4, b4, g4)
        da4d14=da1D12(a4, b4, g4, a1, b1, g1)
        db4d14=db1D12(a4, b4, g4, a1, b1, g1)
        dg4d14=dg1D12(a4, b4, g4, a1, b1, g1)
        # Second derivatives of D function
        da1d14a1=da1D12a1(a1, b1, g1, a4, b4, g4)
        db1d14a1=db1D12a1(a1, b1, g1, a4, b4, g4)
        dg1d14a1=dg1D12a1(a1, b1, g1, a4, b4, g4)
        da4d14a1=da2D12a1(a1, b1, g1, a4, b4, g4)
        db4d14a1=db2D12a1(a1, b1, g1, a4, b4, g4)
        dg4d14a1=dg2D12a1(a1, b1, g1, a4, b4, g4)
        
        da1d14b1=db1d14a1
        db1d14b1=db1D12b1(a1, b1, g1, a4, b4, g4)
        dg1d14b1=dg1D12b1(a1, b1, g1, a4, b4, g4)
        da4d14b1 = db2D12a1(a4, b4, g4, a1, b1, g1)
        db4d14b1=db2D12b1(a1, b1, g1, a4, b4, g4)
        dg4d14b1=dg2D12b1(a1, b1, g1, a4, b4, g4)

        da1d14g1=dg1d14a1
        db1d14g1=dg1d14b1
        dg1d14g1=dg1D12g1(a1, b1, g1, a4, b4, g4)
        da4d14g1 = dg2D12a1(a4, b4, g4, a1, b1, g1)
        db4d14g1 = dg2D12b1(a4, b4, g4, a1, b1, g1)
        dg4d14g1=dg2D12g1(a1, b1, g1, a4, b4, g4)

        da1d14a4=da4d14a1
        db1d14a4=da4d14b1
        dg1d14a4=da4d14g1
        da4d14a4 = da1D12a1(a4, b4, g4, a1, b1, g1)
        db4d14a4 = db1D12a1(a4, b4, g4, a1, b1, g1)
        dg4d14a4 = dg1D12a1(a4, b4, g4, a1, b1, g1)

        da1d14b4=db4d14a1
        db1d14b4=db4d14b1
        dg1d14b4=db4d14g1
        da4d14b4=db4d14a4
        db4d14b4 = db1D12b1(a4, b4, g4, a1, b1, g1)
        dg4d14b4 = dg1D12b1(a4, b4, g4, a1, b1, g1)

        da1d14g4=dg4d14a1
        db1d14g4=dg4d14b1
        dg1d14g4=dg4d14g1
        da4d14g4=dg4d14a4
        db4d14g4=dg4d14b4
        dg4d14g4 = dg1D12g1(a4, b4, g4, a1, b1, g1)


        ### Spring 2-3 spring_arr[3]###
        # D function
        d23=D12(a2, b2, g2, a3, b3, g3)
        # First derivatives of D function
        da2d23=da1D12(a2, b2, g2, a3, b3, g3)
        db2d23=db1D12(a2, b2, g2, a3, b3, g3)
        dg2d23=dg1D12(a2, b2, g2, a3, b3, g3)
        da3d23=da1D12(a3, b3, g3, a2, b2, g2)
        db3d23=db1D12(a3, b3, g3, a2, b2, g2)
        dg3d23=dg1D12(a3, b3, g3, a2, b2, g2)
        # Second derivatives of D function
        da2d23a2=da1D12a1(a2, b2, g2, a3, b3, g3)
        db2d23a2=db1D12a1(a2, b2, g2, a3, b3, g3)
        dg2d23a2=dg1D12a1(a2, b2, g2, a3, b3, g3)
        da3d23a2=da2D12a1(a2, b2, g2, a3, b3, g3)
        db3d23a2=db2D12a1(a2, b2, g2, a3, b3, g3)
        dg3d23a2=dg2D12a1(a2, b2, g2, a3, b3, g3)
        
        da2d23b2=db2d23a2
        db2d23b2=db1D12b1(a2, b2, g2, a3, b3, g3)
        dg2d23b2=dg1D12b1(a2, b2, g2, a3, b3, g3)
        da3d23b2 = db2D12a1(a3, b3, g3, a2, b2, g2)
        db3d23b2=db2D12b1(a2, b2, g2, a3, b3, g3)
        dg3d23b2=dg2D12b1(a2, b2, g2, a3, b3, g3)

        da2d23g2=dg2d23a2
        db2d23g2=dg2d23b2
        dg2d23g2=dg1D12g1(a2, b2, g2, a3, b3, g3)
        da3d23g2 = dg2D12a1(a3, b3, g3, a2, b2, g2)
        db3d23g2 = dg2D12b1(a3, b3, g3, a2, b2, g2)
        dg3d23g2=dg2D12g1(a2, b2, g2, a3, b3, g3)

        da2d23a3=da3d23a2
        db2d23a3=da3d23b2
        dg2d23a3=da3d23g2
        da3d23a3 = da1D12a1(a3, b3, g3, a2, b2, g2)
        db3d23a3 = db1D12a1(a3, b3, g3, a2, b2, g2)
        dg3d23a3 = dg1D12a1(a3, b3, g3, a2, b2, g2)

        da2d23b3=db3d23a2
        db2d23b3=db3d23b2
        dg2d23b3=db3d23g2
        da3d23b3=db3d23a3
        db3d23b3 = db1D12b1(a3, b3, g3, a2, b2, g2)
        dg3d23b3 = dg1D12b1(a3, b3, g3, a2, b2, g2)

        da2d23g3=dg3d23a2
        db2d23g3=dg3d23b2
        dg2d23g3=dg3d23g2
        da3d23g3=dg3d23a3
        db3d23g3=dg3d23b3
        dg3d23g3 = dg1D12g1(a3, b3, g3, a2, b2, g2)

        ### Spring 2-4 spring_arr[4]###
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
        

        return array([

            # ---------- #
            #     V1     #
            # ---------- #
            
            # ad1 a1
            [-spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da1d12/(d12**2. - 1.) - da1d12a1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da1d13/(d13**2. - 1.) - da1d13a1) ) + 
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d14**2. - 1. ))*( (da1d14*da1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da1d14*0./1. + d14*da1d14*da1d14/(d14**2. - 1.) - da1d14a1) ),

            # ad1 b1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db1d12/(d12**2. - 1.) - db1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db1d13/(d13**2. - 1.) - db1d13a1) ) +
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d14**2. - 1. ))*( (da1d14*db1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da1d14*0./1. + d14*da1d14*db1d14/(d14**2. - 1.) - db1d14a1) ),

            # ad1 g1
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg1d12/(d12**2. - 1.) - dg1d12a1) ) +
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg1d13/(d13**2. - 1.) - dg1d13a1) ) +
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d14**2. - 1. ))*( (da1d14*dg1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da1d14*0./1. + d14*da1d14*dg1d14/(d14**2. - 1.) - dg1d14a1) ),

            # ad1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*da2d12/(d12**2. - 1.) - da2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*db2d12/(d12**2. - 1.) - db2d12a1) ),
            -spring_arr[0][0]/(mass_arr[0]*1.*sqrt( d12**2. - 1. ))*( (da1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da1d12*0./1. + d12*da1d12*dg2d12/(d12**2. - 1.) - dg2d12a1) ),

            # ad1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*da3d13/(d13**2. - 1.) - da3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*db3d13/(d13**2. - 1.) - db3d13a1) ),
            -spring_arr[1][0]/(mass_arr[0]*1.*sqrt( d13**2. - 1. ))*( (da1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da1d13*0./1. + d13*da1d13*dg3d13/(d13**2. - 1.) - dg3d13a1) ),

            # ad1 a4,b4,g4
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d14**2. - 1. ))*( (da1d14*da4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da1d14*0./1. + d14*da1d14*da4d14/(d14**2. - 1.) - da4d14a1) ),
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d14**2. - 1. ))*( (da1d14*db4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da1d14*0./1. + d14*da1d14*db4d14/(d14**2. - 1.) - db4d14a1) ),
            -spring_arr[2][0]/(mass_arr[0]*1.*sqrt( d14**2. - 1. ))*( (da1d14*dg4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da1d14*0./1. + d14*da1d14*dg4d14/(d14**2. - 1.) - dg4d14a1) )
            ],

            # ---------- #

            # bd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*sinh(2.*a1)/(sinh(a1)*sinh(a1)) + d12*db1d12*da1d12/(d12**2. - 1.) - da1d12b1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*sinh(2.*a1)/(sinh(a1)*sinh(a1)) + d13*db1d13*da1d13/(d13**2. - 1.) - da1d13b1) ) + 
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d14**2. - 1. ))*( (db1d14*da1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db1d14*sinh(2.*a1)/(sinh(a1)*sinh(a1)) + d14*db1d14*da1d14/(d14**2. - 1.) - da1d14b1) ),

            # bd1 b1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*db1d12/(d12**2. - 1.) - db1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*db1d13/(d13**2. - 1.) - db1d13b1) ) +
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d14**2. - 1. ))*( (db1d14*db1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db1d14*0./(sinh(a1)*sinh(a1)) + d14*db1d14*db1d14/(d14**2. - 1.) - db1d14b1) ),

            # bd1 g1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*dg1d12/(d12**2. - 1.) - dg1d12b1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*dg1d13/(d13**2. - 1.) - dg1d13b1) ) +
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d14**2. - 1. ))*( (db1d14*dg1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db1d14*0./(sinh(a1)*sinh(a1)) + d14*db1d14*dg1d14/(d14**2. - 1.) - dg1d14b1) ),

            # bd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*da2d12/(d12**2. - 1.) - da2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*db2d12/(d12**2. - 1.) - db2d12b1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d12**2. - 1. ))*( (db1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db1d12*0./(sinh(a1)*sinh(a1)) + d12*db1d12*dg2d12/(d12**2. - 1.) - dg2d12b1) ),

            # bd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*da3d13/(d13**2. - 1.) - da3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*db3d13/(d13**2. - 1.) - db3d13b1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d13**2. - 1. ))*( (db1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db1d13*0./(sinh(a1)*sinh(a1)) + d13*db1d13*dg3d13/(d13**2. - 1.) - dg3d13b1) ),

            # bd1 a4,b4,g4
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d14**2. - 1. ))*( (db1d14*da4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db1d14*0./(sinh(a1)*sinh(a1)) + d14*db1d14*da4d14/(d14**2. - 1.) - da4d14b1) ),
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d14**2. - 1. ))*( (db1d14*db4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db1d14*0./(sinh(a1)*sinh(a1)) + d14*db1d14*db4d14/(d14**2. - 1.) - db4d14b1) ),
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sqrt( d14**2. - 1. ))*( (db1d14*dg4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db1d14*0./(sinh(a1)*sinh(a1)) + d14*db1d14*dg4d14/(d14**2. - 1.) - dg4d14b1) )
            ],

            # ---------- #

            # gd1 a1
            [-spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sinh(2.*a1)*sin(b1)*sin(b1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*da1d12/(d12**2. - 1.) - da1d12g1) ) + 
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sinh(2.*a1)*sin(b1)*sin(b1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*da1d13/(d13**2. - 1.) - da1d13g1) ) + 
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d14**2. - 1. ))*( (dg1d14*da1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg1d14*sinh(2.*a1)*sin(b1)*sin(b1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d14*dg1d14*da1d14/(d14**2. - 1.) - da1d14g1) ),

            # gd1 b1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*sin(2.*b1)*sinh(a1)*sinh(a1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*db1d12/(d12**2. - 1.) - db1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*sin(2.*b1)*sinh(a1)*sinh(a1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*db1d13/(d13**2. - 1.) - db1d13g1) ) +
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d14**2. - 1. ))*( (dg1d14*db1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg1d14*sin(2.*b1)*sinh(a1)*sinh(a1)/(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d14*dg1d14*db1d14/(d14**2. - 1.) - db1d14g1) ),

            # gd1 g1
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*dg1d12/(d12**2. - 1.) - dg1d12g1) ) +
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*dg1d13/(d13**2. - 1.) - dg1d13g1) ) +
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d14**2. - 1. ))*( (dg1d14*dg1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg1d14*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d14*dg1d14*dg1d14/(d14**2. - 1.) - dg1d14g1) ),

            # gd1 a2,b2,g2
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*da2d12/(d12**2. - 1.) - da2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*db2d12/(d12**2. - 1.) - db2d12g1) ),
            -spring_arr[0][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d12**2. - 1. ))*( (dg1d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg1d12*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d12*dg1d12*dg2d12/(d12**2. - 1.) - dg2d12g1) ),

            # gd1 a3,b3,g3
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*da3d13/(d13**2. - 1.) - da3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*db3d13/(d13**2. - 1.) - db3d13g1) ),
            -spring_arr[1][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d13**2. - 1. ))*( (dg1d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg1d13*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d13*dg1d13*dg3d13/(d13**2. - 1.) - dg3d13g1) ),

            # gd1 a4,b4,g4
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d14**2. - 1. ))*( (dg1d14*da4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg1d14*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d14*dg1d14*da4d14/(d14**2. - 1.) - da4d14g1) ),
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d14**2. - 1. ))*( (dg1d14*db4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg1d14*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d14*dg1d14*db4d14/(d14**2. - 1.) - db4d14g1) ),
            -spring_arr[2][0]/(mass_arr[0]*sinh(a1)*sinh(a1)*sin(b1)*sin(b1)*sqrt( d14**2. - 1. ))*( (dg1d14*dg4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg1d14*0./(sinh(a1)*sinh(a1)*sin(b1)*sin(b1)) + d14*dg1d14*dg4d14/(d14**2. - 1.) - dg4d14g1) )
            ],

            # ---------- #
            #     V2     #
            # ---------- #

            # ad2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da1d12/(d12**2. - 1.) - da1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db1d12/(d12**2. - 1.) - db1d12a2) ),
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg1d12/(d12**2. - 1.) - dg1d12a2) ),

            # ad2 a2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*da2d12/(d12**2. - 1.) - da2d12a2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da2d23*0./1. + d23*da2d23*da2d23/(d23**2. - 1.) - da2d23a2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da2d24*0./1. + d24*da2d24*da2d24/(d24**2. - 1.) - da2d24a2) ),

            # ad2 b2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*db2d12/(d12**2. - 1.) - db2d12a2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da2d23*0./1. + d23*da2d23*db2d23/(d23**2. - 1.) - db2d23a2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da2d24*0./1. + d24*da2d24*db2d24/(d24**2. - 1.) - db2d24a2) ),

            # ad2 g2
            -spring_arr[0][0]/(mass_arr[1]*1.*sqrt( d12**2. - 1. ))*( (da2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(da2d12*0./1. + d12*da2d12*dg2d12/(d12**2. - 1.) - dg2d12a2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da2d23*0./1. + d23*da2d23*dg2d23/(d23**2. - 1.) - dg2d23a2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da2d24*0./1. + d24*da2d24*dg2d24/(d24**2. - 1.) - dg2d24a2) ),

            # ad2 a3,b3,g3
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da2d23*0./1. + d23*da2d23*da3d23/(d23**2. - 1.) - da3d23a2) ),
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da2d23*0./1. + d23*da2d23*db3d23/(d23**2. - 1.) - db3d23a2) ),
            -spring_arr[3][0]/(mass_arr[1]*1.*sqrt( d23**2. - 1. ))*( (da2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da2d23*0./1. + d23*da2d23*dg3d23/(d23**2. - 1.) - dg3d23a2) ),

            # ad2 a4,b4,g4
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da2d24*0./1. + d24*da2d24*da4d24/(d24**2. - 1.) - da4d24a2) ),
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da2d24*0./1. + d24*da2d24*db4d24/(d24**2. - 1.) - db4d24a2) ),
            -spring_arr[4][0]/(mass_arr[1]*1.*sqrt( d24**2. - 1. ))*( (da2d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da2d24*0./1. + d24*da2d24*dg4d24/(d24**2. - 1.) - dg4d24a2) )
            ],

            # ---------- #

            # bd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*da1d12/(d12**2. - 1.) - da1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*db1d12/(d12**2. - 1.) - db1d12b2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*dg1d12/(d12**2. - 1.) - dg1d12b2) ),

            # bd2 a2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*sinh(2.*a2)/(sinh(a2)*sinh(a2)) + d12*db2d12*da2d12/(d12**2. - 1.) - da2d12b2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db2d23*sinh(2.*a2)/(sinh(a2)*sinh(a2)) + d23*db2d23*da2d23/(d23**2. - 1.) - da2d23b2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db2d24*sinh(2.*a2)/(sinh(a2)*sinh(a2)) + d24*db2d24*da2d24/(d24**2. - 1.) - da2d24b2) ),

            # bd2 b2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*db2d12/(d12**2. - 1.) - db2d12b2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*db2d23/(d23**2. - 1.) - db2d23b2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db2d24*0./(sinh(a2)*sinh(a2)) + d24*db2d24*db2d24/(d24**2. - 1.) - db2d24b2) ),

            # bd2 g2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d12**2. - 1. ))*( (db2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(db2d12*0./(sinh(a2)*sinh(a2)) + d12*db2d12*dg2d12/(d12**2. - 1.) - dg2d12b2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*dg2d23/(d23**2. - 1.) - dg2d23b2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db2d24*0./(sinh(a2)*sinh(a2)) + d24*db2d24*dg2d24/(d24**2. - 1.) - dg2d24b2) ),

            # bd2 a3,b3,g3
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*da3d23/(d23**2. - 1.) - da3d23b2) ),
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*db3d23/(d23**2. - 1.) - db3d23b2) ),
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d23**2. - 1. ))*( (db2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db2d23*0./(sinh(a2)*sinh(a2)) + d23*db2d23*dg3d23/(d23**2. - 1.) - dg3d23b2) ),

            # bd2 a4,b4,g4
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db2d24*0./(sinh(a2)*sinh(a2)) + d24*db2d24*da4d24/(d24**2. - 1.) - da4d24b2) ),
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db2d24*0./(sinh(a2)*sinh(a2)) + d24*db2d24*db4d24/(d24**2. - 1.) - db4d24b2) ),
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sqrt( d24**2. - 1. ))*( (db2d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db2d24*0./(sinh(a2)*sinh(a2)) + d24*db2d24*dg4d24/(d24**2. - 1.) - dg4d24b2) )
            ],

            # ---------- #

            # gd2 a1,b1,g1
            [-spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*da1d12/(d12**2. - 1.) - da1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*db1d12/(d12**2. - 1.) - db1d12g2) ),
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg1d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*dg1d12/(d12**2. - 1.) - dg1d12g2) ),

            # gd2 a2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*da2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sinh(2.*a2)*sin(b2)*sin(b2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*da2d12/(d12**2. - 1.) - da2d12g2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg2d23*sinh(2.*a2)*sin(b2)*sin(b2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*da2d23/(d23**2. - 1.) - da2d23g2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg2d24*sinh(2.*a2)*sin(b2)*sin(b2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d24*dg2d24*da2d24/(d24**2. - 1.) - da2d24g2) ),

            # gd2 b2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*db2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*sin(2.*b2)*sinh(a2)*sinh(a2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*db2d12/(d12**2. - 1.) - db2d12g2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg2d23*sin(2.*b2)*sinh(a2)*sinh(a2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*db2d23/(d23**2. - 1.) - db2d23g2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg2d24*sin(2.*b2)*sinh(a2)*sinh(a2)/(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d24*dg2d24*db2d24/(d24**2. - 1.) - db2d24g2) ),

            # gd2 g2
            -spring_arr[0][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d12**2. - 1. ))*( (dg2d12*dg2d12)/sqrt( d12**2. - 1.) - ( arccosh(d12) - spring_arr[0][1] )*(dg2d12*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d12*dg2d12*dg2d12/(d12**2. - 1.) - dg2d12g2) ) + 
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*dg2d23/(d23**2. - 1.) - dg2d23g2) ) + 
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg2d24*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d24*dg2d24*dg2d24/(d24**2. - 1.) - dg2d24g2) ),

            # gd2 a3,b3,g3
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*da3d23/(d23**2. - 1.) - da3d23g2) ),
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*db3d23/(d23**2. - 1.) - db3d23g2) ),
            -spring_arr[3][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d23**2. - 1. ))*( (dg2d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg2d23*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d23*dg2d23*dg3d23/(d23**2. - 1.) - dg3d23g2) ),

            # gd2 a4,b4,g4
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg2d24*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d24*dg2d24*da4d24/(d24**2. - 1.) - da4d24g2) ),
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg2d24*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d24*dg2d24*db4d24/(d24**2. - 1.) - db4d24g2) ),
            -spring_arr[4][0]/(mass_arr[1]*sinh(a2)*sinh(a2)*sin(b2)*sin(b2)*sqrt( d24**2. - 1. ))*( (dg2d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg2d24*0./(sinh(a2)*sinh(a2)*sin(b2)*sin(b2)) + d24*dg2d24*dg4d24/(d24**2. - 1.) - dg4d24g2) )
            ],

            # ---------- #
            #     V3     # 
            # ---------- #

            # ad3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da1d13/(d13**2. - 1.) - da1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db1d13/(d13**2. - 1.) - db1d13a3) ),
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg1d13/(d13**2. - 1.) - dg1d13a3) ),

            # ad3 a2,b2,g2
            -spring_arr[3][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da3d23*0./1. + d23*da3d23*da2d23/(d23**2. - 1.) - da2d23a3) ),
            -spring_arr[3][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da3d23*0./1. + d23*da3d23*db2d23/(d23**2. - 1.) - db2d23a3) ),
            -spring_arr[3][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da3d23*0./1. + d23*da3d23*dg2d23/(d23**2. - 1.) - dg2d23a3) ),

            # ad3 a3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*da3d13/(d13**2. - 1.) - da3d13a3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da3d23*0./1. + d23*da3d23*da3d23/(d23**2. - 1.) - da3d23a3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*da3d34/(d34**2. - 1.) - da3d34a3) ),

            # ad3 b3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*db3d13/(d13**2. - 1.) - db3d13a3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da3d23*0./1. + d23*da3d23*db3d23/(d23**2. - 1.) - db3d23a3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*db3d34/(d34**2. - 1.) - db3d34a3) ),

            # ad3 g3
            -spring_arr[1][0]/(mass_arr[2]*1.*sqrt( d13**2. - 1. ))*( (da3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(da3d13*0./1. + d13*da3d13*dg3d13/(d13**2. - 1.) - dg3d13a3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*1.*sqrt( d23**2. - 1. ))*( (da3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(da3d23*0./1. + d23*da3d23*dg3d23/(d23**2. - 1.) - dg3d23a3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*dg3d34/(d34**2. - 1.) - dg3d34a3) ),

            # ad3 a4,b4,g4
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*da4d34/(d34**2. - 1.) - da4d34a3) ),
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*db4d34/(d34**2. - 1.) - db4d34a3) ),
            -spring_arr[5][0]/(mass_arr[2]*1.*sqrt( d34**2. - 1. ))*( (da3d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da3d34*0./1. + d34*da3d34*dg4d34/(d34**2. - 1.) - dg4d34a3) )
            ],

            # ---------- #

            # bd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*da1d13/(d13**2. - 1.) - da1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*db1d13/(d13**2. - 1.) - db1d13b3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*dg1d13/(d13**2. - 1.) - dg1d13b3) ),

            # bd3 a2,b2,g2
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db3d23*0./(sinh(a3)*sinh(a3)) + d23*db3d23*da2d23/(d23**2. - 1.) - da2d23b3) ),
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db3d23*0./(sinh(a3)*sinh(a3)) + d23*db3d23*db2d23/(d23**2. - 1.) - db2d23b3) ),
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db3d23*0./(sinh(a3)*sinh(a3)) + d23*db3d23*dg2d23/(d23**2. - 1.) - dg2d23b3) ),

            # bd3 a3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*sinh(2.*a3)/(sinh(a3)*sinh(a3)) + d13*db3d13*da3d13/(d13**2. - 1.) - da3d13b3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db3d23*sinh(2.*a3)/(sinh(a3)*sinh(a3)) + d23*db3d23*da3d23/(d23**2. - 1.) - da3d23b3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*sinh(2.*a3)/(sinh(a3)*sinh(a3)) + d34*db3d34*da3d34/(d34**2. - 1.) - da3d34b3) ),

            # bd3 b3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*db3d13/(d13**2. - 1.) - db3d13b3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db3d23*0./(sinh(a3)*sinh(a3)) + d23*db3d23*db3d23/(d23**2. - 1.) - db3d23b3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(sinh(a3)*sinh(a3)) + d34*db3d34*db3d34/(d34**2. - 1.) - db3d34b3) ),

            # bd3 g3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d13**2. - 1. ))*( (db3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(db3d13*0./(sinh(a3)*sinh(a3)) + d13*db3d13*dg3d13/(d13**2. - 1.) - dg3d13b3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d23**2. - 1. ))*( (db3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(db3d23*0./(sinh(a3)*sinh(a3)) + d23*db3d23*dg3d23/(d23**2. - 1.) - dg3d23b3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(sinh(a3)*sinh(a3)) + d34*db3d34*dg3d34/(d34**2. - 1.) - dg3d34b3) ),

            # bd3 a4,b4,g4
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(sinh(a3)*sinh(a3)) + d34*db3d34*da4d34/(d34**2. - 1.) - da4d34b3) ),
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(sinh(a3)*sinh(a3)) + d34*db3d34*db4d34/(d34**2. - 1.) - db4d34b3) ),
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sqrt( d34**2. - 1. ))*( (db3d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db3d34*0./(sinh(a3)*sinh(a3)) + d34*db3d34*dg4d34/(d34**2. - 1.) - dg4d34b3) )
            ],

            # ---------- #

            # gd3 a1,b1,g1
            [-spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*da1d13/(d13**2. - 1.) - da1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*db1d13/(d13**2. - 1.) - db1d13g3) ),
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg1d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*dg1d13/(d13**2. - 1.) - dg1d13g3) ),

            # gd3 a2,b2,g2
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*da2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*da2d23/(d23**2. - 1.) - da2d23g3) ),
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*db2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*db2d23/(d23**2. - 1.) - db2d23g3) ),
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*dg2d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*dg2d23/(d23**2. - 1.) - dg2d23g3) ),

            # gd3 a3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*da3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sinh(2.*a3)*sin(b3)*sin(b3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*da3d13/(d13**2. - 1.) - da3d13g3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*da3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg3d23*sinh(2.*a3)*sin(b3)*sin(b3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*da3d23/(d23**2. - 1.) - da3d23g3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*sinh(2.*a3)*sin(b3)*sin(b3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d34*dg3d34*da3d34/(d34**2. - 1.) - da3d34g3) ),

            # gd3 b3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*db3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*sin(2.*b3)*sinh(a3)*sinh(a3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*db3d13/(d13**2. - 1.) - db3d13g3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*db3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg3d23*sin(2.*b3)*sinh(a3)*sinh(a3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*db3d23/(d23**2. - 1.) - db3d23g3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*sin(2.*b3)*sinh(a3)*sinh(a3)/(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d34*dg3d34*db3d34/(d34**2. - 1.) - db3d34g3) ),

            # gd3 g3
            -spring_arr[1][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d13**2. - 1. ))*( (dg3d13*dg3d13)/sqrt( d13**2. - 1.) - ( arccosh(d13) - spring_arr[1][1] )*(dg3d13*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d13*dg3d13*dg3d13/(d13**2. - 1.) - dg3d13g3) ) + 
            -spring_arr[3][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d23**2. - 1. ))*( (dg3d23*dg3d23)/sqrt( d23**2. - 1.) - ( arccosh(d23) - spring_arr[3][1] )*(dg3d23*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d23*dg3d23*dg3d23/(d23**2. - 1.) - dg3d23g3) ) + 
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d34*dg3d34*dg3d34/(d34**2. - 1.) - dg3d34g3) ),

            # gd3 a4,b4,g4
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d34*dg3d34*da4d34/(d34**2. - 1.) - da4d34g3) ),
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d34*dg3d34*db4d34/(d34**2. - 1.) - db4d34g3) ),
            -spring_arr[5][0]/(mass_arr[2]*sinh(a3)*sinh(a3)*sin(b3)*sin(b3)*sqrt( d34**2. - 1. ))*( (dg3d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg3d34*0./(sinh(a3)*sinh(a3)*sin(b3)*sin(b3)) + d34*dg3d34*dg4d34/(d34**2. - 1.) - dg4d34g3) )
            ],

            # ---------- #
            #     V4     #
            # ---------- #

            # ad4 a1,b1,g1
            [-spring_arr[2][0]/(mass_arr[3]*1.*sqrt( d14**2. - 1. ))*( (da4d14*da1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da4d14*0./1. + d14*da4d14*da1d14/(d14**2. - 1.) - da1d14a4) ),
            -spring_arr[2][0]/(mass_arr[3]*1.*sqrt( d14**2. - 1. ))*( (da4d14*db1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da4d14*0./1. + d14*da4d14*db1d14/(d14**2. - 1.) - db1d14a4) ),
            -spring_arr[2][0]/(mass_arr[3]*1.*sqrt( d14**2. - 1. ))*( (da4d14*dg1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da4d14*0./1. + d14*da4d14*dg1d14/(d14**2. - 1.) - dg1d14a4) ),

            # ad4 a2,b2,g2
            -spring_arr[4][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da4d24*0./1. + d24*da4d24*da2d24/(d24**2. - 1.) - da2d24a4) ),
            -spring_arr[4][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da4d24*0./1. + d24*da4d24*db2d24/(d24**2. - 1.) - db2d24a4) ),
            -spring_arr[4][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da4d24*0./1. + d24*da4d24*dg2d24/(d24**2. - 1.) - dg2d24a4) ),

            # ad4 a3,b3,g3
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*da3d34/(d34**2. - 1.) - da3d34a4) ),
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*db3d34/(d34**2. - 1.) - db3d34a4) ),
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*dg3d34/(d34**2. - 1.) - dg3d34a4) ),

            # ad4 a4
            -spring_arr[2][0]/(mass_arr[3]*1.*sqrt( d14**2. - 1. ))*( (da4d14*da4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da4d14*0./1. + d14*da4d14*da4d14/(d14**2. - 1.) - da4d14a4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da4d24*0./1. + d24*da4d24*da4d24/(d24**2. - 1.) - da4d24a4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*da4d34/(d34**2. - 1.) - da4d34a4) ),

            # ad4 b4
            -spring_arr[2][0]/(mass_arr[3]*1.*sqrt( d14**2. - 1. ))*( (da4d14*db4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da4d14*0./1. + d14*da4d14*db4d14/(d14**2. - 1.) - db4d14a4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da4d24*0./1. + d24*da4d24*db4d24/(d24**2. - 1.) - db4d24a4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*db4d34/(d34**2. - 1.) - db4d34a4) ),

            # ad4 g4
            -spring_arr[2][0]/(mass_arr[3]*1.*sqrt( d14**2. - 1. ))*( (da4d14*dg4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(da4d14*0./1. + d14*da4d14*dg4d14/(d14**2. - 1.) - dg4d14a4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*1.*sqrt( d24**2. - 1. ))*( (da4d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(da4d24*0./1. + d24*da4d24*dg4d24/(d24**2. - 1.) - dg4d24a4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*1.*sqrt( d34**2. - 1. ))*( (da4d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(da4d34*0./1. + d34*da4d34*dg4d34/(d34**2. - 1.) - dg4d34a4) )
            ],

            # ---------- #

            # bd4 a1,b1,g1
            [-spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d14**2. - 1. ))*( (db4d14*da1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db4d14*0./(sinh(a4)*sinh(a4)) + d14*db4d14*da1d14/(d14**2. - 1.) - da1d14b4) ),
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d14**2. - 1. ))*( (db4d14*db1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db4d14*0./(sinh(a4)*sinh(a4)) + d14*db4d14*db1d14/(d14**2. - 1.) - db1d14b4) ),
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d14**2. - 1. ))*( (db4d14*dg1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db4d14*0./(sinh(a4)*sinh(a4)) + d14*db4d14*dg1d14/(d14**2. - 1.) - dg1d14b4) ),

            # bd4 a2,b2,g2
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db4d24*0./(sinh(a4)*sinh(a4)) + d24*db4d24*da2d24/(d24**2. - 1.) - da2d24b4) ),
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db4d24*0./(sinh(a4)*sinh(a4)) + d24*db4d24*db2d24/(d24**2. - 1.) - db2d24b4) ),
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db4d24*0./(sinh(a4)*sinh(a4)) + d24*db4d24*dg2d24/(d24**2. - 1.) - dg2d24b4) ),

            # bd4 a3,b3,g3
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(sinh(a4)*sinh(a4)) + d34*db4d34*da3d34/(d34**2. - 1.) - da3d34b4) ),
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(sinh(a4)*sinh(a4)) + d34*db4d34*db3d34/(d34**2. - 1.) - db3d34b4) ),
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(sinh(a4)*sinh(a4)) + d34*db4d34*dg3d34/(d34**2. - 1.) - dg3d34b4) ),

            # bd4 a4
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d14**2. - 1. ))*( (db4d14*da4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db4d14*sinh(2.*a4)/(sinh(a4)*sinh(a4)) + d14*db4d14*da4d14/(d13**2. - 1.) - da4d14b4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db4d24*sinh(2.*a4)/(sinh(a4)*sinh(a4)) + d24*db4d24*da4d24/(d23**2. - 1.) - da4d24b4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*sinh(2.*a4)/(sinh(a4)*sinh(a4)) + d34*db4d34*da4d34/(d34**2. - 1.) - da4d34b4) ),

            # bd4 b4
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d14**2. - 1. ))*( (db4d14*db4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db4d14*0./(sinh(a4)*sinh(a4)) + d14*db4d14*db4d14/(d14**2. - 1.) - db4d14b4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db4d24*0./(sinh(a4)*sinh(a4)) + d24*db4d24*db4d24/(d24**2. - 1.) - db4d24b4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(sinh(a4)*sinh(a4)) + d34*db4d34*db4d34/(d34**2. - 1.) - db4d34b4) ),

            # bd4 g4
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d14**2. - 1. ))*( (db4d14*dg4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(db4d14*0./(sinh(a4)*sinh(a4)) + d14*db4d14*dg4d14/(d14**2. - 1.) - dg4d14b4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d24**2. - 1. ))*( (db4d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(db4d24*0./(sinh(a4)*sinh(a4)) + d24*db4d24*dg4d24/(d24**2. - 1.) - dg4d24b4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sqrt( d34**2. - 1. ))*( (db4d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(db4d34*0./(sinh(a4)*sinh(a4)) + d34*db4d34*dg4d34/(d34**2. - 1.) - dg4d34b4) )
            ],

            # ---------- #

            # gd4 a1,b1,g1
            [-spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d14**2. - 1. ))*( (dg4d14*da1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg4d14*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d14*dg4d14*da1d14/(d14**2. - 1.) - da1d14g4) ),
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d14**2. - 1. ))*( (dg4d14*db1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg4d14*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d14*dg4d14*db1d14/(d14**2. - 1.) - db1d14g4) ),
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d14**2. - 1. ))*( (dg4d14*dg1d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg4d14*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d14*dg4d14*dg1d14/(d14**2. - 1.) - dg1d14g4) ),

            # gd4 a2,b2,g2
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*da2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg4d24*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d24*dg4d24*da2d24/(d24**2. - 1.) - da2d24g4) ),
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*db2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg4d24*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d24*dg4d24*db2d24/(d24**2. - 1.) - db2d24g4) ),
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*dg2d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg4d24*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d24*dg4d24*dg2d24/(d24**2. - 1.) - dg2d24g4) ),

            # gd4 a3,b3,g3
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*da3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d34*dg4d34*da3d34/(d34**2. - 1.) - da3d34g4) ),
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*db3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d34*dg4d34*db3d34/(d34**2. - 1.) - db3d34g4) ),
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*dg3d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d34*dg4d34*dg3d34/(d34**2. - 1.) - dg3d34g4) ),

            # gd4 a4
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d14**2. - 1. ))*( (dg4d14*da4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg4d14*sinh(2.*a4)*sin(b4)*sin(b4)/(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d14*dg4d14*da4d14/(d14**2. - 1.) - da4d14g4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*da4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg4d24*sinh(2.*a4)*sin(b4)*sin(b4)/(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d24*dg4d24*da4d24/(d24**2. - 1.) - da4d24g4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*da4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*sinh(2.*a4)*sin(b4)*sin(b4)/(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d34*dg4d34*da4d34/(d34**2. - 1.) - da4d34g4) ),

            # gd4 b4
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d14**2. - 1. ))*( (dg4d14*db4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg4d14*sin(2.*b4)*sinh(a4)*sinh(a4)/(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d14*dg4d14*db4d14/(d14**2. - 1.) - db4d14g4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*db4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg4d24*sin(2.*b4)*sinh(a4)*sinh(a4)/(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d24*dg4d24*db4d24/(d24**2. - 1.) - db4d24g4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*db4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*sin(2.*b4)*sinh(a4)*sinh(a4)/(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d34*dg4d34*db4d34/(d34**2. - 1.) - db4d34g4) ),

            # gd4 g4
            -spring_arr[2][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d14**2. - 1. ))*( (dg4d14*dg4d14)/sqrt( d14**2. - 1.) - ( arccosh(d14) - spring_arr[2][1] )*(dg4d14*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d14*dg4d14*dg4d14/(d14**2. - 1.) - dg4d14g4) ) + 
            -spring_arr[4][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d24**2. - 1. ))*( (dg4d24*dg4d24)/sqrt( d24**2. - 1.) - ( arccosh(d24) - spring_arr[4][1] )*(dg4d24*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d24*dg4d24*dg4d24/(d24**2. - 1.) - dg4d24g4) ) + 
            -spring_arr[5][0]/(mass_arr[3]*sinh(a4)*sinh(a4)*sin(b4)*sin(b4)*sqrt( d34**2. - 1. ))*( (dg4d34*dg4d34)/sqrt( d34**2. - 1.) - ( arccosh(d34) - spring_arr[5][1] )*(dg4d34*0./(sinh(a4)*sinh(a4)*sin(b4)*sin(b4)) + d34*dg4d34*dg4d34/(d34**2. - 1.) - dg4d34g4) )
            ]

        ])
   
    def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return an1 - an - .5*h*(adn + adn1)

    def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
        return bn1 - bn - .5*h*(bdn + bdn1)

    def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
        return gn1 - gn - .5*h*(gdn + gdn1)        

    def con4(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, spoke3_pos, spoke3_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13, sp14):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        a4n,b4n,g4n=spoke3_pos
        a4n1,b4n1,g4n1=spoke3_pos_guess
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*sin(b1n)**2.)*sinh(a1n)*cosh(a1n) - 
            sp12[0]/m1*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*cosh(a2n) - cosh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - cosh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.) -
            sp13[0]/m1*( 
            arccosh(cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n)) - sp13[1])*
            (sinh(a1n)*cosh(a3n) - cosh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - cosh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))**2.) -
            sp14[0]/m1*( 
            arccosh(cosh(a1n)*cosh(a4n) - sinh(a1n)*cos(b1n)*sinh(a4n)*cos(b4n) - sinh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n)) - sp14[1])*
            (sinh(a1n)*cosh(a4n) - cosh(a1n)*cos(b1n)*sinh(a4n)*cos(b4n) - cosh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a4n) - sinh(a1n)*cos(b1n)*sinh(a4n)*cos(b4n) - sinh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n))**2.)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*sin(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) - 
            sp12[0]/m1*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*cosh(a2n1) - cosh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - cosh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.) -
            sp13[0]/m1*( 
            arccosh(cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1)) - sp13[1])*
            (sinh(a1n1)*cosh(a3n1) - cosh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - cosh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))**2.) -
            sp14[0]/m1*( 
            arccosh(cosh(a1n1)*cosh(a4n1) - sinh(a1n1)*cos(b1n1)*sinh(a4n1)*cos(b4n1) - sinh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1)) - sp14[1])*
            (sinh(a1n1)*cosh(a4n1) - cosh(a1n1)*cos(b1n1)*sinh(a4n1)*cos(b4n1) - cosh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a4n1) - sinh(a1n1)*cos(b1n1)*sinh(a4n1)*cos(b4n1) - sinh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1))**2.)
            ))

    def con5(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, spoke3_pos, spoke3_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13, sp14):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        a4n,b4n,g4n=spoke3_pos
        a4n1,b4n1,g4n1=spoke3_pos_guess
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sin(b1n)*cos(b1n) - 2.*ad1n*bd1n/tanh(a1n) - 
            sp12[0]/(m1*sinh(a1n)*sinh(a1n))*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*sin(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.) -
            sp13[0]/(m1*sinh(a1n)*sinh(a1n))*( 
            arccosh(cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n)) - sp13[1])*
            (sinh(a1n)*sin(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))**2.) -
            sp14[0]/(m1*sinh(a1n)*sinh(a1n))*( 
            arccosh(cosh(a1n)*cosh(a4n) - sinh(a1n)*cos(b1n)*sinh(a4n)*cos(b4n) - sinh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n)) - sp14[1])*
            (sinh(a1n)*sin(b1n)*sinh(a4n)*cos(b4n) - sinh(a1n)*cos(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a4n) - sinh(a1n)*cos(b1n)*sinh(a4n)*cos(b4n) - sinh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n))**2.)
            + 
            gd1n1*gd1n1*sin(b1n1)*cos(b1n1) - 2.*ad1n1*bd1n1/tanh(a1n1) - 
            sp12[0]/(m1*sinh(a1n1)*sinh(a1n1))*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.) -
            sp13[0]/(m1*sinh(a1n1)*sinh(a1n1))*( 
            arccosh(cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1)) - sp13[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))**2.) -
            sp14[0]/(m1*sinh(a1n1)*sinh(a1n1))*( 
            arccosh(cosh(a1n1)*cosh(a4n1) - sinh(a1n1)*cos(b1n1)*sinh(a4n1)*cos(b4n1) - sinh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1)) - sp14[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a4n1)*cos(b4n1) - sinh(a1n1)*cos(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a4n1) - sinh(a1n1)*cos(b1n1)*sinh(a4n1)*cos(b4n1) - sinh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1))**2.)
            ))

    def con6(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, spoke3_pos, spoke3_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13, sp14):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        a4n,b4n,g4n=spoke3_pos
        a4n1,b4n1,g4n1=spoke3_pos_guess

        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n/tanh(a1n) - 2.*bd1n*gd1n/tan(b1n) - 
            sp12[0]/(m1*sinh(a1n)*sinh(a1n)*sin(b1n)*sin(b1n))*( 
            arccosh(cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n)) - sp12[1])*
            (sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*sin(g1n - g2n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a2n) - sinh(a1n)*cos(b1n)*sinh(a2n)*cos(b2n) - sinh(a1n)*sin(b1n)*sinh(a2n)*sin(b2n)*cos(g1n - g2n))**2.) -
            sp13[0]/(m1*sinh(a1n)*sinh(a1n)*sin(b1n)*sin(b1n))*( 
            arccosh(cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n)) - sp13[1])*
            (sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*sin(g1n - g3n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a3n) - sinh(a1n)*cos(b1n)*sinh(a3n)*cos(b3n) - sinh(a1n)*sin(b1n)*sinh(a3n)*sin(b3n)*cos(g1n - g3n))**2.) -
            sp14[0]/(m1*sinh(a1n)*sinh(a1n)*sin(b1n)*sin(b1n))*( 
            arccosh(cosh(a1n)*cosh(a4n) - sinh(a1n)*cos(b1n)*sinh(a4n)*cos(b4n) - sinh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n)) - sp14[1])*
            (sinh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*sin(g1n - g4n))/sqrt(-1. + 
            (cosh(a1n)*cosh(a4n) - sinh(a1n)*cos(b1n)*sinh(a4n)*cos(b4n) - sinh(a1n)*sin(b1n)*sinh(a4n)*sin(b4n)*cos(g1n - g4n))**2.)
            + 
            -2.*ad1n1*gd1n1/tanh(a1n1) - 2.*bd1n1*gd1n1/tan(b1n1) - 
            sp12[0]/(m1*sinh(a1n1)*sinh(a1n1)*sin(b1n1)*sin(b1n1))*( 
            arccosh(cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1)) - sp12[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*sin(g1n1 - g2n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a2n1) - sinh(a1n1)*cos(b1n1)*sinh(a2n1)*cos(b2n1) - sinh(a1n1)*sin(b1n1)*sinh(a2n1)*sin(b2n1)*cos(g1n1 - g2n1))**2.) -
            sp13[0]/(m1*sinh(a1n1)*sinh(a1n1)*sin(b1n1)*sin(b1n1))*( 
            arccosh(cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1)) - sp13[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*sin(g1n1 - g3n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a3n1) - sinh(a1n1)*cos(b1n1)*sinh(a3n1)*cos(b3n1) - sinh(a1n1)*sin(b1n1)*sinh(a3n1)*sin(b3n1)*cos(g1n1 - g3n1))**2.) -
            sp14[0]/(m1*sinh(a1n1)*sinh(a1n1)*sin(b1n1)*sin(b1n1))*( 
            arccosh(cosh(a1n1)*cosh(a4n1) - sinh(a1n1)*cos(b1n1)*sinh(a4n1)*cos(b4n1) - sinh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1)) - sp14[1])*
            (sinh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*sin(g1n1 - g4n1))/sqrt(-1. + 
            (cosh(a1n1)*cosh(a4n1) - sinh(a1n1)*cos(b1n1)*sinh(a4n1)*cos(b4n1) - sinh(a1n1)*sin(b1n1)*sinh(a4n1)*sin(b4n1)*cos(g1n1 - g4n1))**2.)
            )) 
    
    def jacobian(positions, velocities, mass_arr, h, spring_arr):
        a1n1,b1n1,g1n1=positions[0]
        a2n1,b2n1,g2n1=positions[1]
        a3n1,b3n1,g3n1=positions[2]
        a4n1,b4n1,g4n1=positions[3]
        ad1n1,bd1n1,gd1n1=velocities[0]
        ad2n1,bd2n1,gd2n1=velocities[1]
        ad3n1,bd3n1,gd3n1=velocities[2]
        ad4n1,bd4n1,gd4n1=velocities[3]
        spring_terms=jacobi_sp_terms(positions, mass_arr, spring_arr)
        #print(spring_terms)
        return array([
            [1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0., 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0.],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0.],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0.],

            [0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.],
            [0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.],
            [0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.],

            [0.,0.,0., 0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1+sin(b1n1)*sin(b1n1)*gd1n1*gd1n1)*cosh(2.*a1n1) + spring_terms[0][0],
            -.25*h*sinh(2.*a1n1)*sin(2.*b1n1)*gd1n1*gd1n1 + spring_terms[0][1],
            0. + spring_terms[0][2], 
            
            spring_terms[0][3],
            spring_terms[0][4],
            spring_terms[0][5],

            spring_terms[0][6],
            spring_terms[0][7],
            spring_terms[0][8],

            spring_terms[0][9],
            spring_terms[0][10],
            spring_terms[0][11],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            -.5*h*sinh(2.*a1n1)*sin(b1n1)*sin(b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # bd1 update
            [-h*ad1n1*bd1n1/(sinh(a1n1)*sinh(a1n1)) + spring_terms[1][0],
            -.5*h*cos(2.*b1n1)*gd1n1*gd1n1 + spring_terms[1][1],
            0. + spring_terms[1][2], 
            
            spring_terms[1][3],
            spring_terms[1][4],
            spring_terms[1][5],

            spring_terms[1][6],
            spring_terms[1][7],
            spring_terms[1][8],

            spring_terms[1][9],
            spring_terms[1][10],
            spring_terms[1][11],
            
            h/tanh(a1n1)*bd1n1,
            1.+h/tanh(a1n1)*ad1n1,
            -.5*h*sin(2.*b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # gd1 update
            [-h*ad1n1*gd1n1/(sinh(a1n1)*sinh(a1n1)) + spring_terms[2][0],
            -h*bd1n1*gd1n1/(sin(b1n1)*sin(b1n1)) + spring_terms[2][1],
            0. + spring_terms[2][2],
             
            spring_terms[2][3],
            spring_terms[2][4],
            spring_terms[2][5],

            spring_terms[2][6],
            spring_terms[2][7],
            spring_terms[2][8],

            spring_terms[2][9],
            spring_terms[2][10],
            spring_terms[2][11],

            h/tanh(a1n1)*gd1n1,
            h/tan(b1n1)*gd1n1,
            1.+h/tanh(a1n1)*ad1n1+h/tan(b1n1)*bd1n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0.,
            ],

            # ad2 update
            [spring_terms[3][0],
            spring_terms[3][1],
            spring_terms[3][2],

            -.5*h*(bd2n1*bd2n1+sin(b2n1)*sin(b2n1)*gd2n1*gd2n1)*cosh(2.*a2n1) + spring_terms[3][3],
            -.25*h*sinh(2.*a2n1)*sin(2.*b2n1)*gd2n1*gd2n1 + spring_terms[3][4],
            0. + spring_terms[3][5],

            spring_terms[3][6],
            spring_terms[3][7],
            spring_terms[3][8],

            spring_terms[3][9],
            spring_terms[3][10],
            spring_terms[3][11],

            0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            -.5*h*sinh(2.*a2n1)*sin(b2n1)*sin(b2n1)*gd2n1,

            0.,0.,0., 0.,0.,0.,
            ],

            # bd2 update
            [spring_terms[4][0],
            spring_terms[4][1],
            spring_terms[4][2],

            -h*ad2n1*bd2n1/(sinh(a2n1)*sinh(a2n1)) + spring_terms[4][3],
            -.5*h*cos(2.*b2n1)*gd2n1*gd2n1 + spring_terms[4][4],
            0. + spring_terms[4][5],

            spring_terms[4][6],
            spring_terms[4][7],
            spring_terms[4][8],

            spring_terms[4][9],
            spring_terms[4][10],
            spring_terms[4][11],

            0.,0.,0.,

            h/tanh(a2n1)*bd2n1,
            1.+h/tanh(a2n1)*ad2n1,
            -.5*h*sin(2.*b2n1)*gd2n1,
            
            0.,0.,0., 0.,0.,0.,
            ],

            # gd2 update
            [spring_terms[5][0],
            spring_terms[5][1],
            spring_terms[5][2],

            -h*ad2n1*gd2n1/(sinh(a2n1)*sinh(a2n1)) + spring_terms[5][3],
            -h*bd2n1*gd2n1/(sin(b2n1)*sin(b2n1)) + spring_terms[5][4],
            0. + spring_terms[5][5],

            spring_terms[5][6],
            spring_terms[5][7],
            spring_terms[5][8],

            spring_terms[5][9],
            spring_terms[5][10],
            spring_terms[5][11],

            0.,0.,0.,

            h/tanh(a2n1)*gd2n1,
            h/tan(b2n1)*gd2n1,
            1.+h/tanh(a2n1)*ad2n1+h/tan(b2n1)*bd2n1,
            
            0.,0.,0., 0.,0.,0.,
            ],

            # ad3 update
            [spring_terms[6][0],
            spring_terms[6][1],
            spring_terms[6][2],

            spring_terms[6][3],
            spring_terms[6][4],
            spring_terms[6][5],

            -.5*h*(bd3n1*bd3n1+sin(b3n1)*sin(b3n1)*gd3n1*gd3n1)*cosh(2.*a3n1) + spring_terms[6][6],
            -.25*h*sinh(2.*a3n1)*sin(2.*b3n1)*gd3n1*gd3n1 + spring_terms[6][7],
            0. + spring_terms[6][8],

            spring_terms[6][9],
            spring_terms[6][10],
            spring_terms[6][11],

            0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a3n1)*bd3n1,
            -.5*h*sinh(2.*a3n1)*sin(b3n1)*sin(b3n1)*gd3n1,

            0.,0.,0.
            ],

            # bd3 update
            [spring_terms[7][0],
            spring_terms[7][1],
            spring_terms[7][2],

            spring_terms[7][3],
            spring_terms[7][4],
            spring_terms[7][5],

            -h*ad3n1*bd3n1/(sinh(a3n1)*sinh(a3n1)) + spring_terms[7][6],
            -.5*h*cos(2.*b3n1)*gd3n1*gd3n1 + spring_terms[7][7],
            0. + spring_terms[7][8],

            spring_terms[7][9],
            spring_terms[7][10],
            spring_terms[7][11],

            0.,0.,0., 0.,0.,0.,

            h/tanh(a3n1)*bd3n1,
            1.+h/tanh(a3n1)*ad3n1,
            -.5*h*sin(2.*b3n1)*gd3n1,

            0.,0.,0.
            ],

            # gd3 update
            [spring_terms[8][0],
            spring_terms[8][1],
            spring_terms[8][2],

            spring_terms[8][3],
            spring_terms[8][4],
            spring_terms[8][5],

            -h*ad3n1*gd3n1/(sinh(a3n1)*sinh(a3n1)) + spring_terms[8][6],
            -h*bd3n1*gd3n1/(sin(b3n1)*sin(b3n1)) + spring_terms[8][7],
            0. + spring_terms[8][8],

            spring_terms[8][9],
            spring_terms[8][10],
            spring_terms[8][11],

            0.,0.,0., 0.,0.,0.,

            h/tanh(a3n1)*gd3n1,
            h/tan(b3n1)*gd3n1,
            1.+h/tanh(a3n1)*ad3n1+h/tan(b3n1)*bd3n1,

            0.,0.,0.
            ],

            # ad4 update
            [spring_terms[9][0],
            spring_terms[9][1],
            spring_terms[9][2],

            spring_terms[9][3],
            spring_terms[9][4],
            spring_terms[9][5],

            spring_terms[9][6],
            spring_terms[9][7],
            spring_terms[9][8],

            -.5*h*(bd4n1*bd4n1+sin(b4n1)*sin(b4n1)*gd4n1*gd4n1)*cosh(2.*a4n1) + spring_terms[9][9],
            -.25*h*sinh(2.*a4n1)*sin(2.*b4n1)*gd4n1*gd4n1 + spring_terms[9][10],
            0. + spring_terms[9][11],

            0.,0.,0., 0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a4n1)*bd4n1,
            -.5*h*sinh(2.*a4n1)*sin(b4n1)*sin(b4n1)*gd4n1
            ],

            # bd4 update
            [spring_terms[10][0],
            spring_terms[10][1],
            spring_terms[10][2],

            spring_terms[10][3],
            spring_terms[10][4],
            spring_terms[10][5],

            spring_terms[10][6],
            spring_terms[10][7],
            spring_terms[10][8],

            -h*ad4n1*bd4n1/(sinh(a4n1)*sinh(a4n1)) + spring_terms[10][9],
            -.5*h*cos(2.*b4n1)*gd4n1*gd4n1 + spring_terms[10][10],
            0. + spring_terms[10][11],

            0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h/tanh(a4n1)*bd4n1,
            1.+h/tanh(a4n1)*ad4n1,
            -.5*h*sin(2.*b4n1)*gd4n1
            ],

            # gd4 update
            [spring_terms[11][0],
            spring_terms[11][1],
            spring_terms[11][2],

            spring_terms[11][3],
            spring_terms[11][4],
            spring_terms[11][5],

            spring_terms[11][6],
            spring_terms[11][7],
            spring_terms[11][8],

            -h*ad4n1*gd4n1/(sinh(a4n1)*sinh(a4n1)) + spring_terms[11][9],
            -h*bd4n1*gd4n1/(sin(b4n1)*sin(b4n1)) + spring_terms[11][10],
            0. + spring_terms[11][11],

            0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h/tanh(a4n1)*gd4n1,
            h/tan(b4n1)*gd4n1,
            1.+h/tanh(a4n1)*ad4n1+h/tan(b4n1)*bd4n1
            ]
        ])

    #print(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr))
    diff1=linalg.solve(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr),-array([
        #p1
        con1(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con2(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con3(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        #p2
        con1(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con2(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con3(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        #p3
        con1(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con2(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con3(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        #p4
        con1(posn_arr[3][0],posn_arr[3][0], posn_arr[3][1], posn_arr[3][1], posn_arr[3][2], posn_arr[3][2], veln_arr[3][0], veln_arr[3][0], veln_arr[3][1], veln_arr[3][1], veln_arr[3][2], veln_arr[3][2], step),
        con2(posn_arr[3][0],posn_arr[3][0], posn_arr[3][1], posn_arr[3][1], posn_arr[3][2], posn_arr[3][2], veln_arr[3][0], veln_arr[3][0], veln_arr[3][1], veln_arr[3][1], veln_arr[3][2], veln_arr[3][2], step),
        con3(posn_arr[3][0],posn_arr[3][0], posn_arr[3][1], posn_arr[3][1], posn_arr[3][2], posn_arr[3][2], veln_arr[3][0], veln_arr[3][0], veln_arr[3][1], veln_arr[3][1], veln_arr[3][2], veln_arr[3][2], step),

        #v1
        con4(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[3],posn_arr[3],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
        con5(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[3],posn_arr[3],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
        con6(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[3],posn_arr[3],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
        #v2
        con4(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],posn_arr[3],posn_arr[3],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
        con5(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],posn_arr[3],posn_arr[3],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
        con6(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[2],posn_arr[2],posn_arr[3],posn_arr[3],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
        #v3
        con4(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[3],posn_arr[3],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[3],spring_arr[5]),
        con5(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[3],posn_arr[3],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[3],spring_arr[5]),
        con6(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[3],posn_arr[3],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[3],spring_arr[5]),
        #v4
        con4(posn_arr[3],posn_arr[3],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[3],veln_arr[3],mass_arr[3],step,spring_arr[2],spring_arr[4],spring_arr[5]),
        con5(posn_arr[3],posn_arr[3],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[3],veln_arr[3],mass_arr[3],step,spring_arr[2],spring_arr[4],spring_arr[5]),
        con6(posn_arr[3],posn_arr[3],posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],veln_arr[3],veln_arr[3],mass_arr[3],step,spring_arr[2],spring_arr[4],spring_arr[5])      
    ]))
    val1 = array([
            posn_arr[0][0]+diff1[0], posn_arr[0][1]+diff1[1], posn_arr[0][2]+diff1[2], 
            posn_arr[1][0]+diff1[3], posn_arr[1][1]+diff1[4], posn_arr[1][2]+diff1[5],
            posn_arr[2][0]+diff1[6], posn_arr[2][1]+diff1[7], posn_arr[2][2]+diff1[8],
            posn_arr[3][0]+diff1[9], posn_arr[3][1]+diff1[10], posn_arr[3][2]+diff1[11],

            veln_arr[0][0]+diff1[12], veln_arr[0][1]+diff1[13], veln_arr[0][2]+diff1[14], 
            veln_arr[1][0]+diff1[15], veln_arr[1][1]+diff1[16], veln_arr[1][2]+diff1[17],
            veln_arr[2][0]+diff1[18], veln_arr[2][1]+diff1[19], veln_arr[2][2]+diff1[20],
            veln_arr[3][0]+diff1[21], veln_arr[3][1]+diff1[22], veln_arr[3][2]+diff1[23]
            ])    
    x = 0
    while(x < 7):
        new_pos_arr=array([val1[0:3],val1[3:6],val1[6:9],val1[9:12]])
        new_vel_arr=array([val1[12:15],val1[15:18],val1[18:21],val1[21:24]])
        diff2=linalg.solve(jacobian(new_pos_arr, new_vel_arr, mass_arr, step, spring_arr),-array([
            #p1
            con1(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con2(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con3(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            #p2
            con1(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con2(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con3(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            #p3
            con1(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con2(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con3(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            #p4
            con1(posn_arr[3][0],new_pos_arr[3][0], posn_arr[3][1], new_pos_arr[3][1], posn_arr[3][2], new_pos_arr[3][2], veln_arr[3][0], new_vel_arr[3][0], veln_arr[3][1], new_vel_arr[3][1], veln_arr[3][2], new_vel_arr[3][2], step),
            con2(posn_arr[3][0],new_pos_arr[3][0], posn_arr[3][1], new_pos_arr[3][1], posn_arr[3][2], new_pos_arr[3][2], veln_arr[3][0], new_vel_arr[3][0], veln_arr[3][1], new_vel_arr[3][1], veln_arr[3][2], new_vel_arr[3][2], step),
            con3(posn_arr[3][0],new_pos_arr[3][0], posn_arr[3][1], new_pos_arr[3][1], posn_arr[3][2], new_pos_arr[3][2], veln_arr[3][0], new_vel_arr[3][0], veln_arr[3][1], new_vel_arr[3][1], veln_arr[3][2], new_vel_arr[3][2], step),

            #v1
            con4(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[3],new_pos_arr[3],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
            con5(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[3],new_pos_arr[3],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
            con6(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[3],new_pos_arr[3],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
            #v2
            con4(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],posn_arr[3],new_pos_arr[3],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
            con5(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],posn_arr[3],new_pos_arr[3],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
            con6(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[2],new_pos_arr[2],posn_arr[3],new_pos_arr[3],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
            #v3
            con4(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[3],new_pos_arr[3],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[3],spring_arr[5]),
            con5(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[3],new_pos_arr[3],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[3],spring_arr[5]),
            con6(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[3],new_pos_arr[3],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[3],spring_arr[5]),
            #v4
            con4(posn_arr[3],new_pos_arr[3],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[3],new_vel_arr[3],mass_arr[3],step,spring_arr[2],spring_arr[4],spring_arr[5]),
            con5(posn_arr[3],new_pos_arr[3],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[3],new_vel_arr[3],mass_arr[3],step,spring_arr[2],spring_arr[4],spring_arr[5]),
            con6(posn_arr[3],new_pos_arr[3],posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],veln_arr[3],new_vel_arr[3],mass_arr[3],step,spring_arr[2],spring_arr[4],spring_arr[5])    
        ]))      
        val2 = array([
            val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], 
            val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5],
            val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8],
            val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11],

            val1[12]+diff2[12], val1[13]+diff2[13], val1[14]+diff2[14], 
            val1[15]+diff2[15], val1[16]+diff2[16], val1[17]+diff2[17],
            val1[18]+diff2[18], val1[19]+diff2[19], val1[20]+diff2[20],
            val1[21]+diff2[21], val1[22]+diff2[22], val1[23]+diff2[23]
            ])
        val1 = val2
        x=x+1
    #print(diff2)
    return val1

#################################################
### H2xE 3-space Spring Potential (Dumb-Bell) ###
#################################################

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

    def jacobi_sp_terms(positions, mass_arr, spring_arr):
        a1,b1,g1=positions[0]
        a2,b2,g2=positions[1]
        a3,b3,g3=positions[2]
        a4,b4,g4=positions[3]
        a5,b5,g5=positions[4]
        a6,b6,g6=positions[5]
        a7,b7,g7=positions[6]
        a8,b8,g8=positions[7]
        
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
            -spring_arr[11][0]/(mass_arr[7]*1.*sqrt( d78**2. - 1. ))*( (da8d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(da8d78*0./1. + d78*da8d78*dg7d78/(d78**2. - 1.) - dg7d78a8) ),

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
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*sqrt( d78**2. - 1. ))*( (db8d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(db8d78*0./(cosh(a8)*cosh(a8)) + d78*db8d78*dg7d78/(d78**2. - 1.) - dg7d78b8) ),

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
            -spring_arr[11][0]/(mass_arr[7]*cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)*sqrt( d78**2. - 1. ))*( (dg8d78*dg7d78)/sqrt( d78**2. - 1.) - ( arccosh(d78) - spring_arr[11][1] )*(dg8d78*0./(cosh(a8)*cosh(a8)*cosh(b8)*cosh(b8)) + d78*dg8d78*dg7d78/(d78**2. - 1.) - dg7d78g8) ),

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

    def con4(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, spoke3_pos, spoke3_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13, sp14):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        a4n,b4n,g4n=spoke3_pos
        a4n1,b4n1,g4n1=spoke3_pos_guess
        return (ad1n1 - ad1n - .5*h*(
            (bd1n*bd1n + gd1n*gd1n*cosh(b1n)**2.)*sinh(a1n)*cosh(a1n) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - sp12[1])*
            (-cosh(a1n)*sinh(a2n) - sinh(a1n)*sinh(b1n)*cosh(a2n)*sinh(b2n) + sinh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n))) - sp13[1])*
            (-cosh(a1n)*sinh(a3n) - sinh(a1n)*sinh(b1n)*cosh(a3n)*sinh(b3n) + sinh(a1n)*cosh(b1n)*cosh(a3n)*cosh(b3n)*cosh(g1n - g3n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n)))**2.) -
            sp14[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a4n) + cosh(a1n)*cosh(a4n)*(cosh(b1n)*cosh(b4n)*cosh(g1n - g4n) - sinh(b1n)*sinh(b4n))) - sp14[1])*
            (-cosh(a1n)*sinh(a4n) - sinh(a1n)*sinh(b1n)*cosh(a4n)*sinh(b4n) + sinh(a1n)*cosh(b1n)*cosh(a4n)*cosh(b4n)*cosh(g1n - g4n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a4n) + cosh(a1n)*cosh(a4n)*(cosh(b1n)*cosh(b4n)*cosh(g1n - g4n) - sinh(b1n)*sinh(b4n)))**2.)
            + 
            (bd1n1*bd1n1 + gd1n1*gd1n1*cosh(b1n1)**2.)*sinh(a1n1)*cosh(a1n1) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - sp12[1])*
            (-cosh(a1n1)*sinh(a2n1) - sinh(a1n1)*sinh(b1n1)*cosh(a2n1)*sinh(b2n1) + sinh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1))) - sp13[1])*
            (-cosh(a1n1)*sinh(a3n1) - sinh(a1n1)*sinh(b1n1)*cosh(a3n1)*sinh(b3n1) + sinh(a1n1)*cosh(b1n1)*cosh(a3n1)*cosh(b3n1)*cosh(g1n1 - g3n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1)))**2.) -
            sp14[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a4n1) + cosh(a1n1)*cosh(a4n1)*(cosh(b1n1)*cosh(b4n1)*cosh(g1n1 - g4n1) - sinh(b1n1)*sinh(b4n1))) - sp14[1])*
            (-cosh(a1n1)*sinh(a4n1) - sinh(a1n1)*sinh(b1n1)*cosh(a4n1)*sinh(b4n1) + sinh(a1n1)*cosh(b1n1)*cosh(a4n1)*cosh(b4n1)*cosh(g1n1 - g4n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a4n1) + cosh(a1n1)*cosh(a4n1)*(cosh(b1n1)*cosh(b4n1)*cosh(g1n1 - g4n1) - sinh(b1n1)*sinh(b4n1)))**2.)
            ))

    def con5(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, spoke3_pos, spoke3_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13, sp14):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        a4n,b4n,g4n=spoke3_pos
        a4n1,b4n1,g4n1=spoke3_pos_guess
        return (bd1n1 - bd1n - .5*h*(
            gd1n*gd1n*sinh(b1n)*cosh(b1n) - 2.*ad1n*bd1n*tanh(a1n) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - sp12[1])*
            (-cosh(a1n)*cosh(b1n)*cosh(a2n)*sinh(b2n) + cosh(a1n)*sinh(b1n)*cosh(a2n)*cosh(b2n)*cosh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n))) - sp13[1])*
            (-cosh(a1n)*cosh(b1n)*cosh(a3n)*sinh(b3n) + cosh(a1n)*sinh(b1n)*cosh(a3n)*cosh(b3n)*cosh(g1n - g3n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n)))**2.) -
            sp14[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a4n) + cosh(a1n)*cosh(a4n)*(cosh(b1n)*cosh(b4n)*cosh(g1n - g4n) - sinh(b1n)*sinh(b4n))) - sp14[1])*
            (-cosh(a1n)*cosh(b1n)*cosh(a4n)*sinh(b4n) + cosh(a1n)*sinh(b1n)*cosh(a4n)*cosh(b4n)*cosh(g1n - g4n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a4n) + cosh(a1n)*cosh(a4n)*(cosh(b1n)*cosh(b4n)*cosh(g1n - g4n) - sinh(b1n)*sinh(b4n)))**2.)
            + 
            gd1n1*gd1n1*sinh(b1n1)*cosh(b1n1) - 2.*ad1n1*bd1n1*tanh(a1n1) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - sp12[1])*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*sinh(b2n1) + cosh(a1n1)*sinh(b1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1))) - sp13[1])*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a3n1)*sinh(b3n1) + cosh(a1n1)*sinh(b1n1)*cosh(a3n1)*cosh(b3n1)*cosh(g1n1 - g3n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1)))**2.) -
            sp14[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a4n1) + cosh(a1n1)*cosh(a4n1)*(cosh(b1n1)*cosh(b4n1)*cosh(g1n1 - g4n1) - sinh(b1n1)*sinh(b4n1))) - sp14[1])*
            (-cosh(a1n1)*cosh(b1n1)*cosh(a4n1)*sinh(b4n1) + cosh(a1n1)*sinh(b1n1)*cosh(a4n1)*cosh(b4n1)*cosh(g1n1 - g4n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a4n1) + cosh(a1n1)*cosh(a4n1)*(cosh(b1n1)*cosh(b4n1)*cosh(g1n1 - g4n1) - sinh(b1n1)*sinh(b4n1)))**2.) 
            ))

    def con6(base_pos, base_pos_guess, spoke1_pos, spoke1_pos_guess, spoke2_pos, spoke2_pos_guess, spoke3_pos, spoke3_pos_guess, base_vel, base_vel_guess, m1, h, sp12, sp13, sp14):
        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        a2n,b2n,g2n=spoke1_pos
        a2n1,b2n1,g2n1=spoke1_pos_guess
        a3n,b3n,g3n=spoke2_pos
        a3n1,b3n1,g3n1=spoke2_pos_guess
        a4n,b4n,g4n=spoke3_pos
        a4n1,b4n1,g4n1=spoke3_pos_guess

        return (gd1n1 - gd1n - .5*h*(
            -2.*ad1n*gd1n*tanh(a1n) - 2.*bd1n*gd1n*tanh(b1n) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n))) - sp12[1])*
            (cosh(a1n)*cosh(b1n)*cosh(a2n)*cosh(b2n)*sinh(g1n - g2n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a2n) + cosh(a1n)*cosh(a2n)*(cosh(b1n)*cosh(b2n)*cosh(g1n - g2n) - sinh(b1n)*sinh(b2n)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n))) - sp13[1])*
            (cosh(a1n)*cosh(b1n)*cosh(a3n)*cosh(b3n)*sinh(g1n - g3n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a3n) + cosh(a1n)*cosh(a3n)*(cosh(b1n)*cosh(b3n)*cosh(g1n - g3n) - sinh(b1n)*sinh(b3n)))**2.) -
            sp14[0]/m1*( 
            arccosh(-sinh(a1n)*sinh(a4n) + cosh(a1n)*cosh(a4n)*(cosh(b1n)*cosh(b4n)*cosh(g1n - g4n) - sinh(b1n)*sinh(b4n))) - sp14[1])*
            (cosh(a1n)*cosh(b1n)*cosh(a4n)*cosh(b4n)*sinh(g1n - g4n))/sqrt(-1. + 
            (-sinh(a1n)*sinh(a4n) + cosh(a1n)*cosh(a4n)*(cosh(b1n)*cosh(b4n)*cosh(g1n - g4n) - sinh(b1n)*sinh(b4n)))**2.)
            + 
            -2.*ad1n1*gd1n1*tanh(a1n1) - 2.*bd1n1*gd1n1*tanh(b1n1) - 
            sp12[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1))) - sp12[1])*
            (cosh(a1n1)*cosh(b1n1)*cosh(a2n1)*cosh(b2n1)*sinh(g1n1 - g2n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a2n1) + cosh(a1n1)*cosh(a2n1)*(cosh(b1n1)*cosh(b2n1)*cosh(g1n1 - g2n1) - sinh(b1n1)*sinh(b2n1)))**2.) -
            sp13[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1))) - sp13[1])*
            (cosh(a1n1)*cosh(b1n1)*cosh(a3n1)*cosh(b3n1)*sinh(g1n1 - g3n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a3n1) + cosh(a1n1)*cosh(a3n1)*(cosh(b1n1)*cosh(b3n1)*cosh(g1n1 - g3n1) - sinh(b1n1)*sinh(b3n1)))**2.) -
            sp14[0]/m1*( 
            arccosh(-sinh(a1n1)*sinh(a4n1) + cosh(a1n1)*cosh(a4n1)*(cosh(b1n1)*cosh(b4n1)*cosh(g1n1 - g4n1) - sinh(b1n1)*sinh(b4n1))) - sp14[1])*
            (cosh(a1n1)*cosh(b1n1)*cosh(a4n1)*cosh(b4n1)*sinh(g1n1 - g4n1))/sqrt(-1. + 
            (-sinh(a1n1)*sinh(a4n1) + cosh(a1n1)*cosh(a4n1)*(cosh(b1n1)*cosh(b4n1)*cosh(g1n1 - g4n1) - sinh(b1n1)*sinh(b4n1)))**2.)
            )) 
    
    def jacobian(positions, velocities, mass_arr, h, spring_arr):
        a1n1,b1n1,g1n1=positions[0]
        a2n1,b2n1,g2n1=positions[1]
        a3n1,b3n1,g3n1=positions[2]
        a4n1,b4n1,g4n1=positions[3]
        a5n1,b5n1,g5n1=positions[4]
        a6n1,b6n1,g6n1=positions[5]
        a7n1,b7n1,g7n1=positions[6]
        a8n1,b8n1,g8n1=positions[7]
        ad1n1,bd1n1,gd1n1=velocities[0]
        ad2n1,bd2n1,gd2n1=velocities[1]
        ad3n1,bd3n1,gd3n1=velocities[2]
        ad4n1,bd4n1,gd4n1=velocities[3]
        ad5n1,bd5n1,gd5n1=velocities[4]
        ad6n1,bd6n1,gd6n1=velocities[5]
        ad7n1,bd7n1,gd7n1=velocities[6]
        ad8n1,bd8n1,gd8n1=velocities[7]
        spring_terms=jacobi_sp_terms(positions, mass_arr, spring_arr)
        #print(spring_terms[0][0])
        return array([
            [1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.],
            [0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.],
            [0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.],

            [0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],

            [0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],

            [0.,0.,0., 0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],

            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0., 0.,0.,0.,],

            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0., 0.,0.,0.,],

            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0., 0.,0.,0.,],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h, 0.,0.,0.,],

            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 1.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., -.5*h,0.,0.],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,1.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,-.5*h,0.],
            [0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,1., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,-.5*h],

            # ad1 update
            [-.5*h*(bd1n1*bd1n1+cosh(b1n1)*cosh(b1n1)*gd1n1*gd1n1)*cosh(2.*a1n1) + spring_terms[0][0],
            -.25*h*sinh(2.*a1n1)*sinh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[0][1],
            0. + spring_terms[0][2], 
            
            spring_terms[0][3],
            spring_terms[0][4],
            spring_terms[0][5],

            spring_terms[0][6],
            spring_terms[0][7],
            spring_terms[0][8],

            spring_terms[0][9],
            spring_terms[0][10],
            spring_terms[0][11],

            spring_terms[0][12],
            spring_terms[0][13],
            spring_terms[0][14],

            spring_terms[0][15],
            spring_terms[0][16],
            spring_terms[0][17],

            spring_terms[0][18],
            spring_terms[0][19],
            spring_terms[0][20],

            spring_terms[0][21],
            spring_terms[0][22],
            spring_terms[0][23],
            
            1.,
            -.5*h*sinh(2.*a1n1)*bd1n1,
            -.5*h*sinh(2.*a1n1)*cosh(b1n1)*cosh(b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # bd1 update
            [h*ad1n1*bd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[1][0],
            -.5*h*cosh(2.*b1n1)*gd1n1*gd1n1 + spring_terms[1][1],
            0. + spring_terms[1][2], 
            
            spring_terms[1][3],
            spring_terms[1][4],
            spring_terms[1][5],

            spring_terms[1][6],
            spring_terms[1][7],
            spring_terms[1][8],

            spring_terms[1][9],
            spring_terms[1][10],
            spring_terms[1][11],

            spring_terms[1][12],
            spring_terms[1][13],
            spring_terms[1][14],

            spring_terms[1][15],
            spring_terms[1][16],
            spring_terms[1][17],

            spring_terms[1][18],
            spring_terms[1][19],
            spring_terms[1][20],

            spring_terms[1][21],
            spring_terms[1][22],
            spring_terms[1][23],
            
            h*tanh(a1n1)*bd1n1,
            1.+h*tanh(a1n1)*ad1n1,
            -.5*h*sinh(2.*b1n1)*gd1n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # gd1 update
            [h*ad1n1*gd1n1/(cosh(a1n1)*cosh(a1n1)) + spring_terms[2][0],
            h*bd1n1*gd1n1/(cosh(b1n1)*cosh(b1n1)) + spring_terms[2][1],
            0. + spring_terms[2][2],
             
            spring_terms[2][3],
            spring_terms[2][4],
            spring_terms[2][5],

            spring_terms[2][6],
            spring_terms[2][7],
            spring_terms[2][8],

            spring_terms[2][9],
            spring_terms[2][10],
            spring_terms[2][11],

            spring_terms[2][12],
            spring_terms[2][13],
            spring_terms[2][14],

            spring_terms[2][15],
            spring_terms[2][16],
            spring_terms[2][17],

            spring_terms[2][18],
            spring_terms[2][19],
            spring_terms[2][20],

            spring_terms[2][21],
            spring_terms[2][22],
            spring_terms[2][23],

            h*tanh(a1n1)*gd1n1,
            h*tanh(b1n1)*gd1n1,
            1.+h*tanh(a1n1)*ad1n1+h*tanh(b1n1)*bd1n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # ad2 update
            [spring_terms[3][0],
            spring_terms[3][1],
            spring_terms[3][2],

            -.5*h*(bd2n1*bd2n1+cosh(b2n1)*cosh(b2n1)*gd2n1*gd2n1)*cosh(2.*a2n1) + spring_terms[3][3],
            -.25*h*sinh(2.*a2n1)*sinh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[3][4],
            0. + spring_terms[3][5],

            spring_terms[3][6],
            spring_terms[3][7],
            spring_terms[3][8],

            spring_terms[3][9],
            spring_terms[3][10],
            spring_terms[3][11],

            spring_terms[3][12],
            spring_terms[3][13],
            spring_terms[3][14],

            spring_terms[3][15],
            spring_terms[3][16],
            spring_terms[3][17],

            spring_terms[3][18],
            spring_terms[3][19],
            spring_terms[3][20],

            spring_terms[3][21],
            spring_terms[3][22],
            spring_terms[3][23],

            0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a2n1)*bd2n1,
            -.5*h*sinh(2.*a2n1)*cosh(b2n1)*cosh(b2n1)*gd2n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # bd2 update
            [spring_terms[4][0],
            spring_terms[4][1],
            spring_terms[4][2],

            h*ad2n1*bd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[4][3],
            -.5*h*cosh(2.*b2n1)*gd2n1*gd2n1 + spring_terms[4][4],
            0. + spring_terms[4][5],

            spring_terms[4][6],
            spring_terms[4][7],
            spring_terms[4][8],

            spring_terms[4][9],
            spring_terms[4][10],
            spring_terms[4][11],

            spring_terms[4][12],
            spring_terms[4][13],
            spring_terms[4][14],

            spring_terms[4][15],
            spring_terms[4][16],
            spring_terms[4][17],

            spring_terms[4][18],
            spring_terms[4][19],
            spring_terms[4][20],

            spring_terms[4][21],
            spring_terms[4][22],
            spring_terms[4][23],

            0.,0.,0.,

            h*tanh(a2n1)*bd2n1,
            1.+h*tanh(a2n1)*ad2n1,
            -.5*h*sinh(2.*b2n1)*gd2n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # gd2 update
            [spring_terms[5][0],
            spring_terms[5][1],
            spring_terms[5][2],

            h*ad2n1*gd2n1/(cosh(a2n1)*cosh(a2n1)) + spring_terms[5][3],
            h*bd2n1*gd2n1/(cosh(b2n1)*cosh(b2n1)) + spring_terms[5][4],
            0. + spring_terms[5][5],

            spring_terms[5][6],
            spring_terms[5][7],
            spring_terms[5][8],

            spring_terms[5][9],
            spring_terms[5][10],
            spring_terms[5][11],

            spring_terms[5][12],
            spring_terms[5][13],
            spring_terms[5][14],

            spring_terms[5][15],
            spring_terms[5][16],
            spring_terms[5][17],

            spring_terms[5][18],
            spring_terms[5][19],
            spring_terms[5][20],

            spring_terms[5][21],
            spring_terms[5][22],
            spring_terms[5][23],

            0.,0.,0.,

            h*tanh(a2n1)*gd2n1,
            h*tanh(b2n1)*gd2n1,
            1.+h*tanh(a2n1)*ad2n1+h*tanh(b2n1)*bd2n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # ad3 update
            [spring_terms[6][0],
            spring_terms[6][1],
            spring_terms[6][2],

            spring_terms[6][3],
            spring_terms[6][4],
            spring_terms[6][5],

            -.5*h*(bd3n1*bd3n1+cosh(b3n1)*cosh(b3n1)*gd3n1*gd3n1)*cosh(2.*a3n1) + spring_terms[6][6],
            -.25*h*sinh(2.*a3n1)*sinh(2.*b3n1)*gd3n1*gd3n1 + spring_terms[6][7],
            0. + spring_terms[6][8],

            spring_terms[6][9],
            spring_terms[6][10],
            spring_terms[6][11],

            spring_terms[6][12],
            spring_terms[6][13],
            spring_terms[6][14],

            spring_terms[6][15],
            spring_terms[6][16],
            spring_terms[6][17],

            spring_terms[6][18],
            spring_terms[6][19],
            spring_terms[6][20],

            spring_terms[6][21],
            spring_terms[6][22],
            spring_terms[6][23],

            0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a3n1)*bd3n1,
            -.5*h*sinh(2.*a3n1)*cosh(b3n1)*cosh(b3n1)*gd3n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # bd3 update
            [spring_terms[7][0],
            spring_terms[7][1],
            spring_terms[7][2],

            spring_terms[7][3],
            spring_terms[7][4],
            spring_terms[7][5],

            h*ad3n1*bd3n1/(cosh(a3n1)*cosh(a3n1)) + spring_terms[7][6],
            -.5*h*cosh(2.*b3n1)*gd3n1*gd3n1 + spring_terms[7][7],
            0. + spring_terms[7][8],

            spring_terms[7][9],
            spring_terms[7][10],
            spring_terms[7][11],

            spring_terms[7][12],
            spring_terms[7][13],
            spring_terms[7][14],

            spring_terms[7][15],
            spring_terms[7][16],
            spring_terms[7][17],

            spring_terms[7][18],
            spring_terms[7][19],
            spring_terms[7][20],

            spring_terms[7][21],
            spring_terms[7][22],
            spring_terms[7][23],

            0.,0.,0., 0.,0.,0.,

            h*tanh(a3n1)*bd3n1,
            1.+h*tanh(a3n1)*ad3n1,
            -.5*h*sinh(2.*b3n1)*gd3n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # gd3 update
            [spring_terms[8][0],
            spring_terms[8][1],
            spring_terms[8][2],

            spring_terms[8][3],
            spring_terms[8][4],
            spring_terms[8][5],

            h*ad3n1*gd3n1/(cosh(a3n1)*cosh(a3n1)) + spring_terms[8][6],
            h*bd3n1*gd3n1/(cosh(b3n1)*cosh(b3n1)) + spring_terms[8][7],
            0. + spring_terms[8][8],

            spring_terms[8][9],
            spring_terms[8][10],
            spring_terms[8][11],

            spring_terms[8][12],
            spring_terms[8][13],
            spring_terms[8][14],

            spring_terms[8][15],
            spring_terms[8][16],
            spring_terms[8][17],

            spring_terms[8][18],
            spring_terms[8][19],
            spring_terms[8][20],

            spring_terms[8][21],
            spring_terms[8][22],
            spring_terms[8][23],

            0.,0.,0., 0.,0.,0.,

            h*tanh(a3n1)*gd3n1,
            h*tanh(b3n1)*gd3n1,
            1.+h*tanh(a3n1)*ad3n1+h*tanh(b3n1)*bd3n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # ad4 update
            [spring_terms[9][0],
            spring_terms[9][1],
            spring_terms[9][2],

            spring_terms[9][3],
            spring_terms[9][4],
            spring_terms[9][5],

            spring_terms[9][6],
            spring_terms[9][7],
            spring_terms[9][8],

            -.5*h*(bd4n1*bd4n1+cosh(b4n1)*cosh(b4n1)*gd4n1*gd4n1)*cosh(2.*a4n1) + spring_terms[9][9],
            -.25*h*sinh(2.*a4n1)*sinh(2.*b4n1)*gd4n1*gd4n1 + spring_terms[9][10],
            0. + spring_terms[9][11],

            spring_terms[9][12],
            spring_terms[9][13],
            spring_terms[9][14],

            spring_terms[9][15],
            spring_terms[9][16],
            spring_terms[9][17],

            spring_terms[9][18],
            spring_terms[9][19],
            spring_terms[9][20],

            spring_terms[9][21],
            spring_terms[9][22],
            spring_terms[9][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a4n1)*bd4n1,
            -.5*h*sinh(2.*a4n1)*cosh(b4n1)*cosh(b4n1)*gd4n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # bd4 update
            [spring_terms[10][0],
            spring_terms[10][1],
            spring_terms[10][2],

            spring_terms[10][3],
            spring_terms[10][4],
            spring_terms[10][5],

            spring_terms[10][6],
            spring_terms[10][7],
            spring_terms[10][8],

            h*ad4n1*bd4n1/(cosh(a4n1)*cosh(a4n1)) + spring_terms[10][9],
            -.5*h*cosh(2.*b4n1)*gd4n1*gd4n1 + spring_terms[10][10],
            0. + spring_terms[10][11],

            spring_terms[10][12],
            spring_terms[10][13],
            spring_terms[10][14],

            spring_terms[10][15],
            spring_terms[10][16],
            spring_terms[10][17],

            spring_terms[10][18],
            spring_terms[10][19],
            spring_terms[10][20],

            spring_terms[10][21],
            spring_terms[10][22],
            spring_terms[10][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a4n1)*bd4n1,
            1.+h*tanh(a4n1)*ad4n1,
            -.5*h*sinh(2.*b4n1)*gd4n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # gd4 update
            [spring_terms[11][0],
            spring_terms[11][1],
            spring_terms[11][2],

            spring_terms[11][3],
            spring_terms[11][4],
            spring_terms[11][5],

            spring_terms[11][6],
            spring_terms[11][7],
            spring_terms[11][8],

            h*ad4n1*gd4n1/(cosh(a4n1)*cosh(a4n1)) + spring_terms[11][9],
            h*bd4n1*gd4n1/(cosh(b4n1)*cosh(b4n1)) + spring_terms[11][10],
            0. + spring_terms[11][11],

            spring_terms[11][12],
            spring_terms[11][13],
            spring_terms[11][14],

            spring_terms[11][15],
            spring_terms[11][16],
            spring_terms[11][17],

            spring_terms[11][18],
            spring_terms[11][19],
            spring_terms[11][20],

            spring_terms[11][21],
            spring_terms[11][22],
            spring_terms[11][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a4n1)*gd4n1,
            h*tanh(b4n1)*gd4n1,
            1.+h*tanh(a4n1)*ad4n1+h*tanh(b4n1)*bd4n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # ad5 update
            [spring_terms[12][0],
            spring_terms[12][1],
            spring_terms[12][2],

            spring_terms[12][3],
            spring_terms[12][4],
            spring_terms[12][5],

            spring_terms[12][6],
            spring_terms[12][7],
            spring_terms[12][8],

            spring_terms[12][9],
            spring_terms[12][10],
            spring_terms[12][11],

            -.5*h*(bd5n1*bd5n1+cosh(b5n1)*cosh(b5n1)*gd5n1*gd5n1)*cosh(2.*a5n1) + spring_terms[12][12],
            -.25*h*sinh(2.*a5n1)*sinh(2.*b5n1)*gd5n1*gd5n1 + spring_terms[12][13],
            0. + spring_terms[12][14],

            spring_terms[12][15],
            spring_terms[12][16],
            spring_terms[12][17],

            spring_terms[12][18],
            spring_terms[12][19],
            spring_terms[12][20],

            spring_terms[12][21],
            spring_terms[12][22],
            spring_terms[12][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a5n1)*bd5n1,
            -.5*h*sinh(2.*a5n1)*cosh(b5n1)*cosh(b5n1)*gd5n1,

            0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # bd5 update
            [spring_terms[13][0],
            spring_terms[13][1],
            spring_terms[13][2],

            spring_terms[13][3],
            spring_terms[13][4],
            spring_terms[13][5],

            spring_terms[13][6],
            spring_terms[13][7],
            spring_terms[13][8],

            spring_terms[13][9],
            spring_terms[13][10],
            spring_terms[13][11],

            h*ad5n1*bd5n1/(cosh(a5n1)*cosh(a5n1)) + spring_terms[13][12],
            -.5*h*cosh(2.*b5n1)*gd5n1*gd5n1 + spring_terms[13][13],
            0. + spring_terms[13][14],

            spring_terms[13][15],
            spring_terms[13][16],
            spring_terms[13][17],

            spring_terms[13][18],
            spring_terms[13][19],
            spring_terms[13][20],

            spring_terms[13][21],
            spring_terms[13][22],
            spring_terms[13][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a5n1)*bd5n1,
            1.+h*tanh(a5n1)*ad5n1,
            -.5*h*sinh(2.*b5n1)*gd5n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # gd5 update
            [spring_terms[14][0],
            spring_terms[14][1],
            spring_terms[14][2],

            spring_terms[14][3],
            spring_terms[14][4],
            spring_terms[14][5],

            spring_terms[14][6],
            spring_terms[14][7],
            spring_terms[14][8],

            spring_terms[14][9],
            spring_terms[14][10],
            spring_terms[14][11],

            h*ad5n1*gd5n1/(cosh(a5n1)*cosh(a5n1)) + spring_terms[14][12],
            h*bd5n1*gd5n1/(cosh(b5n1)*cosh(b5n1)) + spring_terms[14][13],
            0. + spring_terms[14][14],

            spring_terms[14][15],
            spring_terms[14][16],
            spring_terms[14][17],

            spring_terms[14][18],
            spring_terms[14][19],
            spring_terms[14][20],

            spring_terms[14][21],
            spring_terms[14][22],
            spring_terms[14][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a5n1)*gd5n1,
            h*tanh(b5n1)*gd5n1,
            1.+h*tanh(a5n1)*ad5n1+h*tanh(b5n1)*bd5n1,
            
            0.,0.,0., 0.,0.,0., 0.,0.,0.
            ],

            # ad6 update
            [spring_terms[15][0],
            spring_terms[15][1],
            spring_terms[15][2],

            spring_terms[15][3],
            spring_terms[15][4],
            spring_terms[15][5],

            spring_terms[15][6],
            spring_terms[15][7],
            spring_terms[15][8],

            spring_terms[15][9],
            spring_terms[15][10],
            spring_terms[15][11],

            spring_terms[15][12],
            spring_terms[15][13],
            spring_terms[15][14],

            -.5*h*(bd6n1*bd6n1+cosh(b6n1)*cosh(b6n1)*gd6n1*gd6n1)*cosh(2.*a6n1) + spring_terms[15][15],
            -.25*h*sinh(2.*a6n1)*sinh(2.*b6n1)*gd6n1*gd6n1 + spring_terms[15][16],
            0. + spring_terms[15][17],

            spring_terms[15][18],
            spring_terms[15][19],
            spring_terms[15][20],

            spring_terms[15][21],
            spring_terms[15][22],
            spring_terms[15][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a6n1)*bd6n1,
            -.5*h*sinh(2.*a6n1)*cosh(b6n1)*cosh(b6n1)*gd6n1,

            0.,0.,0., 0.,0.,0.
            ],

            # bd6 update
            [spring_terms[16][0],
            spring_terms[16][1],
            spring_terms[16][2],

            spring_terms[16][3],
            spring_terms[16][4],
            spring_terms[16][5],

            spring_terms[16][6],
            spring_terms[16][7],
            spring_terms[16][8],

            spring_terms[16][9],
            spring_terms[16][10],
            spring_terms[16][11],
            
            spring_terms[16][12],
            spring_terms[16][13],
            spring_terms[16][14],

            h*ad6n1*bd6n1/(cosh(a6n1)*cosh(a6n1)) + spring_terms[16][15],
            -.5*h*cosh(2.*b6n1)*gd6n1*gd6n1 + spring_terms[16][16],
            0. + spring_terms[16][17],

            spring_terms[16][18],
            spring_terms[16][19],
            spring_terms[16][20],

            spring_terms[16][21],
            spring_terms[16][22],
            spring_terms[16][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a6n1)*bd6n1,
            1.+h*tanh(a6n1)*ad6n1,
            -.5*h*sinh(2.*b6n1)*gd6n1,
            
            0.,0.,0., 0.,0.,0.
            ],

            # gd6 update
            [spring_terms[17][0],
            spring_terms[17][1],
            spring_terms[17][2],

            spring_terms[17][3],
            spring_terms[17][4],
            spring_terms[17][5],

            spring_terms[17][6],
            spring_terms[17][7],
            spring_terms[17][8],

            spring_terms[17][9],
            spring_terms[17][10],
            spring_terms[17][11],

            spring_terms[17][12],
            spring_terms[17][13],
            spring_terms[17][14],

            h*ad6n1*gd6n1/(cosh(a6n1)*cosh(a6n1)) + spring_terms[17][15],
            h*bd6n1*gd6n1/(cosh(b6n1)*cosh(b6n1)) + spring_terms[17][16],
            0. + spring_terms[17][17],

            spring_terms[17][18],
            spring_terms[17][19],
            spring_terms[17][20],

            spring_terms[17][21],
            spring_terms[17][22],
            spring_terms[17][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a6n1)*gd6n1,
            h*tanh(b6n1)*gd6n1,
            1.+h*tanh(a6n1)*ad6n1+h*tanh(b6n1)*bd6n1,
            
            0.,0.,0., 0.,0.,0.
            ],

            # ad7 update
            [spring_terms[18][0],
            spring_terms[18][1],
            spring_terms[18][2],

            spring_terms[18][3],
            spring_terms[18][4],
            spring_terms[18][5],

            spring_terms[18][6],
            spring_terms[18][7],
            spring_terms[18][8],

            spring_terms[18][9],
            spring_terms[18][10],
            spring_terms[18][11],

            spring_terms[18][12],
            spring_terms[18][13],
            spring_terms[18][14],

            spring_terms[18][15],
            spring_terms[18][16],
            spring_terms[18][17],

            -.5*h*(bd7n1*bd7n1+cosh(b7n1)*cosh(b7n1)*gd7n1*gd7n1)*cosh(2.*a7n1) + spring_terms[18][18],
            -.25*h*sinh(2.*a7n1)*sinh(2.*b7n1)*gd7n1*gd7n1 + spring_terms[18][19],
            0. + spring_terms[18][20],

            spring_terms[18][21],
            spring_terms[18][22],
            spring_terms[18][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a7n1)*bd7n1,
            -.5*h*sinh(2.*a7n1)*cosh(b7n1)*cosh(b7n1)*gd7n1,

            0.,0.,0.
            ],

            # bd7 update
            [spring_terms[19][0],
            spring_terms[19][1],
            spring_terms[19][2],

            spring_terms[19][3],
            spring_terms[19][4],
            spring_terms[19][5],

            spring_terms[19][6],
            spring_terms[19][7],
            spring_terms[19][8],

            spring_terms[19][9],
            spring_terms[19][10],
            spring_terms[19][11],
            
            spring_terms[19][12],
            spring_terms[19][13],
            spring_terms[19][14],

            spring_terms[19][15],
            spring_terms[19][16],
            spring_terms[19][17],

            h*ad7n1*bd7n1/(cosh(a7n1)*cosh(a7n1)) + spring_terms[19][18],
            -.5*h*cosh(2.*b7n1)*gd7n1*gd7n1 + spring_terms[19][19],
            0. + spring_terms[19][20],

            spring_terms[19][21],
            spring_terms[19][22],
            spring_terms[19][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a7n1)*bd7n1,
            1.+h*tanh(a7n1)*ad7n1,
            -.5*h*sinh(2.*b7n1)*gd7n1,
            
            0.,0.,0.
            ],

            # gd7 update
            [spring_terms[20][0],
            spring_terms[20][1],
            spring_terms[20][2],

            spring_terms[20][3],
            spring_terms[20][4],
            spring_terms[20][5],

            spring_terms[20][6],
            spring_terms[20][7],
            spring_terms[20][8],

            spring_terms[20][9],
            spring_terms[20][10],
            spring_terms[20][11],

            spring_terms[20][12],
            spring_terms[20][13],
            spring_terms[20][14],

            spring_terms[20][15],
            spring_terms[20][16],
            spring_terms[20][17],

            h*ad7n1*gd7n1/(cosh(a7n1)*cosh(a7n1)) + spring_terms[20][18],
            h*bd7n1*gd7n1/(cosh(b7n1)*cosh(b7n1)) + spring_terms[20][19],
            0. + spring_terms[20][20],

            spring_terms[20][21],
            spring_terms[20][22],
            spring_terms[20][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a7n1)*gd7n1,
            h*tanh(b7n1)*gd7n1,
            1.+h*tanh(a7n1)*ad7n1+h*tanh(b7n1)*bd7n1,
            
            0.,0.,0.
            ],

            # ad8 update
            [spring_terms[21][0],
            spring_terms[21][1],
            spring_terms[21][2],

            spring_terms[21][3],
            spring_terms[21][4],
            spring_terms[21][5],

            spring_terms[21][6],
            spring_terms[21][7],
            spring_terms[21][8],

            spring_terms[21][9],
            spring_terms[21][10],
            spring_terms[21][11],

            spring_terms[21][12],
            spring_terms[21][13],
            spring_terms[21][14],

            spring_terms[21][15],
            spring_terms[21][16],
            spring_terms[21][17],

            spring_terms[21][18],
            spring_terms[21][19],
            spring_terms[21][20],

            -.5*h*(bd8n1*bd8n1+cosh(b8n1)*cosh(b8n1)*gd8n1*gd8n1)*cosh(2.*a8n1) + spring_terms[21][21],
            -.25*h*sinh(2.*a8n1)*sinh(2.*b8n1)*gd8n1*gd8n1 + spring_terms[21][22],
            0. + spring_terms[21][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            1.,
            -.5*h*sinh(2.*a8n1)*bd8n1,
            -.5*h*sinh(2.*a8n1)*cosh(b8n1)*cosh(b8n1)*gd8n1
            ],

            # bd8 update
            [spring_terms[22][0],
            spring_terms[22][1],
            spring_terms[22][2],

            spring_terms[22][3],
            spring_terms[22][4],
            spring_terms[22][5],

            spring_terms[22][6],
            spring_terms[22][7],
            spring_terms[22][8],

            spring_terms[22][9],
            spring_terms[22][10],
            spring_terms[22][11],
            
            spring_terms[22][12],
            spring_terms[22][13],
            spring_terms[22][14],

            spring_terms[22][15],
            spring_terms[22][16],
            spring_terms[22][17],

            spring_terms[22][18],
            spring_terms[22][19],
            spring_terms[22][20],

            h*ad8n1*bd8n1/(cosh(a8n1)*cosh(a8n1)) + spring_terms[22][21],
            -.5*h*cosh(2.*b8n1)*gd8n1*gd8n1 + spring_terms[22][22],
            0. + spring_terms[22][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a8n1)*bd8n1,
            1.+h*tanh(a8n1)*ad8n1,
            -.5*h*sinh(2.*b8n1)*gd8n1
            ],

            # gd8 update
            [spring_terms[23][0],
            spring_terms[23][1],
            spring_terms[23][2],

            spring_terms[23][3],
            spring_terms[23][4],
            spring_terms[23][5],

            spring_terms[23][6],
            spring_terms[23][7],
            spring_terms[23][8],

            spring_terms[23][9],
            spring_terms[23][10],
            spring_terms[23][11],

            spring_terms[23][12],
            spring_terms[23][13],
            spring_terms[23][14],

            spring_terms[23][15],
            spring_terms[23][16],
            spring_terms[23][17],

            spring_terms[23][18],
            spring_terms[23][19],
            spring_terms[23][20],

            h*ad8n1*gd8n1/(cosh(a8n1)*cosh(a8n1)) + spring_terms[23][21],
            h*bd8n1*gd8n1/(cosh(b8n1)*cosh(b8n1)) + spring_terms[23][22],
            0. + spring_terms[23][23],

            0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0., 0.,0.,0.,

            h*tanh(a8n1)*gd8n1,
            h*tanh(b8n1)*gd8n1,
            1.+h*tanh(a8n1)*ad8n1+h*tanh(b8n1)*bd8n1
            ]
        ])

    # print(jacobian(pos1n[0], pos1n[1], pos1n[2], pos2n[0], pos2n[1], pos2n[2], vel1n[0], vel1n[1], vel1n[2], vel2n[0], vel2n[1], vel2n[2], m1, m2, step, sprcon, eqdist)[6:,:])
    diff1=linalg.solve(jacobian(posn_arr, veln_arr, mass_arr, step, spring_arr),-array([
        #p1
        con1(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con2(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        con3(posn_arr[0][0],posn_arr[0][0], posn_arr[0][1], posn_arr[0][1], posn_arr[0][2], posn_arr[0][2], veln_arr[0][0], veln_arr[0][0], veln_arr[0][1], veln_arr[0][1], veln_arr[0][2], veln_arr[0][2], step),
        #p2
        con1(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con2(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        con3(posn_arr[1][0],posn_arr[1][0], posn_arr[1][1], posn_arr[1][1], posn_arr[1][2], posn_arr[1][2], veln_arr[1][0], veln_arr[1][0], veln_arr[1][1], veln_arr[1][1], veln_arr[1][2], veln_arr[1][2], step),
        #p3
        con1(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con2(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        con3(posn_arr[2][0],posn_arr[2][0], posn_arr[2][1], posn_arr[2][1], posn_arr[2][2], posn_arr[2][2], veln_arr[2][0], veln_arr[2][0], veln_arr[2][1], veln_arr[2][1], veln_arr[2][2], veln_arr[2][2], step),
        #p4
        con1(posn_arr[3][0],posn_arr[3][0], posn_arr[3][1], posn_arr[3][1], posn_arr[3][2], posn_arr[3][2], veln_arr[3][0], veln_arr[3][0], veln_arr[3][1], veln_arr[3][1], veln_arr[3][2], veln_arr[3][2], step),
        con2(posn_arr[3][0],posn_arr[3][0], posn_arr[3][1], posn_arr[3][1], posn_arr[3][2], posn_arr[3][2], veln_arr[3][0], veln_arr[3][0], veln_arr[3][1], veln_arr[3][1], veln_arr[3][2], veln_arr[3][2], step),
        con3(posn_arr[3][0],posn_arr[3][0], posn_arr[3][1], posn_arr[3][1], posn_arr[3][2], posn_arr[3][2], veln_arr[3][0], veln_arr[3][0], veln_arr[3][1], veln_arr[3][1], veln_arr[3][2], veln_arr[3][2], step),
        #p5
        con1(posn_arr[4][0],posn_arr[4][0], posn_arr[4][1], posn_arr[4][1], posn_arr[4][2], posn_arr[4][2], veln_arr[4][0], veln_arr[4][0], veln_arr[4][1], veln_arr[4][1], veln_arr[4][2], veln_arr[4][2], step),
        con2(posn_arr[4][0],posn_arr[4][0], posn_arr[4][1], posn_arr[4][1], posn_arr[4][2], posn_arr[4][2], veln_arr[4][0], veln_arr[4][0], veln_arr[4][1], veln_arr[4][1], veln_arr[4][2], veln_arr[4][2], step),
        con3(posn_arr[4][0],posn_arr[4][0], posn_arr[4][1], posn_arr[4][1], posn_arr[4][2], posn_arr[4][2], veln_arr[4][0], veln_arr[4][0], veln_arr[4][1], veln_arr[4][1], veln_arr[4][2], veln_arr[4][2], step),
        #p6
        con1(posn_arr[5][0],posn_arr[5][0], posn_arr[5][1], posn_arr[5][1], posn_arr[5][2], posn_arr[5][2], veln_arr[5][0], veln_arr[5][0], veln_arr[5][1], veln_arr[5][1], veln_arr[5][2], veln_arr[5][2], step),
        con2(posn_arr[5][0],posn_arr[5][0], posn_arr[5][1], posn_arr[5][1], posn_arr[5][2], posn_arr[5][2], veln_arr[5][0], veln_arr[5][0], veln_arr[5][1], veln_arr[5][1], veln_arr[5][2], veln_arr[5][2], step),
        con3(posn_arr[5][0],posn_arr[5][0], posn_arr[5][1], posn_arr[5][1], posn_arr[5][2], posn_arr[5][2], veln_arr[5][0], veln_arr[5][0], veln_arr[5][1], veln_arr[5][1], veln_arr[5][2], veln_arr[5][2], step),
        #p7
        con1(posn_arr[6][0],posn_arr[6][0], posn_arr[6][1], posn_arr[6][1], posn_arr[6][2], posn_arr[6][2], veln_arr[6][0], veln_arr[6][0], veln_arr[6][1], veln_arr[6][1], veln_arr[6][2], veln_arr[6][2], step),
        con2(posn_arr[6][0],posn_arr[6][0], posn_arr[6][1], posn_arr[6][1], posn_arr[6][2], posn_arr[6][2], veln_arr[6][0], veln_arr[6][0], veln_arr[6][1], veln_arr[6][1], veln_arr[6][2], veln_arr[6][2], step),
        con3(posn_arr[6][0],posn_arr[6][0], posn_arr[6][1], posn_arr[6][1], posn_arr[6][2], posn_arr[6][2], veln_arr[6][0], veln_arr[6][0], veln_arr[6][1], veln_arr[6][1], veln_arr[6][2], veln_arr[6][2], step),
        #p8
        con1(posn_arr[7][0],posn_arr[7][0], posn_arr[7][1], posn_arr[7][1], posn_arr[7][2], posn_arr[7][2], veln_arr[7][0], veln_arr[7][0], veln_arr[7][1], veln_arr[7][1], veln_arr[7][2], veln_arr[7][2], step),
        con2(posn_arr[7][0],posn_arr[7][0], posn_arr[7][1], posn_arr[7][1], posn_arr[7][2], posn_arr[7][2], veln_arr[7][0], veln_arr[7][0], veln_arr[7][1], veln_arr[7][1], veln_arr[7][2], veln_arr[7][2], step),
        con3(posn_arr[7][0],posn_arr[7][0], posn_arr[7][1], posn_arr[7][1], posn_arr[7][2], posn_arr[7][2], veln_arr[7][0], veln_arr[7][0], veln_arr[7][1], veln_arr[7][1], veln_arr[7][2], veln_arr[7][2], step),

        #v1
        con4(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[4],posn_arr[4],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
        con5(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[4],posn_arr[4],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
        con6(posn_arr[0],posn_arr[0],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[4],posn_arr[4],veln_arr[0],veln_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
        #v2
        con4(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[3],posn_arr[3],posn_arr[5],posn_arr[5],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
        con5(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[3],posn_arr[3],posn_arr[5],posn_arr[5],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
        con6(posn_arr[1],posn_arr[1],posn_arr[0],posn_arr[0],posn_arr[3],posn_arr[3],posn_arr[5],posn_arr[5],veln_arr[1],veln_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
        #v3
        con4(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[3],posn_arr[3],posn_arr[6],posn_arr[6],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[5],spring_arr[6]),
        con5(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[3],posn_arr[3],posn_arr[6],posn_arr[6],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[5],spring_arr[6]),
        con6(posn_arr[2],posn_arr[2],posn_arr[0],posn_arr[0],posn_arr[3],posn_arr[3],posn_arr[6],posn_arr[6],veln_arr[2],veln_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[5],spring_arr[6]),
        #v4
        con4(posn_arr[3],posn_arr[3],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[7],posn_arr[7],veln_arr[3],veln_arr[3],mass_arr[3],step,spring_arr[3],spring_arr[5],spring_arr[7]),
        con5(posn_arr[3],posn_arr[3],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[7],posn_arr[7],veln_arr[3],veln_arr[3],mass_arr[3],step,spring_arr[3],spring_arr[5],spring_arr[7]),
        con6(posn_arr[3],posn_arr[3],posn_arr[1],posn_arr[1],posn_arr[2],posn_arr[2],posn_arr[7],posn_arr[7],veln_arr[3],veln_arr[3],mass_arr[3],step,spring_arr[3],spring_arr[5],spring_arr[7]),
        #v5
        con4(posn_arr[4],posn_arr[4],posn_arr[0],posn_arr[0],posn_arr[5],posn_arr[5],posn_arr[6],posn_arr[6],veln_arr[4],veln_arr[4],mass_arr[4],step,spring_arr[2],spring_arr[8],spring_arr[9]),
        con5(posn_arr[4],posn_arr[4],posn_arr[0],posn_arr[0],posn_arr[5],posn_arr[5],posn_arr[6],posn_arr[6],veln_arr[4],veln_arr[4],mass_arr[4],step,spring_arr[2],spring_arr[8],spring_arr[9]),
        con6(posn_arr[4],posn_arr[4],posn_arr[0],posn_arr[0],posn_arr[5],posn_arr[5],posn_arr[6],posn_arr[6],veln_arr[4],veln_arr[4],mass_arr[4],step,spring_arr[2],spring_arr[8],spring_arr[9]),
        #v6
        con4(posn_arr[5],posn_arr[5],posn_arr[1],posn_arr[1],posn_arr[4],posn_arr[4],posn_arr[7],posn_arr[7],veln_arr[5],veln_arr[5],mass_arr[5],step,spring_arr[4],spring_arr[8],spring_arr[10]),
        con5(posn_arr[5],posn_arr[5],posn_arr[1],posn_arr[1],posn_arr[4],posn_arr[4],posn_arr[7],posn_arr[7],veln_arr[5],veln_arr[5],mass_arr[5],step,spring_arr[4],spring_arr[8],spring_arr[10]),
        con6(posn_arr[5],posn_arr[5],posn_arr[1],posn_arr[1],posn_arr[4],posn_arr[4],posn_arr[7],posn_arr[7],veln_arr[5],veln_arr[5],mass_arr[5],step,spring_arr[4],spring_arr[8],spring_arr[10]),
        #v7
        con4(posn_arr[6],posn_arr[6],posn_arr[2],posn_arr[2],posn_arr[4],posn_arr[4],posn_arr[7],posn_arr[7],veln_arr[6],veln_arr[6],mass_arr[6],step,spring_arr[6],spring_arr[9],spring_arr[11]),
        con5(posn_arr[6],posn_arr[6],posn_arr[2],posn_arr[2],posn_arr[4],posn_arr[4],posn_arr[7],posn_arr[7],veln_arr[6],veln_arr[6],mass_arr[6],step,spring_arr[6],spring_arr[9],spring_arr[11]),
        con6(posn_arr[6],posn_arr[6],posn_arr[2],posn_arr[2],posn_arr[4],posn_arr[4],posn_arr[7],posn_arr[7],veln_arr[6],veln_arr[6],mass_arr[6],step,spring_arr[6],spring_arr[9],spring_arr[11]),
        #v8
        con4(posn_arr[7],posn_arr[7],posn_arr[3],posn_arr[3],posn_arr[5],posn_arr[5],posn_arr[6],posn_arr[6],veln_arr[7],veln_arr[7],mass_arr[7],step,spring_arr[7],spring_arr[10],spring_arr[11]),
        con5(posn_arr[7],posn_arr[7],posn_arr[3],posn_arr[3],posn_arr[5],posn_arr[5],posn_arr[6],posn_arr[6],veln_arr[7],veln_arr[7],mass_arr[7],step,spring_arr[7],spring_arr[10],spring_arr[11]),
        con6(posn_arr[7],posn_arr[7],posn_arr[3],posn_arr[3],posn_arr[5],posn_arr[5],posn_arr[6],posn_arr[6],veln_arr[7],veln_arr[7],mass_arr[7],step,spring_arr[7],spring_arr[10],spring_arr[11])       
    ]))
    val1 = array([
            posn_arr[0][0]+diff1[0], posn_arr[0][1]+diff1[1], posn_arr[0][2]+diff1[2], posn_arr[1][0]+diff1[3], posn_arr[1][1]+diff1[4], posn_arr[1][2]+diff1[5],
            posn_arr[2][0]+diff1[6], posn_arr[2][1]+diff1[7], posn_arr[2][2]+diff1[8], posn_arr[3][0]+diff1[9], posn_arr[3][1]+diff1[10], posn_arr[3][2]+diff1[11],
            posn_arr[4][0]+diff1[12], posn_arr[4][1]+diff1[13], posn_arr[4][2]+diff1[14], posn_arr[5][0]+diff1[15], posn_arr[5][1]+diff1[16], posn_arr[5][2]+diff1[17],
            posn_arr[6][0]+diff1[18], posn_arr[6][1]+diff1[19], posn_arr[6][2]+diff1[20], posn_arr[7][0]+diff1[21], posn_arr[7][1]+diff1[22], posn_arr[7][2]+diff1[23],

            veln_arr[0][0]+diff1[24], veln_arr[0][1]+diff1[25], veln_arr[0][2]+diff1[26], veln_arr[1][0]+diff1[27], veln_arr[1][1]+diff1[28], veln_arr[1][2]+diff1[29],
            veln_arr[2][0]+diff1[30], veln_arr[2][1]+diff1[31], veln_arr[2][2]+diff1[32], veln_arr[3][0]+diff1[33], veln_arr[3][1]+diff1[34], veln_arr[3][2]+diff1[35],
            veln_arr[4][0]+diff1[36], veln_arr[4][1]+diff1[37], veln_arr[4][2]+diff1[38], veln_arr[5][0]+diff1[39], veln_arr[5][1]+diff1[40], veln_arr[5][2]+diff1[41],
            veln_arr[6][0]+diff1[42], veln_arr[6][1]+diff1[43], veln_arr[6][2]+diff1[44], veln_arr[7][0]+diff1[45], veln_arr[7][1]+diff1[46], veln_arr[7][2]+diff1[47]
            ])    
    x = 0
    while(x < 7):
        new_pos_arr=array([val1[0:3],val1[3:6],val1[6:9],val1[9:12],val1[12:15],val1[15:18],val1[18:21],val1[21:24]])
        new_vel_arr=array([val1[24:27],val1[27:30],val1[30:33],val1[33:36],val1[36:39],val1[39:42],val1[42:45],val1[45:48]])
        diff2=linalg.solve(jacobian(new_pos_arr, new_vel_arr, mass_arr, step, spring_arr),-array([
            #p1
            con1(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con2(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            con3(posn_arr[0][0],new_pos_arr[0][0], posn_arr[0][1], new_pos_arr[0][1], posn_arr[0][2], new_pos_arr[0][2], veln_arr[0][0], new_vel_arr[0][0], veln_arr[0][1], new_vel_arr[0][1], veln_arr[0][2], new_vel_arr[0][2], step),
            #p2
            con1(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con2(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            con3(posn_arr[1][0],new_pos_arr[1][0], posn_arr[1][1], new_pos_arr[1][1], posn_arr[1][2], new_pos_arr[1][2], veln_arr[1][0], new_vel_arr[1][0], veln_arr[1][1], new_vel_arr[1][1], veln_arr[1][2], new_vel_arr[1][2], step),
            #p3
            con1(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con2(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            con3(posn_arr[2][0],new_pos_arr[2][0], posn_arr[2][1], new_pos_arr[2][1], posn_arr[2][2], new_pos_arr[2][2], veln_arr[2][0], new_vel_arr[2][0], veln_arr[2][1], new_vel_arr[2][1], veln_arr[2][2], new_vel_arr[2][2], step),
            #p4
            con1(posn_arr[3][0],new_pos_arr[3][0], posn_arr[3][1], new_pos_arr[3][1], posn_arr[3][2], new_pos_arr[3][2], veln_arr[3][0], new_vel_arr[3][0], veln_arr[3][1], new_vel_arr[3][1], veln_arr[3][2], new_vel_arr[3][2], step),
            con2(posn_arr[3][0],new_pos_arr[3][0], posn_arr[3][1], new_pos_arr[3][1], posn_arr[3][2], new_pos_arr[3][2], veln_arr[3][0], new_vel_arr[3][0], veln_arr[3][1], new_vel_arr[3][1], veln_arr[3][2], new_vel_arr[3][2], step),
            con3(posn_arr[3][0],new_pos_arr[3][0], posn_arr[3][1], new_pos_arr[3][1], posn_arr[3][2], new_pos_arr[3][2], veln_arr[3][0], new_vel_arr[3][0], veln_arr[3][1], new_vel_arr[3][1], veln_arr[3][2], new_vel_arr[3][2], step),
            #p5
            con1(posn_arr[4][0],new_pos_arr[4][0], posn_arr[4][1], new_pos_arr[4][1], posn_arr[4][2], new_pos_arr[4][2], veln_arr[4][0], new_vel_arr[4][0], veln_arr[4][1], new_vel_arr[4][1], veln_arr[4][2], new_vel_arr[4][2], step),
            con2(posn_arr[4][0],new_pos_arr[4][0], posn_arr[4][1], new_pos_arr[4][1], posn_arr[4][2], new_pos_arr[4][2], veln_arr[4][0], new_vel_arr[4][0], veln_arr[4][1], new_vel_arr[4][1], veln_arr[4][2], new_vel_arr[4][2], step),
            con3(posn_arr[4][0],new_pos_arr[4][0], posn_arr[4][1], new_pos_arr[4][1], posn_arr[4][2], new_pos_arr[4][2], veln_arr[4][0], new_vel_arr[4][0], veln_arr[4][1], new_vel_arr[4][1], veln_arr[4][2], new_vel_arr[4][2], step),
            #p6
            con1(posn_arr[5][0],new_pos_arr[5][0], posn_arr[5][1], new_pos_arr[5][1], posn_arr[5][2], new_pos_arr[5][2], veln_arr[5][0], new_vel_arr[5][0], veln_arr[5][1], new_vel_arr[5][1], veln_arr[5][2], new_vel_arr[5][2], step),
            con2(posn_arr[5][0],new_pos_arr[5][0], posn_arr[5][1], new_pos_arr[5][1], posn_arr[5][2], new_pos_arr[5][2], veln_arr[5][0], new_vel_arr[5][0], veln_arr[5][1], new_vel_arr[5][1], veln_arr[5][2], new_vel_arr[5][2], step),
            con3(posn_arr[5][0],new_pos_arr[5][0], posn_arr[5][1], new_pos_arr[5][1], posn_arr[5][2], new_pos_arr[5][2], veln_arr[5][0], new_vel_arr[5][0], veln_arr[5][1], new_vel_arr[5][1], veln_arr[5][2], new_vel_arr[5][2], step),
            #p7
            con1(posn_arr[6][0],new_pos_arr[6][0], posn_arr[6][1], new_pos_arr[6][1], posn_arr[6][2], new_pos_arr[6][2], veln_arr[6][0], new_vel_arr[6][0], veln_arr[6][1], new_vel_arr[6][1], veln_arr[6][2], new_vel_arr[6][2], step),
            con2(posn_arr[6][0],new_pos_arr[6][0], posn_arr[6][1], new_pos_arr[6][1], posn_arr[6][2], new_pos_arr[6][2], veln_arr[6][0], new_vel_arr[6][0], veln_arr[6][1], new_vel_arr[6][1], veln_arr[6][2], new_vel_arr[6][2], step),
            con3(posn_arr[6][0],new_pos_arr[6][0], posn_arr[6][1], new_pos_arr[6][1], posn_arr[6][2], new_pos_arr[6][2], veln_arr[6][0], new_vel_arr[6][0], veln_arr[6][1], new_vel_arr[6][1], veln_arr[6][2], new_vel_arr[6][2], step),
            #p8
            con1(posn_arr[7][0],new_pos_arr[7][0], posn_arr[7][1], new_pos_arr[7][1], posn_arr[7][2], new_pos_arr[7][2], veln_arr[7][0], new_vel_arr[7][0], veln_arr[7][1], new_vel_arr[7][1], veln_arr[7][2], new_vel_arr[7][2], step),
            con2(posn_arr[7][0],new_pos_arr[7][0], posn_arr[7][1], new_pos_arr[7][1], posn_arr[7][2], new_pos_arr[7][2], veln_arr[7][0], new_vel_arr[7][0], veln_arr[7][1], new_vel_arr[7][1], veln_arr[7][2], new_vel_arr[7][2], step),
            con3(posn_arr[7][0],new_pos_arr[7][0], posn_arr[7][1], new_pos_arr[7][1], posn_arr[7][2], new_pos_arr[7][2], veln_arr[7][0], new_vel_arr[7][0], veln_arr[7][1], new_vel_arr[7][1], veln_arr[7][2], new_vel_arr[7][2], step),

            #v1
            con4(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[4],new_pos_arr[4],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
            con5(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[4],new_pos_arr[4],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
            con6(posn_arr[0],new_pos_arr[0],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[4],new_pos_arr[4],veln_arr[0],new_vel_arr[0],mass_arr[0],step,spring_arr[0],spring_arr[1],spring_arr[2]),
            #v2
            con4(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[3],new_pos_arr[3],posn_arr[5],new_pos_arr[5],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
            con5(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[3],new_pos_arr[3],posn_arr[5],new_pos_arr[5],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
            con6(posn_arr[1],new_pos_arr[1],posn_arr[0],new_pos_arr[0],posn_arr[3],new_pos_arr[3],posn_arr[5],new_pos_arr[5],veln_arr[1],new_vel_arr[1],mass_arr[1],step,spring_arr[0],spring_arr[3],spring_arr[4]),
            #v3
            con4(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[3],new_pos_arr[3],posn_arr[6],new_pos_arr[6],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[5],spring_arr[6]),
            con5(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[3],new_pos_arr[3],posn_arr[6],new_pos_arr[6],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[5],spring_arr[6]),
            con6(posn_arr[2],new_pos_arr[2],posn_arr[0],new_pos_arr[0],posn_arr[3],new_pos_arr[3],posn_arr[6],new_pos_arr[6],veln_arr[2],new_vel_arr[2],mass_arr[2],step,spring_arr[1],spring_arr[5],spring_arr[6]),
            #v4
            con4(posn_arr[3],new_pos_arr[3],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[7],new_pos_arr[7],veln_arr[3],new_vel_arr[3],mass_arr[3],step,spring_arr[3],spring_arr[5],spring_arr[7]),
            con5(posn_arr[3],new_pos_arr[3],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[7],new_pos_arr[7],veln_arr[3],new_vel_arr[3],mass_arr[3],step,spring_arr[3],spring_arr[5],spring_arr[7]),
            con6(posn_arr[3],new_pos_arr[3],posn_arr[1],new_pos_arr[1],posn_arr[2],new_pos_arr[2],posn_arr[7],new_pos_arr[7],veln_arr[3],new_vel_arr[3],mass_arr[3],step,spring_arr[3],spring_arr[5],spring_arr[7]),
            #v5
            con4(posn_arr[4],new_pos_arr[4],posn_arr[0],new_pos_arr[0],posn_arr[5],new_pos_arr[5],posn_arr[6],new_pos_arr[6],veln_arr[4],new_vel_arr[4],mass_arr[4],step,spring_arr[2],spring_arr[8],spring_arr[9]),
            con5(posn_arr[4],new_pos_arr[4],posn_arr[0],new_pos_arr[0],posn_arr[5],new_pos_arr[5],posn_arr[6],new_pos_arr[6],veln_arr[4],new_vel_arr[4],mass_arr[4],step,spring_arr[2],spring_arr[8],spring_arr[9]),
            con6(posn_arr[4],new_pos_arr[4],posn_arr[0],new_pos_arr[0],posn_arr[5],new_pos_arr[5],posn_arr[6],new_pos_arr[6],veln_arr[4],new_vel_arr[4],mass_arr[4],step,spring_arr[2],spring_arr[8],spring_arr[9]),
            #v6
            con4(posn_arr[5],new_pos_arr[5],posn_arr[1],new_pos_arr[1],posn_arr[4],new_pos_arr[4],posn_arr[7],new_pos_arr[7],veln_arr[5],new_vel_arr[5],mass_arr[5],step,spring_arr[4],spring_arr[8],spring_arr[10]),
            con5(posn_arr[5],new_pos_arr[5],posn_arr[1],new_pos_arr[1],posn_arr[4],new_pos_arr[4],posn_arr[7],new_pos_arr[7],veln_arr[5],new_vel_arr[5],mass_arr[5],step,spring_arr[4],spring_arr[8],spring_arr[10]),
            con6(posn_arr[5],new_pos_arr[5],posn_arr[1],new_pos_arr[1],posn_arr[4],new_pos_arr[4],posn_arr[7],new_pos_arr[7],veln_arr[5],new_vel_arr[5],mass_arr[5],step,spring_arr[4],spring_arr[8],spring_arr[10]),
            #v7
            con4(posn_arr[6],new_pos_arr[6],posn_arr[2],new_pos_arr[2],posn_arr[4],new_pos_arr[4],posn_arr[7],new_pos_arr[7],veln_arr[6],new_vel_arr[6],mass_arr[6],step,spring_arr[6],spring_arr[9],spring_arr[11]),
            con5(posn_arr[6],new_pos_arr[6],posn_arr[2],new_pos_arr[2],posn_arr[4],new_pos_arr[4],posn_arr[7],new_pos_arr[7],veln_arr[6],new_vel_arr[6],mass_arr[6],step,spring_arr[6],spring_arr[9],spring_arr[11]),
            con6(posn_arr[6],new_pos_arr[6],posn_arr[2],new_pos_arr[2],posn_arr[4],new_pos_arr[4],posn_arr[7],new_pos_arr[7],veln_arr[6],new_vel_arr[6],mass_arr[6],step,spring_arr[6],spring_arr[9],spring_arr[11]),
            #v8
            con4(posn_arr[7],new_pos_arr[7],posn_arr[3],new_pos_arr[3],posn_arr[5],new_pos_arr[5],posn_arr[6],new_pos_arr[6],veln_arr[7],new_vel_arr[7],mass_arr[7],step,spring_arr[7],spring_arr[10],spring_arr[11]),
            con5(posn_arr[7],new_pos_arr[7],posn_arr[3],new_pos_arr[3],posn_arr[5],new_pos_arr[5],posn_arr[6],new_pos_arr[6],veln_arr[7],new_vel_arr[7],mass_arr[7],step,spring_arr[7],spring_arr[10],spring_arr[11]),
            con6(posn_arr[7],new_pos_arr[7],posn_arr[3],new_pos_arr[3],posn_arr[5],new_pos_arr[5],posn_arr[6],new_pos_arr[6],veln_arr[7],new_vel_arr[7],mass_arr[7],step,spring_arr[7],spring_arr[10],spring_arr[11])      
        ]))      
        val2 = array([
            val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5],
            val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11],
            val1[12]+diff2[12], val1[13]+diff2[13], val1[14]+diff2[14], val1[15]+diff2[15], val1[16]+diff2[16], val1[17]+diff2[17],
            val1[18]+diff2[18], val1[19]+diff2[19], val1[20]+diff2[20], val1[21]+diff2[21], val1[22]+diff2[22], val1[23]+diff2[23],

            val1[24]+diff2[24], val1[25]+diff2[25], val1[26]+diff2[26], val1[27]+diff2[27], val1[28]+diff2[28], val1[29]+diff2[29],
            val1[30]+diff2[30], val1[31]+diff2[31], val1[32]+diff2[32], val1[33]+diff2[33], val1[34]+diff2[34], val1[35]+diff2[35],
            val1[36]+diff2[36], val1[37]+diff2[37], val1[38]+diff2[38], val1[39]+diff2[39], val1[40]+diff2[40], val1[41]+diff2[41],
            val1[42]+diff2[42], val1[43]+diff2[43], val1[44]+diff2[44], val1[45]+diff2[45], val1[46]+diff2[46], val1[47]+diff2[47]
            ])
        val1 = val2
        x=x+1
    # print(val1)
    return val1






