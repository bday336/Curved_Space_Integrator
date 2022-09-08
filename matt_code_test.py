import numpy as np
from symint_bank import imph2georot
from numpy import append,cosh,sinh
import matplotlib.pyplot as plt



def velocity( state ):
    # Assuming state = [r1 ... rn v1 ... vn], this function computes the value of the velocity
    r1=state[0:2]
    v1=state[2:4]
    
    v = np.concatenate( [v1, [v1[1]*v1[1]*np.sinh(r1[0])*np.cosh(r1[0]), -v1[0]*v1[1]/np.tanh(r1[0]) ] ] )
    return v


def dvelocity(state):
    # This function computes the Jacobian of the velocity
    # I'm calling it dvelocity for differential of velocity
    r1 = state[0:2]
    v1 = state[2:4]
    
    dv = np.zeros([4,4]) #store the jacobian in this
    
    #Do the simple momenta ones
    dv[0,2] = 1
    dv[1,3] = 1

    dv[2,0] = v1[1]*v1[1]*np.cosh(2*r1[0])
    dv[2,1] = 0.
    dv[2,2] = 0.
    dv[2,3] = 2.*v1[1]*np.sinh(r1[0])*np.cosh(r1[0])

    dv[3,0] = -v1[0]*v1[1]/(np.sinh(r1[0])*np.sinh(r1[0]))
    dv[3,1] = 0.
    dv[3,2] = -v1[1]/np.tanh(r1[0])
    dv[3,3] = -v1[0]/np.tanh(r1[0])

    return dv
   
    
    
def TSGLRK3(x, dt, min_error=1e-13):
    # This is a method I made up (but probably exists in the literature)
    # Time Symmetric Gauss-Legendre Runge Kutta with 3 stages
    # This method requires finding three velocities, but gives sixth order scaling,
    # with the bonus of being time-reversible
    # See this wikipedia article: https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_method#Time-Symmetric_Variants
    
    k1 = velocity(x)
    k2 = velocity(x)
    k3 = velocity(x)
    xf = x + dt*(5*k1 + 8*k2 + 5*k3)/18
    
    #three constants are used
    c15 = np.sqrt(15)/15*dt
    c30 = np.sqrt(15)/30*dt
    c24 = np.sqrt(15)/24*dt
    
    b  = np.zeros(16)      # have four 8D constraints to satisfy
    db = np.zeros([16,16]) # Jacobian of the above constraints
    I  = np.identity(4)
    while True:
        #x1 and x2 are your intermediate nodes for the Gaussian quadrature
        x1 = (x + xf)/2          - c15*k2 - c30*k3   
        x2 = (x + xf)/2 + c24*k1          - c24*k3
        x3 = (x + xf)/2 + c30*k1 + c15*k2
        
        #three implicit constraints of the method defined below
        b[0 : 4] = k1 - velocity( x1 )
        b[4 : 8] = k2 - velocity( x2 )
        b[8 :12] = k3 - velocity( x3 )
        b[12:16] = (xf - x)/dt - (5*k1 + 8*k2 + 5*k3)/18
        
        if( np.linalg.norm(b) < min_error ):
            break
        
        #If you made it this far, you need to converge. Compute the Jacobian
        j1 = dvelocity(x1)
        j2 = dvelocity(x2)
        j3 = dvelocity(x3)
        
        db[0:4, 0 : 4] = I       #derivative of first constraint w.r.t. k1
        db[0:4, 4 : 8] = c15*j1  #derivative of first constraint w.r.t. k2
        db[0:4, 8 :12] = c30*j1  #derivative of first constraint w.r.t. k3
        db[0:4, 12:16] = -j1/2   #derivative of first constraint w.r.t. xf
        
        db[4: 8, 0 : 4] = -c24*j2#derivative of second constraint w.r.t. k1
        db[4: 8, 4 : 8] = I      #derivative of second constraint w.r.t. k2
        db[4: 8, 8 :12] = c24*j2 #derivative of second constraint w.r.t. k3
        db[4: 8, 12:16] = -j2/2  #derivative of second constraint w.r.t. xf
        
        db[8 :12, 0 : 4] = -c30*j3#derivative of second constraint w.r.t. k1
        db[8 :12, 4 : 8] = -c15*j3#derivative of second constraint w.r.t. k2
        db[8 :12, 8 :12] = I      #derivative of second constraint w.r.t. k3
        db[8 :12, 12:16] = -j3/2  #derivative of second constraint w.r.t. xf

        
        db[12:16, 0 : 4] = -5*I/18  #derivative of fourth constraint w.r.t. k1
        db[12:16, 4 : 8] = -8*I/18  #derivative of fourth constraint w.r.t. k2
        db[12:16, 8 :12] = -5*I/18  #derivative of fourth constraint w.r.t. k3
        db[12:16, 12:16] =  I/dt    #derivative of fourth constraint w.r.t. xf
        
        #solve for the Newton step
        step = np.linalg.solve(db,b)
        step = step/2. #damp the step
        
        k1 = k1 - step[0 : 4]
        k2 = k2 - step[4 : 8]
        k3 = k3 - step[8 :12]
        xf = xf - step[12:16]
    
    # Once you have a satisfactory next step, we can solve for the Jacobian
    # dxf = \partial xf / \partial x  by differentiating the constraints w.r.t x
    # and solving the system of equations
    x1 = (x + xf)/2          - c15*k2 - c30*k3   
    x2 = (x + xf)/2 + c24*k1          - c24*k3
    x3 = (x + xf)/2 + c30*k1 + c15*k2
        
    j1 = dvelocity(x1)
    j2 = dvelocity(x2)
    j3 = dvelocity(x3)
    
    A  = np.zeros([16,16])
    b  = np.zeros([16,4 ])
    
    A[0:4, 0 : 4] = I       #derivative of first constraint w.r.t. k1
    A[0:4, 4 : 8] = c15*j1  #derivative of first constraint w.r.t. k2
    A[0:4, 8 :12] = c30*j1  #derivative of first constraint w.r.t. k3
    A[0:4, 12:16] = -j1/2   #derivative of first constraint w.r.t. xf
    
    A[4: 8, 0 : 4] = -c24*j2#derivative of second constraint w.r.t. k1
    A[4: 8, 4 : 8] = I      #derivative of second constraint w.r.t. k2
    A[4: 8, 8 :12] = c24*j2 #derivative of second constraint w.r.t. k3
    A[4: 8, 12:16] = -j2/2  #derivative of second constraint w.r.t. xf
    
    A[8 :12, 0 : 4] = -c30*j3#derivative of second constraint w.r.t. k1
    A[8 :12, 4 : 8] = -c15*j3#derivative of second constraint w.r.t. k2
    A[8 :12, 8 :12] = I      #derivative of second constraint w.r.t. k3
    A[8 :12, 12:16] = -j3/2  #derivative of second constraint w.r.t. xf

    
    A[12:16, 0 : 4] = -5*I/18  #derivative of fourth constraint w.r.t. k1
    A[12:16, 4 : 8] = -8*I/18  #derivative of fourth constraint w.r.t. k2
    A[12:16, 8 :12] = -5*I/18  #derivative of fourth constraint w.r.t. k3
    A[12:16, 12:16] =  I/dt    #derivative of fourth constraint w.r.t. xf
    
    b[0:4]   = j1/2
    b[4:8]  = j2/2
    b[8:12] = j3/2
    b[12:16] = I/dt
    
    b = np.linalg.solve(A,b)
    dxf = b[12:16, :] #the part of the Jacobian we care about is at the end
    
    #return the next position, and the Jacobian
    return xf, dxf

xm = np.array([
        .5, #a0
        np.pi/2., #b0
        0.,            #ad0
        -0.5             #bd0
        ])

xs = np.array([
        .5, #a0
        np.pi/2., #b0
        0.,            #ad0
        -0.5             #bd0
        ])

data_m=[]
data_s=[]
q=0
data_m.append(xm)
data_s.append(xs)
while(q<1000):
    xf,_=TSGLRK3(xm, .01, min_error=1e-13)
    stepdata=imph2georot(xs[0:2],xs[2:4],.01)
    data_m.append(xf)
    data_s.append(stepdata)
    xm=xf
    xs=stepdata
    q=q+1

# Transform into Poincare disk model for plotting
gut=[]
gvt=[]
for a in range(len(data_m)):
    gut.append(sinh(data_m[a][0])*np.cos(data_m[a][1])/(cosh(data_m[a][0]) + 1.))
    gvt.append(sinh(data_m[a][0])*np.sin(data_m[a][1])/(cosh(data_m[a][0]) + 1.)) 
    gut.append(sinh(data_s[a][0])*np.cos(data_s[a][1])/(cosh(data_s[a][0]) + 1.))
    gvt.append(sinh(data_s[a][0])*np.sin(data_s[a][1])/(cosh(data_s[a][0]) + 1.))    	     		

#This is for the horizon circle.
# Theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# Compute x1 and x2 for horizon
xc = np.cos(theta)
yc = np.sin(theta)  

#This is the particle trajectories in the Poincare model
    
#Plot
plt.figure(figsize=(5,5))

plt.plot(xc,yc)
plt.plot(gut[0::2],gvt[0::2],label="matt")
plt.plot(gut[1::2],gvt[1::2],label="solver")
plt.legend(loc='lower left')	

plt.show()
    
