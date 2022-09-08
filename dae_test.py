from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arccos,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,linalg,add

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

def trapcollocorder2(posn, veln, step):
    
    def con1(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h):
        return an1 - an - h*adn - h*h/6.*(2.*bdn*bdn*sinh(an)*cosh(an) + bdn1*bdn1*sinh(an1)*cosh(an1))

    def con2(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bn1 - bn - h*bdn - h*h/6.*(-2.*2.*adn*bdn/tanh(an) - 2.*adn1*bdn1/tanh(an1))

    def con3(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return adn1 - adn - .5*h*(bdn*bdn*sinh(an)*cosh(an) + bdn1*bdn1*sinh(an1)*cosh(an1))

    def con4(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bdn1 - bdn - .5*h*(-2.*adn*bdn/tanh(an) - 2.*adn1*bdn1/tanh(an1))

    def jacobian(an1, bn1, adn1, bdn1, h):
        return array([
                    [1.-h*h/6.*bdn1*bdn1*cosh(2*an1),0.,0.,-h*h/6.*bdn1*sinh(2.*an1)],
                    [-h*h/3.*adn1*bdn1/(sinh(an1)**2.),1.,h*h/3./tanh(an1)*bdn1,h*h/3./tanh(an1)*adn1],
                    [-.5*h*bdn1*bdn1*cosh(2*an1),0.,1.,-h*sinh(an1)*cosh(an1)*bdn1],
                    [-h*adn1*bdn1/(sinh(an1)*sinh(an1)),0.,h/tanh(an1)*bdn1,1.+h/tanh(an1)*adn1]
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

def testtranport(posn, veln, step):
    
    def con1(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h):
        return an1 - an - .5*h*( (adn - bdn*h*cosh(an)*sin(bn-bn1)*sinh(an1)) + adn1)

    def con2(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bn1 - bn - .5*h*( ( bdn + adn*h/tanh(an)/sinh(an)*sin(bn-bn1)*sinh(an1) - bdn*h*cosh(an)*( cos(bn-bn1)*cosh(an)*sinh(an1) - cosh(an1)*sinh(an) )/sqrt(cosh(an)*cosh(an) - 1.) ) + bdn1)

    def con3(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return adn1 - adn - .5*h*( ( bdn*bdn*cosh(an)*sinh(an) + 2.*adn*bdn*h*cosh(an)/tanh(an)*sin(bn - bn1)*sinh(an1) ) + bdn1*bdn1*cosh(an1)*sinh(an1))

    def con4(an, an1, bn, bn1, adn, adn1, bdn, bdn1, h): 
        return bdn1 - bdn - .5*h*( ( -2.*adn*bdn/tanh(an) + bdn*bdn*h*cosh(an)/tanh(an)*sin(bn - bn1)*sinh(an1) + (2.*adn*bdn*h*cosh(an)/tanh(an)*( -cosh(an1)*sinh(an) + cos(bn - bn1)*cosh(an)*sinh(an1) ))/ sqrt( cosh(an)*cosh(an) - 1. ) ) - 2.*adn1*bdn1/tanh(an1))

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










def interp_segmentq(xk, xdk, xdk1):
    interp_data=[]
    h=.1
    N=1000
    t_arr=np.linspace(0.,h, N)
    for a in t_arr:
        interp_data.append(xk + xdk * a + a * a / 2. * h * (xdk1 - xdk))

    return interp_data

def interp_segmentqd(xk, xdk, xdk1):
    interp_data=[]
    h=.1
    N=1000
    t_arr=np.linspace(0.,h, N)
    for a in t_arr:
        interp_data.append(xdk +  a / h * (xdk1 - xdk))

    return interp_data

def interp_qdminusvo1(xk,xk1,vk,vk1,h):
    # print(xk,xk1,vk,vk1)
    vdk=[
        vk[1]*vk[1]*cosh(xk[0])*sinh(xk[0]),
        -2.*vk[0]*vk[1]/tanh(xk[0])
        ]
    vdk1=[
        vk1[1]*vk1[1]*cosh(xk1[0])*sinh(xk1[0]),
        -2.*vk1[0]*vk1[1]/tanh(xk1[0])
        ]
    # print(xk,xk1,vk,vk1,vdk,vdk1)
    interp_dataa=[]
    interp_datab=[]
    # h=.1
    N=1000
    t_arr=np.linspace(0.,h, N)
    for a in t_arr:
        interp_dataa.append(vk[0] +  a / h * (vk1[0] - vk[0]) - ( vk[0] + vdk[0] * a + a * a / (2. * h) * (vdk1[0] - vdk[0]) ))
        interp_datab.append(vk[1] +  a / h * (vk1[1] - vk[1]) - ( vk[1] + vdk[1] * a + a * a / (2. * h) * (vdk1[1] - vdk[1]) ))
    return interp_dataa,interp_datab

def interp_qddminusao1(xk,xk1,vk,vk1,h):
    # print(xk,xk1,vk,vk1,vdk,vdk1)
    interp_dataa=[]
    interp_datab=[]
    # h=.1
    N=1000
    t_arr=np.linspace(0.,h, N)
    for a in t_arr:
        # print(vdk * a)
        q = xk + vk * a + a * a / (2. * h) * (vk1 - vk)
        qd = vk + a / h * (vk1 - vk)
        interp_dataa.append(1. / h * (vk1[0] - vk[0]) - (qd[1] * qd[1] * .5 * sinh(2. * q[0])) )
        interp_datab.append(1. / h * (vk1[1] - vk[1]) - (-2. * qd[0] * qd[1] / tanh(q[0])) )
    return interp_dataa,interp_datab

def interp_qddminusao2(xk,xk1,vk,vk1,h):
    # print(xk,xk1,vk,vk1)
    vdk=np.array([
        vk[1]*vk[1]*cosh(xk[0])*sinh(xk[0]),
        -2.*vk[0]*vk[1]/tanh(xk[0])
        ])
    vdk1=np.array([
        vk1[1]*vk1[1]*cosh(xk1[0])*sinh(xk1[0]),
        -2.*vk1[0]*vk1[1]/tanh(xk1[0])
        ])
    # print(xk,xk1,vk,vk1,vdk,vdk1)
    interp_dataa=[]
    interp_datab=[]
    # h=.1
    N=1000
    t_arr=np.linspace(0.,h, N)
    for a in t_arr:
        # print(vdk * a)
        q = xk + vk * a + vdk * a * a / 2. + a * a * a / (6. * h) * (vdk1 - vdk)
        qd = vk + vdk * a + a * a / (2. * h) * (vdk1 - vdk)
        interp_dataa.append(vdk[0] +  a / h * (vdk1[0] - vdk[0]) - (qd[1] * qd[1] * .5 * sinh(2. * q[0])) )
        interp_datab.append(vdk[1] +  a / h * (vdk1[1] - vdk[1]) - (-2. * qd[0] * qd[1] / tanh(q[0])) )
    return interp_dataa,interp_datab

        


#Initialize the particles in the simulation
particles=array([
    [.5,0.*pi/2.,
    0.,1.]        #particle 1
    ])

#Intialize the time stepping for the integrator.
delT=.01
maxT=1+delT

nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
data1=[]
data2=[]

# Position in translational parameterization then rotational
positions = array([particles[0][:2]])
# Velocity given in translational parameterization then rotational
velocities = array([particles[0][2:4]])

data1.append(np.concatenate((positions,velocities)))
data2.append(np.concatenate((positions,velocities)))

# Numerical Integration step
step_data=array([
    imph2georot(positions[0], velocities[0], delT),
	trapcollocorder2(positions[0], velocities[0], delT)
    
	])

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump):
    nextpos1 = array([step_data[0][:2]])
    nextdot1 = array([step_data[0][2:4]])

    nextpos2 = array([step_data[1][:2]])
    nextdot2 = array([step_data[1][2:4]])

    data1.append(np.concatenate((nextpos1,nextdot1)))
    data2.append(np.concatenate((nextpos2,nextdot2)))

    step_data=array([
        #imprk4h2geotrans(nextpos[0], nextdot[0], delT) 
        imph2georot(nextpos1[0], nextdot1[0], delT),
        trapcollocorder2(nextpos2[0], nextdot2[0], delT)
        ])


    q=q+1

a1=np.array([])
b1=np.array([])
a2=np.array([])
b2=np.array([])
for a in data1:
    a1=np.append(a1,a[0][0])
    b1=np.append(b1,a[0][1])

for b in data2:
    a2=np.append(a2,b[0][0])
    b2=np.append(b2,b[0][1])

#This is for the horizon circle.
# Theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)

# Compute x1 and x2 for horizon
xc = np.cos(theta)
yc = np.sin(theta)  

#Plot
# plt.figure(figsize=(5,5))


# plt.plot(xc,yc)
# plt.plot(sinh(a1)*cos(b1)/(cosh(a1)+1.),sinh(a1)*sin(b1)/(cosh(a1)+1.),label="my solver")
# plt.plot(sinh(a2)*cos(b2)/(cosh(a2)+1.),sinh(a2)*sin(b2)/(cosh(a2)+1.),label="order 2")
# plt.legend(loc='lower left')	

# plt.show()

# knot_list=[]
# for a in range(int(nump)):
#     knot_list.append([a/nump,0])

# knot_list=np.array(knot_list)

# knot_listx=np.linspace(0.,maxT-delT,int(1/delT))
# knot_listy=np.zeros(timearr.shape)


t_arr=np.linspace(0.,delT, 1000)

diffa1_dat=[] # o1 first d error
diffb1_dat=[] # o1 first d error
diffa2_dat=[] # o1 second d error
diffb2_dat=[] # o1 second d error
diffa3_dat=[] # o2 second d error
diffb3_dat=[] # o2 second d error


for a in range(len(data1)-1):
    diffa1_temp,diffb1_temp=interp_qdminusvo1(data1[a][0],data1[a+1][0],data1[a][1],data1[a+1][1],delT) 
    diffa1_dat.append(diffa1_temp)
    diffb1_dat.append(diffb1_temp)

for a in range(len(data1)-1):
    diffa2_temp,diffb2_temp=interp_qddminusao1(data1[a][0],data1[a+1][0],data1[a][1],data1[a+1][1],delT) 
    diffa2_dat.append(diffa2_temp)
    diffb2_dat.append(diffb2_temp)

for a in range(len(data2)-1):
    diffa3_temp,diffb3_temp=interp_qddminusao2(data2[a][0],data2[a+1][0],data2[a][1],data2[a+1][1],delT) 
    diffa3_dat.append(diffa3_temp)
    diffb3_dat.append(diffb3_temp)



# diffa2,diffb2=interp_qdminusv(data2[0][0],data2[1][0],data2[0][1],data2[1][1]) 

# Plot Trajectory with error
fig , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,8))

ax1.scatter(timearr, np.zeros(timearr.shape), c="red", s=10)
ax1.plot(t_arr+timearr[0],diffa1_dat[0],label="a_dat",color="Black")
ax1.plot(t_arr+timearr[0],diffb1_dat[0],label="b_dat",color="Blue")
for b in np.arange(1,len(timearr)-1):
    ax1.plot(t_arr+timearr[b],diffa1_dat[b],color="Black")
    ax1.plot(t_arr+timearr[b],diffb1_dat[b],color="Blue")
ax1.set_title("Error in first derivative for o1")
ax1.legend(loc='upper right')	

ax2.scatter(timearr, np.zeros(timearr.shape), c="red", s=10)
ax2.plot(t_arr+timearr[0],diffa2_dat[0],label="a_dat",color="Black")
ax2.plot(t_arr+timearr[0],diffb2_dat[0],label="b_dat",color="Blue")
for b in np.arange(1,len(timearr)-1):
    ax2.plot(t_arr+timearr[b],diffa2_dat[b],color="Black")
    ax2.plot(t_arr+timearr[b],diffb2_dat[b],color="Blue")
ax2.set_title("Error in second derivative for o1")
ax2.legend(loc='upper right')

ax3.scatter(timearr, np.zeros(timearr.shape), c="red", s=10)
ax3.plot(t_arr+timearr[0],np.zeros(t_arr.shape),label="a_dat",color="Black")
ax3.plot(t_arr+timearr[0],np.zeros(t_arr.shape),label="b_dat",color="Blue")
for b in np.arange(1,len(timearr)-1):
    ax3.plot(t_arr+timearr[b],np.zeros(t_arr.shape),color="Black")
    ax3.plot(t_arr+timearr[b],np.zeros(t_arr.shape),color="Blue")
ax3.set_title("Error in first derivative for o2")
ax3.legend(loc='upper right')	

ax4.scatter(timearr, np.zeros(timearr.shape), c="red", s=10)
ax4.plot(t_arr+timearr[0],diffa3_dat[0],label="a_dat",color="Black")
ax4.plot(t_arr+timearr[0],diffb3_dat[0],label="b_dat",color="Blue")
for b in np.arange(1,len(timearr)-1):
    ax4.plot(t_arr+timearr[b],diffa3_dat[b],color="Black")
    ax4.plot(t_arr+timearr[b],diffb3_dat[b],color="Blue")
ax4.set_title("Error in second derivative for o2")
ax4.legend(loc='upper right')


fig.tight_layout()	

plt.show()