import scipy.integrate as spi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

def diff_eqs_boomerang(t,Y):   
    INP = Y
    dY = np.zeros((9))
    dY[0] = (1/I3) * (Kz*L*(INP[1]**2+(L*INP[0])**2))
    dY[1] = -(lamb/m)*INP[1]
    dY[2] = -(1/(m * INP[1])) * ( Kz*L*(INP[1]**2+(L*INP[0])**2) + m*g) + (mu/I3)/INP[0]
    dY[3] = (1/(I3*INP[0]))*(-mu*INP[0]*np.sin(INP[6]))
    dY[4] = (1/(I3*INP[0]*np.sin(INP[3]))) * (mu*INP[0]*np.cos(INP[5]))
    dY[5] = -np.cos(INP[3])*INP[4]
    dY[6] = INP[1]*(-np.cos(INP[5])*np.cos(INP[4]) + np.sin(INP[5])*np.sin(INP[4])*np.cos(INP[3]))
    dY[7] = INP[1]*(-np.cos(INP[5])*np.sin(INP[4]) - np.sin(INP[5])*np.cos(INP[4])*np.cos(INP[3]))
    dY[8] = INP[1]*(-np.sin(INP[5])*np.sin(INP[3]))
    return dY   

def diff_eqs_pendulum(t,Y): 
    dY = np.zeros((3))
    dY[0] =  Y[1]
    dY[1] = -Y[0]
    dY[2] =  Y[0]*Y[1]
    return dY

def system(t,Y): 
    l1=Y[0]
    l2=Y[1]
    h1=Y[2]
    h2=Y[3]
    dl1=l2
    dl2=cosh(l1)*sinh(l1)*((((vf+2*vb*cosh(x/2.))-h2)**2.)/(((1.+2.*cosh(l1)*cosh(l1))**2.)) - 2.*k/m*(arccosh(cosh(l1)*cosh(h1))-x)/sqrt(cosh(l1)*cosh(l1)*cosh(h1)*cosh(h1)-1.) ) - k/m*(2.*l1-x)
    dh1=h2
    dh2=cosh(l1)*( 1. + ( 1./( 1. + (1./cosh(l1))**2.) ) )*( k/m*(arccosh(cosh(l1)*cosh(h1))-x)*(sinh(h1)/sqrt(cosh(l1)*cosh(l1)*cosh(h1)*cosh(h1)-1.)) + 2*(h2-(vf+2*vb*cosh(x/2.)))*(l2*sinh(l1)/((1.+2.*cosh(l1)*cosh(l1))**2.)) )

    return [dl1,dl2,dh1,dh2]


t_start, t_end = 0.0, 10.0

case = 'C'

x=1.
m=1.
k=1.
vf=1.
vb=1.


if case == 'A':         # pendulum
    Y = np.array([0.1, 1.0, 0.0]); 
    Yres = spi.solve_ivp(diff_eqs_pendulum, [t_start, t_end], Y, method='RK45', max_step=0.01)

if case == 'B':          # boomerang
    Y = np.array([omega0, V0, Psi0, theta0, phi0, psi0, X0, Y0, Z0])
    print('Y initial:'); print(Y); print()
    Yres = spi.solve_ivp(diff_eqs_boomerang, [t_start, t_end], Y, method='RK45', max_step=0.01)

if case == 'C':          # triangle
    Y = np.array([x/2.,0.,arccosh(cosh(x)/cosh(x/2.)),0.])
    print('Y initial:'); print(Y); print()
    Yres = spi.solve_ivp(system, [t_start, t_end], Y, method='RK45', max_step=0.01)

#---- graphics ---------------------
yy = pd.DataFrame(Yres.y).T
tt = np.linspace(t_start,t_end,yy.shape[0])
with plt.style.context('fivethirtyeight'): 
    plt.figure(1, figsize=(20,5))
    plt.plot(tt,yy,lw=8, alpha=0.5);
    plt.grid(axis='y')
    for j in range(3):
        plt.fill_between(tt,yy[j],0, alpha=0.2, label='y['+str(j)+']')
    plt.legend(prop={'size':20})
plt.show()