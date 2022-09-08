import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp

### Van der Pols System ###

# ### Which is given by the second order ODE
# ### x'' - mu ( 1 - x^2 ) x' + x = 0

# ### Break into system of first oder ODEs
# ### x' = v
# ### v' = mu ( 1 - x^2 ) v - x

# def vdp_derivatives(t, y):
#     x = y[0]
#     v = y[1]
#     return [v, mu*(1-x*x)*v-x]

# def jac_func(t, y):
#     x = y[0]
#     v = y[1]
#     return [
#         [0,1],
#         [-1-2*mu*x*v, mu*(1-x*x)]
#         ]
    

# mu=0
# t=np.linspace(0,10,500)

# sol=solve_ivp(fun=vdp_derivatives, t_span=[t[0],t[-1]], y0=[1,0], t_eval=t, method="Radau", jac=jac_func)

# fig=plt.figure(figsize=(10,5))

# plt.subplot(1,2,1)
# plt.plot(sol.y[0],sol.y[1])
# plt.xlabel("Position, x")
# plt.ylabel("Velocity, dx/dt")

# plt.subplot(1,2,2)
# plt.plot(sol.t,sol.y[1])
# plt.xlabel("Time, t")
# plt.ylabel("Position, x")

# fig.tight_layout()

# plt.show()


### Hyperbolic Geodesic in H2 (rotparam)

### Which is given by the system of second order ODEs
### a'' = b'^2 sinh(a) cosh(a)
### b'' = - a' b' coth(a)

### Break into system of first oder ODEs
### y0' = y2
### y1' = y3
### y2' = y3^2 sinh(y0) cosh(y0)
### y3' = - y2 y3 coth(y0)

def h2georot_derivatives(t, y):
    return [y[2], y[3], y[3]*y[3]*np.sinh(y[0])*np.cosh(y[0]), -y[2]*y[3]/np.tanh(y[0])]

def jac_func(t, y):
    return [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [y[3]*y[3]*np.cosh(2*y[0]), 0, 0, y[3]*np.sinh(2*y[0])],
        [y[2]*y[3]/(np.sinh(y[0])*np.sinh(y[0])), 0, -y[3]/np.tanh(y[0]), -y[2]/np.tanh(y[0])]
        ]
    

t=np.linspace(0,10,5000)

sol=solve_ivp(fun=h2georot_derivatives, t_span=[t[0],t[-1]], y0=[.5, 0., 0, .0], t_eval=t, method='Radau', jac=jac_func)

angle = np.linspace( 0 , 2 * np.pi , 150 ) 
 
radius = 1.
 
x = radius * np.cos( angle ) 
y = radius * np.sin( angle ) 

fig=plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(x,y)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.plot( np.sinh( sol.y[0] ) * np. cos( sol.y[1] ) / ( 1. + np.cosh(sol.y[0]) ), np.sinh( sol.y[0] )  * np. sin( sol.y[1] ) / ( 1. + np.cosh(sol.y[0]) ))
plt.xlabel("Position, x") 
plt.ylabel("Velocity, dx/dt")

plt.subplot(1,2,2)
plt.plot(sol.t,sol.y[1])
plt.xlabel("Time, t")
plt.ylabel("Position, x")

fig.tight_layout()

plt.show()