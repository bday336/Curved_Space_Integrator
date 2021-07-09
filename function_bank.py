import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append

#########

#Function Bank

#########

# Convert from hyperboloid model to poincare model

def hyper2poin2d(point): 
	return array([point[0]/(point[2] + 1.), point[1]/(point[2] + 1.)])

def hyper2poin3d(point): 
	return array([point[0]/(point[3] + 1.), point[1]/(point[3] + 1.), point[2]/(point[3] + 1.)])

# Convert position to be used with SO(n,1) matrices

def formatvec2d(point): 
	return array([point[2], point[0], point[1]])

def unformatvec2d(point): 
	return array([point[1], point[2], point[0]])

def formatvec3d(point): 
	return array([point[3], point[0], point[1], point[2]])

def unformatvec3d(point): 
	return array([point[1], point[2], point[3], point[0]])

# SO(n,1) matrices

# SO(2,1)

def boostx2d(u): 
	return array([
	   [cosh(u), sinh(u), 0.],
	   [sinh(u), cosh(u), 0.],
	   [0., 0., 1.]
		])

def boosty2d(u): 
	return array([
	   [cosh(u), 0., sinh(u)],
	   [0., 1., 0.],
	   [sinh(u), 0., cosh(u)]
		])

def rotz2d(u): 
	return array([
	   [1., 0., 0.],
	   [0., cos(u), -sin(u)],
	   [0., sin(u), cos(u)]
		])

# SO(3,1)   		

def boostx3d(u): 
	return array([
	   [cosh(u), sinh(u), 0., 0.],
	   [sinh(u), cosh(u), 0., 0.],
	   [0., 0., 1., 0.],
	   [0., 0., 0., 1.]
		])

def boosty3d(u): 
	return array([
	   [cosh(u), 0., sinh(u), 0.],
	   [0., 1., 0., 0.],
	   [sinh(u), 0., cosh(u), 0.],
	   [0., 0., 0., 1.]
		])

def boostz3d(u): 
	return array([
	   [cosh(u), 0., 0., sinh(u)],
	   [0., 1., 0., 0.],
	   [0., 0., 1., 0.],
	   [sinh(u), 0., 0., cosh(u)]
		])

def rotx3d(u): 
	return array([
	   [1., 0., 0., 0.],
	   [0., 1., 0., 0.],
	   [0., 0., cos(u), -sin(u)],
	   [0., 0., sin(u), cos(u)]
		])

def roty3d(u): 
	return array([
	   [1., 0., 0., 0.],
	   [0., cos(u), 0., sin(u)],
	   [0., 0., 1., 0.],
	   [0., -sin(u), 0., cos(u)]
		])

def rotz3d(u): 
	return array([
	   [1., 0., 0., 0.],
	   [0., cos(u), -sin(u), 0.],
	   [0., sin(u), cos(u), 0.],
	   [0., 0., 0., 1.]
		])

'''
These are the matrix exponentials of Lie algebra generators for \
translations.

Translation generators for SO(2,1)
{
{0,v1*t,v2*t},
{v1*t,0,0},
{v2*t,0,0}
}

Translation generators for SO(3,1)
{
{0,v1*t,v2*t,v3*t},
{v1*t,0,0,0},
{v2*t,0,0,0},
{v3*t,0,0,0}
}
'''
def motionmat2dh(v1, v2, t): 
	return array([
		[cosh(t*sqrt(v1**2. + v2**2.)), (v1*sinh(t*sqrt(v1**2. + v2**2.)))/sqrt(v1**2. + v2**2.), (v2*sinh(t*sqrt(v1**2. + v2**2.)))/sqrt(v1**2. + v2**2.)], 
			[(v1*sinh(t*sqrt(v1**2. + v2**2.)))/sqrt(v1**2. + v2**2.), (v2**2. + v1**2.*cosh(t*sqrt(v1**2. + v2**2.)))/(v1**2. + v2**2.), (v1*v2*(-1. + cosh(t*sqrt(v1**2. + v2**2.))))/(v1**2. + v2**2.)], 
			[(v2*sinh(t*sqrt(v1**2. + v2**2.)))/sqrt(v1**2. + v2**2.), (v1*v2*(-1. + cosh(t*sqrt(v1**2. + v2**2.))))/(v1**2. + v2**2.), (v1**2. + v2**2.*cosh(t*sqrt(v1**2. + v2**2.)))/(v1**2. + v2**2.)]
		])

def motionmat3dh(v1, v2, v3, t): 
	return array([
		[cosh(t*sqrt(v1**2. + v2**2. + v3**2.)), (v1*sinh(t*sqrt(v1**2. + v2**2. + v3**2.)))/sqrt(v1**2. + v2**2. + v3**2.), (v2*sinh(t*sqrt(v1**2. + v2**2. + v3**2.)))/sqrt(v1**2. + v2**2. + v3**2.), (v3*sinh(t*sqrt(v1**2. + v2**2. + v3**2.)))/sqrt(v1**2. + v2**2. + v3**2.)], 
			[(v1*sinh(t*sqrt(v1**2. + v2**2. + v3**2.)))/sqrt(v1**2. + v2**2. + v3**2.), (v2**2. + v3**2. + v1**2.*cosh(t*sqrt(v1**2. + v2**2. + v3**2.)))/(v1**2. + v2**2. + v3**2.), (v1*v2*(-1. + cosh(t*sqrt(v1**2. + v2**2. + v3**2.))))/(v1**2. + v2**2. + v3**2.), (v1*v3*(-1. + cosh(t*sqrt(v1**2. + v2**2. + v3**2.))))/(v1**2. + v2**2. + v3**2.)], 
			[(v2*sinh(t*sqrt(v1**2. + v2**2. + v3**2.)))/sqrt(v1**2. + v2**2. + v3**2.), (v1*v2*(-1. + cosh(t*sqrt(v1**2. + v2**2. + v3**2.))))/(v1**2. + v2**2. + v3**2.), (v1**2. + v3**2. + v2**2.*cosh(t*sqrt(v1**2. + v2**2. + v3**2.)))/(v1**2. + v2**2. + v3**2.), (v2*v3*(-1. + cosh(t*sqrt(v1**2. + v2**2. + v3**2.))))/(v1**2. + v2**2. + v3**2.)], 
			[(v3*sinh(t*sqrt(v1**2. + v2**2. + v3**2.)))/sqrt(v1**2. + v2**2. + v3**2.), (v1*v3*(-1. + cosh(t*sqrt(v1**2. + v2**2. + v3**2.))))/(v1**2. + v2**2. + v3**2.), (v2*v3*(-1. + cosh(t*sqrt(v1**2. + v2**2. + v3**2.))))/(v1**2. + v2**2. + v3**2.), (v1**2. + v2**2. + v3**2.*cosh(t*sqrt(v1**2. + v2**2. + v3**2.)))/(v1**2. + v2**2. + v3**2.)]
		])


#Functions for python integration algorithms

def py2dhfreetrans(indVars, t):
	u, pu, v, pv = indVars
	return [pu, (pv**2.)*sinh(u)*cosh(u), pv, -2.*pu*pv*tanh(u)]

def xfunc2dhtrans(u,v):
	return sinh(u)
def yfunc2dhtrans(u,v):
	return cosh(u)*sinh(v)
def zfunc2dhtrans(u,v):
	return cosh(u)*cosh(v)	

def py3dhfreetrans(indVars, t):
	u, pu, v, pv, b, pb = indVars
	return [pu, (cosh(v)*cosh(v)*(pb**2.)+(pv**2.))*sinh(u)*cosh(u), pv, sinh(v)*cosh(v)*(pb**2.)-2.*pu*pv*tanh(u), pb, -2.*pu*pb*tanh(u)-2.*pb*pv*tanh(v)]

def xfunc3dhtrans(u,v,b):
	return sinh(u)
def yfunc3dhtrans(u,v,b):
	return cosh(u)*sinh(v)
def zfunc3dhtrans(u,v,b):
	return cosh(u)*cosh(v)*sinh(b)	
def wfunc3dhtrans(u,v,b):
	return cosh(u)*cosh(v)*cosh(b)	

def py3dhfreerot(indVars, t):
	u, pu, v, pv, b, pb = indVars
	return [pu, (sin(v)*sin(v)*(pb**2.)+(pv**2.))*sinh(u)*cosh(u), pv, sin(v)*cos(v)*(pb**2.)-2.*pu*pv*(1./tanh(u)), pb, -2.*pu*pb*(1./tanh(u))-2.*pb*pv*(1./tan(v))]

def xfunc3dhrot(u,v,b):
	return sinh(u)*sin(v)*cos(b)
def yfunc3dhrot(u,v,b):
	return sinh(u)*sin(v)*sin(b)
def zfunc3dhrot(u,v,b):
	return sinh(u)*cos(v)
def wfunc3dhrot(u,v,b):
	return cosh(u)	

def py3dhgrav(indVars, t, sourcemass, testmass):
	u, pu, v, pv, b, pb = indVars
	return [pu, (sin(v)*sin(v)*(pb**2.)+(pv**2.))*sinh(u)*cosh(u)-(sourcemass*testmass)/(sinh(u)**2.), pv, sin(v)*cos(v)*(pb**2.)-2.*pu*pv*(1./tanh(u)), pb, -2.*pu*pb*(1./tanh(u))-2.*pb*pv*(1./tan(v))]

def py3dhefreetrans(indVars, t):
	u, pu, v, pv, b, pb = indVars
	return [pu, (pv**2.)*sinh(u)*cosh(u), pv, -2.*pu*pv*tanh(u), pb, 0.]

def xfunc3dhetrans(u,v,b):
	return sinh(u)
def yfunc3dhetrans(u,v,b):
	return cosh(u)*sinh(v)
def zfunc3dhetrans(u,v,b):
	return b
def wfunc3dhetrans(u,v,b):
	return cosh(u)*cosh(v)

def py3dhefreerot(indVars, t):
	u, pu, v, pv, b, pb = indVars
	return [pu, (pv**2.)*sinh(u)*cosh(u), pv, -2.*pu*pv*(1./tanh(u)), pb, 0.]

def xfunc3dherot(u,v,b):
	return sinh(u)*cos(v)
def yfunc3dherot(u,v,b):
	return sinh(u)*sin(v)
def zfunc3dherot(u,v,b):
	return b
def wfunc3dherot(u,v,b):
	return cosh(u)	


def py3dhspring(indVars, t, k, xo, m1, m2):
	u1, pu1, v1, pv1, b1, pb1, u2, pu2, v2, pv2, b2, pb2 = indVars
	return [pu1,     (-(1./m1)*(-0.5*m1*(2.*pv1*pv1*cosh(u1)*sinh(u1) + 
      2.*pb1*pb1*cosh(u1)*cosh(v1)**2.*sinh(u1)) + (k*(-xo + 
        arccosh(cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2)))*(cosh(b1)*cosh(b2)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(u1) - 
        cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2)*sinh(u1) - cosh(u1)*sinh(u2) - 
        cosh(u2)*sinh(u1)*sinh(v1)*sinh(v2)))/(sqrt(-1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pv1,     (-(1./m1)*(1./cosh(u1)**2.)*(2.*m1*pu1*pv1*cosh(u1)*sinh(u1) - 
   m1*pb1*pb1*cosh(u1)**2.*cosh(v1)*sinh(v1) + (k*(-xo + 
        arccosh(cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2)))*(cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v2)*sinh(v1) - 
        cosh(u1)*cosh(u2)*cosh(v2)*sinh(b1)*sinh(b2)*sinh(v1) - 
        cosh(u1)*cosh(u2)*cosh(v1)*sinh(v2)))/(sqrt(-1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pb1,     (-(1./m1)*(1./cosh(u1)**2.)*(1./cosh(v1)**2.)*(2.*m1*pb1*pu1*cosh(u1)*cosh(v1)**2.*sinh(u1) + 
   2.*m1*pb1*pv1*cosh(u1)**2.*cosh(v1)*sinh(v1) + (k*(-xo + 
        arccosh(cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2)))*(cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1) - 
        cosh(b1)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b2)))/(sqrt(-1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) -
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pu2,     (-(1./m2)*(-0.5*m2*(2.*pv2*pv2*cosh(u2)*sinh(u2) + 
      2.*pb2*pb2*cosh(u2)*cosh(v2)**2.*sinh(u2)) + (k*(-xo + 
        arccosh(cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2)))*(-cosh(u2)*sinh(u1) + cosh(b1)*cosh(b2)*cosh(u1)*cosh(v1)*cosh(v2)*sinh(u2) - 
        cosh(u1)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2)*sinh(u2) - cosh(u1)*sinh(u2)*sinh(v1)*sinh(v2)))/(sqrt(-1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pv2,     (-(1./m2)*(1./cosh(u2)**2.)*(2.*m2*pu2*pv2*cosh(u2)*sinh(u2) - 
   m2*pb2*pb2*cosh(u2)**2.*cosh(v2)*sinh(v2) + (k*(-xo + 
        arccosh(cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2)))*(-cosh(u1)*cosh(u2)*cosh(v2)*sinh(v1) + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*sinh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*sinh(b1)*sinh(b2)*sinh(v2)))/(sqrt(-1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pb2,     (-(1/m2)*(1./cosh(u2)**2.)*(1./cosh(v2)**2.)*(2.*m2*pb2*pu2*cosh(u2)*cosh(v2)**2.*sinh(u2) + 
   2.*m2*pb2*pv2*cosh(u2)**2.*cosh(v2)*sinh(v2) + (k*(-xo + 
        arccosh(cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2)))*(-cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1) + 
        cosh(b1)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b2)))/(sqrt(-1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(b1)*cosh(b2)*cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2)*sinh(b1)*sinh(b2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2)))))]



def py3dhespring(indVars, t, k, xo, m1, m2):
	u1, pu1, v1, pv1, b1, pb1, u2, pu2, v2, pv2, b2, pb2 = indVars
	return [pu1,     (-(1./m1)*(-m1*pv1*pv1*cosh(u1)*sinh(u1) + (k*arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*(-xo + sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.))*(cosh(u2)*cosh(v1)*cosh(v2)*sinh(u1) - cosh(u1)*sinh(u2) - 
        cosh(u2)*sinh(u1)*sinh(v1)*sinh(v2)))/(sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.)*sqrt(-1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pv1,     (-(1./m1)*(1./cosh(u1)**2.)*(2.*m1*pu1*pv1*cosh(u1)*sinh(u1) + (k*arccosh(
       cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*(-xo + sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.))*(cosh(u1)*cosh(u2)*cosh(v2)*sinh(v1) - 
        cosh(u1)*cosh(u2)*cosh(v1)*sinh(v2)))/(sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.)*sqrt(-1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pb1,     (((-b1 + b2)*k*(-xo + sqrt((-b1 + b2)**2. + 
       arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.)))/(m1*sqrt((-b1 + b2)**2. + 
      	arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.))), pu2,     (-(1./m2)*(-m2*pv2*pv2*cosh(u2)*sinh(u2) + (k*arccosh(
       cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*(-xo + sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.))*(-cosh(u2)*sinh(u1) + cosh(u1)*cosh(v1)*cosh(v2)*sinh(u2) - 
        cosh(u1)*sinh(u2)*sinh(v1)*sinh(v2)))/(sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.)*sqrt(-1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pv2,     (-(1./m2)*(1./cosh(u2)**2.)*(2.*m2*pu2*pv2*cosh(u2)*sinh(u2) + (k*arccosh(
       cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*(-xo + sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.))*(-cosh(u1)*cosh(u2)*cosh(v2)*sinh(v1) + 
        cosh(u1)*cosh(u2)*cosh(v1)*sinh(v2)))/(sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.)*sqrt(-1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))*sqrt(1. + 
        cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))))), pb2,     (-(((-b1 + b2)*k*(-xo + sqrt((-b1 + b2)**2. + 
       arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.)))/(m2*sqrt((-b1 + b2)**2. + 
        arccosh(cosh(u1)*cosh(u2)*cosh(v1)*cosh(v2) - 
        sinh(u1)*sinh(u2) - 
        cosh(u1)*cosh(u2)*sinh(v1)*sinh(v2))**2.))))]



















# def imph3sptrans(posn, posn1i, veln, veln1i, step, massvec, sprcon, eqdist):
    
#     def con1(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
#         return an1 - an - .5*h*(adn + adn1)

#     def con2(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h):
#         return bn1 - bn - .5*h*(bdn + bdn1)

#     def con3(an, an1, bn, bn1, gn, gn1, adn, adn1, bdn, bdn1, gdn, gdn1, h): 
#         return gn1 - gn - .5*h*(gdn + gdn1)        

#     def con4(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, h, m1, k, xo):
#         return (ad1n1 - ad1n - .5*h*(-(1./m1)*(-0.5*m1*(2.*bd1n*bd1n*cosh(a1n)*sinh(a1n) + 2.*gd1n*gd1n*cosh(a1n)*sinh(a1n)*cosh(b1n)**2.) + (k*(-xo + 
#             arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))*(cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n)*sinh(a1n) - cosh(a1n)*sinh(a2n) - 
#             cosh(a2n)*sinh(a1n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(a1n)*sinh(g1n)*sinh(g2n)))/(sqrt(-1. + 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n))*sqrt(1. + 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))) - 

#             (1./m1)*(-0.5*m1*(2.*bd1n1*bd1n1*cosh(a1n1)*sinh(a1n1) + 2.*gd1n1*gd1n1*cosh(a1n1)*sinh(a1n1)*cosh(b1n1)**2.) + (k*(-xo + 
#             arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1)))*(cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1)*sinh(a1n1) - cosh(a1n1)*sinh(a2n1) - 
#             cosh(a2n1)*sinh(a1n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(a1n1)*sinh(g1n1)*sinh(g2n1)))/(sqrt(-1. + 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))*sqrt(1. + cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))))))

#     def con5(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, h, m1, k, xo):
#         return (bd1n1 - bd1n - .5*h*(-(1./m1)*(1./cosh(a1n)**2.)*(2.*m1*ad1n*bd1n*cosh(a1n)*sinh(a1n) - gd1n*gd1n*m1*cosh(b1n)*sinh(b1n)*cosh(a1n)**2. + (k*(-xo + 
#             arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))*(cosh(a1n)*cosh(a2n)*cosh(b2n)*cosh(g1n)*cosh(g2n)*sinh(b1n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b2n)*sinh(b1n)*sinh(g1n)*sinh(g2n)))/(sqrt(-1. + 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n))*sqrt(1. + 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))) - 

#             (1./m1)*(1./cosh(a1n1)**2.)*(2.*m1*ad1n1*bd1n1*cosh(a1n1)*sinh(a1n1) - gd1n1*gd1n1*m1*cosh(b1n1)*sinh(b1n1)*cosh(a1n1)**2. + (k*(-xo + 
#             arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1)))*(cosh(a1n1)*cosh(a2n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1)*sinh(b1n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b2n1)*sinh(b1n1)*sinh(g1n1)*sinh(g2n1)))/(sqrt(-1. + 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))*sqrt(1. + 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))))))

#     def con6(a1n, a1n1, b1n, b1n1, g1n, g1n1, a2n, a2n1, b2n, b2n1, g2n, g2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, h, m1, k, xo):
#         return (gd1n1 - gd1n - .5*h*(-(1./m1)*(1./cosh(a1n)**2.)*(1./cosh(b1n)**2.)*(2.*gd1n*m1*ad1n*cosh(a1n)*sinh(a1n)*cosh(b1n)**2. + 2.*gd1n*m1*bd1n*cosh(b1n)*sinh(b1n)*cosh(a1n)**.2 + (k*(-xo + 
#             arccosh(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))*(cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g2n)*sinh(g1n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*sinh(g2n)))/(sqrt(-1. + 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n))*sqrt(1. + 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*cosh(g1n)*cosh(g2n) - sinh(a1n)*sinh(a2n) - 
#             cosh(a1n)*cosh(a2n)*sinh(b1n)*sinh(b2n) - 
#             cosh(a1n)*cosh(a2n)*cosh(b1n)*cosh(b2n)*sinh(g1n)*sinh(g2n)))) - 

#             (1./m1)*(1./cosh(a1n1)**2.)*(1./cosh(b1n1)**2.)*(2.*gd1n1*m1*ad1n1*cosh(a1n1)*sinh(a1n1)*cosh(b1n1)**2. + 2.*gd1n1*m1*bd1n1*cosh(b1n1)*sinh(b1n1)*cosh(a1n1)**.2 + (k*(-xo + 
#             arccosh(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1)))*(cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g2n1)*sinh(g1n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*sinh(g2n1)))/(sqrt(-1. + 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))*sqrt(1. + 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*cosh(g1n1)*cosh(g2n1) - sinh(a1n1)*sinh(a2n1) - 
#             cosh(a1n1)*cosh(a2n1)*sinh(b1n1)*sinh(b2n1) - 
#             cosh(a1n1)*cosh(a2n1)*cosh(b1n1)*cosh(b2n1)*sinh(g1n1)*sinh(g2n1))))))        
    
#     h = step
#     mui = 10e-10
#     mat=array([
#         [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#         [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
#     ])

#     diff1=linalg.solve(mat,-array([
#         con1(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
#         con2(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
#         con3(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], h),
#         con1(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),
#         con2(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),
#         con3(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], h),

#         con4(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
#         con5(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
#         con6(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], massvec[0], h, sprcon, eqdist),
#         con4(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),
#         con5(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),
#         con6(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], massvec[1], h, sprcon, eqdist),       
#     ]))
#     val1 = array([pos1n1i[0]+diff1[0], pos1n1i[1]+diff1[1], pos1n1i[2]+diff1[2], pos2n1i[0]+diff1[3], pos2n1i[1]+diff1[4], pos2n1i[2]+diff1[5], vel1n1i[0]+diff1[6], vel1n1i[1]+diff1[7], vel1n1i[2]+diff1[8], vel2n1i[0]+diff1[9], vel2n1i[1]+diff1[10], vel2n1i[2]+diff1[11]])    
#     x = 0
#     while(x < 7):
#         mat=array([
#             [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#             [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
#         ])

#         diff2=linalg.solve(mat,-array([
#             con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
#             con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
#             con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], h),
#             con1(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),
#             con2(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),
#             con3(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], h),

#             con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
#             con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
#             con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], massvec[0], h, sprcon, eqdist),
#             con4(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),
#             con5(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),
#             con6(pos2n[0], val1[3], pos2n[1], val1[4], pos2n[2], val1[5], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], vel2n[0], val1[9], vel2n[1], val1[10], vel2n[2], val1[11], vel1n[0], val1[6], vel1n[1], val1[7], vel1n[2], val1[8], massvec[1], h, sprcon, eqdist),       
#         ]))
#         val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11]])       
#         val1 = val2
#         x=x+1
#     return[val1]


# def symh3sptrans(pos1n, pos1n1i, pos2n, pos2n1i, vel1n, vel1n1i, vel2n, vel2n1i, step, massvec, sprcon, eqdist):
    
#     def con1(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h):
#         return xn1 - xn - 2.*mu*(xn1 + xn) - .5*h*(adn*cosh(arcsinh(xn)) + adn1*cosh(arcsinh(xn1)))

#     def con2(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
#         return yn1 - yn - 2.*mu*(yn1 + yn) - .5*h*(adn*sinh(arcsinh(xn))*sinh(arcsinh(yn/(cosh(arcsinh(xn))))) + bdn*cosh(arcsinh(yn/(cosh(arcsinh(xn))))) + adn1*sinh(arcsinh(xn1))*sinh(arcsinh(yn1/(cosh(arcsinh(xn1))))) + bdn1*cosh(arcsinh(yn1/(cosh(arcsinh(xn1))))))

#     def con3(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
#         return zn1 - zn - 2.*mu*(zn1 + zn) - .5*h*(adn*sinh(arcsinh(xn))*cosh(arcsinh(yn/(cosh(arcsinh(xn)))))*sinh(arctanh(zn/wn)) + bdn*sinh(arcsinh(yn/(cosh(arcsinh(xn)))))*sinh(arctanh(zn/wn)) + gdn*cosh(arctanh(zn/wn)) + adn1*sinh(arcsinh(xn1))*cosh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*sinh(arctanh(zn1/wn1)) + bdn1*sinh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*sinh(arctanh(zn1/wn1)) + gdn1*cosh(arctanh(zn1/wn1)))

#     def con4(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
#         return wn1 - wn + 2.*mu*(wn1 + wn) - .5*h*(adn*sinh(arcsinh(xn))*cosh(arcsinh(yn/(cosh(arcsinh(xn)))))*cosh(arctanh(zn/wn)) + bdn*sinh(arcsinh(yn/(cosh(arcsinh(xn)))))*cosh(arctanh(zn/wn)) + gdn*sinh(arctanh(zn/wn)) + adn1*sinh(arcsinh(xn1))*cosh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*cosh(arctanh(zn1/wn1)) + bdn1*sinh(arcsinh(yn1/(cosh(arcsinh(xn1)))))*cosh(arctanh(zn1/wn1)) + gdn1*sinh(arctanh(zn1/wn1)))    

#     def con5(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, h, m1, k, xo): 
#         return (ad1n1 - ad1n - .5*h*(-(1./m1)*(-0.5*m1*(2.*bd1n*bd1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n)) + 2.*gd1n*gd1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))**2.) + (k*(-xo + 
#             arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))*(cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n))*sinh(arcsinh(x1n)) - cosh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x2n))*sinh(arcsinh(x1n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arcsinh(x1n))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))/(sqrt(-1. + 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n)))*sqrt(1. + 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))) - 

#             (1./m1)*(-0.5*m1*(2.*bd1n1*bd1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1)) + 2.*gd1n1*gd1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))**2.) + (k*(-xo + 
#             arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))*(cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1))*sinh(arcsinh(x1n1)) - cosh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x2n1))*sinh(arcsinh(x1n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arcsinh(x1n1))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))/(sqrt(-1. + 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))*sqrt(1. + 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))))))

#     def con6(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, h, m1, k, xo):
#         return (bd1n1 - bd1n - .5*h*(-(1./m1)*(1./cosh(arcsinh(x1n))**2.)*(2.*m1*ad1n*bd1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n)) - gd1n*gd1n*m1*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(x1n))**2. + (k*(-xo + 
#             arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))*(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))/(sqrt(-1. + 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n)))*sqrt(1. + 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))) - 

#             (1./m1)*(1./cosh(arcsinh(x1n1))**2.)*(2.*m1*ad1n1*bd1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1)) - gd1n1*gd1n1*m1*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(x1n1))**2. + (k*(-xo + 
#             arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))*(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))/(sqrt(-1. + 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))*sqrt(1. + 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))))))

#     def con7(x1n, x1n1, y1n, y1n1, z1n, z1n1, w1n, w1n1, x2n, x2n1, y2n, y2n1, z2n, z2n1, w2n, w2n1, ad1n, ad1n1, bd1n, bd1n1, gd1n, gd1n1,  ad2n, ad2n1, bd2n, bd2n1, gd2n, gd2n1, mu1, mu2, h, m1, k, xo):
#         return (gd1n1 - gd1n - .5*h*(-(1./m1)*(1./cosh(arcsinh(x1n))**2.)*(1./cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))**2.)*(2.*gd1n*m1*ad1n*cosh(arcsinh(x1n))*sinh(arcsinh(x1n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))**2. + 2.*gd1n*m1*bd1n*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(x1n))**.2 + (k*(-xo + 
#             arccosh(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))*(cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z2n/w2n))*sinh(arctanh(z1n/w1n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))/(sqrt(-1. + 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n)))*sqrt(1. + 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*cosh(arctanh(z1n/w1n))*cosh(arctanh(z2n/w2n)) - sinh(arcsinh(x1n))*sinh(arcsinh(x2n)) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*sinh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*sinh(arcsinh(y2n/(cosh(arcsinh(x2n))))) - 
#             cosh(arcsinh(x1n))*cosh(arcsinh(x2n))*cosh(arcsinh(y1n/(cosh(arcsinh(x1n)))))*cosh(arcsinh(y2n/(cosh(arcsinh(x2n)))))*sinh(arctanh(z1n/w1n))*sinh(arctanh(z2n/w2n))))) - 

#             (1./m1)*(1./cosh(arcsinh(x1n1))**2.)*(1./cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))**2.)*(2.*gd1n1*m1*ad1n1*cosh(arcsinh(x1n1))*sinh(arcsinh(x1n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))**2. + 2.*gd1n1*m1*bd1n1*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(x1n1))**.2 + (k*(-xo + 
#             arccosh(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))*(cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z2n1/w2n1))*sinh(arctanh(z1n1/w1n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1))))/(sqrt(-1. + 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))*sqrt(1. + 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*cosh(arctanh(z1n1/w1n1))*cosh(arctanh(z2n1/w2n1)) - sinh(arcsinh(x1n1))*sinh(arcsinh(x2n1)) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*sinh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*sinh(arcsinh(y2n1/(cosh(arcsinh(x2n1))))) - 
#             cosh(arcsinh(x1n1))*cosh(arcsinh(x2n1))*cosh(arcsinh(y1n1/(cosh(arcsinh(x1n1)))))*cosh(arcsinh(y2n1/(cosh(arcsinh(x2n1)))))*sinh(arctanh(z1n1/w1n1))*sinh(arctanh(z2n1/w2n1)))))))     

#     def con8(xn, xn1, yn, yn1, zn, zn1, wn, wn1, adn, adn1, bdn, bdn1, gdn, gdn1, mu, h): 
#         return xn1*xn1 + yn1*yn1 + zn1*zn1 - wn1*wn1 + 1.
    
#     h = step
#     mu1i = 10e-10
#     mu2i = 10e-10
#     mat=array([
#         [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[0],0.],
#         [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[1],0.],
#         [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos1n[2],0.],
#         [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.*pos1n[3],0.],
#         [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[0]],
#         [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[1]],
#         [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*pos2n[2]],
#         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,4.*pos2n[3]],               
#         [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
#         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],              
#         [2.*pos1n[0],2.*pos1n[1],2.*pos1n[2],-2.*pos1n[3],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#         [0.,0.,0.,0.,2.*pos2n[0],2.*pos2n[1],2.*pos2n[2],-2.*pos2n[3],0.,0.,0.,0.,0.,0.,0.,0.]
#     ])
#     print(mat)
#     diff1=linalg.solve(mat,-array([
#         con1(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
#         con2(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
#         con3(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
#         con4(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
#         con1(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
#         con2(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
#         con3(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),
#         con4(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h),

#         con5(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),
#         con6(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),
#         con7(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu1i, mu2i, massvec[0], h, sprcon, eqdist),

#         con5(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),
#         con6(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),
#         con7(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu2i, mu1i, massvec[1], h, sprcon, eqdist),

#         con8(pos1n[0], pos1n1i[0], pos1n[1], pos1n1i[1], pos1n[2], pos1n1i[2], pos1n[3], pos1n1i[3], vel1n[0], vel1n1i[0], vel1n[1], vel1n1i[1], vel1n[2], vel1n1i[2], mu1i, h),
#         con8(pos2n[0], pos2n1i[0], pos2n[1], pos2n1i[1], pos2n[2], pos2n1i[2], pos2n[3], pos2n1i[3], vel2n[0], vel2n1i[0], vel2n[1], vel2n1i[1], vel2n[2], vel2n1i[2], mu2i, h)        
#     ]))
#     val1 = array([pos1n1i[0]+diff1[0], pos1n1i[1]+diff1[1], pos1n1i[2]+diff1[2], pos1n1i[3]+diff1[3], pos2n1i[0]+diff1[4], pos2n1i[1]+diff1[5], pos2n1i[2]+diff1[6], pos2n1i[3]+diff1[7], vel1n1i[0]+diff1[8], vel1n1i[1]+diff1[9], vel1n1i[2]+diff1[10], vel2n1i[0]+diff1[11], vel2n1i[1]+diff1[12], vel2n1i[2]+diff1[13], mu1i+diff1[14], mu2i+diff1[15]])
#     print(val1)
#     x = 0
#     while(x < 7):
#         mat=array([
#             [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[0],0.],
#             [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[1],0.],
#             [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[2],0.],
#             [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.*val1[3],0.],
#             [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[4]],
#             [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[5]],
#             [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,-4.*val1[6]],
#             [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,4.*val1[7]],               
#             [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
#             [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],              
#             [2.*val1[0],2.*val1[1],2.*val1[2],-2.*val1[3],0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
#             [0.,0.,0.,0.,2.*val1[4],2.*val1[5],2.*val1[6],-2.*val1[7],0.,0.,0.,0.,0.,0.,0.,0.]
#         ])
#         diff2=linalg.solve(mat,-array([
#             con1(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
#             con2(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
#             con3(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
#             con4(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
#             con1(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
#             con2(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
#             con3(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),
#             con4(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h),

#             con5(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
#             con6(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
#             con7(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[14], val1[15], massvec[0], h, sprcon, eqdist),
#             con5(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),
#             con6(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),
#             con7(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[15], val1[14], massvec[1], h, sprcon, eqdist),

#             con8(pos1n[0], val1[0], pos1n[1], val1[1], pos1n[2], val1[2], pos1n[3], val1[3], vel1n[0], val1[8], vel1n[1], val1[9], vel1n[2], val1[10], val1[14], h),
#             con8(pos2n[0], val1[4], pos2n[1], val1[5], pos2n[2], val1[6], pos2n[3], val1[7], vel2n[0], val1[11], vel2n[1], val1[12], vel2n[2], val1[13], val1[15], h)        
#         ]))
#         val2 = array([val1[0]+diff2[0], val1[1]+diff2[1], val1[2]+diff2[2], val1[3]+diff2[3], val1[4]+diff2[4], val1[5]+diff2[5], val1[6]+diff2[6], val1[7]+diff2[7], val1[8]+diff2[8], val1[9]+diff2[9], val1[10]+diff2[10], val1[11]+diff2[11], val1[12]+diff2[12], val1[13]+diff2[13], val1[14]+diff2[14], val1[15]+diff2[15]])        
#         val1 = val2
#         x=x+1    

#     return[val1] 

    








