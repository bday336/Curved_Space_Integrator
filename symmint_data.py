from symint_bank import symh2geotrans,imph2geotrans,imprk4h2geotrans,symh2georot,imph2georot,symh3geotrans,imph3geotrans,symh2xegeotrans,imph2xegeotrans,sympbhgeo,imppbhgeo,symh3cprot,imph3cprot,sympbhcp,imppbhcp,symh3sptrans,imph3sptrans
from function_bank import hyper2poin2d,hyper2poin3d,formatvec2d,unformatvec2d,formatvec3d,unformatvec3d,boostx2d,boosty2d,rotz2d,boostx3d,boosty3d,boostz3d,rotx3d,roty3d,rotz3d,motionmat2dh,motionmat3dh,py2dhfreetrans,xfunc2dhtrans,yfunc2dhtrans,zfunc2dhtrans,py3dhfreetrans,xfunc3dhtrans,yfunc3dhtrans,zfunc3dhtrans,wfunc3dhtrans,py3dhfreerot,xfunc3dhrot,yfunc3dhrot,zfunc3dhrot,wfunc3dhrot,py3dhgrav,py3dhefreetrans,xfunc3dhetrans,yfunc3dhetrans,zfunc3dhetrans,wfunc3dhetrans,py3dhefreerot,xfunc3dherot,yfunc3dherot,zfunc3dherot,wfunc3dherot,py3dhspring,py3dhespring
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

#Initial position (position in rotational / velocity in chosen parameterization)
ai=.5
bi=.0*np.pi/2.
gi=0.
adi=.0
bdi=.5
gdi=0. 
massvec=array([10.,1.])
sprcon=1.
eqdist=1.
version="h2geoboth"
plot="datatraj"
delT=.01
maxT=10+delT

a1i=-.5
b1i=0.
g1i=0.
ad1i=0.
bd1i=0.
gd1i=.5 

a2i=.5
b2i=0.
g2i=0.
ad2i=0.
bd2i=0.
gd2i=.5 

# Geodesic Trajectories

if(version=="h2geotrans"):
	# Do the process
	ab = array([ai, bi])
	# Position is given in rotational parameterization
	posi = array([sinh(ab[0])*cos(ab[1]), sinh(ab[0])*sin(ab[1]), cosh(ab[0])])
	# The implicit method requires the position to be in translational parameterization
	abt = array([arcsinh(sinh(ab[0])*cos(ab[1])), arctanh(tanh(ab[0])*sin(ab[1]))])
	posguess = posi
	abdoti = array([adi, bdi])
	abdotguess = abdoti
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testm = []
	gx = []
	gy = []
	gz = []
	testp = []
	ga = []
	gb = []
	testprk4 = []
	grk4a = []
	grk4b = []	

	gx=append(gx, posi[0])
	gy=append(gy, posi[1])
	gz=append(gz, posi[2])
	ga=append(ga, abt[0])
	gb=append(gb, abt[1])
	grk4a=append(grk4a, abt[0])
	grk4b=append(grk4b, abt[1])	
	testm.append(symh2geotrans(posi, posguess, abdoti, abdotguess, delT))
	testp.append(imph2geotrans(abt, abt, abdoti, abdotguess, delT))	
	testprk4.append(imprk4h2geotrans(abt, abdoti, delT))		
	gx=append(gx, testm[0][0][0])
	gy=append(gy, testm[0][0][1])
	gz=append(gz, testm[0][0][2])
	ga=append(ga, testp[0][0][0])
	gb=append(gb, testp[0][0][1])
	grk4a=append(grk4a, testprk4[0][0][0])
	grk4b=append(grk4b, testprk4[0][0][1])		
	q=q+1
	while(q < nump-1):
		nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2]])
		nextdotm = array([testm[q - 1][0][3], testm[q - 1][0][4]])
		nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1]])
		nextdotp = array([testp[q - 1][0][2], testp[q - 1][0][3]])
		nextposprk4 = array([testprk4[q - 1][0][0], testprk4[q - 1][0][1]])
		nextdotprk4 = array([testprk4[q - 1][0][2], testprk4[q - 1][0][3]])					
		testm.append(symh2geotrans(nextposm, nextposm, nextdotm, nextdotm, delT))
		testp.append(imph2geotrans(nextposp, nextposp, nextdotp, nextdotp, delT))
		testprk4.append(imprk4h2geotrans(nextposprk4, nextdotprk4, delT))		
		gx=append(gx, testm[q][0][0])
		gy=append(gy, testm[q][0][1])
		gz=append(gz, testm[q][0][2])
		ga=append(ga, testp[q][0][0])
		gb=append(gb, testp[q][0][1])
		grk4a=append(grk4a, testprk4[q][0][0])
		grk4b=append(grk4b, testprk4[q][0][1])				
		q=q+1

	gu=[]
	gv=[]
	gpu=[]
	gpv=[]
	gprk4u=[]
	gprk4v=[]	
	for b in range(len(gx)):
	    gu=append(gu,gx[b]/(gz[b] + 1.))
	    gv=append(gv,gy[b]/(gz[b] + 1.)) 
	    gpu=append(gpu,sinh(ga[b])/(cosh(ga[b])*cosh(gb[b]) + 1.))
	    gpv=append(gpv,cosh(ga[b])*sinh(gb[b])/(cosh(ga[b])*cosh(gb[b]) + 1.)) 
	    gprk4u=append(gprk4u,sinh(grk4a[b])/(cosh(grk4a[b])*cosh(grk4b[b]) + 1.))
	    gprk4v=append(gprk4v,cosh(grk4a[b])*sinh(grk4b[b])/(cosh(grk4a[b])*cosh(grk4b[b]) + 1.)) 	    	     

	#Lie generator trajectories

	checkhdata=[]
	checkemdata=[]
	for c in timearr:
		checkhdata.append(hyper2poin2d(unformatvec2d(matmul(rotz2d(ab[1]),matmul(boostx2d(ab[0]),matmul(rotz2d(-ab[1]),matmul(motionmat2dh(abdoti[0], abdoti[1], c),formatvec2d(array([0., 0., 1.])))))))))
		checkemdata.append(unformatvec2d(matmul(rotz2d(ab[1]),matmul(boostx2d(ab[0]),matmul(rotz2d(-ab[1]),matmul(motionmat2dh(abdoti[0], abdoti[1], c),formatvec2d(array([0., 0., 1.]))))))))


	#This formats the data to be compared with the integrator results

	checku = []
	checkv = []
	checkx = []
	checky = []
	checkz = []	
	for d in range(len(timearr)):
		checku.append(checkhdata[d][0])
		checkv.append(checkhdata[d][1])
		checkx.append(checkemdata[d][0])
		checky.append(checkemdata[d][1])
		checkz.append(checkemdata[d][2])		

	if(plot=="datatraj"):	


		#This is for the horizon circle.
		# Theta goes from 0 to 2pi
		theta = np.linspace(0, 2*np.pi, 100)

		# Compute x1 and x2 for horizon
		xc = np.cos(theta)
		yc = np.sin(theta)    		

		spemerr=[]
		slemerr=[]
		lpemerr=[]
		pprk4emerr=[]		
		# Using the Euclidean distance function to determine error in embedding space
		for j in range(len(checku)):
			spemerr.append(sqrt((gx[j]-sinh(ga[j]))**2. + (gy[j]-cosh(ga[j])*sinh(gb[j]))**2. + (gz[j]-cosh(ga[j])*cosh(gb[j]))**2.))
			slemerr.append(sqrt((gx[j]-checkx[j])**2. + (gy[j]-checky[j])**2. + (gz[j]-checkz[j])**2.))
			lpemerr.append(sqrt(abs((checkx[j]-sinh(ga[j]))**2. + (checky[j]-cosh(ga[j])*sinh(gb[j]))**2. + (checkz[j]-cosh(ga[j])*cosh(gb[j]))**2.)))
			pprk4emerr.append(sqrt((sinh(grk4a[j])-sinh(ga[j]))**2. + (cosh(grk4a[j])*sinh(grk4b[j])-cosh(ga[j])*sinh(gb[j]))**2. + (cosh(grk4a[j])*cosh(grk4b[j])-cosh(ga[j])*cosh(gb[j]))**2.))		

		#This is the particle trajectories in the Poincare model
		    
		# #Plot
		fig= plt.figure(figsize=(10,5))

		plt.subplot(1,2,1)
		plt.plot(xc,yc)
		plt.plot(gu,gv,label="sym")
		plt.plot(gpu,gpv,label="parameter")
		plt.plot(checku,checkv,label="lie")
		#plt.plot(gprk4u,gprk4v,label="prk4")
		plt.legend(loc='lower left')

		plt.subplot(1,2,2)
		plt.plot(timearr,spemerr,label="sym/par")
		plt.plot(timearr,slemerr,label="sym/lie")
		plt.plot(timearr,lpemerr,label="lie/par")
		#plt.plot(timearr,pprk4emerr,label="prk4/par")		
		plt.legend(loc='upper left')		

	elif(plot=="datasheet"):

		#Differences

		checksymu=[]
		checkparu=[]
		checkvalu=[]
		checksymv=[]
		checkparv=[]
		checkvalv=[]
		checker=0
		for j in checku:
			checksymu.append(abs(gu[checker]-j))
			checkparu.append(abs(gpu[checker]-j))
			checkvalu.append(abs(j-j))
			checksymv.append(abs(gv[checker]-checkv[checker]))
			checkparv.append(abs(gpv[checker]-checkv[checker]))
			checkvalv.append(abs(checkv[checker]-checkv[checker]))	
			checker=checker+1				

		#This is for the horizon circle.
		# Theta goes from 0 to 2pi
		theta = np.linspace(0, 2*np.pi, 100)

		# Compute x1 and x2 for horizon
		xc = np.cos(theta)
		yc = np.sin(theta) 	

		fig = plt.figure(figsize=(15,8))
		fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

		gs = gridspec.GridSpec(3, 4, width_ratios=[6, 2, 2, 2], height_ratios=[6, 2, 2])
		ax1 = fig.add_subplot(gs[0])
		ax1.set_aspect("equal")
		ax1.set_xlabel('x', fontsize=12)
		ax1.set_ylabel('y', fontsize=12)
		ax1.set_title('Trajectory', fontsize=14)
		ax1.plot(xc,yc)
		ax1.plot(gu,gv, label='symmetric')
		ax1.plot(checku,checkv, label='Lie')
		ax1.plot(gpu,gpv, label='paramater')
		ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
		ax2 = fig.add_subplot(gs[5])
		ax2.set_xlabel('x', fontsize=12)
		ax2.set_ylabel('y', fontsize=12)
		ax2.set_title('x coordinate', fontsize=14)
		ax2.plot(timearr,gu, label='symmetric')
		ax2.plot(timearr,checku, label='Lie')
		ax2.plot(timearr,gpu, label='paramater')
		ax2.legend(loc='upper center', bbox_to_anchor=(-.7, .9))	
		ax3 = fig.add_subplot(gs[6])
		ax3.set_xlabel('x', fontsize=12)
		ax3.set_ylabel('y', fontsize=12)
		ax3.set_title('y coordinate', fontsize=14)
		ax3.plot(timearr,gv, label='symmetric')
		ax3.plot(timearr,checkv, label='Lie')
		ax3.plot(timearr,gpv, label='paramater')
		ax5 = fig.add_subplot(gs[9])
		ax5.set_xlabel('x', fontsize=12)
		ax5.set_ylabel('y', fontsize=12)
		ax5.set_title('x error', fontsize=14)
		# ax5.plot(timearr,checksymu, label='symmetric vs. Lie')
		# ax5.plot(timearr,checkvalu, label='Lie')
		ax5.plot(timearr,checkparu, label='paramater vs. Lie')
		ax5.legend(loc='upper center', bbox_to_anchor=(-.7, .9))
		ax6 = fig.add_subplot(gs[10])
		ax6.set_xlabel('x', fontsize=12)
		ax6.set_ylabel('y', fontsize=12)
		ax6.set_title('y error', fontsize=14)
		# ax6.plot(timearr,checksymv, label='symmetric vs. Lie')
		# ax6.plot(timearr,checkvalv, label='Lie')
		ax6.plot(timearr,checkparv, label='paramater vs. Lie')

	elif(plot=="datadiserror"):	

		#Lie generator trajectories

		checkerrdata=[]
		for c in timearr:
			checkerrdata.append(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(ab[1]),matmul(rotz2d(0),matmul(motionmat2dh(abdoti[0], abdoti[1], c),formatvec2d(array([0., 0., 1.]))))))))

		#This formats the data to be compared with the integrator results

		checkx = []
		checky = []
		checkz = []
		for d in range(len(checku)):
			checkx.append(checkerrdata[d][0])
			checky.append(checkerrdata[d][1])
			checkz.append(checkerrdata[d][2])		

		spdiserr=[]
		sldiserr=[]
		lpdiserr=[]
		for j in range(len(checku)):
			spdiserr.append(arccosh(-gx[j]*sinh(ga[j])-gy[j]*cosh(ga[j])*sinh(gb[j])+gz[j]*cosh(ga[j])*cosh(gb[j])))
			sldiserr.append(arccosh(-gx[j]*checkx[j]-gy[j]*checky[j]+gz[j]*checkz[j]))
			lpdiserr.append(arccosh(-checkx[j]*sinh(ga[j])-checky[j]*cosh(ga[j])*sinh(gb[j])+checkz[j]*cosh(ga[j])*cosh(gb[j])))

			# #Plot
		fig= plt.figure(figsize=(5,5))

		# plt.plot(timearr,spdiserr,label="sym/par")
		# plt.plot(timearr,sldiserr,label="sym/lie")
		plt.plot(timearr,lpdiserr,label="lie/par")
		plt.legend()		

	elif(plot=="dataemerror"):	

		#Lie generator trajectories

		checkerrdata=[]
		for c in timearr:
			checkerrdata.append(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(ab[1]),matmul(rotz2d(0),matmul(motionmat2dh(abdoti[0], abdoti[1], c),formatvec2d(array([0., 0., 1.]))))))))

		#This formats the data to be compared with the integrator results

		checkx = []
		checky = []
		checkz = []
		for d in range(len(checku)):
			checkx.append(checkerrdata[d][0])
			checky.append(checkerrdata[d][1])
			checkz.append(checkerrdata[d][2])		

		spemerr=[]
		slemerr=[]
		lpemerr=[]
		pprk4emerr=[]		
		for j in range(len(checku)):
			spemerr.append(sqrt((gx[j]-sinh(ga[j]))**2. + (gy[j]-cosh(ga[j])*sinh(gb[j]))**2. - (gz[j]-cosh(ga[j])*cosh(gb[j]))**2.))
			slemerr.append(sqrt((gx[j]-checkx[j])**2. + (gy[j]-checky[j])**2. - (gz[j]-checkz[j])**2.))
			lpemerr.append(sqrt((checkx[j]-sinh(ga[j]))**2. + (checky[j]-cosh(ga[j])*sinh(gb[j]))**2. - (checkz[j]-cosh(ga[j])*cosh(gb[j]))**2.))
			pprk4emerr.append(sqrt((sinh(grk4a[j])-sinh(ga[j]))**2. + (cosh(grk4a[j])*sinh(grk4b[j])-cosh(ga[j])*sinh(gb[j]))**2. - (cosh(grk4a[j])*cosh(grk4b[j])-cosh(ga[j])*cosh(gb[j]))**2.))


			# #Plot
		fig= plt.figure(figsize=(5,5))

		# plt.plot(timearr,spemerr,label="sym/par")
		# plt.plot(timearr,slemerr,label="sym/lie")
		# plt.plot(timearr,lpemerr,label="lie/par")
		plt.plot(timearr,pprk4emerr,label="prk4/par")		
		plt.legend()		

	fig.tight_layout()	

	plt.show()

if(version=="h2geoboth"):
	# Do the process for both parameterization at the same time to see how they compare
	# Position is given in rotational parameterization
	abri = array([ai, bi])
	# Position in translational parameterization
	abti = array([arcsinh(sinh(abri[0])*cos(abri[1])), arctanh(tanh(abri[0])*sin(abri[1]))])
	posi = array([sinh(abri[0])*cos(abri[1]), sinh(abri[0])*sin(abri[1]), cosh(abri[0])])
	posguess = posi
	# Velocity given in translational parameterization
	abdotti = array([adi, bdi])
	# Velocity in rotational parameterization
	abdotri = array([(abdotti[0]*sinh(abti[0])*cosh(abti[1]) + abdotti[1]*cosh(abti[0])*sinh(abti[1]))/sinh(abri[0]), (cos(abri[1])*(abdotti[0]*sinh(abti[0])*sinh(abti[1]) + abdotti[1]*cosh(abti[0])*cosh(abti[1])) - sin(abri[1])*(abdotti[0]*cosh(abti[0])))/sinh(abri[0])])
	abdottguess = abdotti
	abdotrguess = abdotri
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testmt = []
	gxt = []
	gyt = []
	gzt = []
	testpt = []
	gat = []
	gbt = []

	testmr = []
	gxr = []
	gyr = []
	gzr = []
	testpr = []
	gar = []
	gbr = []	

	gxt=append(gxt, posi[0])
	gyt=append(gyt, posi[1])
	gzt=append(gzt, posi[2])
	gat=append(gat, abti[0])
	gbt=append(gbt, abti[1])
	gxr=append(gxr, posi[0])
	gyr=append(gyr, posi[1])
	gzr=append(gzr, posi[2])
	gar=append(gar, abri[0])
	gbr=append(gbr, abri[1])

	testmt.append(symh2geotrans(posi, posguess, abdotti, abdottguess, delT))
	testpt.append(imph2geotrans(abti, abti, abdotti, abdottguess, delT))
	testmr.append(symh2georot(posi, posguess, abdotri, abdotrguess, delT))
	testpr.append(imph2georot(abri, abri, abdotri, abdotrguess, delT))		
		
	gxt=append(gxt, testmt[0][0][0])
	gyt=append(gyt, testmt[0][0][1])
	gzt=append(gzt, testmt[0][0][2])
	gat=append(gat, testpt[0][0][0])
	gbt=append(gbt, testpt[0][0][1])
	gxr=append(gxr, testmr[0][0][0])
	gyr=append(gyr, testmr[0][0][1])
	gzr=append(gzr, testmr[0][0][2])
	gar=append(gar, testpr[0][0][0])
	gbr=append(gbr, testpr[0][0][1])
	
	q=q+1
	while(q < nump-1):
		nextposmt = array([testmt[q - 1][0][0], testmt[q - 1][0][1], testmt[q - 1][0][2]])
		nextdotmt = array([testmt[q - 1][0][3], testmt[q - 1][0][4]])
		nextpospt = array([testpt[q - 1][0][0], testpt[q - 1][0][1]])
		nextdotpt = array([testpt[q - 1][0][2], testpt[q - 1][0][3]])
		nextposmr = array([testmr[q - 1][0][0], testmr[q - 1][0][1], testmr[q - 1][0][2]])
		nextdotmr = array([testmr[q - 1][0][3], testmr[q - 1][0][4]])
		nextpospr = array([testpr[q - 1][0][0], testpr[q - 1][0][1]])
		nextdotpr = array([testpr[q - 1][0][2], testpr[q - 1][0][3]])	

		testmt.append(symh2geotrans(nextposmt, nextposmt, nextdotmt, nextdotmt, delT))
		testpt.append(imph2geotrans(nextpospt, nextpospt, nextdotpt, nextdotpt, delT))
		testmr.append(symh2georot(nextposmr, nextposmr, nextdotmr, nextdotmr, delT))
		testpr.append(imph2georot(nextpospr, nextpospr, nextdotpr, nextdotpr, delT))
		
		gxt=append(gxt, testmt[q][0][0])
		gyt=append(gyt, testmt[q][0][1])
		gzt=append(gzt, testmt[q][0][2])
		gat=append(gat, testpt[q][0][0])
		gbt=append(gbt, testpt[q][0][1])
		gxr=append(gxr, testmr[q][0][0])
		gyr=append(gyr, testmr[q][0][1])
		gzr=append(gzr, testmr[q][0][2])
		gar=append(gar, testpr[q][0][0])
		gbr=append(gbr, testpr[q][0][1])

		q=q+1

	gut=[]
	gvt=[]
	gput=[]
	gpvt=[]
	gur=[]
	gvr=[]
	gpur=[]
	gpvr=[]
	
	for b in range(len(gxt)):
	    gut=append(gut,gxt[b]/(gzt[b] + 1.))
	    gvt=append(gvt,gyt[b]/(gzt[b] + 1.)) 
	    gput=append(gput,sinh(gat[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.))
	    gpvt=append(gpvt,cosh(gat[b])*sinh(gbt[b])/(cosh(gat[b])*cosh(gbt[b]) + 1.)) 
	    gur=append(gur,gxr[b]/(gzr[b] + 1.))
	    gvr=append(gvr,gyr[b]/(gzr[b] + 1.)) 
	    gpur=append(gpur,sinh(gar[b])*cos(gbr[b])/(cosh(gar[b]) + 1.))
	    gpvr=append(gpvr,sinh(gar[b])*sin(gbr[b])/(cosh(gar[b]) + 1.)) 	    	     

	#Lie generator trajectories

	checkhdata=[]
	checkemdata=[]
	liemag=array([sqrt(abdotti[0]**2. + cosh(abti[0])**2.*abdotti[1]**2.),0])
	for c in timearr:

		checkhdata.append(hyper2poin2d(unformatvec2d(matmul(rotz2d(abri[1]),matmul(boostx2d(abri[0]),matmul(rotz2d(-abri[1]),matmul(rotz2d(arctan2(abdotti[1], abdotti[0])),formatvec2d(array([sinh(liemag[0]*c), cosh(liemag[0]*c)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)])))))))))
		checkemdata.append(unformatvec2d(matmul(rotz2d(abri[1]),matmul(boostx2d(abri[0]),matmul(rotz2d(-abri[1]),matmul(rotz2d(arctan2(abdotti[1], abdotti[0])),formatvec2d(array([sinh(liemag[0]*c), cosh(liemag[0]*c)*sinh(0.), cosh(liemag[0]*c)*cosh(0.)]))))))))


	#This formats the data to be compared with the integrator results

	checku = []
	checkv = []
	checkx = []
	checky = []
	checkz = []	
	for d in range(len(timearr)):
		checku.append(checkhdata[d][0])
		checkv.append(checkhdata[d][1])
		checkx.append(checkemdata[d][0])
		checky.append(checkemdata[d][1])
		checkz.append(checkemdata[d][2])		

	if(plot=="datatraj"):	


		#This is for the horizon circle.
		# Theta goes from 0 to 2pi
		theta = np.linspace(0, 2*np.pi, 100)

		# Compute x1 and x2 for horizon
		xc = np.cos(theta)
		yc = np.sin(theta)    		

		sptemerr=[]
		sltemerr=[]
		lptemerr=[]
		spremerr=[]
		slremerr=[]
		lpremerr=[]
	
		# Using the Euclidean distance function to determine error in embedding space
		for j in range(len(checku)):
			sptemerr.append(sqrt((gxt[j]-sinh(gat[j]))**2. + (gyt[j]-cosh(gat[j])*sinh(gbt[j]))**2. + (gzt[j]-cosh(gat[j])*cosh(gbt[j]))**2.))
			sltemerr.append(sqrt((gxt[j]-checkx[j])**2. + (gyt[j]-checky[j])**2. + (gzt[j]-checkz[j])**2.))
			lptemerr.append(sqrt(abs((checkx[j]-sinh(gat[j]))**2. + (checky[j]-cosh(gat[j])*sinh(gbt[j]))**2. + (checkz[j]-cosh(gat[j])*cosh(gbt[j]))**2.)))	
			spremerr.append(sqrt((gxr[j]-sinh(gar[j])*cos(gbr[j]))**2. + (gyr[j]-sinh(gar[j])*sin(gbr[j]))**2. + (gzr[j]-cosh(gar[j]))**2.))
			slremerr.append(sqrt((gxr[j]-checkx[j])**2. + (gyr[j]-checky[j])**2. + (gzr[j]-checkz[j])**2.))
			lpremerr.append(sqrt(abs((checkx[j]-sinh(gar[j])*cos(gbr[j]))**2. + (checky[j]-sinh(gar[j])*sin(gbr[j]))**2. + (checkz[j]-cosh(gar[j]))**2.)))	

		#This is the particle trajectories in the Poincare model
		    
		# #Plot
		fig= plt.figure(figsize=(8,8))

		plt.subplot(2,2,1)
		plt.plot(xc,yc)
		plt.plot(gut,gvt,label="sym T")
		plt.plot(gput,gpvt,label="para T")
		plt.plot(gur,gvr,label="sym R")
		plt.plot(gpur,gpvr,label="para R")
		plt.plot(checku,checkv,label="lie")
		plt.legend(loc='lower left')

		plt.subplot(2,2,2)
		plt.plot(timearr,sptemerr,label="sym/par T")
		plt.plot(timearr,sltemerr,label="sym/lie T")
		plt.plot(timearr,lptemerr,label="lie/par T")
		plt.plot(timearr,spremerr,label="sym/par R")
		plt.plot(timearr,slremerr,label="sym/lie R")
		plt.plot(timearr,lpremerr,label="lie/par R")	
		plt.legend(loc='upper left')	

		plt.subplot(2,2,3)
		plt.plot(timearr,sptemerr,label="sym/par T")
		plt.plot(timearr,sltemerr,label="sym/lie T")
		plt.plot(timearr,lptemerr,label="lie/par T")	
		plt.legend(loc='upper left')	

		plt.subplot(2,2,4)
		plt.plot(timearr,spremerr,label="sym/par R")
		plt.plot(timearr,slremerr,label="sym/lie R")
		plt.plot(timearr,lpremerr,label="lie/par R")	
		plt.legend(loc='upper left')		

	elif(plot=="datasheet"):

		#Differences

		checksymu=[]
		checkparu=[]
		checkvalu=[]
		checksymv=[]
		checkparv=[]
		checkvalv=[]
		checker=0
		for j in checku:
			checksymu.append(abs(gu[checker]-j))
			checkparu.append(abs(gpu[checker]-j))
			checkvalu.append(abs(j-j))
			checksymv.append(abs(gv[checker]-checkv[checker]))
			checkparv.append(abs(gpv[checker]-checkv[checker]))
			checkvalv.append(abs(checkv[checker]-checkv[checker]))	
			checker=checker+1				

		#This is for the horizon circle.
		# Theta goes from 0 to 2pi
		theta = np.linspace(0, 2*np.pi, 100)

		# Compute x1 and x2 for horizon
		xc = np.cos(theta)
		yc = np.sin(theta) 	

		fig = plt.figure(figsize=(15,8))
		fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

		gs = gridspec.GridSpec(3, 4, width_ratios=[6, 2, 2, 2], height_ratios=[6, 2, 2])
		ax1 = fig.add_subplot(gs[0])
		ax1.set_aspect("equal")
		ax1.set_xlabel('x', fontsize=12)
		ax1.set_ylabel('y', fontsize=12)
		ax1.set_title('Trajectory', fontsize=14)
		ax1.plot(xc,yc)
		ax1.plot(gu,gv, label='symmetric')
		ax1.plot(checku,checkv, label='Lie')
		ax1.plot(gpu,gpv, label='paramater')
		ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
		ax2 = fig.add_subplot(gs[5])
		ax2.set_xlabel('x', fontsize=12)
		ax2.set_ylabel('y', fontsize=12)
		ax2.set_title('x coordinate', fontsize=14)
		ax2.plot(timearr,gu, label='symmetric')
		ax2.plot(timearr,checku, label='Lie')
		ax2.plot(timearr,gpu, label='paramater')
		ax2.legend(loc='upper center', bbox_to_anchor=(-.7, .9))	
		ax3 = fig.add_subplot(gs[6])
		ax3.set_xlabel('x', fontsize=12)
		ax3.set_ylabel('y', fontsize=12)
		ax3.set_title('y coordinate', fontsize=14)
		ax3.plot(timearr,gv, label='symmetric')
		ax3.plot(timearr,checkv, label='Lie')
		ax3.plot(timearr,gpv, label='paramater')
		ax5 = fig.add_subplot(gs[9])
		ax5.set_xlabel('x', fontsize=12)
		ax5.set_ylabel('y', fontsize=12)
		ax5.set_title('x error', fontsize=14)
		# ax5.plot(timearr,checksymu, label='symmetric vs. Lie')
		# ax5.plot(timearr,checkvalu, label='Lie')
		ax5.plot(timearr,checkparu, label='paramater vs. Lie')
		ax5.legend(loc='upper center', bbox_to_anchor=(-.7, .9))
		ax6 = fig.add_subplot(gs[10])
		ax6.set_xlabel('x', fontsize=12)
		ax6.set_ylabel('y', fontsize=12)
		ax6.set_title('y error', fontsize=14)
		# ax6.plot(timearr,checksymv, label='symmetric vs. Lie')
		# ax6.plot(timearr,checkvalv, label='Lie')
		ax6.plot(timearr,checkparv, label='paramater vs. Lie')

	elif(plot=="datadiserror"):	

		#Lie generator trajectories

		checkerrdata=[]
		for c in timearr:
			checkerrdata.append(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(ab[1]),matmul(rotz2d(0),matmul(motionmat2dh(abdoti[0], abdoti[1], c),formatvec2d(array([0., 0., 1.]))))))))

		#This formats the data to be compared with the integrator results

		checkx = []
		checky = []
		checkz = []
		for d in range(len(checku)):
			checkx.append(checkerrdata[d][0])
			checky.append(checkerrdata[d][1])
			checkz.append(checkerrdata[d][2])		

		spdiserr=[]
		sldiserr=[]
		lpdiserr=[]
		for j in range(len(checku)):
			spdiserr.append(arccosh(-gx[j]*sinh(ga[j])-gy[j]*cosh(ga[j])*sinh(gb[j])+gz[j]*cosh(ga[j])*cosh(gb[j])))
			sldiserr.append(arccosh(-gx[j]*checkx[j]-gy[j]*checky[j]+gz[j]*checkz[j]))
			lpdiserr.append(arccosh(-checkx[j]*sinh(ga[j])-checky[j]*cosh(ga[j])*sinh(gb[j])+checkz[j]*cosh(ga[j])*cosh(gb[j])))

			# #Plot
		fig= plt.figure(figsize=(5,5))

		# plt.plot(timearr,spdiserr,label="sym/par")
		# plt.plot(timearr,sldiserr,label="sym/lie")
		plt.plot(timearr,lpdiserr,label="lie/par")
		plt.legend()		

	elif(plot=="dataemerror"):	

		#Lie generator trajectories

		checkerrdata=[]
		for c in timearr:
			checkerrdata.append(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(ab[1]),matmul(rotz2d(0),matmul(motionmat2dh(abdoti[0], abdoti[1], c),formatvec2d(array([0., 0., 1.]))))))))

		#This formats the data to be compared with the integrator results

		checkx = []
		checky = []
		checkz = []
		for d in range(len(checku)):
			checkx.append(checkerrdata[d][0])
			checky.append(checkerrdata[d][1])
			checkz.append(checkerrdata[d][2])		

		spemerr=[]
		slemerr=[]
		lpemerr=[]
		pprk4emerr=[]		
		for j in range(len(checku)):
			spemerr.append(sqrt((gx[j]-sinh(ga[j]))**2. + (gy[j]-cosh(ga[j])*sinh(gb[j]))**2. - (gz[j]-cosh(ga[j])*cosh(gb[j]))**2.))
			slemerr.append(sqrt((gx[j]-checkx[j])**2. + (gy[j]-checky[j])**2. - (gz[j]-checkz[j])**2.))
			lpemerr.append(sqrt((checkx[j]-sinh(ga[j]))**2. + (checky[j]-cosh(ga[j])*sinh(gb[j]))**2. - (checkz[j]-cosh(ga[j])*cosh(gb[j]))**2.))
			pprk4emerr.append(sqrt((sinh(grk4a[j])-sinh(ga[j]))**2. + (cosh(grk4a[j])*sinh(grk4b[j])-cosh(ga[j])*sinh(gb[j]))**2. - (cosh(grk4a[j])*cosh(grk4b[j])-cosh(ga[j])*cosh(gb[j]))**2.))


			# #Plot
		fig= plt.figure(figsize=(5,5))

		# plt.plot(timearr,spemerr,label="sym/par")
		# plt.plot(timearr,slemerr,label="sym/lie")
		# plt.plot(timearr,lpemerr,label="lie/par")
		plt.plot(timearr,pprk4emerr,label="prk4/par")		
		plt.legend()		

	fig.tight_layout()	

	plt.show()

if(version=="h3geotrans"):
	#Do the process
	abg = array([ai, bi, gi])
	posi = array([sinh(abg[0]), cosh(abg[0])*sinh(abg[1]), cosh(abg[0])*cosh(abg[1])*sinh(abg[2]), cosh(abg[0])*cosh(abg[1])*cosh(abg[2])])
	posguess = posi
	abgdoti = array([adi, bdi, gdi])
	abgdotguess = abgdoti
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testm = []
	gx = []
	gy = []
	gz = []
	gw = []
	testp = []
	ga = []
	gb = []
	gg = []	

	gx=append(gx, posi[0])
	gy=append(gy, posi[1])
	gz=append(gz, posi[2])
	gw=append(gw, posi[3])
	ga=append(ga, abg[0])
	gb=append(gb, abg[1])
	gg=append(gg, abg[2])
	testm.append(symh3geotrans(posi, posguess, abgdoti, abgdotguess, delT))
	testp.append(imph3geotrans(abg, abg, abgdoti, abgdotguess, delT))		
	gx=append(gx, testm[0][0][0])
	gy=append(gy, testm[0][0][1])
	gz=append(gz, testm[0][0][2])
	gw=append(gw, testm[0][0][3])
	ga=append(ga, testp[0][0][0])
	gb=append(gb, testp[0][0][1])
	gg=append(gg, testp[0][0][2])		
	q=q+1
	while(q < nump-1):
		nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
		nextdotm = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6]])
		nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
		nextdotp = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])				
		testm.append(symh3geotrans(nextposm, nextposm, nextdotm, nextdotm, delT))
		testp.append(imph3geotrans(nextposp, nextposp, nextdotp, nextdotp, delT))		
		gx=append(gx, testm[q][0][0])
		gy=append(gy, testm[q][0][1])
		gz=append(gz, testm[q][0][2])
		gw=append(gw, testm[q][0][3])
		ga=append(ga, testp[q][0][0])
		gb=append(gb, testp[q][0][1])
		gg=append(gg, testp[q][0][2])				
		q=q+1

	gu=[]
	gv=[]
	gs=[]
	gpu=[]
	gpv=[]
	gps=[]	
	for b in range(len(gx)):
	    gu=append(gu,gx[b]/(gw[b] + 1.))
	    gv=append(gv,gy[b]/(gw[b] + 1.))
	    gs=append(gs,gz[b]/(gw[b] + 1.)) 
	    gpu=append(gpu,sinh(ga[b])/(cosh(ga[b])*cosh(gb[b])*cosh(gg[b]) + 1.))
	    gpv=append(gpv,cosh(ga[b])*sinh(gb[b])/(cosh(ga[b])*cosh(gb[b])*cosh(gg[b]) + 1.))
	    gps=append(gps,cosh(ga[b])*cosh(gb[b])*sinh(gg[b])/(cosh(ga[b])*cosh(gb[b])*cosh(gg[b]) + 1.)) 	    	     

	#Lie generator trajectories

	checkhdata=[]
	checkemdata=[]
	for c in timearr:
		checkhdata.append(hyper2poin3d(unformatvec3d(matmul(rotz3d(0.),matmul(roty3d(0.),matmul(boostz3d(abg[2]),matmul(motionmat3dh(abgdoti[0], abgdoti[1], abgdoti[2], c),formatvec3d(array([0., 0., 0., 1.])))))))))
		checkemdata.append(unformatvec3d(matmul(rotz3d(0.),matmul(roty3d(0.),matmul(boostz3d(abg[2]),matmul(motionmat3dh(abgdoti[0], abgdoti[1], abgdoti[2], c),formatvec3d(array([0., 0., 0., 1.]))))))))


	#This formats the data to be compared with the integrator results

	checku = []
	checkv = []
	checks = []
	checkx = []
	checky = []
	checkz = []
	checkw = []	
	for d in range(len(timearr)):
		checku.append(checkhdata[d][0])
		checkv.append(checkhdata[d][1])
		checks.append(checkhdata[d][2])
		checkx.append(checkemdata[d][0])
		checky.append(checkemdata[d][1])
		checkz.append(checkemdata[d][2])
		checkw.append(checkemdata[d][3])		

	if(plot=="datatraj"):	   		

		spemerr=[]
		slemerr=[]
		lpemerr=[]		
		for j in range(len(checku)):
			spemerr.append(sqrt((gx[j]-sinh(ga[j]))**2. + (gy[j]-cosh(ga[j])*sinh(gb[j]))**2. + (gz[j]-cosh(ga[j])*cosh(gb[j])*sinh(gg[j]))**2. - (gw[j]-cosh(ga[j])*cosh(gb[j])*cosh(gg[j]))**2.))
			slemerr.append(sqrt((gx[j]-checkx[j])**2. + (gy[j]-checky[j])**2. + (gz[j]-checkz[j])**2.- (gw[j]-checkw[j])**2.))
			lpemerr.append(sqrt(abs((checkx[j]-sinh(ga[j]))**2. + (checky[j]-cosh(ga[j])*sinh(gb[j]))**2. + (checkz[j]-cosh(ga[j])*cosh(gb[j])*sinh(gg[j]))**2. - (checkw[j]-cosh(ga[j])*cosh(gb[j])*cosh(gg[j]))**2.)))		

		#This is the particle trajectories in the Poincare model
		    
		#Plot
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_aspect("equal")

		#draw sphere
		u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
		x = np.sin(u)*np.cos(v)
		y = np.sin(u)*np.sin(v)
		z = np.cos(u)
		ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
		ax1.set_xlim3d(-1,1)
		ax1.set_ylim3d(-1,1)
		ax1.set_zlim3d(-1,1)

		#draw trajectory
		ax1.plot3D(gu,gv,gs, label="sym")
		ax1.plot3D(gpu,gpv,gps, label="parameter")
		ax1.plot3D(checku,checkv,checks, label="lie")
		ax1.legend(loc= 'lower left')

		ax2=fig.add_subplot(122)
		# ax2.plot(timearr,spemerr,label="sym/par")
		# ax2.plot(timearr,slemerr,label="sym/lie")
		ax2.plot(timearr,lpemerr,label="lie/par")
		# plt.plot(timearr,pprk4emerr,label="prk4/par")		
		ax2.legend(loc='upper left')		

	fig.tight_layout()	

	plt.show()	

if(version=="h2xegeotrans"):
	#Do the process
	abg = array([ai, bi, gi])
	posi = array([sinh(abg[0]), cosh(abg[0])*sinh(abg[1]), abg[2], cosh(abg[0])*cosh(abg[1])])
	posguess = posi
	abgdoti = array([adi, bdi, gdi])
	abgdotguess = abgdoti
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testm = []
	gx = []
	gy = []
	gz = []
	gw = []
	testp = []
	ga = []
	gb = []
	gg = []	

	gx=append(gx, posi[0])
	gy=append(gy, posi[1])
	gz=append(gz, posi[2])
	gw=append(gw, posi[3])
	ga=append(ga, abg[0])
	gb=append(gb, abg[1])
	gg=append(gg, abg[2])
	testm.append(symh2xegeotrans(posi, posguess, abgdoti, abgdotguess, delT))
	testp.append(imph2xegeotrans(abg, abg, abgdoti, abgdotguess, delT))		
	gx=append(gx, testm[0][0][0])
	gy=append(gy, testm[0][0][1])
	gz=append(gz, testm[0][0][2])
	gw=append(gw, testm[0][0][3])
	ga=append(ga, testp[0][0][0])
	gb=append(gb, testp[0][0][1])
	gg=append(gg, testp[0][0][2])		
	q=q+1
	while(q < nump-1):
		nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
		nextdotm = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6]])
		nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
		nextdotp = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])				
		testm.append(symh2xegeotrans(nextposm, nextposm, nextdotm, nextdotm, delT))
		testp.append(imph2xegeotrans(nextposp, nextposp, nextdotp, nextdotp, delT))		
		gx=append(gx, testm[q][0][0])
		gy=append(gy, testm[q][0][1])
		gz=append(gz, testm[q][0][2])
		gw=append(gw, testm[q][0][3])
		ga=append(ga, testp[q][0][0])
		gb=append(gb, testp[q][0][1])
		gg=append(gg, testp[q][0][2])				
		q=q+1

	gu=[]
	gv=[]
	gs=[]
	gpu=[]
	gpv=[]
	gps=[]	
	for b in range(len(gx)):
	    gu=append(gu,gx[b]/(gw[b] + 1.))
	    gv=append(gv,gy[b]/(gw[b] + 1.))
	    gs=append(gs,gz[b]) 
	    gpu=append(gpu,sinh(ga[b])/(cosh(ga[b])*cosh(gb[b]) + 1.))
	    gpv=append(gpv,cosh(ga[b])*sinh(gb[b])/(cosh(ga[b])*cosh(gb[b]) + 1.))
	    gps=append(gps,gg[b]) 	    	     

	#Lie generator trajectories

	checkhdata=[]
	checkemdata=[]
	for c in timearr:
		checkhdata.append(hyper2poin2d(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(abg[1]),matmul(rotz2d(0),matmul(motionmat2dh(abgdoti[0], abgdoti[1], c),formatvec2d(array([0., 0., 1.])))))))))
		checkemdata.append(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(abg[1]),matmul(rotz2d(0),matmul(motionmat2dh(abgdoti[0], abgdoti[1], c),formatvec2d(array([0., 0., 1.]))))))))


	#This formats the data to be compared with the integrator results

	checku = []
	checkv = []
	checks = []
	checkx = []
	checky = []
	checkz = []
	checkw = []	
	for d in range(len(timearr)):
		checku.append(checkhdata[d][0])
		checkv.append(checkhdata[d][1])
		checks.append(gi+gdi*timearr[d])
		checkx.append(checkemdata[d][0])
		checky.append(checkemdata[d][1])
		checkz.append(gi+gdi*timearr[d])
		checkw.append(checkemdata[d][2])		

	if(plot=="datatraj"):	   		

		spemerr=[]
		slemerr=[]
		lpemerr=[]		
		for j in range(len(checku)):
			spemerr.append(sqrt((gx[j]-sinh(ga[j]))**2. + (gy[j]-cosh(ga[j])*sinh(gb[j]))**2. + (gz[j]-gg[j])**2. - (gw[j]-cosh(ga[j])*cosh(gb[j]))**2.))
			slemerr.append(sqrt((gx[j]-checkx[j])**2. + (gy[j]-checky[j])**2. + (gz[j]-checkz[j])**2.- (gw[j]-checkw[j])**2.))
			lpemerr.append(sqrt(abs((checkx[j]-sinh(ga[j]))**2. + (checky[j]-cosh(ga[j])*sinh(gb[j]))**2. + (checkz[j]-gg[j])**2. - (checkw[j]-cosh(ga[j])*cosh(gb[j]))**2.)))		

		#This is the particle trajectories in the Poincare model
		    
		#Plot
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_aspect("equal")

		#draw cylinder
		u, v = np.mgrid[-2.:2.+.2:.2, 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
		x = np.cos(v)
		y = np.sin(v)
		z = u
		ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
		ax1.set_xlim3d(-1,1)
		ax1.set_ylim3d(-1,1)
		ax1.set_zlim3d(-2,2)

		#draw trajectory
		ax1.plot3D(gu,gv,gs, label="sym")
		ax1.plot3D(gpu,gpv,gps, label="parameter")
		ax1.plot3D(checku,checkv,checks, label="lie")
		ax1.legend(loc= 'lower left')

		ax2=fig.add_subplot(122)
		ax2.plot(timearr,spemerr,label="sym/par")
		ax2.plot(timearr,slemerr,label="sym/lie")
		ax2.plot(timearr,lpemerr,label="lie/par")
		# plt.plot(timearr,pprk4emerr,label="prk4/par")		
		ax2.legend(loc='upper left')		

	fig.tight_layout()	

	plt.show()		

if(version=="pbhgeo"):
	#Do the process
	abg = array([ai, bi, gi])
	posi = array([exp(abg[0])*sin(abg[1])*cos(abg[2]), exp(abg[0])*sin(abg[1])*sin(abg[2]), exp(abg[0])*cos(abg[1]), abg[0]])
	posguess = posi
	abgdoti = array([adi, bdi, gdi])
	abgdotguess = abgdoti
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testm = []
	gx = []
	gy = []
	gz = []
	gw = []
	testp = []
	ga = []
	gb = []
	gg = []	

	gx=append(gx, posi[0])
	gy=append(gy, posi[1])
	gz=append(gz, posi[2])
	gw=append(gw, posi[3])
	ga=append(ga, abg[0])
	gb=append(gb, abg[1])
	gg=append(gg, abg[2])
	testm.append(sympbhgeo(posi, posguess, abgdoti, abgdotguess, delT))
	testp.append(imppbhgeo(abg, abg, abgdoti, abgdotguess, delT))		
	gx=append(gx, testm[0][0][0])
	gy=append(gy, testm[0][0][1])
	gz=append(gz, testm[0][0][2])
	gw=append(gw, testm[0][0][3])
	ga=append(ga, testp[0][0][0])
	gb=append(gb, testp[0][0][1])
	gg=append(gg, testp[0][0][2])		
	q=q+1
	while(q < nump-1):
		nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
		nextdotm = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6]])
		nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
		nextdotp = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])				
		testm.append(sympbhgeo(nextposm, nextposm, nextdotm, nextdotm, delT))
		testp.append(imppbhgeo(nextposp, nextposp, nextdotp, nextdotp, delT))		
		gx=append(gx, testm[q][0][0])
		gy=append(gy, testm[q][0][1])
		gz=append(gz, testm[q][0][2])
		gw=append(gw, testm[q][0][3])
		ga=append(ga, testp[q][0][0])
		gb=append(gb, testp[q][0][1])
		gg=append(gg, testp[q][0][2])				
		q=q+1

	gu=[]
	gv=[]
	gs=[]
	gpu=[]
	gpv=[]
	gps=[]	
	for b in range(len(gx)):
	    gu=append(gu,gx[b])
	    gv=append(gv,gy[b])
	    gs=append(gs,gz[b]) 
	    gpu=append(gpu,exp(ga[b])*sin(gb[b])*cos(gg[b]))
	    gpv=append(gpv,exp(ga[b])*sin(gb[b])*sin(gg[b]))
	    gps=append(gps,exp(ga[b])*cos(gb[b])) 	    	     		

	if(plot=="datatraj"):	   		

		spemerr=[]	
		for j in range(len(gu)):
			spemerr.append(sqrt((gx[j]-exp(ga[j])*sin(gb[j])*cos(gg[j]))**2. + (gy[j]-exp(ga[j])*sin(gb[j])*sin(gg[j]))**2. + (gz[j]-exp(ga[j])*cos(gb[j]))**2. + (gw[j]-ga[j])**2.))		

		#This is the particle trajectories in the Poincare model
		    
		#Plot
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_aspect("equal")

		#draw sphere
		u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
		x = np.sin(u)*np.cos(v)
		y = np.sin(u)*np.sin(v)
		z = np.cos(u)
		# ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
		ax1.set_xlim3d(-2,2)
		ax1.set_ylim3d(-2,2)
		ax1.set_zlim3d(-2,2)

		#draw trajectory
		ax1.plot3D(gu,gv,gs, label="sym")
		ax1.plot3D(gpu,gpv,gps, label="parameter")
		ax1.legend(loc= 'lower left')

		ax2=fig.add_subplot(122)
		ax2.plot(timearr,spemerr,label="sym/par")		
		ax2.legend(loc='upper left')		

	fig.tight_layout()	

	plt.show()	

# Central Potential Trajectories

if(version=="h3cprot"):
	#Do the process
	abg = array([ai, bi, gi])
	posi = array([sinh(abg[0])*sin(abg[1])*cos(abg[2]), sinh(abg[0])*sin(abg[1])*sin(abg[2]), sinh(abg[0])*cos(abg[1]), cosh(abg[0])])
	posguess = posi
	abgdoti = array([adi, bdi, gdi])
	abgdotguess = abgdoti
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testm = []
	gx = []
	gy = []
	gz = []
	gw = []
	testp = []
	ga = []
	gb = []
	gg = []	

	gx=append(gx, posi[0])
	gy=append(gy, posi[1])
	gz=append(gz, posi[2])
	gw=append(gw, posi[3])
	ga=append(ga, abg[0])
	gb=append(gb, abg[1])
	gg=append(gg, abg[2])
	testm.append(symh3cprot(posi, posguess, abgdoti, abgdotguess, delT, massvec[0], massvec[1]))
	testp.append(imph3cprot(abg, abg, abgdoti, abgdotguess, delT, massvec[0], massvec[1]))		
	gx=append(gx, testm[0][0][0])
	gy=append(gy, testm[0][0][1])
	gz=append(gz, testm[0][0][2])
	gw=append(gw, testm[0][0][3])
	ga=append(ga, testp[0][0][0])
	gb=append(gb, testp[0][0][1])
	gg=append(gg, testp[0][0][2])		
	q=q+1
	while(q < nump-1):
		nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
		nextdotm = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6]])
		nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
		nextdotp = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])				
		testm.append(symh3cprot(nextposm, nextposm, nextdotm, nextdotm, delT, massvec[0], massvec[1]))
		testp.append(imph3cprot(nextposp, nextposp, nextdotp, nextdotp, delT, massvec[0], massvec[1]))		
		gx=append(gx, testm[q][0][0])
		gy=append(gy, testm[q][0][1])
		gz=append(gz, testm[q][0][2])
		gw=append(gw, testm[q][0][3])
		ga=append(ga, testp[q][0][0])
		gb=append(gb, testp[q][0][1])
		gg=append(gg, testp[q][0][2])				
		q=q+1

	gu=[]
	gv=[]
	gs=[]
	gpu=[]
	gpv=[]
	gps=[]	
	for b in range(len(gx)):
	    gu=append(gu,gx[b]/(gw[b] + 1.))
	    gv=append(gv,gy[b]/(gw[b] + 1.))
	    gs=append(gs,gz[b]/(gw[b] + 1.)) 
	    gpu=append(gpu,sinh(ga[b])*sin(gb[b])*cos(gg[b])/(cosh(ga[b]) + 1.))
	    gpv=append(gpv,sinh(ga[b])*sin(gb[b])*sin(gg[b])/(cosh(ga[b]) + 1.))
	    gps=append(gps,sinh(ga[b])*cos(gb[b])/(cosh(ga[b]) + 1.)) 	    	     


	if(plot=="datatraj"):	   		

		spemerr=[]		
		for j in range(len(gu)):
			spemerr.append(sqrt((gx[j]-sinh(ga[j])*sin(gb[j])*cos(gg[j]))**2. + (gy[j]-sinh(ga[j])*sin(gb[j])*sin(gg[j]))**2. + (gz[j]-sinh(ga[j])*cos(gb[j]))**2. - (gw[j]-cosh(ga[j]))**2.))		

		#This is the particle trajectories in the Poincare model
		    
		#Plot
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_aspect("equal")

		#draw sphere
		u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
		x = np.sin(u)*np.cos(v)
		y = np.sin(u)*np.sin(v)
		z = np.cos(u)
		ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
		ax1.set_xlim3d(-1,1)
		ax1.set_ylim3d(-1,1)
		ax1.set_zlim3d(-1,1)

		#draw trajectory
		ax1.plot3D(gu,gv,gs, label="sym")
		ax1.plot3D(gpu,gpv,gps, label="parameter")
		ax1.legend(loc= 'lower left')

		ax2=fig.add_subplot(122)
		ax2.plot(timearr,spemerr,label="sym/par")	
		ax2.legend(loc='upper left')		

	fig.tight_layout()	

	plt.show()	

if(version=="pbhcp"):
	#Do the process
	abg = array([ai, bi, gi])
	posi = array([exp(abg[0])*sin(abg[1])*cos(abg[2]), exp(abg[0])*sin(abg[1])*sin(abg[2]), exp(abg[0])*cos(abg[1]), abg[0]])
	posguess = posi
	abgdoti = array([adi, bdi, gdi])
	abgdotguess = abgdoti
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testm = []
	gx = []
	gy = []
	gz = []
	gw = []
	testp = []
	ga = []
	gb = []
	gg = []	

	gx=append(gx, posi[0])
	gy=append(gy, posi[1])
	gz=append(gz, posi[2])
	gw=append(gw, posi[3])
	ga=append(ga, abg[0])
	gb=append(gb, abg[1])
	gg=append(gg, abg[2])
	testm.append(sympbhcp(posi, posguess, abgdoti, abgdotguess, delT, massvec[0], massvec[1]))
	testp.append(imppbhcp(abg, abg, abgdoti, abgdotguess, delT, massvec[0], massvec[1]))		
	gx=append(gx, testm[0][0][0])
	gy=append(gy, testm[0][0][1])
	gz=append(gz, testm[0][0][2])
	gw=append(gw, testm[0][0][3])
	ga=append(ga, testp[0][0][0])
	gb=append(gb, testp[0][0][1])
	gg=append(gg, testp[0][0][2])		
	q=q+1
	while(q < nump-1):
		nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
		nextdotm = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6]])
		nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
		nextdotp = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])				
		testm.append(sympbhcp(nextposm, nextposm, nextdotm, nextdotm, delT, massvec[0], massvec[1]))
		testp.append(imppbhcp(nextposp, nextposp, nextdotp, nextdotp, delT, massvec[0], massvec[1]))		
		gx=append(gx, testm[q][0][0])
		gy=append(gy, testm[q][0][1])
		gz=append(gz, testm[q][0][2])
		gw=append(gw, testm[q][0][3])
		ga=append(ga, testp[q][0][0])
		gb=append(gb, testp[q][0][1])
		gg=append(gg, testp[q][0][2])				
		q=q+1

	gu=[]
	gv=[]
	gs=[]
	gpu=[]
	gpv=[]
	gps=[]	
	for b in range(len(gx)):
	    gu=append(gu,gx[b])
	    gv=append(gv,gy[b])
	    gs=append(gs,gz[b]) 
	    gpu=append(gpu,exp(ga[b])*sin(gb[b])*cos(gg[b]))
	    gpv=append(gpv,exp(ga[b])*sin(gb[b])*sin(gg[b]))
	    gps=append(gps,exp(ga[b])*cos(gb[b])) 	    	     		

	if(plot=="datatraj"):	   		

		spemerr=[]	
		for j in range(len(gu)):
			spemerr.append(sqrt((gx[j]-exp(ga[j])*sin(gb[j])*cos(gg[j]))**2. + (gy[j]-exp(ga[j])*sin(gb[j])*sin(gg[j]))**2. + (gz[j]-exp(ga[j])*cos(gb[j]))**2. + (gw[j]-ga[j])**2.))		

		#This is the particle trajectories in the Poincare model
		    
		#Plot
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_aspect("equal")

		#draw sphere
		u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
		x = np.sin(u)*np.cos(v)
		y = np.sin(u)*np.sin(v)
		z = np.cos(u)
		# ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
		ax1.set_xlim3d(-2,2)
		ax1.set_ylim3d(-2,2)
		ax1.set_zlim3d(-2,2)

		#draw trajectory
		ax1.plot3D(gu,gv,gs, label="sym")
		ax1.plot3D(gpu,gpv,gps, label="parameter")
		ax1.legend(loc= 'lower left')

		ax2=fig.add_subplot(122)
		ax2.plot(timearr,spemerr,label="sym/par")		
		ax2.legend(loc='upper left')		

	fig.tight_layout()	

	plt.show()	

# Spring Potential Trajectories

if(version=="h3sptrans"):
	#Do the process
	abg1 = array([a1i, b1i, g1i])
	pos1i = array([sinh(abg1[0]), cosh(abg1[0])*sinh(abg1[1]), cosh(abg1[0])*cosh(abg1[1])*sinh(abg1[2]), cosh(abg1[0])*cosh(abg1[1])*cosh(abg1[2])])
	pos1guess = pos1i
	abgdot1i = array([ad1i, bd1i, gd1i])
	abgdot1guess = abgdot1i

	abg2 = array([a2i, b2i, g2i])
	pos2i = array([sinh(abg2[0]), cosh(abg2[0])*sinh(abg2[1]), cosh(abg2[0])*cosh(abg2[1])*sinh(abg2[2]), cosh(abg2[0])*cosh(abg2[1])*cosh(abg2[2])])
	pos2guess = pos2i
	abgdot2i = array([ad2i, bd2i, gd2i])
	abgdot2guess = abgdot2i	

	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	test1m = []
	g1x = []
	g1y = []
	g1z = []
	g1w = []
	test1p = []
	g1a = []
	g1b = []
	g1g = []	

	test2m = []
	g2x = []
	g2y = []
	g2z = []
	g2w = []
	test2p = []
	g2a = []
	g2b = []
	g2g = []	

	g1x=append(g1x, pos1i[0])
	g1y=append(g1y, pos1i[1])
	g1z=append(g1z, pos1i[2])
	g1w=append(g1w, pos1i[3])
	g1a=append(g1a, abg1[0])
	g1b=append(g1b, abg1[1])
	g1g=append(g1g, abg1[2])

	g2x=append(g2x, pos2i[0])
	g2y=append(g2y, pos2i[1])
	g2z=append(g2z, pos2i[2])
	g2w=append(g2w, pos2i[3])
	g2a=append(g2a, abg2[0])
	g2b=append(g2b, abg2[1])
	g2g=append(g2g, abg2[2])

	test1m.append(symh3sptrans(pos1i, pos1guess, pos2i, pos2guess, abgdot1i, abgdot1guess, abgdot2i, abgdot2guess, delT, massvec, sprcon, eqdist))
	test1p.append(imph3sptrans(abg1, abg1, abg2, abg2, abgdot1i, abgdot1guess, abgdot2i, abgdot2guess, delT, massvec, sprcon, eqdist))

	test2m.append(symh3sptrans(pos2i, pos2guess, pos1i, pos1guess, abgdot2i, abgdot2guess, abgdot1i, abgdot1guess, delT, massvec, sprcon, eqdist))
	test2p.append(imph3sptrans(abg2, abg2, abg1, abg1, abgdot2i, abgdot2guess, abgdot1i, abgdot1guess, delT, massvec, sprcon, eqdist))

	g1x=append(g1x, test1m[0][0][0])
	g1y=append(g1y, test1m[0][0][1])
	g1z=append(g1z, test1m[0][0][2])
	g1w=append(g1w, test1m[0][0][3])
	g1a=append(g1a, test1p[0][0][0])
	g1b=append(g1b, test1p[0][0][1])
	g1g=append(g1g, test1p[0][0][2])		

	g2x=append(g2x, test2m[0][0][0])
	g2y=append(g2y, test2m[0][0][1])
	g2z=append(g2z, test2m[0][0][2])
	g2w=append(g2w, test2m[0][0][3])
	g2a=append(g2a, test2p[0][0][0])
	g2b=append(g2b, test2p[0][0][1])
	g2g=append(g2g, test2p[0][0][2])	

	q=q+1
	while(q < nump-1):

		nextpos1m = array([test1m[q - 1][0][0], test1m[q - 1][0][1], test1m[q - 1][0][2], test1m[q - 1][0][3]])
		nextdot1m = array([test1m[q - 1][0][4], test1m[q - 1][0][5], test1m[q - 1][0][6]])
		nextpos1p = array([test1p[q - 1][0][0], test1p[q - 1][0][1], test1p[q - 1][0][2]])
		nextdot1p = array([test1p[q - 1][0][3], test1p[q - 1][0][4], test1p[q - 1][0][5]])

		nextpos2m = array([test2m[q - 1][0][0], test2m[q - 1][0][1], test2m[q - 1][0][2], test2m[q - 1][0][3]])
		nextdot2m = array([test2m[q - 1][0][4], test2m[q - 1][0][5], test2m[q - 1][0][6]])
		nextpos2p = array([test2p[q - 1][0][0], test2p[q - 1][0][1], test2p[q - 1][0][2]])
		nextdot2p = array([test2p[q - 1][0][3], test2p[q - 1][0][4], test2p[q - 1][0][5]])

		test1m.append(symh3sptrans(nextpos1m, nextpos1m, nextdot1m, nextdot1m, delT, massvec, sprcon, eqdist))
		test1p.append(imph3sptrans(nextpos1p, nextpos1p, nextdot1p, nextdot1p, delT, massvec, sprcon, eqdist))

		test2m.append(symh3sptrans(nextpos2m, nextpos2m, nextdot2m, nextdot2m, delT, massvec, sprcon, eqdist))
		test2p.append(imph3sptrans(nextpos2p, nextpos2p, nextdot2p, nextdot2p, delT, massvec, sprcon, eqdist))

		g1x=append(g1x, test1m[q][0][0])
		g1y=append(g1y, test1m[q][0][1])
		g1z=append(g1z, test1m[q][0][2])
		g1w=append(g1w, test1m[q][0][3])
		g1a=append(g1a, test1p[q][0][0])
		g1b=append(g1b, test1p[q][0][1])
		g1g=append(g1g, test1p[q][0][2])

		g2x=append(g2x, test2m[q][0][0])
		g2y=append(g2y, test2m[q][0][1])
		g2z=append(g2z, test2m[q][0][2])
		g2w=append(g2w, test2m[q][0][3])
		g2a=append(g2a, test2p[q][0][0])
		g2b=append(g2b, test2p[q][0][1])
		g2g=append(g2g, test2p[q][0][2])

		q=q+1

	g1u=[]
	g1v=[]
	g1s=[]
	g1pu=[]
	g1pv=[]
	g1ps=[]

	g2u=[]
	g2v=[]
	g2s=[]
	g2pu=[]
	g2pv=[]
	g2ps=[]

	for b in range(len(g1x)):

	    g1u=append(g1u,g1x[b]/(g1w[b] + 1.))
	    g1v=append(g1v,g1y[b]/(g1w[b] + 1.))
	    g1s=append(g1s,g1z[b]/(g1w[b] + 1.)) 
	    g1pu=append(g1pu,sinh(g1a[b])/(cosh(g1a[b])*cosh(g1b[b])*cosh(g1g[b]) + 1.))
	    g1pv=append(g1pv,cosh(g1a[b])*sinh(g1b[b])/(cosh(g1a[b])*cosh(g1b[b])*cosh(g1g[b]) + 1.))
	    g1ps=append(g1ps,cosh(g1a[b])*cosh(g1b[b])*sinh(g1g[b])/(cosh(g1a[b])*cosh(g1b[b])*cosh(g1g[b]) + 1.)) 	

	    g2u=append(g2u,g2x[b]/(g2w[b] + 1.))
	    g2v=append(g2v,g2y[b]/(g2w[b] + 1.))
	    g2s=append(g2s,g2z[b]/(g2w[b] + 1.)) 
	    g2pu=append(g2pu,sinh(g2a[b])/(cosh(g2a[b])*cosh(g2b[b])*cosh(g2g[b]) + 1.))
	    g2pv=append(g2pv,cosh(g2a[b])*sinh(g2b[b])/(cosh(g2a[b])*cosh(g2b[b])*cosh(g2g[b]) + 1.))
	    g2ps=append(g2ps,cosh(g2a[b])*cosh(g2b[b])*sinh(g2g[b])/(cosh(g2a[b])*cosh(g2b[b])*cosh(g2g[b]) + 1.)) 	        	     
		

	if(plot=="datatraj"):	   		

		spemerr1=[]	
		for j in range(len(checku)):
			spemerr1.append(sqrt((g1x[j]-sinh(g1a[j]))**2. + (g1y[j]-cosh(g1a[j])*sinh(g1b[j]))**2. + (g1z[j]-cosh(g1a[j])*cosh(g1b[j])*sinh(g1g[j]))**2. - (g1w[j]-cosh(g1a[j])*cosh(g1b[j])*cosh(g1g[j]))**2.))	

		#This is the particle trajectories in the Poincare model
		    
		#Plot
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_aspect("equal")

		#draw sphere
		u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
		x = np.sin(u)*np.cos(v)
		y = np.sin(u)*np.sin(v)
		z = np.cos(u)
		ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
		ax1.set_xlim3d(-1,1)
		ax1.set_ylim3d(-1,1)
		ax1.set_zlim3d(-1,1)

		#draw trajectory
		ax1.plot3D(g1u,g1v,g1s, label="sym", color="-r")
		ax1.plot3D(g2u,g2v,g2s, color="-r")
		ax1.plot3D(g1pu,g1pv,g1ps, label="parameter", color="-b")
		ax1.plot3D(g2pu,g2pv,g2ps, color="-b")
		ax1.legend(loc= 'lower left')

		ax2=fig.add_subplot(122)
		ax2.plot(timearr,spemerr1,label="sym/par")
		# plt.plot(timearr,pprk4emerr,label="prk4/par")		
		ax2.legend(loc='upper left')		

	fig.tight_layout()	

	plt.show()	

if(version=="h2xegeotrans"):
	#Do the process
	abg = array([ai, bi, gi])
	posi = array([sinh(abg[0]), cosh(abg[0])*sinh(abg[1]), abg[2], cosh(abg[0])*cosh(abg[1])])
	posguess = posi
	abgdoti = array([adi, bdi, gdi])
	abgdotguess = abgdoti
	nump=maxT/delT
	timearr=np.arange(0,maxT,delT)

	q = 0
	testm = []
	gx = []
	gy = []
	gz = []
	gw = []
	testp = []
	ga = []
	gb = []
	gg = []	

	gx=append(gx, posi[0])
	gy=append(gy, posi[1])
	gz=append(gz, posi[2])
	gw=append(gw, posi[3])
	ga=append(ga, abg[0])
	gb=append(gb, abg[1])
	gg=append(gg, abg[2])
	testm.append(symh2xegeotrans(posi, posguess, abgdoti, abgdotguess, delT))
	testp.append(imph2xegeotrans(abg, abg, abgdoti, abgdotguess, delT))		
	gx=append(gx, testm[0][0][0])
	gy=append(gy, testm[0][0][1])
	gz=append(gz, testm[0][0][2])
	gw=append(gw, testm[0][0][3])
	ga=append(ga, testp[0][0][0])
	gb=append(gb, testp[0][0][1])
	gg=append(gg, testp[0][0][2])		
	q=q+1
	while(q < nump-1):
		nextposm = array([testm[q - 1][0][0], testm[q - 1][0][1], testm[q - 1][0][2], testm[q - 1][0][3]])
		nextdotm = array([testm[q - 1][0][4], testm[q - 1][0][5], testm[q - 1][0][6]])
		nextposp = array([testp[q - 1][0][0], testp[q - 1][0][1], testp[q - 1][0][2]])
		nextdotp = array([testp[q - 1][0][3], testp[q - 1][0][4], testp[q - 1][0][5]])				
		testm.append(symh2xegeotrans(nextposm, nextposm, nextdotm, nextdotm, delT))
		testp.append(imph2xegeotrans(nextposp, nextposp, nextdotp, nextdotp, delT))		
		gx=append(gx, testm[q][0][0])
		gy=append(gy, testm[q][0][1])
		gz=append(gz, testm[q][0][2])
		gw=append(gw, testm[q][0][3])
		ga=append(ga, testp[q][0][0])
		gb=append(gb, testp[q][0][1])
		gg=append(gg, testp[q][0][2])				
		q=q+1

	gu=[]
	gv=[]
	gs=[]
	gpu=[]
	gpv=[]
	gps=[]	
	for b in range(len(gx)):
	    gu=append(gu,gx[b]/(gw[b] + 1.))
	    gv=append(gv,gy[b]/(gw[b] + 1.))
	    gs=append(gs,gz[b]) 
	    gpu=append(gpu,sinh(ga[b])/(cosh(ga[b])*cosh(gb[b]) + 1.))
	    gpv=append(gpv,cosh(ga[b])*sinh(gb[b])/(cosh(ga[b])*cosh(gb[b]) + 1.))
	    gps=append(gps,gg[b]) 	    	     

	#Lie generator trajectories

	checkhdata=[]
	checkemdata=[]
	for c in timearr:
		checkhdata.append(hyper2poin2d(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(abg[1]),matmul(rotz2d(0),matmul(motionmat2dh(abgdoti[0], abgdoti[1], c),formatvec2d(array([0., 0., 1.])))))))))
		checkemdata.append(unformatvec2d(matmul(rotz2d(0),matmul(boosty2d(abg[1]),matmul(rotz2d(0),matmul(motionmat2dh(abgdoti[0], abgdoti[1], c),formatvec2d(array([0., 0., 1.]))))))))


	#This formats the data to be compared with the integrator results

	checku = []
	checkv = []
	checks = []
	checkx = []
	checky = []
	checkz = []
	checkw = []	
	for d in range(len(timearr)):
		checku.append(checkhdata[d][0])
		checkv.append(checkhdata[d][1])
		checks.append(gi+gdi*timearr[d])
		checkx.append(checkemdata[d][0])
		checky.append(checkemdata[d][1])
		checkz.append(gi+gdi*timearr[d])
		checkw.append(checkemdata[d][2])		

	if(plot=="datatraj"):	   		

		spemerr=[]
		slemerr=[]
		lpemerr=[]		
		for j in range(len(checku)):
			spemerr.append(sqrt((gx[j]-sinh(ga[j]))**2. + (gy[j]-cosh(ga[j])*sinh(gb[j]))**2. + (gz[j]-gg[j])**2. - (gw[j]-cosh(ga[j])*cosh(gb[j]))**2.))
			slemerr.append(sqrt((gx[j]-checkx[j])**2. + (gy[j]-checky[j])**2. + (gz[j]-checkz[j])**2.- (gw[j]-checkw[j])**2.))
			lpemerr.append(sqrt(abs((checkx[j]-sinh(ga[j]))**2. + (checky[j]-cosh(ga[j])*sinh(gb[j]))**2. + (checkz[j]-gg[j])**2. - (checkw[j]-cosh(ga[j])*cosh(gb[j]))**2.)))		

		#This is the particle trajectories in the Poincare model
		    
		#Plot
		fig = plt.figure(figsize=(10,5))
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_aspect("equal")

		#draw cylinder
		u, v = np.mgrid[-2.:2.+.2:.2, 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
		x = np.cos(v)
		y = np.sin(v)
		z = u
		ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
		ax1.set_xlim3d(-1,1)
		ax1.set_ylim3d(-1,1)
		ax1.set_zlim3d(-2,2)

		#draw trajectory
		ax1.plot3D(gu,gv,gs, label="sym")
		ax1.plot3D(gpu,gpv,gps, label="parameter")
		ax1.plot3D(checku,checkv,checks, label="lie")
		ax1.legend(loc= 'lower left')

		ax2=fig.add_subplot(122)
		ax2.plot(timearr,spemerr,label="sym/par")
		ax2.plot(timearr,slemerr,label="sym/lie")
		ax2.plot(timearr,lpemerr,label="lie/par")
		# plt.plot(timearr,pprk4emerr,label="prk4/par")		
		ax2.legend(loc='upper left')		

	fig.tight_layout()	

	plt.show()		






























