from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convertpos_hyp2roth3,convertpos_rot2hyph3,initial_con
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
from numpy import zeros,array,arange,sqrt,sin,cos,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

###
# We found that the data is only really usable for the case of l<=6. So I will not be plotting the data for
# values of l that are larger than 6.
###

# Load in the data from .dat file
dist_data=np.genfromtxt("sweep_dist_data.csv")
disp_data=np.genfromtxt("sweep_disp_data.csv")
elong_data=np.genfromtxt("sweep_elong_data.csv")

# Format data
dist_data=dist_data.reshape((1100,4))
disp_data=disp_data.reshape((1100,4))
elong_data=elong_data.reshape((1100,4))

# Pull out the data and separate by l_eq
l_data=[
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ],
    [ 
        [],[],[],[],[],[],[],[],[],[],[]
    ]
]
for a in disp_data:
    if a[1]==1.: # l=1
        if a[2]==0.: # k=0
            l_data[0][0].append(a)
        elif a[2]==1.: # k=1
            l_data[0][1].append(a)
        elif a[2]==2.: # k=2
            l_data[0][2].append(a)
        elif a[2]==3.: # k=3
            l_data[0][3].append(a)
        elif a[2]==4.: # k=4
            l_data[0][4].append(a)
        elif a[2]==5.: # k=5
            l_data[0][5].append(a)
        elif a[2]==6.: # k=6
            l_data[0][6].append(a)
        elif a[2]==7.: # k=7
            l_data[0][7].append(a)
        elif a[2]==8.: # k=8
            l_data[0][8].append(a)
        elif a[2]==9.: # k=9
            l_data[0][9].append(a)
        elif a[2]==10.: # k=10
            l_data[0][10].append(a)
        elif a[2]==11.: # k=11
            l_data[0][11].append(a)
    elif a[1]==2.:# l=1
        if a[2]==0.:
            l_data[1][0].append(a)
        elif a[2]==1.:
            l_data[1][1].append(a)
        elif a[2]==2.:
            l_data[1][2].append(a)
        elif a[2]==3.:
            l_data[1][3].append(a)
        elif a[2]==4.:
            l_data[1][4].append(a)
        elif a[2]==5.:
            l_data[1][5].append(a)
        elif a[2]==6.:
            l_data[1][6].append(a)
        elif a[2]==7.:
            l_data[1][7].append(a)
        elif a[2]==8.:
            l_data[1][8].append(a)
        elif a[2]==9.:
            l_data[1][9].append(a)
        elif a[2]==10.:
            l_data[1][10].append(a)
        elif a[2]==11.:
            l_data[1][11].append(a)
    elif a[1]==3.:
        if a[2]==0.:
            l_data[2][0].append(a)
        elif a[2]==1.:
            l_data[2][1].append(a)
        elif a[2]==2.:
            l_data[2][2].append(a)
        elif a[2]==3.:
            l_data[2][3].append(a)
        elif a[2]==4.:
            l_data[2][4].append(a)
        elif a[2]==5.:
            l_data[2][5].append(a)
        elif a[2]==6.:
            l_data[2][6].append(a)
        elif a[2]==7.:
            l_data[2][7].append(a)
        elif a[2]==8.:
            l_data[2][8].append(a)
        elif a[2]==9.:
            l_data[2][9].append(a)
        elif a[2]==10.:
            l_data[2][10].append(a)
        elif a[2]==11.:
            l_data[2][11].append(a)
    elif a[1]==4.:
        if a[2]==0.:
            l_data[3][0].append(a)
        elif a[2]==1.:
            l_data[3][1].append(a)
        elif a[2]==2.:
            l_data[3][2].append(a)
        elif a[2]==3.:
            l_data[3][3].append(a)
        elif a[2]==4.:
            l_data[3][4].append(a)
        elif a[2]==5.:
            l_data[3][5].append(a)
        elif a[2]==6.:
            l_data[3][6].append(a)
        elif a[2]==7.:
            l_data[3][7].append(a)
        elif a[2]==8.:
            l_data[3][8].append(a)
        elif a[2]==9.:
            l_data[3][9].append(a)
        elif a[2]==10.:
            l_data[3][10].append(a)
        elif a[2]==11.:
            l_data[3][11].append(a)
    elif a[1]==5.:
        if a[2]==0.:
            l_data[4][0].append(a)
        elif a[2]==1.:
            l_data[4][1].append(a)
        elif a[2]==2.:
            l_data[4][2].append(a)
        elif a[2]==3.:
            l_data[4][3].append(a)
        elif a[2]==4.:
            l_data[4][4].append(a)
        elif a[2]==5.:
            l_data[4][5].append(a)
        elif a[2]==6.:
            l_data[4][6].append(a)
        elif a[2]==7.:
            l_data[4][7].append(a)
        elif a[2]==8.:
            l_data[4][8].append(a)
        elif a[2]==9.:
            l_data[4][9].append(a)
        elif a[2]==10.:
            l_data[4][10].append(a)
        elif a[2]==11.:
            l_data[4][11].append(a)
    elif a[1]==6.:
        if a[2]==0.:
            l_data[5][0].append(a)
        elif a[2]==1.:
            l_data[5][1].append(a)
        elif a[2]==2.:
            l_data[5][2].append(a)
        elif a[2]==3.:
            l_data[5][3].append(a)
        elif a[2]==4.:
            l_data[5][4].append(a)
        elif a[2]==5.:
            l_data[5][5].append(a)
        elif a[2]==6.:
            l_data[5][6].append(a)
        elif a[2]==7.:
            l_data[5][7].append(a)
        elif a[2]==8.:
            l_data[5][8].append(a)
        elif a[2]==9.:
            l_data[5][9].append(a)
        elif a[2]==10.:
            l_data[5][10].append(a)
        elif a[2]==11.:
            l_data[5][11].append(a)

# Plot
fig = plt.figure(figsize=(17,6))

ax1=fig.add_subplot(2,4,1)
ax2=fig.add_subplot(2,4,2)
ax3=fig.add_subplot(2,4,3)
ax4=fig.add_subplot(2,4,4)
ax5=fig.add_subplot(2,4,5)
ax6=fig.add_subplot(2,4,6)

# This is force as a function of initial velocity

ax1.scatter((np.array(l_data[0][0]).flatten())[0::4],((np.array(l_data[0][0]).flatten())[3::4])*(np.array(l_data[0][0]).flatten())[2::4],label="k=0")
ax1.scatter((np.array(l_data[0][1]).flatten())[0::4],((np.array(l_data[0][1]).flatten())[3::4])*(np.array(l_data[0][1]).flatten())[2::4],label="k=1")
ax1.scatter((np.array(l_data[0][2]).flatten())[0::4],((np.array(l_data[0][2]).flatten())[3::4])*(np.array(l_data[0][2]).flatten())[2::4],label="k=2")
ax1.scatter((np.array(l_data[0][3]).flatten())[0::4],((np.array(l_data[0][3]).flatten())[3::4])*(np.array(l_data[0][3]).flatten())[2::4],label="k=3")
ax1.scatter((np.array(l_data[0][4]).flatten())[0::4],((np.array(l_data[0][4]).flatten())[3::4])*(np.array(l_data[0][4]).flatten())[2::4],label="k=4")
ax1.scatter((np.array(l_data[0][5]).flatten())[0::4],((np.array(l_data[0][5]).flatten())[3::4])*(np.array(l_data[0][5]).flatten())[2::4],label="k=5")
ax1.scatter((np.array(l_data[0][6]).flatten())[0::4],((np.array(l_data[0][6]).flatten())[3::4])*(np.array(l_data[0][6]).flatten())[2::4],label="k=6")
ax1.scatter((np.array(l_data[0][7]).flatten())[0::4],((np.array(l_data[0][7]).flatten())[3::4])*(np.array(l_data[0][7]).flatten())[2::4],label="k=7")
ax1.scatter((np.array(l_data[0][8]).flatten())[0::4],((np.array(l_data[0][8]).flatten())[3::4])*(np.array(l_data[0][8]).flatten())[2::4],label="k=8")
ax1.scatter((np.array(l_data[0][9]).flatten())[0::4],((np.array(l_data[0][9]).flatten())[3::4])*(np.array(l_data[0][9]).flatten())[2::4],label="k=9")
ax1.scatter((np.array(l_data[0][10]).flatten())[0::4],((np.array(l_data[0][10]).flatten())[3::4])*(np.array(l_data[0][10]).flatten())[2::4],label="k=10")
ax1.set_title("l=1")
ax1.set_xlim([0,11])
ax1.set_xlabel('Initial Velocity')
ax1.set_ylabel('Max Force')
# ax1.legend(loc='lower right')

ax2.scatter((np.array(l_data[1][0]).flatten())[0::4],((np.array(l_data[1][0]).flatten())[3::4])*(np.array(l_data[1][0]).flatten())[2::4],label="k=0")
ax2.scatter((np.array(l_data[1][1]).flatten())[0::4],((np.array(l_data[1][1]).flatten())[3::4])*(np.array(l_data[1][1]).flatten())[2::4],label="k=1")
ax2.scatter((np.array(l_data[1][2]).flatten())[0::4],((np.array(l_data[1][2]).flatten())[3::4])*(np.array(l_data[1][2]).flatten())[2::4],label="k=2")
ax2.scatter((np.array(l_data[1][3]).flatten())[0::4],((np.array(l_data[1][3]).flatten())[3::4])*(np.array(l_data[1][3]).flatten())[2::4],label="k=3")
ax2.scatter((np.array(l_data[1][4]).flatten())[0::4],((np.array(l_data[1][4]).flatten())[3::4])*(np.array(l_data[1][4]).flatten())[2::4],label="k=4")
ax2.scatter((np.array(l_data[1][5]).flatten())[0::4],((np.array(l_data[1][5]).flatten())[3::4])*(np.array(l_data[1][5]).flatten())[2::4],label="k=5")
ax2.scatter((np.array(l_data[1][6]).flatten())[0::4],((np.array(l_data[1][6]).flatten())[3::4])*(np.array(l_data[1][6]).flatten())[2::4],label="k=6")
ax2.scatter((np.array(l_data[1][7]).flatten())[0::4],((np.array(l_data[1][7]).flatten())[3::4])*(np.array(l_data[1][7]).flatten())[2::4],label="k=7")
ax2.scatter((np.array(l_data[1][8]).flatten())[0::4],((np.array(l_data[1][8]).flatten())[3::4])*(np.array(l_data[1][8]).flatten())[2::4],label="k=8")
ax2.scatter((np.array(l_data[1][9]).flatten())[0::4],((np.array(l_data[1][9]).flatten())[3::4])*(np.array(l_data[1][9]).flatten())[2::4],label="k=9")
ax2.scatter((np.array(l_data[1][10]).flatten())[0::4],((np.array(l_data[1][10]).flatten())[3::4])*(np.array(l_data[1][10]).flatten())[2::4],label="k=10")
ax2.set_title("l=2")
ax2.set_xlim([0,11])
ax2.set_xlabel('Initial Velocity')
ax2.set_ylabel('Max Force')
# ax2.legend(loc='lower right')

ax3.scatter((np.array(l_data[2][0]).flatten())[0::4],((np.array(l_data[2][0]).flatten())[3::4])*(np.array(l_data[2][0]).flatten())[2::4],label="k=0")
ax3.scatter((np.array(l_data[2][1]).flatten())[0::4],((np.array(l_data[2][1]).flatten())[3::4])*(np.array(l_data[2][1]).flatten())[2::4],label="k=1")
ax3.scatter((np.array(l_data[2][2]).flatten())[0::4],((np.array(l_data[2][2]).flatten())[3::4])*(np.array(l_data[2][2]).flatten())[2::4],label="k=2")
ax3.scatter((np.array(l_data[2][3]).flatten())[0::4],((np.array(l_data[2][3]).flatten())[3::4])*(np.array(l_data[2][3]).flatten())[2::4],label="k=3")
ax3.scatter((np.array(l_data[2][4]).flatten())[0::4],((np.array(l_data[2][4]).flatten())[3::4])*(np.array(l_data[2][4]).flatten())[2::4],label="k=4")
ax3.scatter((np.array(l_data[2][5]).flatten())[0::4],((np.array(l_data[2][5]).flatten())[3::4])*(np.array(l_data[2][5]).flatten())[2::4],label="k=5")
ax3.scatter((np.array(l_data[2][6]).flatten())[0::4],((np.array(l_data[2][6]).flatten())[3::4])*(np.array(l_data[2][6]).flatten())[2::4],label="k=6")
ax3.scatter((np.array(l_data[2][7]).flatten())[0::4],((np.array(l_data[2][7]).flatten())[3::4])*(np.array(l_data[2][7]).flatten())[2::4],label="k=7")
ax3.scatter((np.array(l_data[2][8]).flatten())[0::4],((np.array(l_data[2][8]).flatten())[3::4])*(np.array(l_data[2][8]).flatten())[2::4],label="k=8")
ax3.scatter((np.array(l_data[2][9]).flatten())[0::4],((np.array(l_data[2][9]).flatten())[3::4])*(np.array(l_data[2][9]).flatten())[2::4],label="k=9")
ax3.scatter((np.array(l_data[2][10]).flatten())[0::4],((np.array(l_data[2][10]).flatten())[3::4])*(np.array(l_data[2][10]).flatten())[2::4],label="k=10")
ax3.set_title("l=3")
ax3.set_xlim([0,11])
ax3.set_xlabel('Initial Velocity')
ax3.set_ylabel('Max Force')
# ax3.legend(loc='lower right')

ax4.scatter((np.array(l_data[3][0]).flatten())[0::4],((np.array(l_data[3][0]).flatten())[3::4])*(np.array(l_data[3][0]).flatten())[2::4],label="k=0")
ax4.scatter((np.array(l_data[3][1]).flatten())[0::4],((np.array(l_data[3][1]).flatten())[3::4])*(np.array(l_data[3][1]).flatten())[2::4],label="k=1")
ax4.scatter((np.array(l_data[3][2]).flatten())[0::4],((np.array(l_data[3][2]).flatten())[3::4])*(np.array(l_data[3][2]).flatten())[2::4],label="k=2")
ax4.scatter((np.array(l_data[3][3]).flatten())[0::4],((np.array(l_data[3][3]).flatten())[3::4])*(np.array(l_data[3][3]).flatten())[2::4],label="k=3")
ax4.scatter((np.array(l_data[3][4]).flatten())[0::4],((np.array(l_data[3][4]).flatten())[3::4])*(np.array(l_data[3][4]).flatten())[2::4],label="k=4")
ax4.scatter((np.array(l_data[3][5]).flatten())[0::4],((np.array(l_data[3][5]).flatten())[3::4])*(np.array(l_data[3][5]).flatten())[2::4],label="k=5")
ax4.scatter((np.array(l_data[3][6]).flatten())[0::4],((np.array(l_data[3][6]).flatten())[3::4])*(np.array(l_data[3][6]).flatten())[2::4],label="k=6")
ax4.scatter((np.array(l_data[3][7]).flatten())[0::4],((np.array(l_data[3][7]).flatten())[3::4])*(np.array(l_data[3][7]).flatten())[2::4],label="k=7")
ax4.scatter((np.array(l_data[3][8]).flatten())[0::4],((np.array(l_data[3][8]).flatten())[3::4])*(np.array(l_data[3][8]).flatten())[2::4],label="k=8")
ax4.scatter((np.array(l_data[3][9]).flatten())[0::4],((np.array(l_data[3][9]).flatten())[3::4])*(np.array(l_data[3][9]).flatten())[2::4],label="k=9")
ax4.scatter((np.array(l_data[3][10]).flatten())[0::4],((np.array(l_data[3][10]).flatten())[3::4])*(np.array(l_data[3][10]).flatten())[2::4],label="k=10")
ax4.set_title("l=4")
ax4.set_xlim([0,11])
ax4.set_xlabel('Initial Velocity')
ax4.set_ylabel('Max Force')
# ax4.legend(loc='lower right')

ax5.scatter((np.array(l_data[4][0]).flatten())[0::4],((np.array(l_data[4][0]).flatten())[3::4])*(np.array(l_data[4][0]).flatten())[2::4],label="k=0")
ax5.scatter((np.array(l_data[4][1]).flatten())[0::4],((np.array(l_data[4][1]).flatten())[3::4])*(np.array(l_data[4][1]).flatten())[2::4],label="k=1")
ax5.scatter((np.array(l_data[4][2]).flatten())[0::4],((np.array(l_data[4][2]).flatten())[3::4])*(np.array(l_data[4][2]).flatten())[2::4],label="k=2")
ax5.scatter((np.array(l_data[4][3]).flatten())[0::4],((np.array(l_data[4][3]).flatten())[3::4])*(np.array(l_data[4][3]).flatten())[2::4],label="k=3")
ax5.scatter((np.array(l_data[4][4]).flatten())[0::4],((np.array(l_data[4][4]).flatten())[3::4])*(np.array(l_data[4][4]).flatten())[2::4],label="k=4")
ax5.scatter((np.array(l_data[4][5]).flatten())[0::4],((np.array(l_data[4][5]).flatten())[3::4])*(np.array(l_data[4][5]).flatten())[2::4],label="k=5")
ax5.scatter((np.array(l_data[4][6]).flatten())[0::4],((np.array(l_data[4][6]).flatten())[3::4])*(np.array(l_data[4][6]).flatten())[2::4],label="k=6")
ax5.scatter((np.array(l_data[4][7]).flatten())[0::4],((np.array(l_data[4][7]).flatten())[3::4])*(np.array(l_data[4][7]).flatten())[2::4],label="k=7")
ax5.scatter((np.array(l_data[4][8]).flatten())[0::4],((np.array(l_data[4][8]).flatten())[3::4])*(np.array(l_data[4][8]).flatten())[2::4],label="k=8")
ax5.scatter((np.array(l_data[4][9]).flatten())[0::4],((np.array(l_data[4][9]).flatten())[3::4])*(np.array(l_data[4][9]).flatten())[2::4],label="k=9")
ax5.scatter((np.array(l_data[4][10]).flatten())[0::4],((np.array(l_data[4][10]).flatten())[3::4])*(np.array(l_data[4][10]).flatten())[2::4],label="k=10")
ax5.set_title("l=5")
ax5.set_xlim([0,11])
ax5.set_xlabel('Initial Velocity')
ax5.set_ylabel('Max Force')
# ax5.legend(loc='lower right')

ax6.scatter((np.array(l_data[5][0]).flatten())[0::4],((np.array(l_data[5][0]).flatten())[3::4])*(np.array(l_data[5][0]).flatten())[2::4],label="k=0")
ax6.scatter((np.array(l_data[5][1]).flatten())[0::4],((np.array(l_data[5][1]).flatten())[3::4])*(np.array(l_data[5][1]).flatten())[2::4],label="k=1")
ax6.scatter((np.array(l_data[5][2]).flatten())[0::4],((np.array(l_data[5][2]).flatten())[3::4])*(np.array(l_data[5][2]).flatten())[2::4],label="k=2")
ax6.scatter((np.array(l_data[5][3]).flatten())[0::4],((np.array(l_data[5][3]).flatten())[3::4])*(np.array(l_data[5][3]).flatten())[2::4],label="k=3")
ax6.scatter((np.array(l_data[5][4]).flatten())[0::4],((np.array(l_data[5][4]).flatten())[3::4])*(np.array(l_data[5][4]).flatten())[2::4],label="k=4")
ax6.scatter((np.array(l_data[5][5]).flatten())[0::4],((np.array(l_data[5][5]).flatten())[3::4])*(np.array(l_data[5][5]).flatten())[2::4],label="k=5")
ax6.scatter((np.array(l_data[5][6]).flatten())[0::4],((np.array(l_data[5][6]).flatten())[3::4])*(np.array(l_data[5][6]).flatten())[2::4],label="k=6")
ax6.scatter((np.array(l_data[5][7]).flatten())[0::4],((np.array(l_data[5][7]).flatten())[3::4])*(np.array(l_data[5][7]).flatten())[2::4],label="k=7")
ax6.scatter((np.array(l_data[5][8]).flatten())[0::4],((np.array(l_data[5][8]).flatten())[3::4])*(np.array(l_data[5][8]).flatten())[2::4],label="k=8")
ax6.scatter((np.array(l_data[5][9]).flatten())[0::4],((np.array(l_data[5][9]).flatten())[3::4])*(np.array(l_data[5][9]).flatten())[2::4],label="k=9")
ax6.scatter((np.array(l_data[5][10]).flatten())[0::4],((np.array(l_data[5][10]).flatten())[3::4])*(np.array(l_data[5][10]).flatten())[2::4],label="k=10")
ax6.set_title("l=6")
ax6.set_xlim([0,11])
ax6.set_xlabel('Initial Velocity')
ax6.set_ylabel('Max Force')
# ax6.legend(loc='lower right')
handles, labels = ax6.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')

fig.tight_layout()	

plt.show()