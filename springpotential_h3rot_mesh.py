from asyncio import new_event_loop
from symint_bank import imph3sprot3,imph3sprot3_condense_econ, imph3sprot_mesh
from function_bank import hyper2poinh3,h3dist,boostxh3,rotxh3,rotyh3,rotzh3,hypercirch3,collisionh3,convertpos_hyp2roth3,convertpos_rot2hyph3,initial_con
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation, rc
import numpy as np
import copy as cp
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,pi

# Here we take a mesh data structure as an initial condition for the positions
# and connections acting as the springs for the edges. The generic structure is
# given to be taken in the form of
# [ 
#   array of vertex position (x,y,z) , 
#   array of tuples of vertex indices forming egdes (1,2) ,
#   array of 3-tuple of vertex indices forming facets (1,2,3)
#  ]
# Needs to take care to scale it an appropriate size since the euclidean coordinates
# will be interpreted as coordinates in Poincare disk (probably not best choice, but
# that is what we will use for now). (Rotational parameterization for position and
# velocity values)

# The initial velocities of the vertices will be given through the killing field 
# initializer which describes a general loxodromic killing field along the x-axis.
# This will form the second array needed to initialize the problem. With the third 
# array containing the data concerning the mass of each vertex, and values for stiffness
# and rest length of the spring/edges between each vertex.

# Manually generate mesh data
# mesh=[ # Dumbbell mesh
#     [ # Vertex position 
#         [.5,np.pi/2.,1.*np.pi/2.],      #particle 1
#         [.5,np.pi/2.,3.*np.pi/2.]      #particle 2
#     ]
#     ,
#     [ # Edge connection given by vertex indices
#         [0,1]
#     ]
#     ,
#     [ # Face given by the vertex indices (here for completeness)
        
#     ]
# ]
# mesh=[ # Equilateral triangle mesh
#     [ # Vertex position 
#         [.5,np.pi/2.,0.*2.*np.pi/3.],      #particle 1
#         [.5,np.pi/2.,1.*2.*np.pi/3.],      #particle 2
#         [.5,np.pi/2.,2.*2.*np.pi/3.]       #particle 3
#     ]
#     # [ # Vertex position (Analytic check)
#     #     [arccosh(cosh(1)/cosh(.5)),np.pi/2.,0.],      #particle 1
#     #     [.5,np.pi/2.,1.*np.pi/2.],                    #particle 2
#     #     [.5,np.pi/2.,3.*np.pi/2.]                     #particle 3
#     # ]
#     ,
#     [ # Edge connection given by vertex indices
#         [0,1],
#         [0,2],
#         [1,2]
#     ]
#     ,
#     [ # Face given by the vertex indices (here for completeness)
#         [0,1,2]
#     ]
# ]

# mesh=[ # D20
#     [
#         [0.5, 3.04159, 1.5708], 
#         [0.5, 0.1, -1.5708], 
#         [0.5, 2.03195, 
#         3.09172], [0.5, 1.10965, -0.0498753], [0.5, 
#         2.09147, -0.584217], [0.5, 1.97414, 0.665288], [0.5, 
#         1.16745, -2.4763], [0.5, 1.05012, 2.55738], [0.5, 
#         2.12928, -1.90277], [0.5, 1.93912, 1.87157], [0.5, 
#         1.20247, -1.27002], [0.5, 1.01231, 1.23882]
#     ], 
#     [
#         [1, 11], [11, 
#         7], [7, 1], [7, 6], [6, 1], [6, 10], [10, 1], [10, 3], [3, 1], [3, 
#         11], [4, 8], [8, 0], [0, 4], [5, 4], [0, 5], [9, 5], [0, 9], [2, 
#         9], [0, 2], [8, 2], [11, 9], [9, 7], [7, 2], [2, 6], [6, 8], [8, 
#         10], [10, 4], [4, 3], [3, 5], [5, 11]
#    ], 
#    [
#        [1, 11, 7], [1, 7, 6], [1,
#         6, 10], [1, 10, 3], [1, 3, 11], [4, 8, 0], [5, 4, 0], [9, 5, 
#         0], [2, 9, 0], [8, 2, 0], [11, 9, 7], [7, 2, 6], [6, 8, 10], [10, 
#         4, 3], [3, 5, 11], [4, 10, 8], [5, 3, 4], [9, 11, 5], [2, 7, 
#         9], [8, 6, 2]
#    ]
# ]

mesh=[[[0.5, 0.65887, -3.01165], [0.5, 1.10965, -0.0498753], [0.5, 
   0.1, -1.5708], [0.5, 1.16745, -2.4763], [0.5, 1.05012, 
   2.55738], [0.5, 3.04159, 1.5708], [0.5, 1.20247, -1.27002], [0.5, 
   1.01231, 1.23882], [0.5, 0.748007, -1.85022], [0.5, 0.557986, 
   1.93296], [0.5, 0.598277, 0.511537], [0.5, 
   0.715048, -0.724155], [0.5, 2.48272, 0.129947], [0.5, 
   2.58361, -1.20863], [0.5, 2.39359, 1.29138], [0.5, 2.42654, 
   2.41744], [0.5, 2.54332, -2.63006], [0.5, 
   1.44142, -0.641245], [0.5, 1.47727, -1.88059], [0.5, 1.324, 
   0.610286], [0.5, 1.28709, 1.89249], [0.5, 1.38304, -3.12253], [0.5,
    1.75855, 0.0190642], [0.5, 1.8545, -1.24911], [0.5, 1.66433, 
   1.261], [0.5, 1.81759, -2.53131], [0.5, 1.70017, 2.50035], [0.5, 
   2.09147, -0.584217], [0.5, 2.03195, 3.09172], [0.5, 1.97414, 
   0.665288], [0.5, 2.12928, -1.90277], [0.5, 1.93912, 1.87157], [0.5,
    1.32911, -1.58555], [0.5, 0.955779, -1.51338], [0.5, 
   1.11259, -1.86826], [0.5, 1.12913, 1.58664], [0.5, 0.922447, 
   1.90687], [0.5, 0.756169, 1.50243], [0.5, 2.01246, -1.55495], [0.5,
    2.38542, -1.63916], [0.5, 2.21915, -1.23472], [0.5, 1.81249, 
   1.55604], [0.5, 2.029, 1.27333], [0.5, 2.18581, 1.62822], [0.5, 
   2.86264, -2.79772], [0.5, 2.90853, -1.12771], [0.5, 
   2.68219, -1.94592], [0.5, 1.6708, -1.5708], [0.5, 
   1.80329, -1.89079], [0.5, 2.71922, 1.32684], [0.5, 2.74852, 
   2.3142], [0.5, 2.49225, 1.84282], [0.5, 1.61311, 1.88218], [0.5, 
   1.4708, 1.5708], [0.5, 0.422376, -1.81476], [0.5, 
   0.649338, -1.29877], [0.5, 0.393068, -0.827392], [0.5, 
   1.52848, -1.25941], [0.5, 0.233058, 2.01388], [0.5, 0.278954, 
   0.343876], [0.5, 0.459402, 1.19567], [0.5, 1.3383, 1.2508], [0.5, 
   2.80095, 0.286989], [0.5, 2.6409, -0.482071], [0.5, 2.52448, 
   0.745151], [0.5, 1.30969, -0.945712], [0.5, 
   1.65165, -0.940094], [0.5, 1.14983, 0.902805], [0.5, 1.48994, 
   0.940094], [0.5, 1.26013, -2.81045], [0.5, 
   0.896282, -2.68898], [0.5, 1.02031, -3.07997], [0.5, 1.20135, 
   2.87835], [0.5, 0.823136, 2.85019], [0.5, 
   0.934657, -2.21169], [0.5, 0.617116, -2.39644], [0.5, 0.50069, 
   2.65952], [0.5, 0.780803, 2.3231], [0.5, 1.99177, -2.23879], [0.5, 
   2.18098, -2.56758], [0.5, 2.36864, -2.18976], [0.5, 1.8319, 
   2.19588], [0.5, 2.20011, 2.09569], [0.5, 2.0637, 2.46735], [0.5, 
   0.34064, -2.8546], [0.5, 2.31846, -0.291404], [0.5, 
   2.36079, -0.818489], [0.5, 2.24531, 0.452616], [0.5, 2.20694, 
   0.929901], [0.5, 1.07789, -0.67424], [0.5, 
   0.941479, -1.04591], [0.5, 0.772957, 0.951837], [0.5, 0.960609, 
   0.574014], [0.5, 2.25676, 2.80873], [0.5, 2.57999, 2.98144], [0.5, 
   2.30639, -2.97635], [0.5, 0.835207, 0.165243], [0.5, 
   0.561606, -0.16015], [0.5, 0.884833, -0.332861], [0.5, 
   1.65165, -2.2015], [0.5, 1.48994, 2.2015], [0.5, 
   1.94316, -2.84777], [0.5, 1.87861, 2.78052], [0.5, 
   1.94024, -0.263247], [0.5, 2.12129, 0.0616223], [0.5, 1.88146, 
   0.331144], [0.5, 1.60165, -0.31269], [0.5, 
   1.76652, -0.614636], [0.5, 1.6491, 0.63706], [0.5, 1.53994, 
   0.31269], [0.5, 1.43402, -0.0138102], [0.5, 
   1.26299, -0.361071], [0.5, 1.19843, 0.293827], [0.5, 
   1.31143, -2.16629], [0.5, 1.14798, 2.20745], [0.5, 
   1.99361, -0.934143], [0.5, 1.83016, 0.975305], [0.5, 
   1.60165, -2.8289], [0.5, 1.70757, 3.12778], [0.5, 1.53994, 
   2.8289], [0.5, 1.49249, -2.50453], [0.5, 1.37507, 2.52696]], [[6, 
   33], [33, 32], [32, 6], [33, 34], [34, 32], [34, 18], [18, 
   32], [33, 8], [8, 34], [20, 36], [36, 35], [35, 20], [36, 37], [37,
    35], [37, 7], [7, 35], [36, 9], [9, 37], [30, 39], [39, 38], [38, 
   30], [39, 40], [40, 38], [40, 23], [23, 38], [39, 13], [13, 
   40], [24, 42], [42, 41], [41, 24], [42, 43], [43, 41], [43, 
   31], [31, 41], [42, 14], [14, 43], [5, 45], [45, 44], [44, 5], [45,
    46], [46, 44], [46, 16], [16, 44], [45, 13], [13, 46], [23, 
   47], [47, 38], [47, 48], [48, 38], [48, 30], [47, 18], [18, 
   48], [5, 50], [50, 49], [49, 5], [50, 51], [51, 49], [51, 14], [14,
    49], [50, 15], [15, 51], [20, 53], [53, 52], [52, 20], [53, 
   41], [41, 52], [31, 52], [53, 24], [8, 55], [55, 54], [54, 8], [55,
    56], [56, 54], [56, 2], [2, 54], [55, 11], [11, 56], [32, 
   57], [57, 6], [32, 47], [47, 57], [23, 57], [2, 59], [59, 58], [58,
    2], [59, 60], [60, 58], [60, 9], [9, 58], [59, 10], [10, 60], [7, 
   61], [61, 35], [61, 53], [53, 35], [61, 24], [12, 63], [63, 
   62], [62, 12], [63, 45], [45, 62], [5, 62], [63, 13], [49, 
   62], [49, 64], [64, 62], [64, 12], [14, 64], [57, 65], [65, 
   6], [57, 66], [66, 65], [66, 17], [17, 65], [23, 66], [7, 67], [67,
    61], [67, 68], [68, 61], [68, 24], [67, 19], [19, 68], [3, 
   70], [70, 69], [69, 3], [70, 71], [71, 69], [71, 21], [21, 
   69], [70, 0], [0, 71], [71, 72], [72, 21], [71, 73], [73, 72], [73,
    4], [4, 72], [0, 73], [8, 75], [75, 74], [74, 8], [75, 70], [70, 
   74], [3, 74], [75, 0], [0, 76], [76, 73], [76, 77], [77, 73], [77, 
   4], [76, 9], [9, 77], [25, 79], [79, 78], [78, 25], [79, 80], [80, 
   78], [80, 30], [30, 78], [79, 16], [16, 80], [31, 82], [82, 
   81], [81, 31], [82, 83], [83, 81], [83, 26], [26, 81], [82, 
   15], [15, 83], [2, 84], [84, 54], [84, 75], [75, 54], [84, 0], [58,
    84], [58, 76], [76, 84], [27, 86], [86, 85], [85, 27], [86, 
   63], [63, 85], [12, 85], [86, 13], [64, 87], [87, 12], [64, 
   88], [88, 87], [88, 29], [29, 87], [14, 88], [17, 89], [89, 
   65], [89, 90], [90, 65], [90, 6], [89, 11], [11, 90], [7, 91], [91,
    67], [91, 92], [92, 67], [92, 19], [91, 10], [10, 92], [15, 
   94], [94, 93], [93, 15], [94, 95], [95, 93], [95, 28], [28, 
   93], [94, 16], [16, 95], [10, 97], [97, 96], [96, 10], [97, 
   98], [98, 96], [98, 1], [1, 96], [97, 11], [11, 98], [18, 99], [99,
    48], [99, 78], [78, 48], [99, 25], [26, 100], [100, 81], [100, 
   52], [52, 81], [100, 20], [95, 101], [101, 28], [95, 79], [79, 
   101], [25, 101], [83, 102], [102, 26], [83, 93], [93, 102], [28, 
   102], [85, 103], [103, 27], [85, 104], [104, 103], [104, 22], [22, 
   103], [12, 104], [104, 105], [105, 22], [104, 87], [87, 105], [29, 
   105], [22, 106], [106, 103], [106, 107], [107, 103], [107, 
   27], [106, 17], [17, 107], [29, 108], [108, 105], [108, 109], [109,
    105], [109, 22], [108, 19], [19, 109], [1, 111], [111, 110], [110,
    1], [111, 106], [106, 110], [22, 110], [111, 17], [109, 
   110], [109, 112], [112, 110], [112, 1], [19, 112], [34, 113], [113,
    18], [34, 74], [74, 113], [3, 113], [77, 114], [114, 4], [77, 
   36], [36, 114], [20, 114], [40, 115], [115, 23], [40, 86], [86, 
   115], [27, 115], [88, 116], [116, 29], [88, 42], [42, 116], [24, 
   116], [25, 117], [117, 101], [117, 118], [118, 101], [118, 
   28], [117, 21], [21, 118], [21, 119], [119, 118], [119, 102], [102,
    118], [119, 26], [3, 120], [120, 113], [120, 99], [99, 113], [120,
    25], [114, 121], [121, 4], [114, 100], [100, 121], [26, 121], [66,
    107], [66, 115], [115, 107], [68, 116], [68, 108], [108, 
   116], [98, 111], [98, 89], [89, 111], [92, 112], [92, 96], [96, 
   112], [56, 59], [56, 97], [97, 59], [69, 120], [69, 117], [117, 
   120], [121, 72], [121, 119], [119, 72], [44, 50], [44, 94], [94, 
   50], [46, 80], [46, 39], [39, 80], [51, 43], [51, 82], [82, 
   43], [90, 33], [90, 55], [55, 33], [60, 37], [60, 91], [91, 
   37]], [[6, 33, 32], [32, 33, 34], [32, 34, 18], [33, 8, 34], [20, 
   36, 35], [35, 36, 37], [35, 37, 7], [36, 9, 37], [30, 39, 38], [38,
    39, 40], [38, 40, 23], [39, 13, 40], [24, 42, 41], [41, 42, 
   43], [41, 43, 31], [42, 14, 43], [5, 45, 44], [44, 45, 46], [44, 
   46, 16], [45, 13, 46], [23, 47, 38], [38, 47, 48], [38, 48, 
   30], [47, 18, 48], [5, 50, 49], [49, 50, 51], [49, 51, 14], [50, 
   15, 51], [20, 53, 52], [52, 53, 41], [52, 41, 31], [53, 24, 
   41], [8, 55, 54], [54, 55, 56], [54, 56, 2], [55, 11, 56], [6, 32, 
   57], [57, 32, 47], [57, 47, 23], [32, 18, 47], [2, 59, 58], [58, 
   59, 60], [58, 60, 9], [59, 10, 60], [7, 61, 35], [35, 61, 53], [35,
    53, 20], [61, 24, 53], [12, 63, 62], [62, 63, 45], [62, 45, 
   5], [63, 13, 45], [5, 49, 62], [62, 49, 64], [62, 64, 12], [49, 14,
    64], [6, 57, 65], [65, 57, 66], [65, 66, 17], [57, 23, 66], [7, 
   67, 61], [61, 67, 68], [61, 68, 24], [67, 19, 68], [3, 70, 
   69], [69, 70, 71], [69, 71, 21], [70, 0, 71], [21, 71, 72], [72, 
   71, 73], [72, 73, 4], [71, 0, 73], [8, 75, 74], [74, 75, 70], [74, 
   70, 3], [75, 0, 70], [0, 76, 73], [73, 76, 77], [73, 77, 4], [76, 
   9, 77], [25, 79, 78], [78, 79, 80], [78, 80, 30], [79, 16, 
   80], [31, 82, 81], [81, 82, 83], [81, 83, 26], [82, 15, 83], [2, 
   84, 54], [54, 84, 75], [54, 75, 8], [84, 0, 75], [2, 58, 84], [84, 
   58, 76], [84, 76, 0], [58, 9, 76], [27, 86, 85], [85, 86, 63], [85,
    63, 12], [86, 13, 63], [12, 64, 87], [87, 64, 88], [87, 88, 
   29], [64, 14, 88], [17, 89, 65], [65, 89, 90], [65, 90, 6], [89, 
   11, 90], [7, 91, 67], [67, 91, 92], [67, 92, 19], [91, 10, 
   92], [15, 94, 93], [93, 94, 95], [93, 95, 28], [94, 16, 95], [10, 
   97, 96], [96, 97, 98], [96, 98, 1], [97, 11, 98], [18, 99, 
   48], [48, 99, 78], [48, 78, 30], [99, 25, 78], [26, 100, 81], [81, 
   100, 52], [81, 52, 31], [100, 20, 52], [28, 95, 101], [101, 95, 
   79], [101, 79, 25], [95, 16, 79], [26, 83, 102], [102, 83, 
   93], [102, 93, 28], [83, 15, 93], [27, 85, 103], [103, 85, 
   104], [103, 104, 22], [85, 12, 104], [22, 104, 105], [105, 104, 
   87], [105, 87, 29], [104, 12, 87], [22, 106, 103], [103, 106, 
   107], [103, 107, 27], [106, 17, 107], [29, 108, 105], [105, 108, 
   109], [105, 109, 22], [108, 19, 109], [1, 111, 110], [110, 111, 
   106], [110, 106, 22], [111, 17, 106], [22, 109, 110], [110, 109, 
   112], [110, 112, 1], [109, 19, 112], [18, 34, 113], [113, 34, 
   74], [113, 74, 3], [34, 8, 74], [4, 77, 114], [114, 77, 36], [114, 
   36, 20], [77, 9, 36], [23, 40, 115], [115, 40, 86], [115, 86, 
   27], [40, 13, 86], [29, 88, 116], [116, 88, 42], [116, 42, 
   24], [88, 14, 42], [25, 117, 101], [101, 117, 118], [101, 118, 
   28], [117, 21, 118], [21, 119, 118], [118, 119, 102], [118, 102, 
   28], [119, 26, 102], [3, 120, 113], [113, 120, 99], [113, 99, 
   18], [120, 25, 99], [4, 114, 121], [121, 114, 100], [121, 100, 
   26], [114, 20, 100], [17, 66, 107], [107, 66, 115], [107, 115, 
   27], [66, 23, 115], [24, 68, 116], [116, 68, 108], [116, 108, 
   29], [68, 19, 108], [1, 98, 111], [111, 98, 89], [111, 89, 
   17], [98, 11, 89], [19, 92, 112], [112, 92, 96], [112, 96, 1], [92,
    10, 96], [2, 56, 59], [59, 56, 97], [59, 97, 10], [56, 11, 
   97], [3, 69, 120], [120, 69, 117], [120, 117, 25], [69, 21, 
   117], [4, 121, 72], [72, 121, 119], [72, 119, 21], [121, 26, 
   119], [5, 44, 50], [50, 44, 94], [50, 94, 15], [44, 16, 94], [16, 
   46, 80], [80, 46, 39], [80, 39, 30], [46, 13, 39], [14, 51, 
   43], [43, 51, 82], [43, 82, 31], [51, 15, 82], [6, 90, 33], [33, 
   90, 55], [33, 55, 8], [90, 11, 55], [9, 60, 37], [37, 60, 91], [37,
    91, 7], [60, 10, 91]]]


# Format mesh for use in the solver. Mainly about ordering the edge connection
# array to make it easier to calculate the jacobian in solver. Have the lower
# value vertex index on the left of the tuple.
for a in mesh[1]:
    a.sort()
mesh[1].sort()

# Extract mesh information for indexing moving forward
vert_num=len(mesh[0])
sp_num=len(mesh[1])

# List of connecting vertices for each vertex. This is used in the velocity
# update constraints in the solver.
# (index of stoke vertex in mesh[0], index of spring for stoke in sparr)
conn_list=[]
for b in range(vert_num):
    conn_list.append([])
for c in range(vert_num):
    for d in mesh[1]:
        if c in d:
            if d.index(c)==0:
                conn_list[c].append([d[1],mesh[1].index(d)])
            else:
                conn_list[c].append([d[0],mesh[1].index(d)])

# Generate velocity and mass array
velocities=[]
masses=[]
for e in mesh[0]:
    velocities.append(initial_con(e,1.,.0).tolist())
    masses.append(1.)

# Generate parameter array (we will implicitly take all the vertices to have
# mass of 1 so not included in this array. Can add later if needed). The rest
# length of the springs for the edges are calculated from the mesh vertices
# separation values.
# (spring constant k (all set to 1 here), rest length l_eq)
sparr=[]
for f in mesh[1]:
    sparr.append([
        1.,
        h3dist(convertpos_rot2hyph3(mesh[0][f[0]]),convertpos_rot2hyph3(mesh[0][f[1]]))
        ])

# Intialize the time stepping for the integrator.
delT=.01
maxT=10+delT
nump=maxT/delT
timearr=np.arange(0,maxT,delT)

# Containers for trajectory data
q = 0
gat = []
gbt = []
ggt = []
dist = [] # contains distance data for all spring connections - index in multiples of total springs)
energy_dat=[] 

### Add initial conditions to the containers
# Position and kinetic energy data
kin_energy=0.
for g in range(vert_num):
    gat.append(mesh[0][g][0])
    gbt.append(mesh[0][g][1])
    ggt.append(mesh[0][g][2])
    kin_energy+=.5*masses[g]*( velocities[g][0]**2. + sinh(mesh[0][g][0])**2.*velocities[g][1]**2. + sinh(mesh[0][g][0])**2.*sin(mesh[0][g][1])**2.*velocities[g][2]**2. )

# Distance between vertices and potential energy
pot_energy=0.
for h in range(sp_num):
    dist.append(sparr[h][1])
    pot_energy+=.5*sparr[h][0]*( sparr[h][1] - sparr[h][1] )**2.

# Energy of system
energy_dat.append(kin_energy+pot_energy)

# Copy of mesh (position of vertices will be changed with the integration)
mesh_copy=cp.deepcopy(mesh)

# Make container for the updating velocities
velocities_copy=cp.deepcopy(velocities)

# Numerical Integration step
step_data=[
	imph3sprot_mesh(mesh_copy, velocities_copy, delT, masses, sparr, energy_dat, conn_list)
	]

# Copy of mesh and velocity array (position of vertices and velocity values will be changed with the integration)
new_mesh=cp.deepcopy(mesh)
new_velocities=cp.deepcopy(velocities)

# Store data from first time step
a_dat=step_data[0][0:3*vert_num:3]
b_dat=step_data[0][1:3*vert_num:3]
g_dat=step_data[0][2:3*vert_num:3]
ad_dat=step_data[0][3*vert_num+0:2*3*vert_num:3]
bd_dat=step_data[0][3*vert_num+1:2*3*vert_num:3]
gd_dat=step_data[0][3*vert_num+2:2*3*vert_num:3]

# Position and velocity data
for i in range(vert_num):
    gat.append(a_dat[i])
    gbt.append(b_dat[i])
    ggt.append(g_dat[i])
    new_mesh[0][i]=[a_dat[i],b_dat[i],g_dat[i]]
    new_velocities[i]=[ad_dat[i],bd_dat[i],gd_dat[i]]

# Distance between vertices
for j in range(sp_num):
    dist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][j][1]])))

# Energy of system
energy_dat.append(step_data[0][-1])

q=q+1

# Iterate through each time step using data from the previous step in the trajectory
while(q < nump-1):

    step_data=array([
        imph3sprot_mesh(new_mesh, new_velocities, delT, masses, sparr, energy_dat[-1], conn_list)
        ])

    # Copy of mesh and velocity array (position of vertices and velocity values will be changed with the integration)
    new_mesh=cp.deepcopy(new_mesh)
    new_velocities=cp.deepcopy(new_velocities)

    # Store data from first time step
    a_dat=step_data[0][0:3*vert_num:3]
    b_dat=step_data[0][1:3*vert_num:3]
    g_dat=step_data[0][2:3*vert_num:3]
    ad_dat=step_data[0][3*vert_num+0:2*3*vert_num:3]
    bd_dat=step_data[0][3*vert_num+1:2*3*vert_num:3]
    gd_dat=step_data[0][3*vert_num+2:2*3*vert_num:3]

    # Position and velocity data
    for k in range(vert_num):
        gat.append(a_dat[k])
        gbt.append(b_dat[k])
        ggt.append(g_dat[k])
        new_mesh[0][k]=[a_dat[k],b_dat[k],g_dat[k]]
        new_velocities[k]=[ad_dat[k],bd_dat[k],gd_dat[k]]

    # Distance between vertices
    for l in range(sp_num):
        dist.append(h3dist(convertpos_rot2hyph3(new_mesh[0][new_mesh[1][l][0]]),convertpos_rot2hyph3(new_mesh[0][new_mesh[1][l][1]])))

    # Energy of system
    energy_dat.append(step_data[0][-1])

    q=q+1


# Transform into Poincare disk model for plotting
gut=[]
gvt=[]
grt=[]


for i in range(len(gat)):
    gut.append(sinh(gat[i])*sin(gbt[i])*cos(ggt[i])/(cosh(gat[i]) + 1.))
    gvt.append(sinh(gat[i])*sin(gbt[i])*sin(ggt[i])/(cosh(gat[i]) + 1.))
    grt.append(sinh(gat[i])*cos(gbt[i])/(cosh(gat[i]) + 1.))	

# Save the time series data
# np.savetxt("dist12_data.csv",dist12) 
# np.savetxt("dist13_data.csv",dist13) 
# np.savetxt("dist23_data.csv",dist23)    	     		

#####################
#  PLOTTING SECTION #
#####################

# ------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model ###
# ------------------------------------------------------------------

# #Plot
# fig = plt.figure(figsize=(8,8))
# ax1 = fig.add_subplot(111, projection='3d')
# # ax1.set_aspect("equal")

# #draw sphere
# u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
# x = np.sin(u)*np.cos(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(u)
# ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
# ax1.set_xlim3d(-1,1)
# ax1.set_xlabel('X')
# ax1.set_ylim3d(-1,1)
# ax1.set_ylabel('Y')
# ax1.set_zlim3d(-1,1)
# ax1.set_zlabel('Z')

# # Particle Plot data
# part_plot=[]
# for a in range(vert_num):
#     part_plot.append(hypercirch3(array([sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*cos(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*sin(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*cos(gbt[-(vert_num-a)]),cosh(gat[-(vert_num-a)])]),.1))
#     ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")

# #draw trajectory
# for b in range(vert_num):
#     ax1.plot3D(gut[b::vert_num],gvt[b::vert_num],grt[b::vert_num], label="particle []".format(b))
# ax1.legend(loc= 'lower left')

# plt.show()

# --------------------------------------------------------------------------------------
### Uncomment to just plot trajectory in the Poincare disk model with distance plots ###
# --------------------------------------------------------------------------------------

# # Plot Trajectory with error
# fig = plt.figure(figsize=(14,6))

# ax1=fig.add_subplot(1,3,1,projection='3d')

# #draw sphere
# u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
# x = np.sin(u)*np.cos(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(u)
# ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
# ax1.set_xlim3d(-1,1)
# ax1.set_xlabel('X')
# ax1.set_ylim3d(-1,1)
# ax1.set_ylabel('Y')
# ax1.set_zlim3d(-1,1)
# ax1.set_zlabel('Z')

# # Particle Plot data
# part_plot=[]
# for a in range(vert_num):
#     part_plot.append(hypercirch3(array([sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*cos(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*sin(gbt[-(vert_num-a)])*sin(ggt[-(vert_num-a)]),sinh(gat[-(vert_num-a)])*cos(gbt[-(vert_num-a)]),cosh(gat[-(vert_num-a)])]),.1))
#     ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")

# #draw trajectory
# for b in range(vert_num):
#     ax1.plot3D(gut[b::vert_num],gvt[b::vert_num],grt[b::vert_num], label="particle []".format(b))
# ax1.legend(loc= 'lower left')

# # ax1.plot_surface(part1x, part1y, part1z, color="b")
# # ax1.plot_surface(part2x, part2y, part2z, color="r")
# # ax1.plot_surface(part3x, part3y, part3z, color="k")

# # Displacement Plot
# # ax2=fig.add_subplot(1,3,2)

# # ax2.plot(timearr,(dist12-spring_arr[0][1]),label="Spring 12 Displacement")
# # ax2.plot(timearr,(dist13-spring_arr[1][1]),label="Spring 13 Displacement")
# # ax2.plot(timearr,(dist23-spring_arr[2][1]),label="Spring 23 Displacement")
# # #ax2.axhline(y=spring[3]+sqrt(2.*1./spring[2]*.5*.5), color='b', linestyle='-')
# # #ax2.axhline(y=((dist.max()-spring[3])+(dist.min()-spring[3]))/2., color='r', linestyle='-')
# # #ax2.set_yscale("log",basey=10)	
# # #ax2.set_ylabel('displacement (m)')
# # ax2.set_xlabel('time (s)')
# # ax2.legend(loc='lower right')

# # Distance Plot
# ax2=fig.add_subplot(1,3,2)

# for c in range(sp_num):
#     ax2.plot(timearr,dist[c::sp_num],label="Spring [] Distance".format(c))

# # ax2.plot(timearr,dist[0::3],label="Spring 12 Distance")
# # ax2.plot(timearr,dist[1::3],label="Spring 13 Distance")
# # ax2.plot(timearr,dist[2::3],label="Spring 23 Distance")
# #ax2.axhline(y=spring[3]+sqrt(2.*1./spring[2]*.5*.5), color='b', linestyle='-')
# #ax2.axhline(y=((dist.max()-spring[3])+(dist.min()-spring[3]))/2., color='r', linestyle='-')
# #ax2.set_yscale("log",basey=10)	
# #ax2.set_ylabel('displacement (m)')
# ax2.set_xlabel('time (s)')
# ax2.legend(loc='lower right')

# # Energy Plot
# ax3=fig.add_subplot(1,3,3)

# ax3.plot(timearr,energy_dat,label="Energy")
# # ax3.set_yscale("log",basey=10)
# ax3.set_xlabel('time (s)')	
# ax3.legend(loc='lower right')

# # Force Plot
# # ax3=fig.add_subplot(1,3,3)

# # ax3.plot(timearr,(dist12-spring_arr[0][1])*spring_arr[0][0],label="Spring 12 Force")
# # ax3.plot(timearr,(dist13-spring_arr[1][1])*spring_arr[1][0],label="Spring 13 Force")
# # ax3.plot(timearr,(dist23-spring_arr[2][1])*spring_arr[2][0],label="Spring 23 Force")
# # # ax3.set_yscale("log",basey=10)
# # ax3.set_xlabel('time (s)')	
# # ax3.legend(loc='lower right')	

# fig.tight_layout()	

# plt.show()

# ------------------------------------------------------------------
### Uncomment to just generate gif of trajectory of the particle ###
# ------------------------------------------------------------------

# Generate gif
# create empty lists for the x and y data
x1 = []
y1 = []
z1 = []
x2 = []
y2 = []
z2 = []
x3 = []
y3 = []
z3 = []

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111, projection='3d')
# ax1.set_aspect("equal")

#draw sphere
u, v = np.mgrid[0:np.pi+(np.pi)/15.:(np.pi)/15., 0:2.*np.pi+(2.*np.pi)/15.:(2.*np.pi)/15.]
x = np.sin(u)*np.cos(v)
y = np.sin(u)*np.sin(v)
z = np.cos(u)
ax1.plot_wireframe(x, y, z, color="b", alpha=.1)
ax1.set_xlim3d(-1,1)
ax1.set_xlabel('X')
ax1.set_ylim3d(-1,1)
ax1.set_ylabel('Y')
ax1.set_zlim3d(-1,1)
ax1.set_zlabel('Z')

# Particle Plot data
part_plot=[]
balls=[]
for a in range(vert_num):
    part_plot.append(hypercirch3(array([sinh(gat[a])*sin(gbt[a])*cos(ggt[a]),sinh(gat[a])*sin(gbt[a])*sin(ggt[a]),sinh(gat[a])*cos(gbt[a]),cosh(gat[a])]),.1))
    balls.append([ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")])

# #draw trajectory
# for b in range(vert_num):
#     ax1.plot3D(gut[b::vert_num],gvt[b::vert_num],grt[b::vert_num], label="particle []".format(b))
# ax1.legend(loc= 'lower left')

# part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::3][0])*sin(gbt[0::3][0])*cos(ggt[0::3][0]),sinh(gat[0::3][0])*sin(gbt[0::3][0])*sin(ggt[0::3][0]),sinh(gat[0::3][0])*cos(gbt[0::3][0]),cosh(gat[0::3][0])]),particles[0][7])
# part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::3][1])*sin(gbt[1::3][1])*cos(ggt[1::3][1]),sinh(gat[1::3][1])*sin(gbt[1::3][1])*sin(ggt[1::3][1]),sinh(gat[1::3][1])*cos(gbt[1::3][1]),cosh(gat[1::3][1])]),particles[1][7])
# part3x,part3y,part3z=hypercirch3(array([sinh(gat[2::3][2])*sin(gbt[2::3][2])*cos(ggt[2::3][2]),sinh(gat[2::3][2])*sin(gbt[2::3][2])*sin(ggt[2::3][2]),sinh(gat[2::3][2])*cos(gbt[2::3][2]),cosh(gat[2::3][2])]),particles[2][7])
# ball1=[ax1.plot_surface(part1x,part1y,part1z, color="b")]
# ball2=[ax1.plot_surface(part2x,part2y,part2z, color="r")]
# ball3=[ax1.plot_surface(part3x,part3y,part3z, color="k")]

# animation function. This is called sequentially
frames=50
def animate(i):
    for b in range(vert_num):
        ax1.plot3D(gut[b::vert_num][:int(len(timearr)*i/frames)],gvt[b::vert_num][:int(len(timearr)*i/frames)],grt[b::vert_num][:int(len(timearr)*i/frames)], label="particle {}".format(b))
    # ax1.plot3D(gut[0::3][:int(len(timearr)*i/frames)],gvt[0::3][:int(len(timearr)*i/frames)],grt[0::3][:int(len(timearr)*i/frames)])
    # ax1.plot3D(gut[1::3][:int(len(timearr)*i/frames)],gvt[1::3][:int(len(timearr)*i/frames)],grt[1::3][:int(len(timearr)*i/frames)])
    # ax1.plot3D(gut[2::3][:int(len(timearr)*i/frames)],gvt[2::3][:int(len(timearr)*i/frames)],grt[2::3][:int(len(timearr)*i/frames)])
        part_plot.append(hypercirch3(array([sinh(gat[b::vert_num][int(len(timearr)*i/frames)])*sin(gbt[b::vert_num][int(len(timearr)*i/frames)])*cos(ggt[b::vert_num][int(len(timearr)*i/frames)]),sinh(gat[b::vert_num][int(len(timearr)*i/frames)])*sin(gbt[b::vert_num][int(len(timearr)*i/frames)])*sin(ggt[b::vert_num][int(len(timearr)*i/frames)]),sinh(gat[b::vert_num][int(len(timearr)*i/frames)])*cos(gbt[b::vert_num][int(len(timearr)*i/frames)]),cosh(gat[b::vert_num][int(len(timearr)*i/frames)])]),.1))
        balls[b][0].remove()
        balls[b][0]=ax1.plot_surface(part_plot[-1][0], part_plot[-1][1], part_plot[-1][2], color="b")
    # part1x,part1y,part1z=hypercirch3(array([sinh(gat[0::3][int(len(timearr)*i/frames)])*sin(gbt[0::3][int(len(timearr)*i/frames)])*cos(ggt[0::3][int(len(timearr)*i/frames)]),sinh(gat[0::3][int(len(timearr)*i/frames)])*sin(gbt[0::3][int(len(timearr)*i/frames)])*sin(ggt[0::3][int(len(timearr)*i/frames)]),sinh(gat[0::3][int(len(timearr)*i/frames)])*cos(gbt[0::3][int(len(timearr)*i/frames)]),cosh(gat[0::3][int(len(timearr)*i/frames)])]),particles[0][7])
    # part2x,part2y,part2z=hypercirch3(array([sinh(gat[1::3][int(len(timearr)*i/frames)])*sin(gbt[1::3][int(len(timearr)*i/frames)])*cos(ggt[1::3][int(len(timearr)*i/frames)]),sinh(gat[1::3][int(len(timearr)*i/frames)])*sin(gbt[1::3][int(len(timearr)*i/frames)])*sin(ggt[1::3][int(len(timearr)*i/frames)]),sinh(gat[1::3][int(len(timearr)*i/frames)])*cos(gbt[1::3][int(len(timearr)*i/frames)]),cosh(gat[1::3][int(len(timearr)*i/frames)])]),particles[1][7])
    # part3x,part3y,part3z=hypercirch3(array([sinh(gat[2::3][int(len(timearr)*i/frames)])*sin(gbt[2::3][int(len(timearr)*i/frames)])*cos(ggt[2::3][int(len(timearr)*i/frames)]),sinh(gat[2::3][int(len(timearr)*i/frames)])*sin(gbt[2::3][int(len(timearr)*i/frames)])*sin(ggt[2::3][int(len(timearr)*i/frames)]),sinh(gat[2::3][int(len(timearr)*i/frames)])*cos(gbt[2::3][int(len(timearr)*i/frames)]),cosh(gat[2::3][int(len(timearr)*i/frames)])]),particles[2][7])
    # ball1[0].remove()
    # ball1[0]=ax1.plot_surface(part1x,part1y,part1z, color="b")
    # ball2[0].remove()
    # ball2[0]=ax1.plot_surface(part2x,part2y,part2z, color="r")
    # ball3[0].remove()
    # ball3[0]=ax1.plot_surface(part3x,part3y,part3z, color="k")

# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

# call the animator. blit=True means only re-draw the parts that 
# have changed.
anim = animation.FuncAnimation(fig, animate,frames=frames, interval=50)

anim.save('./h3springmesh_test.gif', writer='imagemagick')

