import numpy as np
import copy as cp
from numpy import zeros,array,arange,sqrt,sin,cos,tan,sinh,cosh,tanh,pi,arccos,arcsinh,arccosh,arctanh,arctan2,matmul,exp,identity,append,linalg,add
from function_bank import boostxh2e,h2dist,boostxh2,rotzh2,convertpos_hyp2paratransh2,convertvel_hyp2paratransh2


def imph3sprot_mesh_rigidcon(mesh, veln_arr, step, mass_arr, lagmut_arr, con_arr, conn_list):

    # These functions return values involving the distance between two particles in the mesh. They
    # are written in terms of the rotational parameterization (a,b,g) of the hyperboloid model of H^3. Here
    # the value D12 is connected to the distance function by 
    #
    # Dist(p1,p2)=arccosh(D12)
    #
    # These values are needed when taking derivatives of the distance function in the equations of
    # motion. Due to the symmetry of the expression D12 the number of functions is reduced for the 
    # derivatives by swapping what is considered p1 and p2.

    ### D12 Functions ###

    # The function D12
    def D12(a1, b1, g1, a2, b2, g2):
        return cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2)

    # These are the first derivatives of the D12 function with respective to a1, b1, and g1. Due to the symmetry
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

    # These are the second derivatives of the D12 function with respective to combinations of a1, b1, g1, a2, b2, g2. Due to the symmetry
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


    ### Jacobian Helper Functions ###
    # These functions are designed to streamline the process of populating the jacobian used by the Newton solver
    # when evolving the system to the next time step.

    ## Jacobian Constraint Terms ##
    # Use the D12 functions to generate the rigid rod constraint terms found the in jacobian matrix.
    # This function constructs three block matrices of values that will be included in the middle left,
    # middle right, and bottom left block of the complete jacobian.
    # This is because they only involve derivatives with respect to the positions and multipliers and not the velocities.

    def jacobi_rigcon_terms(meshn, meshn1, mass_arr, lagmut_arr):

        # The lagmut_arr is the array of lambdas at n+1 of iteration

        # This function returns the derivative with respect to x2 of the derivative with respect to x1 of the constraint
        # Where x can be taken to be a, b, or g
        def da1con12(m, f, d12, da1d12):
            return -da1d12/(m*f*sqrt( d12**2. - 1. ))

        # This function returns the derivative with respect to x2 of the derivative with respect to x1 of the constraint
        # Where x can be taken to be a, b, or g
        def da2da1con12(m, f, lagmut, d12, da1d12, da2d12, df, da2d12da1):
            return -lagmut/(m*f*sqrt( d12**2. - 1. ))*( da2d12da1 - df*da1d12/f - d12*da1d12*da2d12/( d12**2. - 1. ) )

        # This function returns a nested list of the derivative values for each spring in the mesh
        def derivative_terms(part1,part2):
            a1, b1, g1=part1
            a2, b2, g2=part2
            # D12 function
            d12=D12(a1, b1, g1, a2, b2, g2)
            # First derivatives of D12 function
            da1d12=da1D12(a1, b1, g1, a2, b2, g2)
            db1d12=db1D12(a1, b1, g1, a2, b2, g2)
            dg1d12=dg1D12(a1, b1, g1, a2, b2, g2)
            da2d12=da1D12(a2, b2, g2, a1, b1, g1)
            db2d12=db1D12(a2, b2, g2, a1, b1, g1)
            dg2d12=dg1D12(a2, b2, g2, a1, b1, g1)
            # Second derivatives of D12 function
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
                [ # D12 function
                    d12
                ],
                [ # First derivatives of D12 function
                    da1d12, 
                    db1d12, 
                    dg1d12, 
                    da2d12, 
                    db2d12, 
                    dg2d12],
                [ # Second derivatives of D12 function
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
                    ]
                ]
            ]

        # Generate array of derivative information for each rigid rod (step n and n+1)
        rod_datan=[]
        rod_datan1=[]
        for a in meshn[1]:
            rod_datan.append(derivative_terms(meshn[0][a[0]],meshn[0][a[1]]))
            rod_datan1.append(derivative_terms(meshn1[0][a[0]],meshn1[0][a[1]]))
        print("D12")
        print(rod_datan1[0][0])
        print("dD12")
        print(rod_datan1[0][1])
        print("ddD12")
        print(rod_datan1[0][2])

        # Generate matrices of rigid rod terms that will be added to complete
        # jacobian. Matrix initialized with zeros for places where
        # there is no contributions from rods. (when there is no connection between
        # two given vertices)
        midleft_matrix=zeros((3*len(meshn[0]),3*len(meshn[0])))
        midright_matrix=zeros((3*len(meshn[0]),len(lagmut_arr)))
        lowleft_matrix=zeros((len(lagmut_arr),3*len(meshn[0])))

        # Populate rigid rod matrices for each edge in the mesh. This is done by adding the
        # contribution of successive rods to its corresponding elements of the matrix.
        # Each element is the column derivative of the row derivative of the constraint. Thus
        # each edge contributes four block submatrices due to symmetry of the constraint.
        for b in meshn[1]:
            rod_count=meshn[1].index(b)

            # Middle Left Block
            midleft_matrix[3*b[0]+0][3*b[0]+0]+=da2da1con12(mass_arr[b[0]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0], rod_datan1[rod_count][1][0], 0., rod_datan1[rod_count][2][0][0])
            midleft_matrix[3*b[0]+0][3*b[0]+1]+=da2da1con12(mass_arr[b[0]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0], rod_datan1[rod_count][1][1], 0., rod_datan1[rod_count][2][0][1])
            midleft_matrix[3*b[0]+0][3*b[0]+2]+=da2da1con12(mass_arr[b[0]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0], rod_datan1[rod_count][1][2], 0., rod_datan1[rod_count][2][0][2])

            midleft_matrix[3*b[0]+1][3*b[0]+0]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1], rod_datan1[rod_count][1][0], sinh(2.*meshn1[0][b[0]][0]), rod_datan1[rod_count][2][1][0])
            midleft_matrix[3*b[0]+1][3*b[0]+1]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1], rod_datan1[rod_count][1][1], 0.,                          rod_datan1[rod_count][2][1][1])
            midleft_matrix[3*b[0]+1][3*b[0]+2]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1], rod_datan1[rod_count][1][2], 0.,                          rod_datan1[rod_count][2][1][2])

            midleft_matrix[3*b[0]+2][3*b[0]+0]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.*sin(meshn1[0][b[0]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2], rod_datan1[rod_count][1][0], sinh(2.*meshn1[0][b[0]][0])*sin(meshn1[0][b[0]][1])**2., rod_datan1[rod_count][2][2][0])
            midleft_matrix[3*b[0]+2][3*b[0]+1]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.*sin(meshn1[0][b[0]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2], rod_datan1[rod_count][1][1], sin(2.*meshn1[0][b[0]][1])*sinh(meshn1[0][b[0]][0])**2., rod_datan1[rod_count][2][2][1])
            midleft_matrix[3*b[0]+2][3*b[0]+2]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.*sin(meshn1[0][b[0]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2], rod_datan1[rod_count][1][2], 0.,                                                      rod_datan1[rod_count][2][2][2])


            midleft_matrix[3*b[0]+0][3*b[1]+0]+=da2da1con12(mass_arr[b[0]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0], rod_datan1[rod_count][1][3], 0., rod_datan1[rod_count][2][0][3])
            midleft_matrix[3*b[0]+0][3*b[1]+1]+=da2da1con12(mass_arr[b[0]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0], rod_datan1[rod_count][1][4], 0., rod_datan1[rod_count][2][0][4])
            midleft_matrix[3*b[0]+0][3*b[1]+2]+=da2da1con12(mass_arr[b[0]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0], rod_datan1[rod_count][1][5], 0., rod_datan1[rod_count][2][0][5])

            midleft_matrix[3*b[0]+1][3*b[1]+0]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1], rod_datan1[rod_count][1][3], 0., rod_datan1[rod_count][2][1][3])
            midleft_matrix[3*b[0]+1][3*b[1]+1]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1], rod_datan1[rod_count][1][4], 0., rod_datan1[rod_count][2][1][4])
            midleft_matrix[3*b[0]+1][3*b[1]+2]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1], rod_datan1[rod_count][1][5], 0., rod_datan1[rod_count][2][1][5])

            midleft_matrix[3*b[0]+2][3*b[1]+0]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.*sin(meshn1[0][b[0]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2], rod_datan1[rod_count][1][3], 0., rod_datan1[rod_count][2][2][3])
            midleft_matrix[3*b[0]+2][3*b[1]+1]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.*sin(meshn1[0][b[0]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2], rod_datan1[rod_count][1][4], 0., rod_datan1[rod_count][2][2][4])
            midleft_matrix[3*b[0]+2][3*b[1]+2]+=da2da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.*sin(meshn1[0][b[0]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2], rod_datan1[rod_count][1][5], 0., rod_datan1[rod_count][2][2][5])


            midleft_matrix[3*b[1]+0][3*b[0]+0]+=da2da1con12(mass_arr[b[1]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3], rod_datan1[rod_count][1][0], 0., rod_datan1[rod_count][2][3][0])
            midleft_matrix[3*b[1]+0][3*b[0]+1]+=da2da1con12(mass_arr[b[1]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3], rod_datan1[rod_count][1][1], 0., rod_datan1[rod_count][2][3][1])
            midleft_matrix[3*b[1]+0][3*b[0]+2]+=da2da1con12(mass_arr[b[1]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3], rod_datan1[rod_count][1][2], 0., rod_datan1[rod_count][2][3][2])

            midleft_matrix[3*b[1]+1][3*b[0]+0]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4], rod_datan1[rod_count][1][0], 0., rod_datan1[rod_count][2][4][0])
            midleft_matrix[3*b[1]+1][3*b[0]+1]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4], rod_datan1[rod_count][1][1], 0., rod_datan1[rod_count][2][4][1])
            midleft_matrix[3*b[1]+1][3*b[0]+2]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4], rod_datan1[rod_count][1][2], 0., rod_datan1[rod_count][2][4][2])

            midleft_matrix[3*b[1]+2][3*b[0]+0]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.*sin(meshn1[0][b[1]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5], rod_datan1[rod_count][1][0], 0., rod_datan1[rod_count][2][5][0])
            midleft_matrix[3*b[1]+2][3*b[0]+1]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.*sin(meshn1[0][b[1]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5], rod_datan1[rod_count][1][1], 0., rod_datan1[rod_count][2][5][1])
            midleft_matrix[3*b[1]+2][3*b[0]+2]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.*sin(meshn1[0][b[1]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5], rod_datan1[rod_count][1][2], 0., rod_datan1[rod_count][2][5][2])


            midleft_matrix[3*b[1]+0][3*b[1]+0]+=da2da1con12(mass_arr[b[1]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3], rod_datan1[rod_count][1][3], 0., rod_datan1[rod_count][2][3][3])
            midleft_matrix[3*b[1]+0][3*b[1]+1]+=da2da1con12(mass_arr[b[1]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3], rod_datan1[rod_count][1][4], 0., rod_datan1[rod_count][2][3][4])
            midleft_matrix[3*b[1]+0][3*b[1]+2]+=da2da1con12(mass_arr[b[1]], 1., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3], rod_datan1[rod_count][1][5], 0., rod_datan1[rod_count][2][3][5])

            midleft_matrix[3*b[1]+1][3*b[1]+0]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4], rod_datan1[rod_count][1][3], sinh(2.*meshn1[0][b[1]][0]), rod_datan1[rod_count][2][4][3])
            midleft_matrix[3*b[1]+1][3*b[1]+1]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4], rod_datan1[rod_count][1][4], 0.,                          rod_datan1[rod_count][2][4][4])
            midleft_matrix[3*b[1]+1][3*b[1]+2]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4], rod_datan1[rod_count][1][5], 0.,                          rod_datan1[rod_count][2][4][5])

            midleft_matrix[3*b[1]+2][3*b[1]+0]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.*sin(meshn1[0][b[1]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5], rod_datan1[rod_count][1][3], sinh(2.*meshn1[0][b[1]][0])*sin(meshn1[0][b[1]][1])**2., rod_datan1[rod_count][2][5][3])
            midleft_matrix[3*b[1]+2][3*b[1]+1]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.*sin(meshn1[0][b[1]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5], rod_datan1[rod_count][1][4], sin(2.*meshn1[0][b[1]][1])*sinh(meshn1[0][b[1]][0])**2., rod_datan1[rod_count][2][5][4])
            midleft_matrix[3*b[1]+2][3*b[1]+2]+=da2da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.*sin(meshn1[0][b[1]][1])**2., lagmut_arr[rod_count], rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5], rod_datan1[rod_count][1][5], 0.,                                                      rod_datan1[rod_count][2][5][5])
        

            # Middle Right Block
            midright_matrix[3*b[0]+0][rod_count]+=( da1con12(mass_arr[b[0]], 1.,                                                     rod_datan[rod_count][0][0], rod_datan[rod_count][1][0]) + da1con12(mass_arr[b[0]], 1.,                                                       rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0]) )
            midright_matrix[3*b[0]+1][rod_count]+=( da1con12(mass_arr[b[0]], sinh(meshn[0][b[0]][0])**2.,                            rod_datan[rod_count][0][0], rod_datan[rod_count][1][1]) + da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.,                             rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1]) )
            midright_matrix[3*b[0]+2][rod_count]+=( da1con12(mass_arr[b[0]], sinh(meshn[0][b[0]][0])**2.*sin(meshn[0][b[0]][1])**2., rod_datan[rod_count][0][0], rod_datan[rod_count][1][2]) + da1con12(mass_arr[b[0]], sinh(meshn1[0][b[0]][0])**2.*sin(meshn1[0][b[0]][1])**2., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2]) )

            midright_matrix[3*b[1]+0][rod_count]+=( da1con12(mass_arr[b[1]], 1.,                                                     rod_datan[rod_count][0][0], rod_datan[rod_count][1][3]) + da1con12(mass_arr[b[1]], 1.,                                                       rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3]) )
            midright_matrix[3*b[1]+1][rod_count]+=( da1con12(mass_arr[b[1]], sinh(meshn[0][b[1]][0])**2.,                            rod_datan[rod_count][0][0], rod_datan[rod_count][1][4]) + da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.,                             rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4]) )
            midright_matrix[3*b[1]+2][rod_count]+=( da1con12(mass_arr[b[1]], sinh(meshn[0][b[1]][0])**2.*sin(meshn[0][b[1]][1])**2., rod_datan[rod_count][0][0], rod_datan[rod_count][1][5]) + da1con12(mass_arr[b[1]], sinh(meshn1[0][b[1]][0])**2.*sin(meshn1[0][b[1]][1])**2., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5]) )

            # Lower Left Block
            lowleft_matrix[rod_count][3*b[0]+0]+=da1con12(1., 1., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][0])
            lowleft_matrix[rod_count][3*b[0]+1]+=da1con12(1., 1., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][1])
            lowleft_matrix[rod_count][3*b[0]+2]+=da1con12(1., 1., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][2])

            lowleft_matrix[rod_count][3*b[1]+0]+=da1con12(1., 1., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][3])
            lowleft_matrix[rod_count][3*b[1]+1]+=da1con12(1., 1., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][4])
            lowleft_matrix[rod_count][3*b[1]+2]+=da1con12(1., 1., rod_datan1[rod_count][0][0], rod_datan1[rod_count][1][5])

        return midleft_matrix, midright_matrix, lowleft_matrix

    ### Jacobian ###
    # This is the function used to generate the jacobian for the dynamical system being evolved
    # which is inverted when evaluating the root solver. The positions and velocities arrays are
    # taken to be the n+1 iteration guess (correspond to meshn1 so should maybe simplify).

    def jacobian(positions, velocities, mass_arr, h, lagmut_arr, meshn, meshn1):

        # Function that returns matrix of derivatives of the geodesic
        # terms of the update equations (Christoffel symbols for velocity
        # update equations)
        def geo_term_arr(pos, vel, h):
            a1,b1,g1=pos
            ad1,bd1,gd1=vel
            return [
                [ # position submatrix (derivatives with respect to position)
                    [   # a1, b1, g1 derivatives of adn1 update constraint
                        -.5*h*(bd1*bd1+sin(b1)*sin(b1)*gd1*gd1)*cosh(2.*a1),
                        -.25*h*sinh(2.*a1)*sin(2.*b1)*gd1*gd1,
                        0.
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
                ],
                [ # velocity submatrix (derivatives with respect to velocity)
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
            ]

        # Generate array of derivative information from Christoffel symbols
        geo_arr=[]
        for a in range(len(positions)):
            geo_arr.append(geo_term_arr(positions[a], velocities[a], h))

        # Generate block matrices of constraint terms
        midleft_terms, midright_terms, lowleft_terms=jacobi_rigcon_terms(meshn, meshn1, mass_arr, lagmut_arr)

        # Contruct the total jacobian matrix
        jmat=zeros((3*len(positions)+3*len(velocities)+len(lagmut_arr),3*len(positions)+3*len(velocities)+len(lagmut_arr)))

        ### Build upper third of matrix

        # Upper left block of jacobian
        iden_third=identity(3*len(positions))                    
        jmat[0:3*len(positions),0:3*len(positions)]=iden_third

        # Upper middle block of jacobian
        h_third=-.5*h*identity(3*len(velocities))                
        jmat[0:3*len(velocities),0+3*len(velocities):3*len(velocities)+3*len(velocities)]=h_third

        # Upper right block of jacobian (all zeros)

        ### Build middle third of matrix

        # Include terms from Christoffel symbol derivatives
        for c in range(len(positions)):
            # geodesic term for positions (middle left block diagonal)
            jmat[3*len(positions)+3*c:3*len(positions)+3+3*c,0+3*c:3+3*c]=geo_arr[c][0]
            # geodesic term for velocities (middle middle block diagonal)
            jmat[3*len(positions)+3*c:3*len(positions)+3+3*c,0+3*len(positions)+3*c:3+3*len(positions)+3*c]=geo_arr[c][1]

        # Include constraint terms (middle left block)
        jmat[3*len(positions):2*3*len(positions),0:3*len(positions)]=np.add(jmat[3*len(positions):2*3*len(positions),0:3*len(positions)],midleft_terms)

        # Middle right block of jacobian
        jmat[3*len(positions):3*len(positions)+3*len(positions),0+3*len(positions)+3*len(velocities):3*len(positions)+3*len(velocities)+len(lagmut_arr)]=midright_terms

        ### Build lower third of matrix

        # Lower left block (only nonzero block of lower third)
        jmat[3*len(positions)+3*len(positions):3*len(positions)+3*len(positions)+len(lagmut_arr),0:3*len(positions)]=lowleft_terms

        print(jmat)
        return jmat

    ### Constraint Equations ###
    # These are the equations that are being used by the Newton root solver. They
    # constitute a system of coupled ODE equations. The solver implies an implcit
    # trapezoidal method in solving the system and thus requires the use of the
    # Newton root solver to determine the n+1 value of the trajectory.
   
   # Position update constraint for a coordinate
    def con1(base_pos, base_pos_guess, base_vel, base_vel_guess, h):
        an,bn,gn=base_pos
        an1,bn1,gn1=base_pos_guess
        adn,bdn,gdn=base_vel
        adn1,bdn1,gdn1=base_vel_guess
        return an1 - an - .5*h*(adn + adn1)

    # Position update constraint for b coordinate
    def con2(base_pos, base_pos_guess, base_vel, base_vel_guess, h):
        an,bn,gn=base_pos
        an1,bn1,gn1=base_pos_guess
        adn,bdn,gdn=base_vel
        adn1,bdn1,gdn1=base_vel_guess
        return bn1 - bn - .5*h*(bdn + bdn1)

    # Position update constraint for g coordinate
    def con3(base_pos, base_pos_guess, base_vel, base_vel_guess, h): 
        an,bn,gn=base_pos
        an1,bn1,gn1=base_pos_guess
        adn,bdn,gdn=base_vel
        adn1,bdn1,gdn1=base_vel_guess
        return gn1 - gn - .5*h*(gdn + gdn1)        

    # Velocity update constraint for a coordinate
    def con4(base_pos, base_pos_guess, spokes_conn_list, base_vel, base_vel_guess, m1, h, lagmut_arr, meshn, meshn1):
        
        # Helper function to generate constraint force constribution
        def geo_con_term_ad(a1, b1, g1, a2, b2, g2, m, lagmut12):
            return (-lagmut12/(m*1.)*(sinh(a1)*cosh(a2) - cosh(a1)*cos(b1)*sinh(a2)*cos(b2) - cosh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.))


        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        nval=(bd1n*bd1n + gd1n*gd1n*sin(b1n)**2.)*sinh(a1n)*cosh(a1n)
        n1val=(bd1n1*bd1n1 + gd1n1*gd1n1*sin(b1n1)**2.)*sinh(a1n1)*cosh(a1n1)

        # Cycle over the springs connected to base point to include spring force
        for a in spokes_conn_list:
            axn,bxn,gxn=meshn[0][a[0]]
            nval+=geo_con_term_ad(a1n, b1n, g1n, axn, bxn, gxn, m1, lagmut_arr[a[1]])
            axn1,bxn1,gxn1=meshn1[0][a[0]]
            n1val+=geo_con_term_ad(a1n1, b1n1, g1n1, axn1, bxn1, gxn1, m1, lagmut_arr[a[1]])

        return (ad1n1 - ad1n - .5*h*(nval+n1val))

    # Velocity update constraint for b coordinate
    def con5(base_pos, base_pos_guess, spokes_conn_list, base_vel, base_vel_guess, m1, h, lagmut_arr, meshn, meshn1):
        
        # Helper function to generate spring force constribution
        def geo_con_term_bd(a1, b1, g1, a2, b2, g2, m, lagmut12):
            return (-lagmut12/(m*sinh(a1)*sinh(a1))*(sinh(a1)*sin(b1)*sinh(a2)*cos(b2) - sinh(a1)*cos(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        nval=gd1n*gd1n*sin(b1n)*cos(b1n) - 2.*ad1n*bd1n/tanh(a1n)
        n1val=gd1n1*gd1n1*sin(b1n1)*cos(b1n1) - 2.*ad1n1*bd1n1/tanh(a1n1)

        # Cycle over the springs connected to base point to include spring force
        for a in spokes_conn_list:
            axn,bxn,gxn=meshn[0][a[0]]
            nval+=geo_con_term_bd(a1n, b1n, g1n, axn, bxn, gxn, m1, lagmut_arr[a[1]])
            axn1,bxn1,gxn1=meshn1[0][a[0]]
            n1val+=geo_con_term_bd(a1n1, b1n1, g1n1, axn1, bxn1, gxn1, m1, lagmut_arr[a[1]])

        return (bd1n1 - bd1n - .5*h*(nval+n1val))

    # Velocity update constraint for g coordinate
    def con6(base_pos, base_pos_guess, spokes_conn_list, base_vel, base_vel_guess, m1, h, lagmut_arr, meshn, meshn1):
        
        # Helper function to generate spring force constribution
        def geo_con_term_gd(a1, b1, g1, a2, b2, g2, m, lagmut12):
            return (-lagmut12/(m*sinh(a1)*sinh(a1)*sin(b1)*sin(b1))*(sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*sin(g1 - g2))/sqrt(-1. + 
            (cosh(a1)*cosh(a2) - sinh(a1)*cos(b1)*sinh(a2)*cos(b2) - sinh(a1)*sin(b1)*sinh(a2)*sin(b2)*cos(g1 - g2))**2.)) 

        a1n,b1n,g1n=base_pos
        a1n1,b1n1,g1n1=base_pos_guess
        ad1n,bd1n,gd1n=base_vel
        ad1n1,bd1n1,gd1n1=base_vel_guess

        nval=-2.*ad1n*gd1n/tanh(a1n) - 2.*bd1n*gd1n/tan(b1n)
        n1val=-2.*ad1n1*gd1n1/tanh(a1n1) - 2.*bd1n1*gd1n1/tan(b1n1)

        # Cycle over the springs connected to base point to include spring force
        for a in spokes_conn_list:
            axn,bxn,gxn=meshn[0][a[0]]
            nval+=geo_con_term_gd(a1n, b1n, g1n, axn, bxn, gxn, m1, lagmut_arr[a[1]])
            axn1,bxn1,gxn1=meshn1[0][a[0]]
            n1val+=geo_con_term_gd(a1n1, b1n1, g1n1, axn1, bxn1, gxn1, m1, lagmut_arr[a[1]])

        return (gd1n1 - gd1n - .5*h*(nval+n1val))

    # Rigid rod constraint
    def con7(pos1_guess, pos2_guess, con_val):

        a1n1,b1n1,g1n1=pos1_guess
        a2n1,b2n1,g2n1=pos2_guess

        return arccosh(D12(a1n1, b1n1, g1n1, a2n1, b2n1, g2n1)) - con_val


    ###############################
    ### Start of body of solver ###
    ###############################

    # Extract mesh information for indexing moving forward
    vert_num=len(mesh[0])
    sp_num=len(mesh[1])

    # Copy the mesh and velocities to avoid calling the wrong instance in functions
    # The only part that will be potentially effect are the positions for mesh
    input_mesh=cp.deepcopy(mesh)    # Initial mesh given to solver
    solver_mesh=cp.deepcopy(mesh)   # Current version of mesh in iterations (version being updated)

    input_velarr=cp.deepcopy(veln_arr)    # Initial array of velocities given to solver
    solver_velarr=cp.deepcopy(veln_arr)   # Current version of array of velocities in iterations (version being updated)

    # Generate the array of constraints for Newton root solver
    constraint_arr=[]
    # Position update constriants
    for a in range(vert_num):
        constraint_arr.append(con1(input_mesh[0][a], input_mesh[0][a], input_velarr[a], solver_velarr[a], step))
        constraint_arr.append(con2(input_mesh[0][a], input_mesh[0][a], input_velarr[a], solver_velarr[a], step))
        constraint_arr.append(con3(input_mesh[0][a], input_mesh[0][a], input_velarr[a], solver_velarr[a], step))

    # Velocity update constraints
    for b in range(vert_num):
        constraint_arr.append(con4(input_mesh[0][b], input_mesh[0][b], conn_list[b], input_velarr[b], solver_velarr[b], mass_arr[b], step, lagmut_arr, input_mesh, solver_mesh))
        constraint_arr.append(con5(input_mesh[0][b], input_mesh[0][b], conn_list[b], input_velarr[b], solver_velarr[b], mass_arr[b], step, lagmut_arr, input_mesh, solver_mesh))
        constraint_arr.append(con6(input_mesh[0][b], input_mesh[0][b], conn_list[b], input_velarr[b], solver_velarr[b], mass_arr[b], step, lagmut_arr, input_mesh, solver_mesh))

    # Rigid rod constraints
    for i in range(sp_num):
        constraint_arr.append(con7(solver_mesh[0][input_mesh[1][i][0]], solver_mesh[0][input_mesh[1][i][1]], con_arr[i]))

    # Run zeroth iteration of the Newton root solver
    diff1=linalg.solve(array(jacobian(input_mesh[0], input_velarr, mass_arr, step, lagmut_arr, input_mesh, solver_mesh),dtype=np.float),-array(constraint_arr,dtype=np.float))
    val1=diff1.copy()
    # Position updates
    for c in range(vert_num):
        val1[3*c+0]+=input_mesh[0][c][0]
        val1[3*c+1]+=input_mesh[0][c][1]
        val1[3*c+2]+=input_mesh[0][c][2]

    # Velocity updates
    for d in range(vert_num):
        val1[3*d+0+3*vert_num]+=input_velarr[d][0]
        val1[3*d+1+3*vert_num]+=input_velarr[d][1]
        val1[3*d+2+3*vert_num]+=input_velarr[d][2]

    # Rigid Rod Constraint
    for j in range(sp_num):
        val1[3*vert_num+3*vert_num+j]+=lagmut_arr[j]

    # Run remaining iterations of the Newton root solver
    # The number of iterations has been hard coded to be 
    # seven since that is when convergence has occurred for
    # test trajectories. (Can be made more rigorous)
    x = 1           # Iteration number
    while(x < 100):
        # Values from the results of the previous root iteration
        new_pos_arr=[]
        new_vel_arr=[]
        new_lagmut_arr=[]
        for e in range(vert_num):
            new_pos_arr.append(val1[0+3*e:3+3*e])
            new_vel_arr.append(val1[3*vert_num+3*e:3+3*vert_num+3*e])

        for i in range(sp_num):
            new_lagmut_arr.append(val1[3*vert_num+3*vert_num+i])

        # Update the current version of the mesh and velocity array
        solver_mesh[0]=new_pos_arr
        solver_velarr=new_vel_arr

        # Generate the new array of constraints
        new_constraint_arr=[]
        # Position update constraints
        for f in range(vert_num):
            new_constraint_arr.append(con1(input_mesh[0][f], solver_mesh[0][f], input_velarr[f], solver_velarr[f], step))
            new_constraint_arr.append(con2(input_mesh[0][f], solver_mesh[0][f], input_velarr[f], solver_velarr[f], step))
            new_constraint_arr.append(con3(input_mesh[0][f], solver_mesh[0][f], input_velarr[f], solver_velarr[f], step))

        # Velocity update constraints
        for g in range(vert_num):
            new_constraint_arr.append(con4(input_mesh[0][g], solver_mesh[0][g], conn_list[g], input_velarr[g], solver_velarr[g], mass_arr[g], step, lagmut_arr, input_mesh, solver_mesh))
            new_constraint_arr.append(con5(input_mesh[0][g], solver_mesh[0][g], conn_list[g], input_velarr[g], solver_velarr[g], mass_arr[g], step, lagmut_arr, input_mesh, solver_mesh))
            new_constraint_arr.append(con6(input_mesh[0][g], solver_mesh[0][g], conn_list[g], input_velarr[g], solver_velarr[g], mass_arr[g], step, lagmut_arr, input_mesh, solver_mesh))

        # Rigid rod constraints
        for i in range(sp_num):
            new_constraint_arr.append(con7(solver_mesh[0][input_mesh[1][i][0]], solver_mesh[0][input_mesh[1][i][1]], con_arr[i]))

        # Run x-th iteration of the Newton root solver
        diff2=linalg.solve(array(jacobian(solver_mesh[0], solver_velarr, mass_arr, step, lagmut_arr, input_mesh, solver_mesh),dtype=np.float),-array(new_constraint_arr,dtype=np.float))
        val2=np.add(array(val1),array(diff2)).tolist()   
        val1 = val2
        print("Iteration %f" % x)
        print(val1)
        x=x+1

    return val1