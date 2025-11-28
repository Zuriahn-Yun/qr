import qr
import numpy as np
import plotly
import math

# Problem 1
# Estimate the Rank of A
# Solve min ||Ax - b||
def problem_1(r):
    x = np.array([1,1,1])
    b = np.array([2,2*r,2*r,2])
    A = np.array([
                [0,2,0],
                [r,r,0],
                [r,0,r],
                [0,1,1]])
    rank = qr.estimate_rank(A)
    x = qr.backsub(A,b)
    residual = qr.norm((A @ x) - b)
    return rank, residual
    
# Problem 2
# Use linear, quadratic and a linearized exponential least squares model to predict winning time in the 2024 olympics
def problem_2():
    b = np.array([12.2,11.9,11.5,11.9,11.5,11.5,11,11.4,11,11.07,11.08,11.06,10.97,10.54,10.82,10.94,10.75,10.93,10.78,10.75,10.71,10.61])
    A = np.array([[1,1928],
                 [1,1932],
                 [1,1936],
                 [1,1948],
                 [1,1952],
                 [1,1956],
                 [1,1960],
                 [1,1964],
                 [1,1968],
                 [1,1972],
                 [1,1976],
                 [1,1980],
                 [1,1984],
                 [1,1988],
                 [1,1992],
                 [1,1996],
                 [1,2000],
                 [1,2004],
                 [1,2008],
                 [1,2012],
                 [1,2016],
                 [1,2021]])
    x = qr.backsub(A.T @ A,A.T @ b)
    # Linear model 
    linear = x[0] + x[1] * 2024
    print("Linear Prediction: " , linear)
    
    # Quadradic Model, Modify A 
    new_col = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        new_col[i] = A[i][1] ** 2
    
    # Add Quadratic column to A 
    new_col = new_col.reshape(-1,1)
    quadratic_A = np.concatenate((A,new_col),axis=1)
    x_quad = qr.backsub(quadratic_A.T @quadratic_A,quadratic_A.T @ b)
    
    # Quadratic Prediction Model 
    quadratic = x_quad[0] + x_quad[1] * 2024 + x_quad[2] * 2024 ** 2
    print("Quadratic Prediction: ", quadratic)
    
    # Linearized exponential least squares model 
    x_exp = qr.backsub(A.T @ A, A.T @ np.log(b))
    x_linearized = x_exp[0] + x_exp[1] * 2024
    print("Linearized Exponential Prediction: " , math.exp(x_linearized))
    
    
    return 
# Problem 3 
def problem_3():
    return

# Problem 4
def problem_4(F):
    A = np.array([[1,1],
                  [1,2],
                  [1,4]
    ])
    b = np.array([5,6,7])
    x = qr.backsub(A,b)
    residual = qr.norm((A @ x) - b)
    L = x[0] + x[1] * F
    return L

# Problem 5
def problem_5():
    A = np.array([
                [1,math.cos(math.pi/6 * 18.9),math.sin(math.pi/6 * 18.9)],
                [1,math.cos(math.pi/6 * 21.1),math.sin(math.pi/6 * 21.1)],
                [1,math.cos(math.pi/6 * 23.3),math.sin(math.pi/6 * 23.3)],
                [1,math.cos(math.pi/6 * 27.8),math.sin(math.pi/6 * 27.8)],
                [1,math.cos(math.pi/6 * 32.2),math.sin(math.pi/6 * 32.2)],
                [1,math.cos(math.pi/6 * 37.2),math.sin(math.pi/6 * 37.2)],
                [1,math.cos(math.pi/6 * 36.1),math.sin(math.pi/6 * 36.1)],
                [1,math.cos(math.pi/6 * 34.4),math.sin(math.pi/6 * 34.4)],
                [1,math.cos(math.pi/6 * 29.4),math.sin(math.pi/6 * 29.4)],
                [1,math.cos(math.pi/6 * 23.3),math.sin(math.pi/6 * 23.3)],
                [1,math.cos(math.pi/6 * 18.9),math.sin(math.pi/6 * 18.9)],
    ])
    # Setup the b array
    # use backsub to predict for month 11 or june or july i think 
    return



if __name__ == "__main__":
    print("--------------------------------------")
    print("Problem 1")
    for i in range(6,13):
        print("Iteration: " , i - 5 )
        r = 10 ** i
        rank,residual = problem_1(r)
        print("R Value : ", i)
        print("Rank: ", rank)
        print("Residual: ", residual)
    print("--------------------------------------")
    print("Problem 2")
    problem_2()
    print("--------------------------------------")
    print("Problem 3")
    print("--------------------------------------")
    print("Problem 4")
    for F in range(1,6):
        L = problem_4(F)
        print("Force: ", F)
        print("Predicted Length: ", L)
    print("--------------------------------------")
    print("Problem 5")
    
        
    
    
    