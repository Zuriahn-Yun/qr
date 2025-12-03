import qr
import numpy as np
import plotly.graph_objects as go
import math

# Problem 1
# Estimate the Rank of A
# Solve min ||Ax - b||
def problem_1(r):
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
    A = np.array([[1,1928],[1,1932],[1,1936],[1,1948],[1,1952],[1,1956],[1,1960],[1,1964],[1,1968],[1,1972],[1,1976],[1,1980],[1,1984],[1,1988],[1,1992],[1,1996],[1,2000],[1,2004],[1,2008],[1,2012],[1,2016],[1,2021]])
    
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
    # Create b 
    b = np.array([[ .7 ** 2 +  4** 2],
                 [ 3.3 ** 2 +  4.7 ** 2],
                 [ 5.6 ** 2 +  4 ** 2],
                 [ 7.1 ** 2 +  1.3 ** 2],
                 [ 6.4 ** 2 +  -1.1 ** 2],
                 [ 4.4** 2 +  -3** 2],
                 [ .3** 2 +  -2.5** 2],
                 [ -1.1** 2 +  1.3 ** 2],
                 ])
    # Create A 
    A = np.array([
                [2* .7, 2 * 4,1],
                [2* 3.3, 2 * 4.7,1],
                [2* 5.6, 2 * 4,1],
                [2* 7.1, 2 * 1.3,1],
                [2* 6.4, 2 * -1.1,1],
                [2* 4.4, 2 * -3,1],
                [2* .3, 2 * -2.5,1],
                [2* -1.1, 2 * 1.3,1],
    ])
    # Backsub to find c 
    c = qr.backsub(A.T @ A, A.T @ b)
    print("C Vector: ", c)
    # Use c to calculate the Radius 
    r = math.sqrt(c[2] + c[1] ** 2 + c[0] ** 2)
    print("Radius: ", r)
    # Use r to calculate points for the circle 
    x_y = [[0.7,4],
           [3.3,4.7],
           [5.6,4],
           [7.1,1.3],
           [6.4,-1.1],
           [4.4,-3],
           [.3,-2.5],
           [-1.1,1.3],]
    circle_x = []
    circle_y = []
    # Create x,y coordinates
    for i in range(100):
        t = i * (2 * math.pi / 100)
        new_x = c[0] + r * math.cos(t)
        new_y = c[1] + r * math.sin(t)
        circle_x.append(new_x)
        circle_y.append(new_y)
        
    scatter_data = go.Scatter(
        x=circle_x + [circle_x[0]],
        y=circle_y + [circle_y[0]],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='circle'),
        name='Problem 3, Best Fit Circle'
    )
    circle_trace = go.Scatter(
        x=circle_x + [circle_x[0]],
        y=circle_y + [circle_y[0]],
        mode='lines+markers',
        marker=dict(size=6, color='red', symbol='square'),
        line=dict(color='red'),
        name=f'Best-Fit Circle (R={r:.2f})'
    )
    
    
    fig = go.Figure(data=[scatter_data,circle_trace])
    fig.update_layout(title_text='Problem 3 Graph')
    fig.write_html("Problem3.html")
    fig.show()
    
    

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
                [1,math.cos(math.pi/6 * 1),math.sin(math.pi/6 * 1)],
                [1,math.cos(math.pi/6 * 2),math.sin(math.pi/6 * 2)],
                [1,math.cos(math.pi/6 * 3),math.sin(math.pi/6 * 3)],
                [1,math.cos(math.pi/6 * 4),math.sin(math.pi/6 * 4)],
                [1,math.cos(math.pi/6 * 5),math.sin(math.pi/6 * 5)],
                [1,math.cos(math.pi/6 * 6),math.sin(math.pi/6 * 6)],
                [1,math.cos(math.pi/6 * 8),math.sin(math.pi/6 * 8)],
                [1,math.cos(math.pi/6 * 9),math.sin(math.pi/6 * 9)],
                [1,math.cos(math.pi/6 * 10),math.sin(math.pi/6 * 10)],
                [1,math.cos(math.pi/6 * 11),math.sin(math.pi/6 * 11)],
                [1,math.cos(math.pi/6 * 12),math.sin(math.pi/6 * 12)],
    ])
    # Setup the b array
    b = np.array([[18.9],
                  [21.1],
                  [23.3],
                  [27.8],
                  [32.2],
                  [37.2],
                  [36.1],
                  [34.4],
                  [29.4],
                  [23.3],
                  [18.9],
                  ])
    x = qr.backsub(A.T @ A,A.T @ b)
    prediction = x[0] + math.cos(x[1] * 7) + math.sin(x[1] * 7)
    return prediction



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
    problem_3()
    print("--------------------------------------")
    print("Problem 4")
    for F in range(1,6):
        L = problem_4(F)
        print("Force: ", F)
        print("Predicted Length: ", L)
    print("--------------------------------------")
    print("Problem 5")
    prediction = problem_5()
    print("July Temperature Prediction: ", prediction)
        
    
    
    