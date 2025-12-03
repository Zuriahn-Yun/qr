import numpy as np
import math

# Check Orthogonality Function
def check_orthogonal(A):
    n = A.shape[0]
    if np.allclose(A @ A.T,np.identity(n)):
        return True
    return False

# Householder Decomposition, returns an array res that contains all Householder Matrices
# Implemented with row swopping to maintain accuracy and deal with rank deficient algorithms
def householder(A):
    res = []
    P = np.eye(A.shape[1])
    for i in range(min(A.shape[0],A.shape[1])):
        # Iterate through Columns of A
        max_col_norm = -1
        max_col_idx = i
        for j in range(i,A.shape[1]):
            if norm(A[i:,j]) > max_col_norm:
                max_col_norm = norm(A[i:,j])
                max_col_idx = j
        # Check if remaining columns are zero 
        if max_col_norm < 1e-12: 
            break
        # If they are not equal we swap 
        if max_col_idx != i:
            # Execute the Swap in A 
            A[:, [i, max_col_idx]] = A[:, [max_col_idx, i]]
            # Record the Swap in P 
            P[:, [i, max_col_idx]] = P[:, [max_col_idx, i]]  
            
        # Current X
        x = A[i:,i]
        
        # v = x - w*ei
        w = - (math.copysign(norm(x),x[0]))
        v = x.copy()
        v[0] -= w
        v_norm = norm(v)
        if v_norm == 0:
            v = np.zeros(v.shape[0])
        else:
            v = v / v_norm
        
        # Create H_i
        H = np.eye(A.shape[0])
        H_sub = np.eye(A.shape[0] - i) - 2 * np.outer(v,v)
        
        H[i:,i:] = H_sub
        # Generate new A 
        A = H @ A
        # Save H 
        res.append(H)
        
    # Where A = R in our final state
    # P is our Permutation Matrix
    # res is a list of Householder Transformation Matrices
    return A, P,res
        
# Euclidean Norm
def norm(x):
    res = 0
    for i in range(len(x)):
        res += x[i] ** 2
    
    if res <= 0:
        return 0
    else:
        return math.sqrt(res)

# Solve Ax = b
def backsub(A,b):
    R, P , res = householder(A.copy())
    x = np.zeros(A.shape[1])
    
    Q = np.eye(A.shape[0])
    for H in res:
        Q = Q @ H
    # Rx = Q.Tb
    c = Q.T @ b
    
    # R is upper triangular so we need to start from the bottom row 
    for i in range(A.shape[1] -1, -1 , -1):
        curr = c[i]
        for j in range(i + 1,A.shape[1]):
            curr -= x[j] * R[i,j]
        x[i] = curr / R[i,i]
    return P @ x
    

def estimate_rank(A):
    R,P,res = householder(A.copy())
    rank = 0
    print(R)
    for i in range(min(R.shape)):
        if abs(R[i,i]) > 1e-4:
            rank +=1
        else:
            break
    return rank
        
if __name__ == "__main__":
    A = np.array([[4,1,3],
                  [2,3,5],
                  [0,4,6]]
                 )
    A_original = A.copy()
    R, P,res = householder(A)
    
    print("Check Orthogonality in Householder Method --------------------------")
    for H in res:
        print((H.T @ H).round(1))
        
    print("Generate Q -----------------------")
    # Generate Q 
    Q = np.eye(A_original.shape[0])
    for H in res:
        Q = Q @ H
    print(Q)
   
    print("Print R -----------------------")
    print(R.round(1))
    print("Compare AP = QR -----------------------")
    # Generate AP = QR
    print((Q @ R).round(10))
    print(A_original @ P)
    print("Check Rank Estimation ----------------------------")
    z = A = np.array([[1,1],
                  [0,0],
                  [1,1]]
                 )
    print(z)
    rank = estimate_rank(z)
    print("Rank Estimation: ", rank)
    print("Size Testing ------------------------------------")
    Test = np.random.randn(200,200)
    print("Generate a random 200 x 200 size matrix")
    x_true = np.random.randn(200)
    print("Generate a True x")
    b = (Test @ x_true )+ 0.01 * np.random.randn(200)
    print("Generate a b by A @ X + 0.01 * random val")
    c = backsub(Test.T @ Test,Test.T @ b)
    print("Compute Relative Error")
    error = norm(c - x_true) / norm(x_true)
    print("Error: ", error)
    
    