import numpy as np

# Requirements????
def check_orthogonal(A):
    n = A.shape[0]
    if A * A.T == np.identity(n):
        return True
    return False
        

def qr_factor(A):
    # cannot compute Q 
    return 