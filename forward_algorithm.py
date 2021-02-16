import numpy as np

def forward_algorithm(A,B,O,c):
    # Forward algorithm for discrete hidden Markov Models with 'm' hidden states, 'n' observable states and 'N' observations.
    # A - m x m (state transition matrix)
    # B - m x n (confusion matrix)
    # O - 1 x N (observations vector, here I could use my algorithm that represents observation vectors to integers).
    # c - 1 x m (initial probabilities vector)

    m = A.shape[0]
    N = len(O)
    
    # Initialization
    Alpha = np.zeros((N, m))
    for i in range(m):
        Alpha[1,i] = c[i]*B[i,O[1]]
    
    # Iterative calculation of forward probabilities, Alpha
    for k in range(2,N):
        for i in range(m):
            S = 0
            for j in range(m):
                S = S + A[j,i]*Alpha[k-1, j] 

            Alpha[k, i] = B[i, O[k]]*S
    
    # Probability of observing O
    P = np.sum(Alpha[N-1, :])
    print(f"Probability of observed sequence: {P}")
    

if __name__ == '__main__':
    A = np.array([[1.0, 0.0],[0.2, 0.8]])
    print(A)
    B = np.array([[1.0, 0.0],[0.1, 0.9]])
    print(B[:, 0])
    print(B)
    c = np.array([1.0, 0.0])
    print(c)
    O = np.array([0, 0, 0, 0, 0, 0])
    print(O)

    forward_algorithm(A,B,O,c)

