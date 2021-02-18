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
        Alpha[0,i] = c[i]*B[i,O[1]]
    
    # Iterative calculation of forward probabilities, Alpha
    for k in range(1,N):
        for i in range(m):
            S = 0
            for j in range(m):
                S = S + A[j,i]*Alpha[k-1, j] 

            Alpha[k, i] = B[i, O[k]]*S
    
    # Probability of observing O
    # P = np.sum(Alpha[N-1, :])
    
    return Alpha

def backward_algorithm(A, B, O):
    # Backward algorithm for hidden Markov Models with 'm' hidden states, 'n' observable states, and 'N' observations.
    # A - m x m (state transition matrix)
    # B - m x n (confusion matrix)
    # O - 1 x N (observations vector, here I could use my algorithm that represents observation vectors to integers).
    m = A.shape[0]
    N = len(O)

    # Initialization
    Beta = np.zeros((N, m))
    for i in range(m):
        Beta[N-1, i] = 1.0
    
    for t in range(N-2, -1, -1):  # N-2 is the start value and is included, -1 is set to be the terminal value but it is not included, i.e., the terminal value is 0, and -1 is the step.
        for i in range(m):
            Beta[t, i] = 0.0
            for j in range(m):
                Beta[t, i] = Beta[t, i] + A[i, j]*B[j, O[t+1]]*Beta[t+1, j]

    return Beta

def compute_gamma(Alpha, Beta):
    (N, m) = Alpha.shape
    # AB = np.matmul(Alpha, np.transpose(Beta))
    # diag_AB = np.diagonal(AB).reshape((N,1)) 
    # P = np.matmul(diag_AB, np.ones((1, m)))
    # elemAB = np.multiply(Alpha, Beta)
    # Gama = np.divide(elemAB, P)
    Gamma = np.zeros((N, m))
    for k in range(N):
        for i in range(m):
            alpha_beta = Alpha[k, i]*Beta[k, i]
            P = 0
            for j in range(m):
                P = P + Alpha[k, j]*Beta[k, j]

            Gamma[k, i] = alpha_beta/P

    return Gamma

def compute_nu(Gamma, B):
    # Return the number of visits to state i
    # m hidden states, n output states and N observations
    #
    # B - m x n: (Confusion matrix)
    # nu -
    (m, n) = B.shape
    sum_gamma_columns = np.sum(Gamma, axis=0).reshape((1, m))
    nu = np.matmul(np.transpose(sum_gamma_columns), np.ones((1, n)))
    return nu

def compute_tau(Alpha, Beta, A, B, O):
    (m, n) = B.shape
    N = np.size(O)
    tau = np.zeros((m, m))
    for k in range(N-1):
        num = A*np.matmul(np.transpose(Alpha[k,:]), Beta[k+1, :])*np.transpose(np.matmul(B[:, O[k+1]].reshape(m, 1), np.ones((1, m))))
        den = np.matmul(np.matmul(np.ones((1,m)), num), np.ones((m, 1)))
        tau = tau + np.divide(num,den)
    return tau

def compute_taui(Gamma, B, O):
    (m, n) = B.shape
    N = np.size(O)
    taui = Gamma[0:N,:]
    taui = np.matmul((np.sum(taui, axis=0).reshape(m, 1)), np.ones((1,m))) 
    return taui

def compute_omega(Gamma, B, O):
    (m, n) = B.shape
    Omega = np.zeros((m, n))
    for j in range(n):
        inx = np.where(O == j)
        if not len(inx) == 0:
            Omega[:, j] = np.transpose(np.sum(Gamma[inx, :], 1)).reshape(2)
        else:
            Omega[:, j] = np.zeros((m,1))
    return Omega

def baum_welch(A, B, O, c):
    Alpha = forward_algorithm(A, B, O, c)
    Beta = backward_algorithm(A, B, O)
    Gamma = compute_gamma(Alpha, Beta)
    tau = compute_tau(Alpha, Beta, A, B, O)
    taui = compute_taui(Gamma, B, O)
    nu = compute_nu(Gamma, B)
    Omega = compute_omega(Gamma, B, O)
    c = Gamma[1, :]
    A = np.divide(tau, taui)
    B = np.divide(Omega, nu)

    return A, B, c

def hidden_gilbert_elliot_generator(A, B, c, N):
    # Generate a sequence of observations from a hidden gilbert-elliot markov model.
    # A - 2 x 2: state transition probabilites
    # B - 2 x 2: confusion matrix 
    # c - 1 x 2: initial state probability distribution
    # N : number of observations collected by the Hidden Gilbert-Elliot model.
    
    # Set initial state s
    r = np.random.uniform(0.0, 1.0)
    if r <= c[0]:
        s = 0
    else:
        s = 1

    # List of states
    S = []
    S.append(s)

    # List of observations
    O = []
    o = generate_observation(B, s)
    O.append(o)

    # Generate a sequence of N state transitions
    for i in range(1, N):
        r = np.random.uniform(0.0, 1.0)
        if s == 0:
            if r < A[0,0]:
                s = 0
            else:
                s = 1
        if s == 1:
            if r < A[1, 1]:
                s = 1
            else:
                s = 0
        S.append(s)
        o = generate_observation(B, s)
        O.append(o)

    return np.array(S), np.array(O)

def generate_observation(B,s):
    # Generate an observation for the two state hidden Gilbert-Eliot model.
    # B - 2 x 2: confusion matrix, the observations are the same as the states.  
    # s : current state of the system
    # o : observation
    r = np.random.uniform(0.0, 1.0)
    if s == 0:
        if r <= B[0, 0]:
            o = 0
        else:
            o = 1
    if s == 1:
        if r <= B[1, 1]:
            o = 1
        else:
            o = 0
    return o

if __name__ == '__main__':
    A = np.array([[0.9,  0.1],
                  [0.01, 0.99]])
    B = np.array([[0.99,  0.01],
                  [0.01, 0.99]])
    c = np.array([0.9, 0.1])
    # print(f'Sum: {np.sum(A, axis=0)}')
    N = 20
    S, O = hidden_gilbert_elliot_generator(A, B, c, N)
    print(S)
    print(O)

    A_hat = np.array([[0.8,  0.2],
                  [0.1, 0.9]])
    B_hat = np.array([[0.9,  0.1],
                  [0.1, 0.9]])
    c_hat = np.array([0.2, 0.3])

    for g in range(1):
        S, O = hidden_gilbert_elliot_generator(A, B, c, N)
        for l in range(1):
            A_hat, B_hat, c_hat = baum_welch(A_hat, B_hat, O, c_hat)
            print(A_hat)
            print('--------------------------')




    # Alpha, P = forward_algorithm(A, B, O, c)
    
    # # print(f"Probability of observed sequence: {P}")
    # Beta = backward_algorithm(A, B, O)
    # # print(Beta)
    # Gamma = compute_gamma(Alpha, Beta)
    # nu = compute_nu(Gamma, B)
    # tau = compute_tau(Alpha, Beta, A, B, O)
    # # print(Alpha)
    # # print(Beta)
    # taui = compute_taui(Gamma, B, O)
    # Omega = compute_omega(Gamma, B, O)
    # print(Omega)





    # A = np.array([[1.0,  0.0],
    #               [0.0, 1.0]])
    # B = np.array([[1.0,  0.0],
    #               [0.0, 1.0]])
    # c = np.array([0.5, 0.5])
    # N = 10
    # S, O = hidden_gilbert_elliot_generator(A, B, c, N)
    # print(S)
    # print(O)
    # forward_algorithm(A, B, O, c)
    # Alpha, P = forward_algorithm(A, B, O, c)
    # print(f"Probability of observed sequence: {P} ")
    # Beta = backward_algorithm(A, B, O)
    # print(Beta)


