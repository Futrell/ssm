""" State-space sequence model """
import random

import torch
import opt_einsum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SSM:
    def __init__(self, A, B, C, phi=None, init=None, bias=None):
        self.A = A
        self.B = B
        self.C = C
        X, X2 = self.A.shape
        X3, U = self.B.shape
        Y, X4 = self.C.shape
        assert X == X2 == X3 == X4
        if bias is None:
            self.bias = torch.zeros(Y)
        else:
            self.bias = bias
        if phi is None:
            self.phi = torch.eye(Y)
        else:
            self.phi = phi
        if init is None:
            self.init = torch.eye(X)[0]
        else:
            self.init = init

    def impulse_response(self, K):
        return torch.stack([
            self.C @ torch.matrix_power(self.A, k) @ self.B
            for k in reversed(range(K))
        ])

    def log_likelihood(self, sequence):
        x = self.init
        score = 0.0
        for symbol in sequence:
            y = self.phi @ self.C @ x + self.bias
            score += torch.log_softmax(y, -1)[symbol]
            x = self.A @ x + self.B @ self.phi[symbol]
        return score

def train(K, S, data, print_every=1000, **kwds):
    A_diag = torch.randn(K, requires_grad=True)
    B = torch.randn(K, S, requires_grad=True)
    C = torch.randn(S, K, requires_grad=True)
    opt = torch.optim.AdamW(params=[A_diag, B, C], **kwds)
    for i, xs in enumerate(data):
        opt.zero_grad()
        A = torch.diag(A_diag)
        model = SSM(A, B, C)
        loss = -model.log_likelihood(xs)
        loss.backward()
        opt.step()
        if i % print_every == 0:
            print(i, loss.item())
    return SSM(A, B, C)

def anbn_ssm():
    """ SSM for a^n b^n. a = 1, b = 2, halt = 0.
    For every length N, the highest-probaiblity string is a^{(N-1)/2} b^{(N-1)/2} #
    """
    A = torch.diag(torch.Tensor([
        1, # dimension for count of a's
        1, # dimension for whether we have seen b
    ]))
    B = torch.Tensor([
        [0, 1, -1],  # increment for a, decrement for b
        [0, 0, 10],   # increase for each b
    ])
    C = torch.Tensor([
        [-10, 0], # halt -- only if no outstanding a's.
        [0, -10], # a -- not allowed if b seen.
        [0, 0],
    ])
    return SSM(A, B, C, init=torch.zeros(2))
        
def random_star_ab(S=3, T=4):
    while True:
        s = [random.choice(range(3)) for _ in range(T)]
        if not any(x == 0 and y == 1 for x, y in zip(s, s[1:])):
             return s

def random_tiptup():
    C1 = random.choice([0,1])
    C2 = random.choice([0,1])
    V = 2 + C1 ^ C2 
    return [C1, V, C2]

def random_xor():
    two = [random.choice(range(2)) for _ in range(2)]
    return two + [two[0] ^ two[1]]
             
def random_anbn(p_halt=1/2, start=1):
    T = start
    while True:
        if random.random() < p_halt:
            break
        else:
            T += 1
    return [1]*T + [2]*T + [0]
            
    
        
            
            
        

