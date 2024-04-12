""" State-space sequence model """
import random

import torch
import opt_einsum

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
            y = self.C @ x + self.bias
            score += torch.log_softmax(y, -1)[symbol]
            x = self.A @ x + self.B @ self.phi[symbol]
        return score

def train(K, S, data, print_every=1000, **kwds):
    A = torch.randn(K, K).requires_grad_(True)
    B = torch.randn(K, S).requires_grad_(True)
    C = torch.randn(S, K).requires_grad_(True)
    opt = torch.optim.AdamW(params=[A, B, C], **kwds)
    for i, xs in enumerate(data):
        opt.zero_grad()
        model = SSM(A, B, C)
        loss = -model.log_likelihood(xs)
        loss.backward()
        opt.step()
        if i % print_every == 0:
            print(i, loss.item())
    return SSM(A, B, C)
        
def random_star_ab(S=3, T=4):
    while True:
        s = [random.choice(range(3)) for _ in range(T)]
        if not any(x == 0 and y == 1 for x, y in zip(s, s[1:])):
             return s

def random_xor():
    two = [random.choice(range(2)) for _ in range(2)]
    return two + [two[0] ^ two[1]]
             

    
    
        
            
            
        

