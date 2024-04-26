""" State-space sequence model """
import random

import torch
import pandas as pd
import opt_einsum

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'    

class BooleanSemiring:
    def mv(self, A, x):
        return torch.any(A & x[None, :], -1)
    
class RealSemiring:
    def mv(self, A, x):
        return A @ x
    
class SSM:
    def __init__(self, A, B, C, init=None, bias=None, device=DEVICE):
        """
        A: state matrix (K x K): determines contribution of state at previous 
           time to state at current time
        B: input matrix (K x S): determines contribution of input at current 
           time to state at current time
        C: output matrix (S x K): maps states to output space
        phi: matrix (Y x S): maps segments to vector representation
        init: initialization vector (K)
        bias: bias term
        device: train on GPU ('cuda') or CPU ('cpu')
        """ 
        self.A = A
        self.B = B
        self.C = C

        # Confirm that A,B,C have correct dimensions
        X, X2 = self.A.shape
        X3, U = self.B.shape
        Y, X4 = self.C.shape
        assert X == X2 == X3 == X4

        if self.A.dtype == torch.bool:
            self.semiring = BooleanSemiring()
            self.phi = torch.eye(U, dtype=torch.bool)
        else:
            self.semiring = RealSemiring()
            self.phi = torch.eye(U)

        if bias is None:
            self.bias = torch.zeros(Y, device=device)
        else:
            self.bias = bias
            
        # NOTE: Consider the logic of the initial state, and its type.
        if init is None:
            self.init = torch.eye(X, device=device)[0]
        else:
            self.init = init


    def impulse_response(self, K): # cool stuff to calculate Log Likelihood fast
        return torch.stack([
            self.C @ torch.matrix_power(self.A, k) @ self.B
            for k in reversed(range(K))
        ])

    def log_likelihood(self, sequence):
        '''
        calculate log likelihood of sequence under this model
        '''
        x = self.init
        score = 0.0
        for symbol in sequence:
            # Get output vector given current state
            y = self.C @ x + self.bias
            # Get log probability distribution over output symbols
            # Add log prob of current symbol to total
            score += torch.log_softmax(y, -1)[symbol]
            # Update state
            x = self.semiring.mv(self.A, x) + self.semiring.mv(self.B, self.phi[symbol])
        return score



def train(K, S, data, A_diag=None, B=None, print_every=1000, device=DEVICE, **kwds):
    '''
    Fit model to a dataset
    K: dimension of state space vector
    S: dimension of input vector
    '''
    if A_diag is None:     # TODO for future capacity of training A_diag
        A_diag = torch.randn(K, requires_grad=True, device=device)
    if B is None:
        B = torch.randn(K, S, requires_grad=True, device=device)
    C = torch.randn(S, K, requires_grad=True, device=device)
    
    opt = torch.optim.AdamW(params=[A_diag, B, C], **kwds)
    
    for i, xs in enumerate(data):
        opt.zero_grad()
        A = torch.diag(A_diag) # convert the vector A to diagonal matrix if 
        model = SSM(A, B, C)
        loss = -model.log_likelihood(xs)
        loss.backward()
        opt.step()
        if i % print_every == 0:
            print(i, loss.item())
    return SSM(A, B, C)

def sl_matrices(S):
    """ A and B matrices for 2-SL """
    return torch.zeros(S, S), torch.eye(S)

def sp_matrices(S):
    """ A and B matrices for 2-SP """
    return torch.eye(S, dtype=torch.bool), torch.eye(S, dtype=torch.bool)

def anbn_ssm():
    """ SSM for a^n b^n. a = 1, b = 2, halt = 0.
    For every length N, the highest-probability string is a^{(N-1)/2} b^{(N-1)/2} #
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

def sl2_ssm():
    '''
    sl2_ssm that prohibits substrings of aa or bb
    '''
    A, B = sl_matrices(2)
    C = torch.Tensor([
        [-10, 0], # a -- not allowed if a seen
        [0, -10] # b -- not allowed if b seen
    ])
    return SSM(A, B, C, init=torch.zeros(2))

def sp2_ssm():
    '''
    sl2_ssm that prohibits subsequences of aa
    '''
    A, B = sp_matrices(2)
    C = torch.Tensor([
        [-10, 0], # a -- not allowed if a seen
        [0, 0] # b -- always fine
    ])
    return SSM(A, B, C, init=torch.zeros(2))


def random_star_ab(S=3, T=4):
    while True:
        s = [random.choice(range(3)) for _ in range(T)]
        if not any(x == 0 and y == 1 for x, y in zip(s, s[1:])):
             return s

def random_and():
    V1 = random.choice([0,1])
    V2 = random.choice([0,1])
    V3 = 1 if V1 and V2 else random.choice([0,1])
    return [V1, V2, V3]

def evaluate_and(model):
    good = [
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,1],
    ]
    bad = [
        [1,1,0],
    ]
    return evaluate_model_unpaired(model, good, bad)

def evaluate_tiptup(model):
    good = [
        [0,2,0],
        [1,2,1],
        [0,3,1],
        [1,3,0],
    ]
    bad = [
        [1,2,0],
        [0,2,1],
        [1,3,1],
        [0,3,0]
    ]
    return evaluate_model_unpaired(model, good, bad)

def evaluate_no_aa_bb(model):
    good = [
        [0],
        [1]
    ] + [
        [0, 1] * x for x in range(1,5)
    ] + [
        [1, 0] * x for x in range(1, 5)
    ]

    bad = [
        [0, 0],
        [1, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    return evaluate_model_unpaired(model, good, bad)

def evaluate_model_unpaired(model, good_strings, bad_strings):
    """ Do a pairwise comparison of log likelihood for all good and bad strings. """
    def gen():
        for good in good_strings:
            for bad in bad_strings:
                yield good, bad, model.log_likelihood(good).item(), model.log_likelihood(bad).item()
    df = pd.DataFrame(list(gen()))
    df.columns = ['good', 'bad', 'good_score', 'bad_score']
    df['diff'] = df['good_score'] - df['bad_score']
    return df

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
            

if __name__ == "__main__":
    model = sl2_ssm()
    results = evaluate_no_aa_bb(model)
    print(results)
    
    
    
    # Evaluation: 
    # 1. SL example that cannot be captured in SP
    # no bb \Sigma*bb\Sigma*
    # vs.  no b...b \Sigma*b\Sigma*b\Sigma*
 

