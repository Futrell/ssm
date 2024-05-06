""" State-space sequence model """
import random
import operator
import itertools
from collections import namedtuple

import tqdm
import torch
import pandas as pd
import einops

INF = float('inf')

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'    

def boolean_mv(A, x):
    return torch.any(A & x[None, :], -1)

def boolean_mm(A, B):
    return torch.any(A & B, -2)

def boolean_vv(x, y):
    return torch.any(x & y)

Semiring = namedtuple("Semiring", ['zero', 'one', 'add', 'mul', 'dot', 'mv', 'mm', 'complement'])

RealSemiring = Semiring(0, 1, operator.add, operator.mul, operator.matmul, operator.matmul, operator.matmul, lambda x: 1-x)
BooleanSemiring = Semiring(False, True, operator.or_, operator.and_, boolean_vv, boolean_mv, boolean_mm, operator.invert)

class SSM:
    def __init__(self, A, B, C, init=None, phi=None, pi=None, device=DEVICE):
        """
        A: state matrix (K x K): determines contribution of state at previous 
           time to state at current time
        B: input matrix (K x S): determines contribution of input at current 
           time to state at current time
        C: output matrix (S x K): maps states to output space

        Optional:
        phi: matrix (Y x S): maps segments to vector representation
        pi: vector (S): projection vector for tier. Segments s with pi[s]=0 are ignored.
        init: initialization vector (K); if unspecified, [0, 0, 0, ..., 0, 1] by default.
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

        self.dtype = self.A.dtype

        self.semiring = BooleanSemiring if self.dtype is torch.bool else RealSemiring
        self.phi = torch.eye(U, dtype=self.dtype, device=device) if phi is None else phi
        self.init = torch.eye(X, dtype=self.dtype, device=device)[0] if init is None else init
        self.pi = torch.ones(X, U, dtype=self.dtype, device=device) if pi is None else pi

        # TODO: Need to distinguish between features that make a segment project, and those features which are projected?
        # Eg, "vowel" feature triggers projection of "frontness" feature.
        # If these are separate, then pi and pi_diag should be separately defined.
            
    def log_likelihood(self, sequence):
        '''
        calculate log likelihood of sequence under this model
        '''
        x = self.init
        score = 0.0
        for symbol in sequence:
            # Get output vector given current state
            y = self.C @ x.float()
            
            # Get log probability distribution over output symbols
            # Add log prob of current symbol to total
            score += torch.log_softmax(y, -1)[symbol]
            
            # Update state
            u = self.phi[symbol]
            proj = self.semiring.mv(self.pi, u)
            update = self.semiring.mv(self.A, x) + self.semiring.mv(self.B, u)
            x = self.semiring.complement(proj)*x + proj*update
        return score

def product(a: SSM, b: SSM) -> SSM:
    # untested
    A = torch.block_diag(a.A, b.A) 
    B = torch.cat([a.B, b.B])
    C = torch.cat([a.C, b.C])
    init = torch.cat([a.init, b.init])
    pi = torch.cat([a.pi, b.pi])
    phi = torch.cat([a.phi, b.phi])
    return SSM(A, B, C, init, pi=pi, phi=phi)

def train(K, S, data, A=None, B=None, C=None, init=None, pi=None, print_every=1000, device=DEVICE, **kwds):
    '''
    Fit model to a dataset
    K: dimension of state space vector
    S: dimension of input vector

    data: An iterable of batches of data.
    '''
    A = torch.randn(K, K, requires_grad=True, device=device) if A is None else A.to(device)
    B = torch.randn(K, S, requires_grad=True, device=device) if B is None else B.to(device)
    C = torch.randn(S, K, requires_grad=True, device=device) if C is None else C.to(device)
    
    if init is not None:
        init = init.to(device)

    if pi is not None:
        pi = pi.to(device)
        
    opt = torch.optim.AdamW(params=[A, B, C], **kwds)
    
    for i, xs in enumerate(data):
        opt.zero_grad()
        model = SSM(A, B, C, init, pi=pi)
        loss = -torch.stack([model.log_likelihood(x) for x in xs]).sum()
        loss.backward()
        opt.step()
        if i % print_every == 0:
            print(i, loss.item())
            
    return SSM(A, B, C, init, pi=pi)

def whole_dataset(data, num_epochs=None):
    return minibatches(data, len(data), num_epochs=num_epochs)

def single_datapoints(data, num_epochs=None):
    return minibatches(data, 1, num_epochs=num_epochs)

def batch(iterable, n=1):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i : min(i+n, l)]

def minibatches(data, batch_size=1, num_epochs=None):
    """
    Generate a stream of data in minibatches of size batch_size.
    Go through the data num_epochs times, each time in a random order.
    If num_epochs not specified, returns an infinite stream of batches.
    """
    data = list(data)
    def gen_epoch():
        return batch(data, batch_size)
    stream = iter(gen_epoch, None)
    return itertools.chain.from_iterable(itertools.islice(stream, None, num_epochs))

def train_tsl(S, projection, data, **kwds):
    A, B, init = sl_matrices(S)
    pi = projection.unsqueeze(0).expand(A.shape[0], -1)
    return train(A.shape[0], S, data, A=A, B=B, init=init, pi=pi, **kwds)

def train_sl(S, data, **kwds):
    A, B, init = sl_matrices(S)
    return train(A.shape[0], S, data, A=A, B=B, init=init, **kwds)

def train_sp(S, data, **kwds):
    A, B, init = sp_matrices(S)
    return train(A.shape[0], S, data, A=A, B=B, init=init, **kwds)

def train_sl_sp(S, data, **kwds):
    A, B, init = sl_sp_matrices(S)
    return train(A.shape[0], S, data, A=A, B=B, init=init, **kwds)

def sl_sp_matrices(S):
    X = S + 1
    A = torch.block_diag(
        torch.zeros(X, X, dtype=torch.bool),  # SL
        torch.eye(X, dtype=torch.bool), # SP
    )
    B = torch.cat([
        torch.eye(X, dtype=torch.bool)[:, 1:], # SL, S -> S + 1
        torch.eye(X, dtype=torch.bool)[:, 1:] # SP, S -> S
    ])
    init = torch.cat([
        torch.eye(X, dtype=torch.bool)[0],
        torch.eye(X, dtype=torch.bool)[0],
    ])
    return A, B, init

def sl_matrices(S):
    """ A and B matrices for 2-SL """
    # S+1 to account for the initial state
    # TODO: also need a constant state dimension for bias?
    X = S + 1
    A = torch.zeros(X, X)
    B = torch.eye(X)[:, 1:]
    init = torch.eye(X)[0]
    return torch.zeros(X, X), torch.eye(X)[:, 1:], torch.eye(X)[0]

def sp_matrices(S):
    """ A and B matrices for 2-SP """
    # S+1 to account for the initial state, which is persistent
    X = S + 1
    A = torch.eye(X, dtype=torch.bool)
    B = torch.eye(X, dtype=torch.bool)[:, 1:]
    init = torch.eye(X, dtype=torch.bool)[0]
    return A, B, init

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

def has_axb(xs):
    # contains sequence 20*3
    for i, x in enumerate(xs):
        if x == 2:
            for y in xs[i+1:]:
                if y == 0:
                    continue
                elif y == 1:
                    break
                elif y == 2:
                    break
                elif y == 3:
                    return True
    else:
        return False

def random_no_axb(n=4):
    # no sequence 20*3
    # 0 is not projected so should be ignored.
    # 1 is projected so it should not be ignored: 213 is ok.
    while True:
        sequence = [random.choice(range(4)) for _ in range(n)]
        if not has_axb(sequence):
            return sequence

def evaluate_no_axb(num_epochs=1000, batch_size=5, n=4, model_type='tsl', **kwds):
    dataset = [random_no_axb(n=n) for _ in range(1000)]
    good = [
        [0,2,0,1,0,3], # matched on SL factors
        [0,0,2,0,0,1,0,0,3], # matched on SL factors
        [0,2,0,1,0,2,0,1,0,3], # matched on (boolean) SP factors: 00,01,02,03,10,11,12,13,20,21,22,23
        [1,1,3,0,2,0],
        [3,3,2,1,3,0],
    ]
    bad = [
        [0,1,0,2,0,3], 
        [0,0,1,0,0,2,0,0,3],
        [0,2,0,1,0,1,0,2,0,3],
        [1,1,2,0,3,0],
        [3,3,2,0,3,0],
    ]
    if model_type == 'tsl':
        model = train_tsl(4, torch.Tensor([0,1,1,1]), minibatches(dataset, 5, num_epochs=20))
    elif model_type == 'sl':
        model = train_sl(4, minibatches(dataset, batch_size, num_epochs=20))
    elif model_type == 'sp':
        model = train_sp(4, minibatches(dataset, batch_size, num_epochs=20))
    elif model_type == 'sl_sp':
        model = train_sl_sp(4, minibatches(dataset, batch_size, num_epochs=20))
    else:
        raise TypeError("Unknown model type %s" % model_type)

    return evaluate_model_paired(model, good, bad)

def evaluate_no_aa_bb(num_epochs=10000, **kwds): # SL2 dataset
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

    model = train_sl(2, whole_dataset(good, num_epochs=num_epochs), **kwds)

    return evaluate_model_unpaired(model, good, bad)

def evaluate_model_paired(model, good_strings, bad_strings):
    def gen():
        for good, bad in zip(good_strings, bad_strings):
            yield good, bad, model.log_likelihood(good).item(), model.log_likelihood(bad).item()
    df = pd.DataFrame(list(gen()))
    df.columns = ['good', 'bad', 'good_score', 'bad_score']
    df['diff'] = df['good_score'] - df['bad_score']
    return df            

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
    results = evaluate_no_aa_bb()
    print(results)
    
    # Evaluation: 
    # 1. SL example that cannot be captured in SP
    # no bb \Sigma*bb\Sigma*
    # vs.  no b...b \Sigma*b\Sigma*b\Sigma*
    # Parity 
