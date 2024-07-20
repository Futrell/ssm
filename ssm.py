""" State-space sequence model """
import sys
import csv
import random
from typing import *
import operator
import itertools
from collections import namedtuple, deque

import tqdm
import torch
import pandas as pd
import numpy as np

INF = float('inf')

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'    

def boolean_mv(A, x):
    """ Boolean matrix-vector multiplication. """
    return torch.any(A & x[None, :], -1)

def boolean_mm(A, B):
    """ Boolean matrix-matrix multiplication. """
    return torch.any(A & B, -2)

Semiring = namedtuple("Semiring", ['zero', 'one', 'add', 'mul', 'mv', 'mm', 'complement'])

RealSemiring = Semiring(0, 1, operator.add, operator.mul, operator.matmul, operator.matmul, lambda x: 1-x)
BooleanSemiring = Semiring(False, True, operator.or_, operator.and_, boolean_mv, boolean_mm, operator.invert)

SSMOutput = namedtuple("SSMOutput", "u proj x y".split())


# Core SSM logic

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
        
        self.phi = torch.eye(U, dtype=self.dtype, device=device) if phi is None else phi # default to identity matrix
        self.init = torch.eye(X, dtype=self.dtype, device=device)[0] if init is None else init # default to [1, 0, 0, 0, ...]

        # The projection matrix pi is a function from input feature i to state feature j.
        # It says, for input feature i, whether state feature j should be sensitive to it.
        # By default, all state features are sensitive to all input features, yielding a standard LTI SSM.
        if pi is None:
            self.pi = torch.ones(X, U, dtype=self.dtype, device=device) # default [[1, 1, ...], ...]
        else:
            self.pi = pi
            assert self.pi.shape[0] == X
            assert self.pi.shape[1] == U

    def log_likelihood(self, sequence, init=None, debug=False):
        '''
        calculate log likelihood of sequence under this model
        '''
        if init is None:
            x = self.init # shape X
        else:
            x = init # shape X
            
        score = 0.0
        for symbol in sequence:
            u = self.phi[symbol]  # for example, input 2  -> vector [0, 0, 1, 0, ...], shape U
            proj = self.semiring.mv(self.pi, u) # pi @ u, shape X
            
            # Get output vector given current state.
            y = (self.C * self.pi.T) @ x.float()

            # Get log probability distribution over output symbols
            # Add log prob of current symbol to total
            score += torch.log_softmax(y, -1)[symbol]
            
            # Update state
            update = self.semiring.mv(self.A, x) + self.semiring.mv(self.B, u)
            x = self.semiring.complement(proj)*x + proj*update

            if debug:
                breakpoint()
        return score

    def output_sequence(self, input, init=None, debug=False):
        T = len(input)
        u = self.phi[input]
        proj = self.semiring.mm(self.pi, u.T).T # einsum("xu,tu->tx", self.pi, u)
        x = torch.zeros(T + 1, self.A.shape[0], dtype=self.dtype)
        x[0] = self.init if init is None else init        
        for t in range(T):
            update = self.semiring.mv(self.A, x[t]) + self.semiring.mv(self.B, u[t])
            x[t+1] = self.semiring.complement(proj[t])*x[t] + proj[t]*update
            if debug:
                breakpoint()
        y = torch.einsum("yx,tx->ty", (self.C * self.pi.T), x[:-1].float())
        return SSMOutput(u, proj, x, y)

    def log_likelihood2(self, sequence, init=None):
        # Alternative implementation
        logits = self.output_sequence(sequence, init).y
        return logits.log_softmax(-1).gather(-1, torch.tensor(sequence).unsqueeze(-1)).sum()

# Classes for trainable phonotactics models

class PhonotacticsModel:
    def __init__(self, A, B, C, init=None, pi=None):
        self.A = A
        self.B = B
        self.C = C
        self.init = init
        self.pi = pi

    @classmethod
    def initialize(cls, X, S, requires_grad=True, device=DEVICE):
        self.A = torch.randn(X, X, requires_grad=requires_grad, device=device)
        self.B = torch.randn(X, S, requires_grad=requires_grad, device=device)
        self.C = torch.randn(S, X, requires_grad=requires_grad, device=device)
        return cls(A, B, C)

    def parameters(self):
        return [self.A, self.B, self.C]
    
    def ssm(self):
        return SSM(self.A, self.B, self.C, init=self.init, pi=self.pi)

    def __add__(self, other):
        return CompoundModel(self, other)    
        
    def train(self,
              data: Iterable[Iterable[int]],
              report_every: int=1000,
              device: str=DEVICE,
              debug: bool=False,
              reporting_window_size: int=100,
              **kwds):
        opt = torch.optim.AdamW(params=self.parameters(), **kwds)
        reporting_window = deque(maxlen=reporting_window_size)
        diagnostics = []
        writer = csv.DictWriter(sys.stderr, "step mean_loss".split())
        writer.writeheader()
        for i, xs in enumerate(data, 1):
            opt.zero_grad()
            model = self.ssm()
            loss = -torch.stack([model.log_likelihood(x, debug=debug) for x in xs]).mean()
            loss.backward()
            opt.step()
            reporting_window.append(loss.detach())
            if i % report_every == 0:
                diagnostic = {
                    'step': i,
                    'mean_loss': np.mean(reporting_window),
                }
                writer.writerow(diagnostic)                
                diagnostics.append(diagnostic)
        return diagnostics

class CompoundModel(PhonotacticsModel):
    def __init__(self, *models):
        self.models = models

    def parameters(self):
        params = []
        for model in self.models:
            params.extend(model.parameters())
        return params

    def ssm(self):
        ssms = [m.ssm() for m in self.models]
        return SSM(
            A=torch.block_diag(*[m.A for m in ssms]),
            B=torch.cat([m.B for m in ssms]),
            C=torch.cat([m.C for m in ssms], dim=-1),
            init=torch.cat([m.init for m in ssms]),
            pi=torch.cat([m.pi for m in ssms]),
        )

class Factor2(PhonotacticsModel):
    """ Phonotactics model whose probabilies are determined by factors of 2 segments """
    @classmethod
    def from_factors(cls, factors, projection=None):
        S, X = factors.shape
        A, B, init = cls.init_matrices(X, device=factors.device)
        return cls(A, B, factors, init=init, pi=projection)

    def parameters(self):
        return [self.C]

    def init_matrices(self, d):
        raise NotImplementedError

    @classmethod
    def initialize(cls, S, projection=None, requires_grad: bool=True, device: str=DEVICE):
        factors = torch.randn(S, S+1, requires_grad=requires_grad, device=device)
        return cls.from_factors(factors, projection=projection)

class SL2(Factor2):
    @classmethod
    def init_matrices(cls, d, device=DEVICE):
        A = torch.zeros(d, d, device=device) # X x X
        B = torch.eye(d, device=device)[:, 1:] # X x S
        init = torch.eye(d, device=device)[0]
        return A, B, init

class SP2(Factor2):
    @classmethod
    def init_matrices(cls, d, device=DEVICE):
        A = torch.eye(d, d, dtype=bool, device=device)
        B = torch.eye(d, dtype=bool, device=device)[:, 1:]
        init = torch.eye(d, dtype=bool, device=device)[0]
        return A, B, init

class SL_SP2(Factor2):
    @classmethod
    def init_matrices(cls, d, device=DEVICE):
        A = torch.block_diag(
            torch.zeros(d, d, dtype=torch.bool, device=device),  # SL
            torch.eye(d, dtype=torch.bool, device=device), # SP
        )
        B = torch.cat([
            torch.eye(d, dtype=torch.bool, device=device)[:, 1:], 
            torch.eye(d, dtype=torch.bool, device=device)[:, 1:] 
        ])
        init = torch.cat([
            torch.eye(d, dtype=torch.bool, device=device)[0],
            torch.eye(d, dtype=torch.bool, device=device)[0],
        ])
        return A, B, init

class TierBased(Factor2):
    def ssm(self):
        return SSM(
            self.A,
            self.B,
            self.C,
            init=self.init,
            pi=self.pi.unsqueeze(0).expand(*self.B.shape),
        )

class SoftTierBased(Factor2):
    def parameters(self):
        return [self.C, self.pi]
    
    @classmethod
    def initialize(cls, S, requires_grad: bool=True, device: str=DEVICE):
        factors = torch.randn(S, S + 1, requires_grad=requires_grad, device=device)
        projection = torch.randn(S, requires_grad=requires_grad, device=device)
        return cls.from_factors(factors, projection)

    def ssm(self):
        return SSM(
            self.A,
            self.B,
            self.C,
            init=self.init,
            pi=torch.sigmoid(self.pi.unsqueeze(0).expand(*self.B.shape)),
        )

class TSL2(SL2, TierBased):
    pass

class TSP2(SP2, TierBased):
    pass

class SoftTSL2(SL2, SoftTierBased):
    pass

# Data handling
    
def whole_dataset(data: Iterable, num_epochs: Optional[int]=None) -> Iterator[Sequence]:
    return minibatches(data, len(data), num_epochs=num_epochs)

def single_datapoints(data: Iterable, num_epochs: Optional[int]=None) -> Iterator[Sequence]:
    return minibatches(data, 1, num_epochs=num_epochs)

def batch(iterable: Sequence, n: int=1) -> Iterator[Sequence]:
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i : min(i+n, l)]

def minibatches(data: Iterable,
                batch_size: int=1,
                num_epochs: Optional[int]=None) -> Iterator[Sequence]:
    """
    Generate a stream of data in minibatches of size batch_size.
    Go through the data num_epochs times, each time in a random order.
    If num_epochs not specified, returns an infinite stream of batches.
    """
    data = list(data)
    def gen_epoch():
        random.shuffle(data)
        return batch(data, batch_size)
    stream = iter(gen_epoch, None)
    return itertools.chain.from_iterable(itertools.islice(stream, None, num_epochs))

# Example SSMs

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


# Generating random samples from languages of interest

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

def random_no_axb(n=4, neg=False):
    # no sequence 20*3
    # 0 is not projected so should be ignored.
    # 1 is projected so it should not be ignored: 213 is ok.
    while True:
        sequence = [random.choice(range(4)) for _ in range(n)]
        tier = [x for x in sequence if x != 0]
        if any(x == 2 and y == 3 for x, y in pairs(tier)) == neg:
            return sequence

def has_subsequence(seq, subseq):
    """Check if subseq is a subsequence of seq."""
    it = iter(seq)

    return all(elem in it for elem in subseq)

def generate_sequence(length, valid_range):
    """Generate a random sequence of given length from the valid range."""
    return [random.choice(valid_range) for _ in range(length)]

def random_no_aa_subseq(n=4):
    while True:
        sequence = generate_sequence(n, range(1, 5))
        if not has_subsequence(sequence, [2, 2]):
            return sequence

def random_no_ab(n=4, neg=False):
    # No substring 23
    while True:
        sequence = [random.choice(range(4)) for _ in range(n)]
        if any(x == 2 and y == 3 for x, y in pairs(sequence)) == neg:
            return sequence

def random_no_ab_subsequence(n=4, neg=False):
    # No subsequence 23
    while True:
        sequence = [random.choice(range(4)) for _ in range(n)]
        bad = False
        for i, sym1 in enumerate(sequence):
            if sym1 == 2:
                for j, sym2 in enumerate(sequence[i+1:]):
                    if sym2 == 3:
                        bad = True
        if not neg and not bad:
            return sequence
        elif neg and bad:
            return sequence

def pairs(xs): # contiguous substrings
    return zip(xs, xs[1:])

def random_two_tiers(n=6):
    # two independent tiers, {012} and {345}.
    # Prohibit *01 and *34 on each tier.
    p1 = {0,1,2}
    p2 = {3,4,5}
    while True:
        sequence = [random.choice(range(2*3)) for _ in range(n)]
        tier1 = [x for x in sequence if x in p1]
        tier2 = [x for x in sequence if x in p2]
        if (not any(x == 0 and y == 1 for x, y in pairs(tier1))
            and not any(x == 3 and y == 4 for x, y in pairs(tier2))):
            return sequence


# Evaluation functions

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

def evaluate_no_axb(num_epochs=20, batch_size=5, n=4, model_type=TSL2, num_samples=1000, num_test=100, **kwds):
    """ Evaluation for 2-TSL """
    dataset = [random_no_axb(n=n) for _ in range(num_samples)]
    # the first two of these comparisons will be exactly zero for SL
    # the third comparison will be exactly zero for SL and SP
    good = [ # no 23 on the tier {1, 2, 3}
         [0,2,0,1,0,3], # matched on SL factors
         [0,0,2,0,0,1,0,0,3], # matched on SL factors
         [0,2,0,1,0,2,0,1,0,3], # matched on (boolean) SP factors: 00,01,02,03,10,11,12,13,20,21,22,23
         [1,1,3,0,2,0],
         [3,3,2,1,3,0],
    ]

    bad = [
        [0,1,0,2,0,3], 
        [0,0,1,0,0,2,0,0,3],
        [0,2,0,1,0,1,0,2,0,3], # same SP factors as the SP example above;  
        [1,1,2,0,3,0],
        [3,3,2,0,3,0],
    ]
    #good = [random_no_axb(n=n+1) for _ in range(num_test)]
    #bad = [random_no_axb(n=n+1, neg=True) for _ in range(num_test)]

    if issubclass(model_type, TierBased):
        model = model_type.initialize(4, torch.Tensor([0,1,1,1]))
    else:
        model = model_type.initialize(4)

    data = minibatches(dataset, batch_size, num_epochs=num_epochs)
    model.train(data, **kwds)
    return evaluate_model_paired(model.ssm(), good, bad), model

def evaluate_no_ab(num_epochs=20, batch_size=5, n=4, model_type=SL2, num_samples=1000, num_test=100, **kwds): # SL2 dataset
    dataset = [random_no_ab(n=n) for _ in range(num_samples)]

    # good = [
    #     [0, 2, 1, 3, 2, 2], # matched on 2-SP factors
    #     [2, 0, 3, 1, 3, 2], # TSL model will fail
    #     [3, 2, 1, 0, 1, 3],
    #     [1, 3, 2, 0, 3, 1], # TSL model will fail
    #     [0, 1, 2, 1, 3, 2],
    # ]
    # bad = [
    #     [0, 2, 1, 2, 3, 2], # matched on 2-SP factors
    #     [2, 3, 0, 1, 3, 2],
    #     [3, 2, 1, 0, 2, 3],
    #     [1, 3, 2, 3, 0, 1], 
    #     [0, 1, 2, 3, 1, 2],
    # ]
    good = [random_no_ab(n=n+1) for _ in range(num_test)]
    bad = [random_no_ab(n=n+1, neg=True) for _ in range(num_test)]

    if issubclass(model_type, TierBased):
        model = model_type.initialize(4, torch.Tensor([0,1,1,1]))
    else:
        model = model_type.initialize(4)
        
    data = minibatches(dataset, batch_size, num_epochs=num_epochs)
    model.train(data, **kwds)
    return evaluate_model_unpaired(model.ssm(), good, bad)

    # 2-SP language *a-a
    # 2-TSL language: *a-a, Tier = {a, b, c, d}

    # For a string like aba
    # - this is prohibited by SP but allowed by TSL
    # but for Tier = {a} then no 

def evaluate_no_ab_subsequence(num_epochs=20, batch_size=5, n=4, model_type=SP2, num_samples=1000, num_test=100, **kwds):
    """ Evaluation for 2-SP """
    dataset = [random_no_ab_subsequence(n=n) for _ in range(num_samples)]
    # the first two of these comparisons will be exactly zero for SL
    # the third comparison will be exactly zero for SL and SP
    # good = [ # no 23 on the tier {1, 2, 3}
    #     [0,2,0,1,0,3], # matched on SL factors
    #     [0,0,2,0,0,1,0,0,3], # matched on SL factors
    #     [0,2,0,1,0,2,0,1,0,3], # matched on (boolean) SP factors: 00,01,02,03,10,11,12,13,20,21,22,23
    #     [1,1,3,0,2,0],
    #     [3,3,2,1,3,0],
    # ]

    # bad = [
    #     [0,1,0,2,0,3], 
    #     [0,0,1,0,0,2,0,0,3],
    #     [0,2,0,1,0,1,0,2,0,3], # same SP factors as the SP example above;  
    #     [1,1,2,0,3,0],
    #     [3,3,2,0,3,0],
    # ]
    good = [random_no_ab_subsequence(n=n+1) for _ in range(num_test)]
    bad = [random_no_ab_subsequence(n=n+1, neg=True) for _ in range(num_test)]

    if issubclass(model_type, TierBased):
        model = model_type.initialize(4, torch.Tensor([0,1,1,1]))
    else:
        model = model_type.initialize(4)

    data = minibatches(dataset, batch_size, num_epochs=num_epochs)
    model.train(data, **kwds)
    return evaluate_model_unpaired(model.ssm(), good, bad)

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

def evaluate_model_simple(model, good_strings, bad_strings):
    df_list = []
    for good in good_strings:
        df_list.append([good, 'good', model.log_likelihood(good).item()])
    for bad in bad_strings:
        df_list.append([bad, 'bad', model.log_likelihood(bad).item()])
    df = pd.DataFrame(df_list)
    df.columns = ['string', 'grammatical', 'score']
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
    print("Training SP model on SL data")
    results_sp_no_ab = evaluate_no_ab(model_type = SP2)
    print("Training SL model on SL data")
    results_sl_no_ab = evaluate_no_ab(model_type = SL2)
    print("Training TSL model on SL data")
    results_tsl_no_ab = evaluate_no_ab(model_type = TSL2)

    sp_no_ab_mean = np.mean(results_sp_no_ab['diff'])
    sl_no_ab_mean = np.mean(results_sl_no_ab['diff'])
    tsl_no_ab_mean = np.mean(results_tsl_no_ab['diff'])

    print("SL datatset: \n SSM-SL mean difference: {}\n SSM-SP mean difference: {}\n SSM-TSL mean difference: {}".format(
        sl_no_ab_mean, sp_no_ab_mean, tsl_no_ab_mean 
    ))
    
    # print("Training SP model on TSL data")
    # results_sp_no_axb = evaluate_no_axb(model_type = 'sp')
    # print("Training SL model on TSL data")
    # results_sl_no_axb = evaluate_no_axb(model_type = 'sl')
    # print("Training TSL model on TSL data")
    # results_tsl_no_axb = evaluate_no_axb(model_type = 'tsl')

    # sl_no_axb_mean = np.mean(results_sl_no_axb['diff'])
    # tsl_no_axb_mean = np.mean(results_tsl_no_axb['diff'])
    # sp_no_axb_mean = np.mean(results_sp_no_axb['diff'])

    # print("TSL dataset: \n SSM-SL mean difference: {}\n SSM-SP mean difference: {}\n SSM-TSL mean difference: {}".format(
    #     sl_no_axb_mean, sp_no_axb_mean, tsl_no_axb_mean
    # ))
    
    # print("Training SP model on SP data")
    # results_sp_no_ab_subsequence = evaluate_no_ab_subsequence(model_type = 'sp')
    # print("Training SL model on SP data")
    # results_sl_no_ab_subsequence = evaluate_no_ab_subsequence(model_type = 'sl')
    # print("Training TSL model on SP data")
    # results_tsl_no_ab_subsequence = evaluate_no_ab_subsequence(model_type = 'tsl')

    # sl_no_ab_subsequence_mean = np.mean(results_sl_no_ab_subsequence['diff'])
    # tsl_no_ab_subsequence_mean = np.mean(results_tsl_no_ab_subsequence['diff'])
    # sp_no_ab_subsequence_mean = np.mean(results_sp_no_ab_subsequence['diff'])

    # print("SP dataset: \n SSM-SL mean difference: {}\n SSM-SP mean difference: {}\n SSM-TSL mean difference: {}".format(
    #     sl_no_ab_subsequence_mean, sp_no_ab_subsequence_mean, tsl_no_ab_subsequence_mean
    # ))


# SP but not captured by SL
# *a…b
# *acccb
# *abbbb

# # SP but not captured by TSL
# *a…b, if tier is {a, b}, 
# *acccb TSL
# *abbbb TSL
# *acacb TSL
# *

# But yes ab on

# abbaccca

# 3-SP language *a-b-c
# 3-TSL language: *a-b-c, Tier = {a, b, c}

# For a string like abbc
# - this is prohibited by SP but allowed by TSL
# this cannot work for any tier


# 2-SP language *a-a
# 2-TSL language: *a-a, Tier = {a, b, c, d}

# For a string like aba
# - this is prohibited by SP but allowed by TSL
# but for Tier = {a} then no
