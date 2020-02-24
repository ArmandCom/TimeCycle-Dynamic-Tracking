import torch
import numpy as np
import pickle as pkl
from functools import reduce
from operator import mul

class TrackerDynBoxes:
    def __init__(self, T0 = 5, T = 2):

        self.T0 = T0
        self.T = T

        self.buffer_past_x = np.zeros((T0, 1))
        self.buffer_past_y = np.zeros((T0, 1))
        self.buffer_future_x = []
        self.buffer_future_y = []

        self.current_T0 = 0 # Init only
        self.current_T = 0 # Init only

        self.past_JBLDs = []
    
    
    def decide(self, *candidates):
        if( self.current_T0 < self.T0 ):
            # Tracker needs more points to compute a reliable JBLD
            if(len(candidates) == 1): # We assume there is only 1 candidate for the T0 frames
                self.buffer_past_x[self.current_T0, 0] candidates[0]
                self.buffer_past_y[self.current_T0, 0] = candidates[1]
                self.current_T0 += 1
        else:
            # Append points to buffers
            ryan_list_x = [] # temp lists
            ryan_list_y = []
            for (x,y) in candidates:
                ryan_list_x.append(x)
                ryan_list_y.append(y)
            
            self.buffer_future_x.append(ryan_list_x)
            self.buffer_future_y.append(ryan_list_y)

            if( len(self.buffer_future_x) == self.T ):
                # We have enough number of points to take a decision, let us build the trees
                seqs_lengths = []
                [seqs_lengths.append(len(y)) for y in x]
                num_of_seqs = reduce(mul, seqs_lengths)
                JBLDS = []
                for i in range(num_of_seqs)):
                    # Build each sequence of tree
                    
                    
                    # Compute JBLD for each 

                
                # Choose the minimum

                # Classify candidate
                
                # if not a suitable candidate
                    # Predict
                
                # update buffers


    def generate_seq_from_tree(tree, idx):
        return sequence
    def classify(cand):
        # do sth with self.past_JBLDs
        return si_o_no

    
                

            



def Hankel(s0, stitch=False, s1=0):
    """
    :param s0: Root sequence
    :param stitch: Boolean to indicate if Hankel must be stitched or not
    :param s1: Sequence to add if Hankel must be stitched
    :return: H: Hankel matrix
    """
    dim = 1  # if x and y want to be treated jointly, change to dim=2
    l0 = s0.shape[0]
    l1 = 0
    if stitch:
        l1 = s1.shape[0]
        s0 = torch.cat([s0, s1])
    if l0 % 2 == 0:  # l is even
        num_rows = int(l0/2) * dim
        num_cols = int(l0/2) + 1 + l1
    else:  # l is odd
        num_rows = int(np.ceil(l0 / 2)) * dim
        num_cols = int(np.ceil(l0 / 2)) + l1
    H = torch.zeros([num_rows, num_cols])
    for i in range(int(num_rows/dim)):
        H[dim * i, :] = (s0[i:i + num_cols]).view(1, num_cols)
    return H


def JBLD(X, Y):
    d = torch.log(torch.det((X + Y)/2)) - 0.5*torch.log(torch.det(torch.matmul(X, Y)))
    return d


def Gram(H, eps):
    N = np.power(eps, 2) * H.shape[0] * torch.eye(H.shape[0])
    G = torch.matmul(H, H.t()) + N
    Gnorm = G/torch.norm(G, 'fro')
    return Gnorm


def compare_dynamics(data_root, data):
    dist = torch.zeros(BS, 2, device=device)
    for n_batch in range(BS):
        for d in range(2):
            H0 = Hankel(data_root[n_batch, :, d])
            H1 = Hankel(data_root[n_batch, :, d], True, data[n_batch, :, d])
            dist[n_batch, d] = JBLD(Gram(H0, eps), Gram(H1, eps))
    dist = torch.mean(dist, 1)
    return dist


def predict_Hankel(H):
    rows, cols = H.size()
    U, S, V = torch.svd(H)
    r = V[:,-1]
    print("H size = ", H.size())
    print("r size = ", r.size())
    last_column_of_H = H[-1,:]
    last_column_of_H = last_column_of_H[1:]
    print("Last column of H size = ", last_column_of_H.size())    
    print(H)
    first_term = torch.matmul(last_column_of_H, r[:-1])/(-r[-1])
    print("PREDICTION = ", first_term)

dtype = torch.float
device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
T0 = 5  # Temporal length of the initial window
eps = 0.0001  # Gram Matrix noise
BS = 100  # Batch Size
L0 = 5  # Longitude of the Root Sequence
L = 1  # Longitude of the Sequence being Tested
dim = 2  # Number of channels (x,y)
eps = 0.01  # Noise epsilon

# Create Random Data
data_root = torch.randn(BS, L0, dim,  device=device, dtype=dtype)  # size: (BS, L0, dim)
data = torch.randn(BS, L, dim, device=device, dtype=dtype)  # size: (BS, L, dim)

# Tracker data
with open('/data/Ponc/tracking/centroids_tree_nhl.obj', 'rb') as f:
    data = pkl.load(f)

s = torch.zeros((len(data), 2))  # Sequence of final points
changes = torch.zeros(len(data))  # Flag that will indicate where changes have occurred
jbld = torch.zeros(len(data))  # JBLD distances for each added point


for t, points in enumerate(data):
    print('\nTIME:', t)
    if t < T0:
        s[t, :] = torch.FloatTensor(points)

    # if there is more than one candidate
    elif len(points) > 1:
        dist = torch.zeros((len(points), 2))
        H0x = Hankel(s[0:t, 0])
        H0y = Hankel(s[0:t, 1])
        for i in range(len(points)):
            p = torch.Tensor(points[i])
            print('point:', p)
            Hix = Hankel(s[0:t, 0], True, p[0, 0].unsqueeze(0))
            Hiy = Hankel(s[0:t, 1], True, p[0, 1].unsqueeze(0))
            Gram1 = Gram(H0x, eps)
            Gram2 = Gram(Hix, eps)
            dist[i, 0] = JBLD(Gram(H0x, eps), Gram(Hix, eps))
            dist[i, 1] = JBLD(Gram(H0y, eps), Gram(Hiy, eps))
        print('distancia', dist)

        # 1. Creo candidats
        # 2. Computo JBLDs
        # 3. Actualitzo s, changes i jbld

    # if there is only one candidate
    else:
        s[t, :] = torch.FloatTensor(points)
        p = torch.Tensor(points).squeeze(0)
        
        print('point', p)
        H0x = Hankel(s[0:t, 0])
        H1x = Hankel(s[0:t, 0], True, p[0, 0].unsqueeze(0))

        Gram1 = Gram(H0x, eps)
        Gram2 = Gram(H1x, eps)
        print("det|X+Y| = ", torch.det(Gram1+Gram2), " det|XY| = ", torch.det(torch.matmul(Gram1, Gram2)))
        dista = JBLD(Gram1, Gram2)
        print('JBLD', dista)
        # 1. L'afegeixo a s
        print(predict_Hankel(H0x))