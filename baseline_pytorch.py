import torch
import numpy as np
import pickle as pkl
from functools import reduce
from operator import mul


class TrackerDynBoxes:
    def __init__(self, T0 = 5, T = 2):

        self.T0 = T0
        self.T = T

        self.buffer_past_x = torch.zeros((T0, 1))
        self.buffer_past_y = torch.zeros((T0, 1))
        self.buffer_future_x = []
        self.buffer_future_y = []

        self.current_T0 = 0 # Init only
        self.current_T = 0 # Init only

        self.past_JBLDs = []
    


    def generate_seq_from_tree(self, seq_lengths, idx):
        """
        :return matrix T,2 = (T, (x,y))
        """
        sequence = np.zeros((1, len(seq_lengths), 2))
        new_idx = np.unravel_index(idx, seq_lengths)
        for frame in range(len(new_idx)):
            sequence[:, frame, 0] = self.buffer_future_x[frame][new_idx[frame]]
            sequence[:, frame, 1] = self.buffer_future_y[frame][new_idx[frame]]
        sequence = torch.from_numpy(sequence)
        return sequence

    def classify(self, cand, thresh = 0.5):
        si_o_no = -1
        if(len(self.past_JBLDs) == 0):
            si_o_no = 1
            return si_o_no
        else:
            mitjana = sum(self.past_JBLDs)/len(self.past_JBLDs)
        if( abs(cand-mitjana) <= thresh):
            si_o_no = 1
        return si_o_no

    def update_buffers(self, new_result):
        self.buffer_past_x[0:-1, 0] = self.buffer_past_x[1:,0]
        self.buffer_past_y[0:-1, 0] = self.buffer_past_y[1:,0]
        self.buffer_past_x[-1, 0] = new_result[0]
        self.buffer_past_y[-1, 0] = new_result[1]
        del self.buffer_future_x[0]
        del self.buffer_future_y[0]
    
    def decide(self, *candidates):
        candidates = candidates[0]
        """
        candidates contains N sublists [ [ [(px11,py11)] ] , [ [(px12,py12)],[(px22,py22)] ] ]
        candidates[1] = [ [(px12,py12)],[(px22,py22)] ]
        candidates[1][0] = [(px12,py12)]

        """
        
        if( self.current_T0 < self.T0 ):
            # Tracker needs more points to compute a reliable JBLD
            # We assume there is only 1 candidate for the T0 frames #NOTE: Corregir
            self.buffer_past_x[self.current_T0, 0] = float(candidates[0][0][0])
            self.buffer_past_y[self.current_T0, 0] = float(candidates[0][0][1])
            self.current_T0 += 1
        else:
            # Append points to buffers
            ryan_list_x = [] # temp lists
            ryan_list_y = []
            for [(x,y)] in candidates:
                ryan_list_x.append(x)
                ryan_list_y.append(y)
            
            self.buffer_future_x.append(ryan_list_x)
            self.buffer_future_y.append(ryan_list_y)

            if( len(self.buffer_future_x) == self.T ):
                # We have enough number of points to take a decision, let us build the trees
                seqs_lengths = []
                [seqs_lengths.append(len(y)) for y in self.buffer_future_x]
                num_of_seqs = reduce(mul, seqs_lengths)
                JBLDs = torch.zeros((num_of_seqs, 1))
                buffers_past = torch.cat([self.buffer_past_x, self.buffer_past_y], dim=1).unsqueeze(0)
                for i in range(num_of_seqs):
                    # Build each sequence of tree
                    seq = self.generate_seq_from_tree(seqs_lengths, i)
                    # Compute JBLD for each
                    # compare_dynamics needs a sequence of (1, T0, 2) and (1, T, 2)
                    JBLDs[i, 0] = compare_dynamics(buffers_past.type(torch.FloatTensor), seq.type(torch.FloatTensor))
                
                # Choose the minimum
                min_idx_jbld = torch.argmin(JBLDs) 
                min_val_jbld = JBLDs[min_idx_jbld, 0]
                point_to_add = self.generate_seq_from_tree(seqs_lengths, min_idx_jbld)
                point_to_add = point_to_add[0, 0, :]

                # Classify candidate
                classification_outcome = self.classify(min_val_jbld) # -1 dolent, 1 bo
                if(classification_outcome == -1):
                    # Predict
                    Hx = Hankel(self.buffer_past_x)
                    Hy = Hankel(self.buffer_past_y)
                    px = predict_Hankel(Hx)
                    py = predict_Hankel(Hy)
                    point_to_add[0] = px
                    point_to_add[1] = py
                else:
                    # We only use the JBLD if it is consistent with past dynamics, i.e, we did not have to predict
                    self.past_JBLDs.append(min_val_jbld)
                # update buffers
                self.update_buffers(point_to_add)


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

def JBLD2(X, Y):
    d = (torch.det((X + Y)/2)) - 0.5*(torch.det(torch.matmul(X, Y)))
    return d


def Gram(H, eps):
    N = np.power(eps, 2) * H.shape[0] * torch.eye(H.shape[0])
    G = torch.matmul(H, H.t()) + N
    Gnorm = G/torch.norm(G, 'fro')
    return Gnorm


def compare_dynamics(data_root, data, BS=1):
    dist = torch.zeros(BS, 2, device=device)
    for n_batch in range(BS):
        for d in range(2):
            H0 = Hankel(data_root[n_batch, :, d])
            H1 = Hankel(data_root[n_batch, :, d], True, data[n_batch, :, d])
            dist[n_batch, d] = JBLD2(Gram(H0, eps), Gram(H1, eps))
    print(dist)
    dist = torch.mean(dist, 1)
    return dist


def predict_Hankel(H):
    rows, cols = H.size()
    U, S, V = torch.svd(H)
    r = V[:,-1]
    last_column_of_H = H[-1,:]
    last_column_of_H = last_column_of_H[1:]
    first_term = torch.matmul(last_column_of_H, r[:-1])/(-r[-1])
    return first_term

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


tracker = TrackerDynBoxes()
# data[0] = [data[0][0],data[0][0]]
# print(data)
# print(len(data[19]))
for t, points in enumerate(data):
    tracker.decide(points)




# for t, points in enumerate(data):
#     print('\nTIME:', t)
#     if t < T0:
#         s[t, :] = torch.FloatTensor(points)

#     # if there is more than one candidate
#     elif len(points) > 1:
#         dist = torch.zeros((len(points), 2))
#         H0x = Hankel(s[0:t, 0])
#         H0y = Hankel(s[0:t, 1])
#         for i in range(len(points)):
#             p = torch.Tensor(points[i])
#             print('point:', p)
#             Hix = Hankel(s[0:t, 0], True, p[0, 0].unsqueeze(0))
#             Hiy = Hankel(s[0:t, 1], True, p[0, 1].unsqueeze(0))
#             Gram1 = Gram(H0x, eps)
#             Gram2 = Gram(Hix, eps)
#             dist[i, 0] = JBLD(Gram(H0x, eps), Gram(Hix, eps))
#             dist[i, 1] = JBLD(Gram(H0y, eps), Gram(Hiy, eps))
#         print('distancia', dist)

#         # 1. Creo candidats
#         # 2. Computo JBLDs
#         # 3. Actualitzo s, changes i jbld

#     # if there is only one candidate
#     else:
#         s[t, :] = torch.FloatTensor(points)
#         p = torch.Tensor(points).squeeze(0)
        
#         print('point', p)
#         H0x = Hankel(s[0:t, 0])
#         H1x = Hankel(s[0:t, 0], True, p[0, 0].unsqueeze(0))

#         Gram1 = Gram(H0x, eps)
#         Gram2 = Gram(H1x, eps)
#         print("det|X+Y| = ", torch.det(Gram1+Gram2), " det|XY| = ", torch.det(torch.matmul(Gram1, Gram2)))
#         dista = JBLD(Gram1, Gram2)
#         print('JBLD', dista)
#         # 1. L'afegeixo a s
#         print(predict_Hankel(H0x))