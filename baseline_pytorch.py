import torch
import numpy as np
import pickle as pkl

class TrackerDynBoxes:
    def __init__(self, T = 7, K = 3):
        """ 
        [0, 1, ... , T-1, T, T+1, ..., T+(K-1)]
         <------- T -------> <------- K ----->
        """
        self.T = T
        self.K = K
        self.points_buffer
        self.tree_buffer
        self.current_K = 0
        self.current_T = 0
    
    def accumulate(self, *candidates):
        
        if(self.current_T < self.T):
            # Tracker needs more points to compute a reliable JBLD
            if(len(candidates) == 1):
                # Tracker is confident
                self.points_buffer[self.current_T] = candidates
                self.current_T += 1
            elif(len(candidates) > 1):
                # tracker 
                build_tree(candidates)
    

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
        H[dim * i, :] = (s0[i:i + num_cols]).t()
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
    rows = H.size()[0]
    u, s , v = torch.svd(H)
    r = V[:,-1]
    print("r ", r)
    print(H)    
    

dtype = torch.float
device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
T0 = 5  # Temporal length of the initial window
eps = 0.0001  # Gram Matrix noise
BS = 100  # Batch Size
L0 = 6  # Longitude of the Root Sequence
L = 2  # Longitude of the Sequence being Tested
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
        p = torch.Tensor([[300.0,500.0]])
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