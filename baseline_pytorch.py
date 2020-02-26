import torch
import numpy as np
import pickle as pkl
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt


class TrackerDynBoxes:
    """ Generates a candidate sequence given an index
    Attributes:
        - T0: size of the past window
        - T: size of the future window
        - buffer_past_x:
        - buffer_past_y:
        - buffer_future_x:
        - buffer_future_y:
        - current_T0:
        - current_T:
        - past_JBLDs:
    """

    def __init__(self, T0=7, T=2):
        """ Inits TrackerDynBoxes"""
        self.T0 = T0
        self.T = T
        self.buffer_past_x = torch.zeros((T0, 1))
        self.buffer_past_y = torch.zeros((T0, 1))
        self.buffer_future_x = []
        self.buffer_future_y = []
        self.current_t = 0
        self.past_JBLDs = []

    def generate_seq_from_tree(self, seq_lengths, idx):
        """ Generates a candidate sequence given an index
        Args:
            - seq_lengths: list containing the number of candidates per frame (T)
            - idx: index of the desired sequence (1)
        Returns:
            - sequence: sequence corresponding to the provided index (1, T, (x,y))
        """
        sequence = np.zeros((1, len(seq_lengths), 2))
        new_idx = np.unravel_index(idx, seq_lengths)
        for frame in range(len(new_idx)):
            sequence[:, frame, 0] = self.buffer_future_x[frame][new_idx[frame]]
            sequence[:, frame, 1] = self.buffer_future_y[frame][new_idx[frame]]
        sequence = torch.from_numpy(sequence)
        return sequence

    def classify(self, cand, thresh = 0.5):
        """ Generates a candidate sequence given an index
        Args:
            - cand:
            - thresh:
        Returns:
            - belongs:
        """
        belongs = -1
        if len(self.past_JBLDs) == 0:
            belongs = 1
            return belongs
        else:
            # mitjana = sum(self.past_JBLDs)/len(self.past_JBLDs)
            pass
        # if abs(cand-mitjana) <= thresh:
        if cand<=thresh:
            belongs = 1
        return belongs

    def update_buffers(self, new_result):
        """ Generates a candidate sequence given an index
        Args:
            - new_result:
        """
        self.buffer_past_x[0:-1, 0] = self.buffer_past_x[1:,0]
        self.buffer_past_y[0:-1, 0] = self.buffer_past_y[1:,0]
        self.buffer_past_x[-1, 0] = new_result[0]
        self.buffer_past_y[-1, 0] = new_result[1]
        del self.buffer_future_x[0]
        del self.buffer_future_y[0]
    
    def decide(self, *candidates):
        print(self.current_t)
        print(candidates)
        """ Generates a candidate sequence given an index
        Args:
            - candidates: list containing the number of candidates per frame (T)
        """
        candidates = candidates[0]
        """
        candidates contains N sublists [ [ [(px11,py11)] ] , [ [(px12,py12)],[(px22,py22)] ] ]
        candidates[1] = [ [(px12,py12)],[(px22,py22)] ]
        candidates[1][0] = [(px12,py12)]
        """
        point_to_add = torch.zeros(2)
        if self.current_t < self.T0:
            # Tracker needs the first T0 points to compute a reliable JBLD
            self.buffer_past_x[self.current_t, 0] = float(candidates[0][0][0])
            self.buffer_past_y[self.current_t, 0] = float(candidates[0][0][1])
            self.current_t += 1
            # point_to_add = torch.tensor([float(candidates[0][0][0]), float(candidates[0][0][1])])
            if len(candidates) > 1:
                raise ValueError('There is more than one candidate in the first T0 frames')
        else:
            # Append points to buffers
            temp_list_x = []
            temp_list_y = []
            for [(x, y)] in candidates:
                temp_list_x.append(x)
                temp_list_y.append(y)
            self.buffer_future_x.append(temp_list_x)
            self.buffer_future_y.append(temp_list_y)

            if len(self.buffer_future_x) == self.T:
                # Buffers are now full
                seqs_lengths = []
                [seqs_lengths.append(len(y)) for y in self.buffer_future_x]
                num_of_seqs = reduce(mul, seqs_lengths)
                JBLDs = torch.zeros((num_of_seqs, 1))
                buffers_past = torch.cat([self.buffer_past_x, self.buffer_past_y], dim=1).unsqueeze(0)
                # print("seqs ",num_of_seqs)
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
                # print("idx of jbld = ", min_idx_jbld)
                # Classify candidate
                classification_outcome = self.classify(min_val_jbld)  # -1 bad, 1 good
                if classification_outcome == -1:
                    print("predict")
                    Hx = Hankel(self.buffer_past_x)
                    Hy = Hankel(self.buffer_past_y)
                    px = predict_Hankel(Hx)
                    py = predict_Hankel(Hy)
                    point_to_add[0] = px
                    point_to_add[1] = py
                else:
                    # We only use the JBLD if it is consistent with past dynamics, i.e, we did not have to predict
                    self.past_JBLDs.append(min_val_jbld)
                    # point_to_add[0] = float(self.buffer_future_x[0][0])
                    # point_to_add[1] = float(self.buffer_future_y[0][0])
                # update buffers
                self.update_buffers(point_to_add)
            
            self.current_t += 1
        return point_to_add


def Hankel(s0, stitch=False, s1=0):
    """ Generates a candidate sequence given an index
    Args:
        - s0: Root sequence
        - switch: Boolean to indicate if Hankel must be stitched or not
        - s1: Sequence to add if Hankel must be stitched
    Returns:
        - H: Hankel matrix
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


def Gram(H, eps):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    N = np.power(eps, 2) * H.shape[0] * torch.eye(H.shape[0])
    G = torch.matmul(H, H.t()) + N
    Gnorm = G/torch.norm(G, 'fro')
    return Gnorm


def JBLD(X, Y, det):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    d = torch.log(torch.det((X + Y)/2)) - 0.5*torch.log(torch.det(torch.matmul(X, Y)))
    if not det:
        d = (torch.det((X + Y) / 2)) - 0.5 * (torch.det(torch.matmul(X, Y)))
    return d


def compare_dynamics(data_root, data, BS=1):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    dist = torch.zeros(BS, 2, device=device)
    for n_batch in range(BS):
        for d in range(2):
            H0 = Hankel(data_root[n_batch, :, d])
            H1 = Hankel(data_root[n_batch, :, d], True, data[n_batch, :, d])
            dist[n_batch, d] = JBLD(Gram(H0, eps), Gram(H1, eps), False)
    dist = torch.mean(dist, 1)
    # print(dist[0].item())
    return dist


def predict_Hankel(H):
    """ Generates a candidate sequence given an index
    Args:
        - seq_lengths: list containing the number of candidates per frame (T)
        - idx: index of the desired sequence (1)
    Returns:
        - sequence: sequence corresponding to the provided index (1, T, (x,y))
    """
    rows, cols = H.size()
    U, S, V = torch.svd(H)
    r = V[:,-1]
    last_column_of_H = H[-1,:]
    last_column_of_H = last_column_of_H[1:]
    first_term = torch.matmul(last_column_of_H, r[:-1])/(-r[-1])
    return first_term


device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
eps = 0.0001  # Gram Matrix noise
directory = '/data/Ponc/tracking/centroids_tree_nhl.obj'

# Tracker data
with open(directory, 'rb') as f:
    data = pkl.load(f)

# tracker = TrackerDynBoxes(T0=7, T=4)
# points_tracked_npy = np.zeros((len(data), 2))
# for t, points in enumerate(data):
    
#     points_tracked = tracker.decide(points)

#     points_tracked_npy[t, :] = np.asarray(points_tracked)


# t = np.arange(len(data))
# plt.scatter(t, points_tracked_npy[:, 0], s=75, c='r', zorder=1, label='x component', alpha=0.5)
# plt.scatter(t, points_tracked_npy[:, 1], s=75, c='b', zorder=1, label='y component', alpha=0.5)
# plt.legend(loc="center left")
# plt.title('Trajectory Evolution just predicting one time step')
# plt.xlabel('Time [s]')
# plt.show()
def plot_candidates_and_trajectory(data, points_tracked_npy, T0, T, count_t = 0):
    
    for t, points in enumerate(data):
        if(t>T0+T-1): #and t<=len(data)-T):
            print("t = ", t)
            if t == 0:
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75, label='candidates')
                # pass
            if len(points) == 1:
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75)
                plt.scatter(t, points[0][0][1], s=50, c='k', zorder=1, alpha=0.75)
                # pass
            else:
                # pass
                for c in range(len(points)):
                    plt.scatter(t, points[c][0][0], s=50, c='k', zorder=1, alpha=0.75)
                    plt.scatter(t, points[c][0][1], s=50, c='k', zorder=1, alpha=0.75)
            
            if(count_t<points_tracked_npy.shape[0]):
                plt.scatter(t-(T), int(points_tracked_npy[count_t,0]), s=25, c='tomato', zorder=2, label='decided x')
                plt.scatter(t-(T), int(points_tracked_npy[count_t,1]), s=25, c='orange', zorder=1, label='decided y')
                count_t = count_t + 1
                print(count_t)
        else:    
            if t == 0:
                # pass
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75, label='candidates')
            if len(points) == 1:
                # pass
                plt.scatter(t, points[0][0][0], s=50, c='k', zorder=1, alpha=0.75)
                plt.scatter(t, points[0][0][1], s=50, c='k', zorder=1, alpha=0.75)
            else:
                # pass
                for c in range(len(points)):
                    plt.scatter(t, points[c][0][0], s=50, c='k', zorder=1, alpha=0.75)
                    plt.scatter(t, points[c][0][1], s=50, c='k', zorder=1, alpha=0.75)
            
    # time = np.arange(len(data)-T-T0+1)
    # plt.scatter(time, points_tracked_npy[:, 0], s=25, c='tomato', zorder=2, label='decided x')
    # plt.scatter(time, points_tracked_npy[:, 1], s=25, c='orange', zorder=1, label='decided y')
    # plt.legend(loc="center left")
    tit = 'Trajectory Evolution with T0=' + str(T0) + ' and T=' + str(T)
    plt.title(tit)
    plt.xlabel('Time')
    plt.show()


device = torch.device('cpu')
# device = torch.device('cuda:0')
# Parameters
eps = 0.0001  # Gram Matrix noise
# directory = '/Users/marinaalonsopoal/PycharmProjects/Marina/Tracker/centroids_tree_nhl.obj'
directory = '/data/Ponc/tracking/centroids_tree_nhl.obj'
# Tracker data
with open(directory, 'rb') as f:
    data = pkl.load(f)
T0 = 10
T = 3
tracker = TrackerDynBoxes(T0=T0, T=T)
points_tracked_npy = np.zeros((len(data)-T0+1, 2))
print("Size of npy = ", points_tracked_npy.shape)

for t, points in enumerate(data):
    print("---------")
    points_tracked = tracker.decide(points)
    if t >= T0+T-1 :
        print("t = ", t)
        print("retorna el tracker = ", points_tracked)
        print("t-T+1 = ", t-T+1)
        print("t-T-T0+1 = ", t-T-T0+1)
        points_tracked_npy[t-T-T0+1, :] = np.asarray(points_tracked)
print(points_tracked_npy)
plot_candidates_and_trajectory(data, points_tracked_npy, T0, T)