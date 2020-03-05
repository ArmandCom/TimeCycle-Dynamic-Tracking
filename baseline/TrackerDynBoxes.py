from utils_dynamics import *
from functools import reduce
from operator import mul
import torch
import numpy as np
device = torch.device('cpu')


class TrackerDynBoxes:
    # Generates a candidate sequence given an index

    def __init__(self, T0=7, T=2, noise=0.0001, coord=0):
        """ Inits TrackerDynBoxes"""
        self.T0 = T0
        self.T = T
        self.noise = noise
        self.buffer_past_x = torch.zeros((T0, 1))
        self.buffer_past_y = torch.zeros((T0, 1))
        self.buffer_future_x = []
        self.buffer_future_y = []
        self.current_t = 0
        self.JBLDs_x = []
        self.count_pred = 0
        self.coord = coord

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

    def classify(self, cand, thresh=0.5):
        """ Determines if a candidate should be kept or a prediction should be performed
        Args:
            - cand: ??
            - thresh: Threshold
        Returns:
            - belongs: Boolean determining if the position must be predicted (-1) or not (1)
        """
        belongs = 1
        th = 0.00045
        past_jbld = self.JBLDs_x[-1]
        frame_to_predict = 18
        # if self.current_t == frame_to_predict or self.current_t == frame_to_predict + 1:
        #     belongs = -1
        # if self.JBLDs_x[-1] > th:
        #     print('PREDICTED!')
        #     belongs = -1
        return belongs

    def update_buffers(self, new_result):
        """ Updates buffers with incoming information
        """
        self.buffer_past_x[0:-1, 0] = self.buffer_past_x[1:, 0]
        self.buffer_past_y[0:-1, 0] = self.buffer_past_y[1:, 0]
        self.buffer_past_x[-1, 0] = new_result[0]
        self.buffer_past_y[-1, 0] = new_result[1]
        del self.buffer_future_x[0]
        del self.buffer_future_y[0]

    def decide(self, *candidates):
        """ Generates a candidate sequence given an index
        Args:
            - candidates: list containing the number of candidates per frame (T)
            - candidates contains N sublists [ [ [(px11,py11)] ] , [ [(px12,py12)],[(px22,py22)] ] ]
            - candidates[1] = [ [(px12,py12)],[(px22,py22)] ]
            - candidates[1][0] = [(px12,py12)]
        Returns:
            - point_to_add: (x,y) decided point at certain sequence
        """
        candidates = candidates[0]
        point_to_add = torch.zeros(2)

        if self.current_t < self.T0:
            # Tracker needs the first T0 points to compute a reliable JBLD
            self.buffer_past_x[self.current_t, 0] = float(candidates[0][0][0])
            self.buffer_past_y[self.current_t, 0] = float(candidates[0][0][1])
            self.current_t += 1
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
                for i in range(num_of_seqs):
                    # Build each sequence of tree
                    seq = self.generate_seq_from_tree(seqs_lengths, i)
                    # Compute JBLD for each
                    # compare_dynamics needs a sequence of (1, T0, 2) and (1, T, 2)
                    JBLDs[i, 0] = compare_dynamics(buffers_past.type(torch.FloatTensor), seq.type(torch.FloatTensor), self.noise, self.coord)

                # Choose the minimum
                min_idx_jbld = torch.argmin(JBLDs)
                min_val_jbld = JBLDs[min_idx_jbld, 0]
                point_to_add = self.generate_seq_from_tree(seqs_lengths, min_idx_jbld)
                point_to_add = point_to_add[0, 0, :]
                self.JBLDs_x.append(min_val_jbld)

                # Classify candidate
                classification_outcome = self.classify(min_val_jbld)  # -1 bad, 1 good
                if classification_outcome == -1:
                    Hx = Hankel(self.buffer_past_x)
                    Hy = Hankel(self.buffer_past_y)
                    px = predict_Hankel(Hx)
                    py = predict_Hankel(Hy)
                    point_to_add[0] = px
                    point_to_add[1] = py

                self.update_buffers(point_to_add)

            self.current_t += 1
        return point_to_add