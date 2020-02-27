from baseline.utils_dynamics import *
from functools import reduce
from operator import mul


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

    def __init__(self, T0=7, T=2, noise=0.0001):
        """ Inits TrackerDynBoxes"""
        self.T0 = T0
        self.T = T
        self.noise = noise
        self.buffer_past_x = torch.zeros((T0, 1))
        self.buffer_past_y = torch.zeros((T0, 1))
        self.buffer_future_x = []
        self.buffer_future_y = []
        self.current_t = 0
        self.past_JBLDs = []
        self.JBLDs_init = []  # Podria ser un array de np.zeros((self.T0-3))

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
        if cand <= thresh:
            belongs = 1
        return belongs

    def update_buffers(self, new_result):
        """ Generates a candidate sequence given an index
        Args:
            - new_result:
        """
        self.buffer_past_x[0:-1, 0] = self.buffer_past_x[1:, 0]
        self.buffer_past_y[0:-1, 0] = self.buffer_past_y[1:, 0]
        self.buffer_past_x[-1, 0] = new_result[0]
        self.buffer_past_y[-1, 0] = new_result[1]
        del self.buffer_future_x[0]
        del self.buffer_future_y[0]

    def decide(self, *candidates):
        # print(self.current_t)
        # print(candidates)
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
            # if(self.current_t > 2):
            #     buffers_past = torch.cat([self.buffer_past_x, self.buffer_past_y], dim=1).unsqueeze(0)
            #     cand_torch = torch.empty((1,1,2))
            #     cand_torch[0,0,0] = float(candidates[0][0][0])
            #     cand_torch[0,0,1] = float(candidates[0][0][1])
            #     jbld_ = compare_dynamics(buffers_past.type(torch.FloatTensor), cand_torch.type(torch.FloatTensor))
            #     self.JBLDs_init.append(jbld_)
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
                    JBLDs[i, 0] = compare_dynamics(buffers_past.type(torch.FloatTensor), seq.type(torch.FloatTensor), self.noise)

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

