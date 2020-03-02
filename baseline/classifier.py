import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):
    def __init__(self):
        super(SequenceClassifier, self).__init__()
        self.lstm = nn.LSTM()    

