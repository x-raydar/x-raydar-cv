import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import model_20210820_XNet38MS.wt_inception as inception

class XNet38_urg(nn.Module):
    # This module implements XNet38 followed by a FCS module to translate probabilities into urgencies 
    def __init__(self):
        super().__init__()

        self.model_XNet38 = inception.Inception3(num_classes=38, # Directly predict the multilabels
                                                 aux_logits=True, # loss should use this
                                                 transform_input=True)


        self.model_FCS = TranslatorCVLogitsToUrgency_fcs(num_labels=38, num_urgencies=4)

    def load_state_dict(self, folder_weights):
    
        model_XNet38_state_fn = os.path.join(folder_weights, 'model_best.pth.tar')
        state = torch.load(model_XNet38_state_fn, map_location=torch.device('cpu'))
        self.model_XNet38.load_state_dict(state['state_dict'])

        model_FCS_state_fn = os.path.join(folder_weights, 'model_TranslatorCVLogitsToUrgency_fcs.pth.tar')
        state = torch.load(model_FCS_state_fn, map_location=torch.device('cpu'))
        self.model_FCS.load_state_dict(state['state_dict'])


    def forward(self, x):
        # First multi-label classifier:
        logits_multi = self.model_XNet38(x) # N x 38
        # print('DEBUG:', 'logits_multi.shape', logits_multi.shape)
        logits_urg = self.model_FCS(logits_multi) # N x 4
        # print('DEBUG:', 'logits_urg.shape', logits_urg.shape)

        return logits_multi, logits_urg



# Translator from CV_logits to urg_logits
class TranslatorCVLogitsToUrgency_fcs(nn.Module):

    def __init__(self, num_labels, num_urgencies):
        super().__init__()

        self.fc1 = nn.Linear(num_labels, num_labels)
        self.fc2 = nn.Linear(num_labels, num_urgencies)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x