import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch


def get_mask(length, max_len, out_shape):
    """
    length shape:[batch_size]
    """
    length = torch.tensor(length)
    range_tensor = torch.arange(max_len).unsqueeze(0)
    mask = range_tensor < length.unsqueeze(1)
    ret = mask.float().reshape(out_shape)

    return ret


class MECPEStep1(nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout=None,
                 use_audio=False,
                 use_video=False):
        """

        :param dropout: If dropout is a list, then it is [dropout_audio, dropout_video, dropout_bert_hidden, dropout_bert_attention, dropout_softmax]. If it is a float, then it is the global dropout.
        :param use_audio:
        :param use_video:
        """
        super(MECPEStep1, self).__init__()
        if dropout is None:
            self.dropout_audio, self.dropout_video, self.dropout_bert_hidden, self.dropout_bert_attention, self.dropout_softmax = [
                0.5, 0.5, 0.1, 0.3, 1]
        elif isinstance(dropout, list):
            self.dropout_audio, self.dropout_video, self.dropout_bert_hidden, self.dropout_bert_attention, self.dropout_softmax = dropout
        else:
            self.dropout_audio, self.dropout_video, self.dropout_bert_hidden, self.dropout_bert_attention, self.dropout_softmax = [dropout] * 5
        self.use_audio = use_audio
        self.use_video = use_video


