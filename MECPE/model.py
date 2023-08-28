import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig


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
                 hidden_dim,
                 max_sent_len,
                 max_utt_num,
                 dropout=None,
                 embeddings=None,
                 use_audio=True,
                 use_video=True
                 ):
        """

        :param hidden_dim:
        :param max_sent_len:
        :param max_utt_num:
        :param dropout: If dropout is a list, then it is [dropout_audio, dropout_video, dropout_bert_hidden, dropout_bert_attention, dropout_softmax]. If it is a float, then it is the global dropout.
        :param use_audio:
        :param use_video:
        :param embeddings:
        """
        super(MECPEStep1, self).__init__()
        if dropout is None:
            dropout = [0.5, 0.5, 0.1, 0.3, 1]
            self.dropout_audio, self.dropout_video, self.dropout_softmax = [nn.Dropout(dropout[i]) for i in [0, 1, 4]]
            dropout_bert_hidden, dropout_bert_attention = dropout[2], dropout[3]
        elif isinstance(dropout, list):
            self.dropout_audio, self.dropout_video, self.dropout_softmax = [nn.Dropout(dropout[i]) for i in [0, 1, 4]]
            dropout_bert_hidden, dropout_bert_attention = dropout[2], dropout[3]
        else:
            self.dropout_audio, self.dropout_video, self.dropout_softmax = [nn.Dropout(dropout) for _ in range(3)]
            dropout_bert_hidden, dropout_bert_attention = dropout, dropout
        self.use_audio = use_audio
        self.use_video = use_video
        if self.use_audio or self.use_video:
            if embeddings is None:
                raise NotImplementedError('The audio and video are not implemented.')
            else:
                self.audio_embeddings, self.video_embeddings = embeddings
        else:
            self.audio_embeddings, self.video_embeddings = None, None
        self.hidden_dim = hidden_dim
        self.max_shape = [max_utt_num, max_sent_len]

        config = BertConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states = True
        config.hidden_dropout_prob = dropout_bert_hidden
        config.attention_probs_dropout_prob = dropout_bert_attention
        self.BERT = BertModel.from_pretrained('bert-base-uncased', config=config)
        # linear settings
        self.linear_audio_emotion = nn.Linear(6373, hidden_dim)
        self.linear_audio_cause = nn.Linear(6373, hidden_dim)
        self.linear_video_emotion = nn.Linear(4096, hidden_dim)
        self.linear_video_cause = nn.Linear(4096, hidden_dim)
        self.linear_fusion_emotion = nn.Linear(hidden_dim * 2 + 768, hidden_dim)
        self.linear_fusion_cause = nn.Linear(hidden_dim * 2 + 768, hidden_dim)

    def forward(self, x_a_v, bert_input, diag_len):
        """

        :param x_a_v: The input of audio and video. x_a_v[0] is the audio input, x_a_v[1] is the video input.
        :param bert_input: The input of bert_sen. bert_input[0] is the input of bert, bert_input[1] is the mask of bert.
        :param diag_len: The length of each dialogue.
        :return: The output of the first step prediction.
        """
        x_audio, x_video = x_a_v
        x_bert_sent, x_bert_sent_mask = bert_input
        # x_audio shape:[batch_size, max_utt_num, 6373]
        # x_video shape:[batch_size, max_utt_num, 4096]
        # x_bert_sent shape:[batch_size, max_utt_num, max_sent_len]
        # x_bert_sent_mask shape:[batch_size, max_utt_num, max_sent_len]
        # diag_len shape:[batch_size]

        x_audio = F.embedding(x_audio, self.audio_embeddings)
        x_video = F.embedding(x_video, self.video_embeddings)
        x_audio = self.dropout_audio(x_audio)
        x_video = self.dropout_video(x_video)
