import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

bert_path = 'F:/python_work/GitHub/bert-base-uncased'


def get_mask(length, max_len, out_shape):
    """
    length shape:[batch_size]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = torch.tensor(length, device=device)
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
    mask = range_tensor < length.unsqueeze(1)
    ret = mask.float().reshape(out_shape)

    return ret


def softmax_with_length_mask(input, length):
    # input shape:[batch_size, 1, max_len]
    input = torch.exp(input.float())
    input *= get_mask(length, input.shape[-1], input.shape)
    exp_sum = torch.sum(input, dim=-1, keepdim=True) + 1e-9
    return input / exp_sum


def concat_feature(tensors):
    return torch.cat(tensors, dim=-1)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, x, length):
        x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(x)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output


class RealTimeTransformer(nn.Module):
    def __init__(self, n_hidden, n_head=1, real_time=True):
        super(RealTimeTransformer, self).__init__()
        # Multi-head Attention mechanism
        self.real_time = real_time
        self.n_head = n_head
        self.self_attn = nn.MultiheadAttention(n_hidden, n_head, batch_first=True)
        # Point-wise Feedforward network
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        # Layer normalization
        self.norm1 = nn.LayerNorm(n_hidden)
        self.norm2 = nn.LayerNorm(n_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch_size, max_len, n_hidden]
        # Multi-head Attention
        key_sum = torch.sum(torch.abs(x), dim=-1)
        key_mask = torch.sign(key_sum).repeat([self.n_head, 1])  # [n_head * batch_size, max_len]
        key_mask = key_mask.unsqueeze(1).repeat([1, x.shape[1], 1])  # [n_head * batch_size, max_len, max_len]
        if self.real_time:
            real_time_mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).to(x.device)
            mask = key_mask * real_time_mask == 0
        else:
            mask = key_mask == 0
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)

        x = self.norm1(attn_output + x)
        # Point-wise Feedforward
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        output = self.norm2(output + x)

        return output

    def init_weights(self, init_range=0.1):
        nn.init.uniform_(self.fc1.weight, -init_range, init_range)
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc2.weight, -init_range, init_range)
        nn.init.zeros_(self.fc2.bias)


class MECPEStep1(nn.Module):
    def __init__(self,
                 hidden_dim,
                 max_sent_len,
                 max_utt_num,
                 dropout=None,
                 embeddings=None,
                 use_audio=True,
                 use_video=True,
                 choose_emo_cat=False,
                 real_time=True,
                 bert_path=bert_path
                 ):
        """
        The first step of MECPE.
        :param dropout: If dropout is a list, then it is [dropout_audio, dropout_video, dropout_bert_hidden, dropout_bert_attention, dropout_softmax]. If it is a float, then it is the global dropout.
        :param embeddings: The list of embeddings [audio_embeddings, video_embeddings]. If one of them is None, then the corresponding modality is not used.
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

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.norm = nn.LayerNorm(hidden_dim)

        config = BertConfig.from_pretrained(bert_path)
        config.output_hidden_states = True
        config.hidden_dropout_prob = dropout_bert_hidden
        config.attention_probs_dropout_prob = dropout_bert_attention
        self.BERT = BertModel.from_pretrained(bert_path, config=config)
        self.transformer = RealTimeTransformer(hidden_dim, n_head=2, real_time=real_time)
        # linear settings
        self.linear_audio_emotion = nn.Linear(6373, hidden_dim)
        self.linear_audio_cause = nn.Linear(6373, hidden_dim)
        self.linear_video_emotion = nn.Linear(4096, hidden_dim)
        self.linear_video_cause = nn.Linear(4096, hidden_dim)
        self.linear_fusion_emotion = nn.Linear(hidden_dim * np.sum([self.use_audio, self.use_video]) + 768, hidden_dim)
        self.linear_fusion_cause = nn.Linear(hidden_dim * np.sum([self.use_audio, self.use_video]) + 768, hidden_dim)
        pred_dim = 7 if choose_emo_cat else 2
        self.linear_emotion_pred = nn.Linear(hidden_dim, pred_dim)
        self.linear_cause_pred = nn.Linear(hidden_dim, pred_dim)

    def forward(self, x_a_v, bert_input, diag_len):
        """

        :param x_a_v: The input of audio and video. x_a_v[0] is the audio input, x_a_v[1] is the video input.
        :param bert_input: The input of bert_sen. bert_input[0] is the input of bert, bert_input[1] is the mask of bert.
        :param diag_len: The length of each dialogue.
        :return: The output of the first step prediction.
        """
        if x_a_v is None:
            x_audio, x_video = None, None
        else:
            x_audio, x_video = x_a_v
        x_bert_sent, x_bert_sent_mask = bert_input
        # x_audio shape:[batch_size, max_utt_num]
        # x_video shape:[batch_size, max_utt_num]
        # x_bert_sent shape:[batch_size, max_utt_num, max_sent_len]
        # x_bert_sent_mask shape:[batch_size, max_utt_num, max_sent_len]
        # diag_len shape:[batch_size]

        feature_mask = get_mask(diag_len, self.max_shape[0], [-1, self.max_shape[0], 1])
        x_bert_sen = x_bert_sent.view(-1, self.max_shape[1])
        x_mask_bert_sen = x_bert_sent_mask.view(-1, self.max_shape[1])

        outputs = self.BERT(x_bert_sen, attention_mask=x_mask_bert_sen)
        s_bert = outputs[1]
        s_bert = s_bert.view(-1, self.max_shape[0], s_bert.shape[-1])
        s_bert = s_bert * feature_mask

        if x_audio is not None:
            x_audio = F.embedding(x_audio, self.audio_embeddings)
            x_audio = self.dropout_audio(x_audio)
            x_audio_emotion = self.relu(self.norm(self.linear_audio_emotion(x_audio)))
            x_audio_cause = self.relu(self.norm(self.linear_audio_cause(x_audio)))

        if x_video is not None:
            x_video = F.embedding(x_video, self.video_embeddings)
            x_video = self.dropout_video(x_video)
            x_video_emotion = self.relu(self.norm(self.linear_video_emotion(x_video)))
            x_video_cause = self.relu(self.norm(self.linear_video_cause(x_video)))

        if self.use_audio and self.use_video:
            tensors_emo = [s_bert, x_audio_emotion, x_video_emotion]
            tensors_cause = [s_bert, x_audio_cause, x_video_cause]
        elif self.use_audio:
            tensors_emo = [s_bert, x_audio_emotion]
            tensors_cause = [s_bert, x_audio_cause]
        elif self.use_video:
            tensors_emo = [s_bert, x_video_emotion]
            tensors_cause = [s_bert, x_video_cause]
        else:
            tensors_emo = [s_bert]
            tensors_cause = [s_bert]

        x_emotion = concat_feature(tensors_emo)
        x_emotion = x_emotion * feature_mask
        x_emotion = self.linear_fusion_emotion(x_emotion)

        x_cause = concat_feature(tensors_cause)
        x_cause = x_cause * feature_mask
        x_cause = self.linear_fusion_cause(x_cause)
        # Get the expression vector of emotion and cause
        x_emotion = self.transformer(x_emotion)  # [batch_size, max_utt_num, hidden_dim]
        x_cause = self.transformer(x_cause)  # [batch_size, max_utt_num, hidden_dim]

        # Get the prediction of emotion and cause
        x_emotion = self.dropout_softmax(x_emotion)
        x_cause = self.dropout_softmax(x_cause)
        x_emotion = self.softmax(self.linear_emotion_pred(x_emotion))
        x_cause = self.softmax(self.linear_cause_pred(x_cause))

        reg = (torch.norm(self.linear_emotion_pred.weight, p=2) + torch.norm(self.linear_emotion_pred.bias, p=2) +
               torch.norm(self.linear_cause_pred.weight, p=2) + torch.norm(self.linear_cause_pred.bias, p=2))

        return x_emotion, x_cause, reg

    def init_weights(self, init_range=0.1):
        def _init_linear(linear):
            nn.init.uniform_(linear.weight, -init_range, init_range)
            nn.init.zeros_(linear.bias)

        _init_linear(self.linear_audio_emotion)
        _init_linear(self.linear_audio_cause)
        _init_linear(self.linear_video_emotion)
        _init_linear(self.linear_video_cause)
        _init_linear(self.linear_fusion_emotion)
        _init_linear(self.linear_fusion_cause)
        _init_linear(self.linear_emotion_pred)
        _init_linear(self.linear_cause_pred)
        self.transformer.init_weights(init_range)


class MECPEStep2(nn.Module):
    def __init__(self, hidden_dim, dropout=None,
                 embeddings=None,
                 use_audio=True,
                 use_video=True,
                 choose_emo_cat=False
                 ):
        """
        The second step of MECPE.
        :param hidden_dim: The output dimension of the Linear layer.
        :param dropout: The dropout rate, it is a list of [dropout_audio, dropout_video, dropout_word, dropout_LSTM, dropout_softmax]. If it is a float, then it is the global dropout.
        :param embeddings: The embeddings of audio, video and word, which is [audio_embeddings, video_embeddings, word_embeddings, position_embeddings]
        :param use_audio:
        :param use_video:
        :param choose_emo_cat: Whether to take the emotion category into consideration.
        """
        super(MECPEStep2, self).__init__()
        self.audio_embeddings, self.video_embeddings, self.word_embeddings, self.position_embeddings = embeddings
        self.use_audio = use_audio
        self.use_video = use_video
        self.choose_emo_cat = choose_emo_cat
        self.hidden_dim = hidden_dim
        if dropout is None:
            dropout = [0.5, 0.5, 0., 0.1, 0.5]
        elif isinstance(dropout, list):
            dropout = dropout
        else:
            dropout = [dropout] * 5

        self.linear_audio = nn.Linear(6373, hidden_dim)
        self.linear_video = nn.Linear(4096, hidden_dim)
        self.linear_attention1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_attention2 = nn.Linear(hidden_dim, 1, bias=False)
        self.linear_softmax = nn.Linear(2 * hidden_dim * (np.sum([self.use_audio, self.use_video]) + 1)
                                        + (2 if self.choose_emo_cat else 1) * self.position_embeddings.shape[-1], 2)
        self.lstm = BiLSTM(hidden_dim, hidden_dim, 1, dropout=dropout[3])

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.tanh = nn.Tanh()

        self.dropout_audio, self.dropout_video, self.dropout_word = [nn.Dropout(dropout[i]) for i in range(3)]
        self.dropout_softmax = nn.Dropout(dropout[4])

    def forward(self, x, x_video, distance, sent_len, x_emo_cat=None):
        if self.choose_emo_cat and x_emo_cat is None:
            raise ValueError('You should input the emotion category.')

        # Embeddings
        # After embedding:
        # x shape:[batch_size, 2, max_sent_len, emb_dim]
        # x_video shape:[batch_size, 2, 4096]
        # x_audio shape:[batch_size, 2, 6373]
        # x_distance shape:[batch_size, pos_emb_dim]
        # x_emo_cat shape:[batch_size, pos_emb_dim]
        x = F.embedding(x, self.word_embeddings)
        x = self.dropout_word(x)
        if self.use_audio:
            x_audio = F.embedding(x_video, self.audio_embeddings)
            x_audio = self.dropout_audio(x_audio)
            x_audio = self.relu(self.norm(self.linear_audio(x_audio)))
        if self.use_video:
            x_video = F.embedding(x_video, self.video_embeddings)
            x_video = self.dropout_video(x_video)
            x_video = self.relu(self.norm(self.linear_video(x_video)))
        x_distance = F.embedding(distance, self.position_embeddings)
        if self.choose_emo_cat:
            x_emo_cat = F.embedding(x_emo_cat, self.position_embeddings)

        # Attention
        sent_len = sent_len.reshape(-1)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [batch_size * 2, max_sent_len, emb_dim]
        x = self.lstm(x, sent_len)  # [batch_size * 2, max_sent_len, hidden_dim]
        tmp = x.reshape(-1, x.shape[-1])  # [batch_size * 2 * max_sent_len, hidden_dim]
        q = self.tanh(self.linear_attention1(tmp))  # [batch_size * 2 * max_sent_len, hidden_dim]
        alpha = self.linear_attention2(q).reshape(-1, 1, x.shape[1])  # [batch_size * 2, 1, max_sent_len]
        alpha = softmax_with_length_mask(alpha, sent_len)  # [batch_size * 2, 1, max_sent_len]
        x = torch.bmm(alpha, x).reshape(-1, 2, x.shape[-1])  # [batch_size, 2, hidden_dim]

        # Fusion
        tensors = [x]
        if self.use_audio:
            tensors.append(x_audio)
        if self.use_video:
            tensors.append(x_video)
        x = concat_feature(tensors)
        x = x.reshape(-1, 2 * x.shape[-1])  # [batch_size, 2*combined_dim]
        tensors = [x, x_distance]
        if self.choose_emo_cat:
            tensors.append(x_emo_cat)
        x = concat_feature(
            tensors)  # [batch_size, 2*combined_dim + pos_emb_dim] or [batch_size, 2*combined_dim + 2*pos_emb_dim]
        x = self.dropout_softmax(x)
        x = self.linear_softmax(x)
        output = self.softmax(x)
        return output
