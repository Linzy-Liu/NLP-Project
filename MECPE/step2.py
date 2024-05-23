import time
import torch
import torch.nn as nn
import numpy as np
import data
import model

epochs = 20  # The number of epochs
batch_size = 4  # The number of dialogues in a batch
hidden_dim = 200  # The dimension of hidden layer
pos_dim = 50  # The dimension of position embedding
learning_rate = 1e-5
warmup_percent = 0.1
max_utt_num = 35
max_sent_len = 35
report_freq = 100
real_time = True
choose_emo_cat = False

bert_path = 'F:/python_work/GitHub/bert-base-uncased'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    # Load data
    word_ebd, pos_ebd, word2idx, idx2word = data.get_glove_embedding('data/ECF_glove_300.txt',
                                                                     'data/all_data_pair.txt',
                                                                     pos_dim)
    var_list = [word2idx, word_ebd, pos_ebd]
    emb_paths = ['data/video_id_mapping.npy', 'data/video_embedding_4096.npy', 'data/audio_embedding_6373.npy']


