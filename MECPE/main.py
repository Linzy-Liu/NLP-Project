import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import data
import model

batch_size = 8  # The number of dialogues in a batch
hidden_dim = 200  # The dimension of hidden layer
max_utt_num = 35
max_sent_len = 35


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    # Load the data
    train_data = data.DataSet(
        'data/train.txt',
        'data/video_id_mapping.npy',
        'data/video_embedding_4096.npy',
        'data/audio_embedding_6373.npy')
    dev_data = data.DataSet(
        'data/dev.txt',
        'data/video_id_mapping.npy',
        'data/video_embedding_4096.npy',
        'data/audio_embedding_6373.npy')
    test_data = data.DataSet(
        'data/test.txt',
        'data/video_id_mapping.npy',
        'data/video_embedding_4096.npy',
        'data/audio_embedding_6373.npy')
    model1 = model.MECPEStep1(hidden_dim, max_utt_num, max_sent_len).to(device)

    def train_step1():
        model1.train()

        train_loss = 0
        criterion = nn.CrossEntropyLoss()
        model1.init_weights()



