import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import numpy as np
import data
import model

epochs = 100  # The number of epochs
batch_size = 8  # The number of dialogues in a batch
hidden_dim = 200  # The dimension of hidden layer
learning_rate = 1e-5
warmup_percent = 0.1
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
        print('Training step1 ...')

        train_loss = 0
        criterion = nn.CrossEntropyLoss()
        model1.init_weights()
        optimizer = torch.optim.AdamW(model1.parameters(), lr=learning_rate)

        num_training_steps = epochs * int(np.ceil(len(train_data) / batch_size))
        num_warmup_steps = int(num_training_steps * warmup_percent)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        for epoch in range(epochs):




