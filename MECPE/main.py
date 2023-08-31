import time
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
report_freq = 100

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
        start_time = time.time()

        train_loss = 0
        criterion = nn.CrossEntropyLoss()
        model1.init_weights()
        model1.to(device)
        optimizer = torch.optim.AdamW(model1.parameters(), lr=learning_rate)

        num_training_steps = epochs * int(np.ceil(len(train_data) / batch_size))
        num_warmup_steps = int(num_training_steps * warmup_percent)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        train_step = 0
        loss_list = []
        for epoch in range(epochs):
            for batch in data.get_batch(train_data, step=1, batch_size=batch_size):
                optimizer.zero_grad()
                train_step += 1
                x_v, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_len = batch
                x_a_v = [torch.from_numpy(x_v).to(device) for _ in range(2)]
                bert_input = [torch.from_numpy(x_bert_sent).to(device), torch.from_numpy(x_bert_sent_mask).to(device)]
                y_emotion = torch.from_numpy(y_emotion).argmax(dim=-1).to(device)
                y_cause = torch.from_numpy(y_cause).argmax(dim=-1).to(device)

                y_emotion_pred, y_cause_pred, reg = model1(x_a_v, bert_input, diag_len)
                loss = criterion(y_emotion_pred, y_emotion) + criterion(y_cause_pred, y_cause) + reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                train_loss += loss.detach().item()

                if train_step % report_freq == 0:
                    cur_loss_avg = train_loss / report_freq
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '.format(
                        epoch + 1, train_step, num_training_steps, scheduler.get_last_lr()[0],
                        elapsed * 1000 / report_freq))
                    train_loss = 0
                    start_time = time.time()
                    loss_list.append(cur_loss_avg)
        return loss_list

    def evaluate_step1(dataset: data.DataSet):
        model1.eval()
        print('Evaluating step1 ...')

