import time
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import numpy as np
import data
import model

epochs = 20  # The number of epochs
batch_size = 4  # The number of dialogues in a batch
hidden_dim = 200  # The dimension of hidden layer
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
    # Load the data
    emb_paths = ['data/video_id_mapping.npy', 'data/video_embedding_4096.npy', 'data/audio_embedding_6373.npy', ]
    train_data = data.DataSet(
        ['data/train.txt'] + emb_paths,
        bert_path=bert_path
    )
    dev_data = data.DataSet(
        ['data/dev.txt'] + emb_paths,
        bert_path=bert_path
    )
    test_data = data.DataSet(
        ['data/test.txt'] + emb_paths,
        bert_path=bert_path
    )
    model1 = model.MECPEStep1(
        hidden_dim, max_utt_num, max_sent_len,
        embeddings=[torch.from_numpy(train_data.audio_embedding).to(device),
                    torch.from_numpy(train_data.video_embedding).to(device)],
        choose_emo_cat=choose_emo_cat,
        bert_path=bert_path
    ).to(device)
    criterion = nn.NLLLoss()  # Since we've used log_softmax in the model, we use NLLLoss here.


    def evaluate_step1(dataset: data.DataSet):
        model1.eval()
        print('Evaluating step1 ...')
        with torch.no_grad():
            x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_len = dataset.x_video, dataset.x_bert_sent, \
                dataset.x_bert_sent_mask, dataset.y_emotion, dataset.y_cause, dataset.diag_len
            x_a_v = [torch.from_numpy(x_video).to(device) for _ in range(2)]
            bert_input = [torch.from_numpy(x_bert_sent).to(device), torch.from_numpy(x_bert_sent_mask).to(device)]
            y_emotion = torch.from_numpy(y_emotion).argmax(dim=-1).to(device)
            y_cause = torch.from_numpy(y_cause).argmax(dim=-1).to(device)

            y_emotion_pred, y_cause_pred, reg = model1(x_a_v, bert_input, diag_len)
            pred_dim = y_emotion_pred.shape[-1]
            loss = (criterion(y_emotion_pred.reshape(-1, pred_dim), y_emotion.reshape(-1))
                    + criterion(y_cause_pred.reshape(-1, pred_dim), y_cause.reshape(-1)) + reg)
            emo_p, emo_r, emo_f1 = data.cal_prf(y_emotion_pred, y_emotion, diag_len, 7 if choose_emo_cat else 2)
            cause_p, cause_r, cause_f1 = data.cal_prf(y_cause_pred, y_cause, diag_len, 2)
        print(
            'Loss: {:.4f}\n Emotion: P: {:.4f}\t R: {:.4f}\t F1: {:.4f}\n Cause: P: {:.4f}\t R: {:.4f}\t F1: {:.4f}'.format(
                loss, emo_p, emo_r, emo_f1, cause_p, cause_r, cause_f1))
        return loss, emo_p, emo_r, emo_f1, cause_p, cause_r, cause_f1


    def train_step1():
        print('Training step1 ...')
        start_time = time.time()

        train_loss = 0
        model1.init_weights()
        optimizer = torch.optim.AdamW(model1.parameters(), lr=learning_rate)

        num_training_steps = epochs * int(np.ceil(len(train_data) / batch_size))
        num_warmup_steps = int(num_training_steps * warmup_percent)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        train_step = 0
        loss_list = []
        emo_list = []
        cause_list = []
        for epoch in range(epochs):
            for batch in data.get_batch(train_data, step=1, batch_size=batch_size):
                model1.train()
                optimizer.zero_grad()
                train_step += 1
                x_v, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_len = batch
                x_a_v = [torch.from_numpy(x_v).to(device) for _ in range(2)]
                bert_input = [torch.from_numpy(x_bert_sent).to(device), torch.from_numpy(x_bert_sent_mask).to(device)]
                y_emotion = torch.from_numpy(y_emotion).argmax(dim=-1).to(device)
                y_cause = torch.from_numpy(y_cause).argmax(dim=-1).to(device)

                y_emotion_pred, y_cause_pred, reg = model1(x_a_v, bert_input, diag_len)
                pred_dim = y_emotion_pred.shape[-1]
                loss = (criterion(y_emotion_pred.reshape(-1, pred_dim), y_emotion.reshape(-1))
                        + criterion(y_cause_pred.reshape(-1, pred_dim), y_cause.reshape(-1)) + reg)
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
                    loss_list.append([train_step, cur_loss_avg])

            with torch.no_grad():
                model1.eval()
                x_v, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_len = dev_data.x_video, dev_data.x_bert_sent, \
                    dev_data.x_bert_sent_mask, dev_data.y_emotion, dev_data.y_cause, dev_data.diag_len
                x_a_v = [torch.from_numpy(x_v).to(device) for _ in range(2)]
                bert_input = [torch.from_numpy(x_bert_sent).to(device), torch.from_numpy(x_bert_sent_mask).to(device)]
                y_emotion = torch.from_numpy(y_emotion).argmax(dim=-1).to(device)
                y_cause = torch.from_numpy(y_cause).argmax(dim=-1).to(device)

                y_emotion_pred, y_cause_pred, reg = model1(x_a_v, bert_input, diag_len)
                emo_p, emo_r, emo_f1 = data.cal_prf(y_emotion_pred, y_emotion, diag_len, 7 if choose_emo_cat else 2)
                cause_p, cause_r, cause_f1 = data.cal_prf(y_cause_pred, y_cause, diag_len, 2)
                print(
                    'Index in Epoch {:d}\n Emotion: P: {:.4f}\t R: {:.4f}\t F1: {:.4f}\n Cause: P: {:.4f}\t R: {:.4f}\t F1: {:.4f}'.format(
                        epoch, emo_p, emo_r, emo_f1, cause_p, cause_r, cause_f1))
                emo_list.append([epoch, emo_p, emo_r, emo_f1])
                cause_list.append([epoch, cause_p, cause_r, cause_f1])
        return loss_list, emo_list, cause_list


    def write_data(dir_path, loss_list, emo_list, cause_list):
        with open(dir_path + '/log/loss.txt', 'w') as f:
            for i in loss_list:
                f.write('Batch: {:d}\t Loss: {:.4f}\n'.format(i[0], i[1]))
        with open(dir_path + '/log/emo.txt', 'w') as f:
            for i in emo_list:
                f.write('Epoch: {:d}\t P: {:.4f}\t R: {:.4f}\t F1: {:.4f}\n'.format(i[0], i[1], i[2], i[3]))
        with open(dir_path + '/log/cause.txt', 'w') as f:
            for i in cause_list:
                f.write('Epoch: {:d}\t P: {:.4f}\t R: {:.4f}\t F1: {:.4f}\n'.format(i[0], i[1], i[2], i[3]))
        with open(dir_path + '/save/model.pkl', 'wb') as f:
            torch.save(model1, f)
        start_time = time.time()
        print('Start to predict train set, dev set and test set ...')
        with (torch.no_grad()):
            def predict(dataset):
                x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_len = dataset.x_video, dataset.x_bert_sent, \
                    dataset.x_bert_sent_mask, dataset.y_emotion, dataset.y_cause, dataset.diag_len
                x_a_v = [torch.from_numpy(x_video).to(device) for _ in range(2)]
                bert_input = [torch.from_numpy(x_bert_sent).to(device), torch.from_numpy(x_bert_sent_mask).to(device)]

                y_emotion_pred, y_cause_pred, _ = model1(x_a_v, bert_input, diag_len)
                return y_emotion_pred, y_cause_pred

            # Predict the train set, dev set and test set
            y_emotion_pred_train, y_cause_pred_train = predict(train_data)
            y_cause_pred_train, y_cause_pred_train = \
                torch.argmax(y_cause_pred_train, dim=-1), torch.argmax(y_cause_pred_train, dim=-1)
            y_emotion_pred_dev, y_cause_pred_dev = predict(dev_data)
            y_emotion_pred_dev, y_cause_pred_dev = \
                torch.argmax(y_emotion_pred_dev, dim=-1), torch.argmax(y_cause_pred_dev, dim=-1)
            y_emotion_pred_test, y_cause_pred_test = predict(test_data)
            y_emotion_pred_test, y_cause_pred_test = \
                torch.argmax(y_emotion_pred_test, dim=-1), torch.argmax(y_cause_pred_test, dim=-1)

            print('Predicting finished. Time: {:.2f}s'.format(time.time() - start_time))
            print('Writing results ...')

            def put_data(dataset: data.DataSet, y_emotion_pred, y_cause_pred, dir_path, file_name):
                with open('{}/save/{}.txt'.format(dir_path, file_name), 'w') as file:
                    for i in range(len(dataset)):
                        file.write('{} {}'.format(dataset.diag_id[i], dataset.diag_len[i]))
                        for j in range(dataset.diag_len[i]):
                            file.write('{} | {} | {}\n'.format(j + 1, y_emotion_pred[i, j], y_cause_pred[i, j]))
            put_data(train_data, y_emotion_pred_train, y_cause_pred_train, dir_path, 'train')
            put_data(dev_data, y_emotion_pred_dev, y_cause_pred_dev, dir_path, 'dev')
            put_data(test_data, y_emotion_pred_test, y_cause_pred_test, dir_path, 'test')
        print('Writing finished.')


    loss_list, emo_list, cause_list = train_step1()
    write_data('step1', loss_list, emo_list, cause_list)
    evaluate_step1(dev_data)
