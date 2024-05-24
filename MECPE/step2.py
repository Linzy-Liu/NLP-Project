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
    train_data = data.DataSet(
        ['data/train.txt', 'step1/save/train.txt'] + emb_paths,
        bert_path=bert_path,
        step=2
    )
    dev_data = data.DataSet(
        ['data/dev.txt', 'step1/save/dev.txt'] + emb_paths,
        bert_path=bert_path,
        step=2
    )
    test_data = data.DataSet(
        ['data/test.txt', 'step1/save/test.txt'] + emb_paths,
        bert_path=bert_path,
        step=2
    )
    model2 = model.MECPEStep2(
        hidden_dim=hidden_dim,
        embeddings=[torch.from_numpy(train_data.audio_embedding).to(device),
                    torch.from_numpy(train_data.video_embedding).to(device),
                    torch.from_numpy(train_data.word_embedding).to(device),
                    torch.from_numpy(train_data.pos_embedding).to(device)],
        choose_emo_cat=choose_emo_cat
    ).to(device)
    if os.path.exists('step1/save/model.pt'):
        model1.load_state_dict(torch.load('step1/save/model.pt'))
        print('Model loaded.')
    criterion = nn.NLLLoss()  # Since we've used log_softmax in the model, we use NLLLoss here.

    # Training
    def evaluate_step2(dataset: data.DataSet):
        model2.eval()
        print('Evaluating Step2...')
        with torch.no_grad():
            pass

    def train_step2():
        pass
