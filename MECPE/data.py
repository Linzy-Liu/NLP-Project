import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer

emotion_idx = dict(zip(['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], range(7)))


def get_video_and_audio(video_id_mapping_file, video_emb_file, audio_emb_file):
    def normalize(x):
        x1 = x[1:, :]
        min_x = np.min(x1, axis=0, keepdims=True)
        max_x = np.max(x1, axis=0, keepdims=True)
        x1 = (x1 - min_x) / (max_x - min_x + 1e-8)
        x[1:, :] = x1
        return x

    v_id_map = eval(str(np.load(video_id_mapping_file, allow_pickle=True)))
    v_emb = normalize(np.load(video_emb_file, allow_pickle=True))
    a_emb = normalize(np.load(audio_emb_file, allow_pickle=True))
    return v_id_map, v_emb, a_emb


def get_glove_embedding(path, words=None):
    with open(path, 'r', encoding='utf-8') as f:
        vocab = {}
        line = f.readline().split(' ')
        vocab_num = int(line[0])
        vocab_dim = int(line[1])
        for i in range(vocab_num):
            line = f.readline().split(' ')
            vocab[line[0]] = np.array(line[1:], dtype=np.float64)
    if words is not None:
        for word in words:
            if word not in vocab:
                vocab[word] = list(np.random.rand(vocab_dim) / 5. - 0.1)
    return vocab


def load_data_step1(paths,
                    max_utt_num: int = 35,
                    max_sent_len: int = 35,
                    bert_path='F:/python_work/GitHub/bert-base-uncased',
                    choose_emo_cat: bool = False,
                    do_lower_case: bool = True):
    """
    Load the data from the given paths: [word_path, video_id_mapping_filename, video_emb_filename, audio_emb_filename]
    :param max_sent_len: The max length of sentence
    :param max_utt_num: The max number of utterances
    :param paths: strings of paths: [word_path, video_id_mapping_filename, video_emb_filename, audio_emb_filename]
    :param bert_path: The path of bert model
    :param choose_emo_cat: Whether to choose the emotion category
    :param do_lower_case: The setting of bert model
    :return: The dataset for step1
    """
    print('Loading data from {} ...'.format(paths[0]))

    word_path, video_id_mapping_file, video_emb_file, audio_emb_file = paths
    v_id_map, v_emb, a_emb = get_video_and_audio(video_id_mapping_file, video_emb_file, audio_emb_file)
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=do_lower_case)

    diag_id, diag_len = [], []
    x_bert_sent, x_bert_sent_mask = [], []
    x_video = []
    y_emotion, y_cause, y_pairs = [], [], []

    with open(word_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline().strip().split(' ')
            if line == '':
                break

            d_id, d_len = int(line[0]), int(line[1])
            diag_id.append(d_id)
            diag_len.append(d_len)

            # store labels
            y_pairs_tmp = eval('[' + f.readline().strip() + ']')
            y_pairs_tmp = [list(i) for i in y_pairs_tmp]
            y_emotion_tmp = np.zeros((max_utt_num, 7)) if choose_emo_cat else np.zeros((max_utt_num, 2))
            y_cause_tmp = np.zeros((max_utt_num, 2))

            has_emo_tmp = [y_pairs_tmp[i][0] for i in range(len(y_pairs_tmp))]
            is_cause_tmp = [y_pairs_tmp[i][1] for i in range(len(y_pairs_tmp))]
            for i in range(d_len):
                if not choose_emo_cat:
                    y_emotion_tmp[i, :] = np.array([0, 1]) if i + 1 in has_emo_tmp else np.array([1, 0])
                y_cause_tmp[i, :] = np.array([0, 1]) if i + 1 in is_cause_tmp else np.array([1, 0])

            # Initialize the shape
            x_bert_sent_tmp, x_bert_sent_mask_tmp = [
                np.zeros((max_utt_num, max_sent_len), dtype=np.int64) for _ in range(2)]
            x_video_tmp = np.zeros(max_utt_num, dtype=np.float64)

            # tokenize utterances
            for i in range(d_len):
                line = f.readline().strip().split(' | ')
                x_video_tmp[i] = v_id_map['dia{}_utt{}'.format(d_id, i + 1)]

                tmp_token = tokenizer.tokenize(line[3])
                if len(tmp_token) > max_sent_len - 2:  # Cut the sentence to the max length; -2 for [CLS] and [SEP]
                    tmp_token = tmp_token[:max_sent_len - 2]
                tmp_token = ['[CLS]'] + tmp_token + ['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(tmp_token)
                x_bert_sent_tmp[i, :len(input_ids)] = input_ids
                x_bert_sent_mask_tmp[i, :len(input_ids)] = 1

                if choose_emo_cat:
                    y_emotion_tmp[i, :] = np.array([1 if i == emotion_idx[line[2]] else 0 for i in range(7)])

            # store data
            x_bert_sent.append(x_bert_sent_tmp)
            x_bert_sent_mask.append(x_bert_sent_mask_tmp)
            x_video.append(x_video_tmp)
            y_emotion.append(y_emotion_tmp)
            y_cause.append(y_cause_tmp)
            y_pairs.append(y_pairs_tmp)

    x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, y_pairs, diag_id, diag_len = map(
        np.array, [x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, y_pairs, diag_id, diag_len]
    )
    print('Finish loading data from {} ...'.format(paths[0]))
    return x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, y_pairs, diag_id, diag_len, v_emb, a_emb


class DataSet(object):
    def __init__(self, word_data_path, video_id_mapping_file, video_emb_file, audio_emb_file,
                 max_utt_num=35, max_sent_len=35,
                 bert_path='F:/python_work/GitHub/bert-base-uncased',
                 choose_emo_cat=False,
                 do_lower_case=True):
        self.choose_emo_cat = choose_emo_cat
        paths = [word_data_path, video_id_mapping_file, video_emb_file, audio_emb_file]

        self.x_video, self.x_bert_sent, self.x_bert_sent_mask, \
            self.y_emotion, self.y_cause, self.y_pairs, \
            self.diag_id, self.diag_len, \
            self.v_emb, self.a_emb = \
            load_data_step1(paths, max_utt_num, max_sent_len,
                            bert_path, choose_emo_cat,
                            do_lower_case)


def get_batch(dataset: DataSet, step=1, batch_size=8):
    if step == 1:
        x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_len = \
            dataset.x_video, dataset.x_bert_sent, dataset.x_bert_sent_mask, dataset.y_emotion, dataset.y_cause, dataset.diag_len
        batch_num = int(np.ceil(len(x_video) / batch_size))
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(x_video))
            yield x_video[start:end], x_bert_sent[start:end], x_bert_sent_mask[start:end], \
                y_emotion[start:end], y_cause[start:end], diag_len[start:end]
    elif step == 2:
        # wait to be implemented
        pass
    else:
        ValueError('The step must be 1 or 2.')
