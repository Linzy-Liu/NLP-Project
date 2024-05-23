import numpy as np
import time
from transformers import BertTokenizer

bert_path = 'F:/python_work/GitHub/bert-base-uncased'
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
    v_emb = normalize(np.load(video_emb_file, allow_pickle=True).astype(np.float32))
    a_emb = normalize(np.load(audio_emb_file, allow_pickle=True).astype(np.float32))
    return v_id_map, v_emb, a_emb


def get_glove_embedding(ebd_path, all_text_path, pos_embedding_dim):
    """
    Get the glove embedding from the given path.
    :param pos_embedding_dim: The dimension of position embedding.
    :param all_text_path: The path which contains all vocabulary.
    :param ebd_path: The path of glove embedding.
    :return: The glove embedding, position embedding, word to index dict and index to word dict.
    """
    with open(ebd_path, 'r', encoding='utf-8') as f:
        vocab = {}
        line = f.readline().split(' ')
        vocab_num = int(line[0])
        vocab_dim = int(line[1])
        for i in range(vocab_num):
            line = f.readline().split(' ')
            vocab[line[0]] = np.array(line[1:], dtype=np.float32)
    with open(all_text_path, 'r', encoding='utf-8') as f:
        words = []
        while True:
            line = f.readline()
            if line == '':
                break

            line = line.strip()
            utt_num = int(line.split(' ')[1])
            f.readline()  # skip labels
            for i in range(utt_num):
                line = f.readline().strip().split(' | ')
                words += [line[2]]
                words += line[3].split(' ')
    words = list(set(words))
    word2idx = {}
    idx2word = {}
    word_ebd = [np.zeros(vocab_dim, dtype=np.float64)]
    for i, word in enumerate(words):
        word2idx[word] = i
        idx2word[i] = word
        if word in vocab:
            word_ebd.append(vocab[word])
        else:
            word_ebd.append(np.random.rand(vocab_dim) / 5. - 0.1)
    word_ebd = np.array(word_ebd, dtype=np.float32)
    pos_ebd = ([np.zeros(pos_embedding_dim, dtype=np.float32)] +
               [np.random.normal(0, 0.1, pos_embedding_dim) for _ in range(200)])
    pos_ebd = np.array(pos_ebd, dtype=np.float32)
    return word_ebd, pos_ebd, word2idx, idx2word


def load_data_step1(paths,
                    max_utt_num: int = 35,
                    max_sent_len: int = 35,
                    bert_path=bert_path,
                    choose_emo_cat: bool = False,
                    do_lower_case: bool = True
                    ):
    """
    Load the data from the given paths: [word_path, video_id_mapping_filename, video_emb_filename, audio_emb_filename]
    :param max_sent_len: The max length of sentence
    :param max_utt_num: The max number of utterances
    :param paths: strings of paths: [word_path, video_id_mapping_filename, video_emb_filename, audio_emb_filename]
    :param bert_path: The path of bert model
    :param choose_emo_cat: Whether to choose the emotion category
    :param do_lower_case: The setting of bert model
    :return: The dataset for model of step 1.
    """
    start_time = time.time()
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
            line = f.readline()
            if line == '':
                break

            line = line.strip().split(' ')
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
                np.zeros((max_utt_num, max_sent_len), dtype=np.int32) for _ in range(2)]
            x_video_tmp = np.zeros(max_utt_num, dtype=np.int32)

            # tokenize utterances
            for i in range(d_len):
                line = f.readline().strip().split(' | ')
                x_video_tmp[i] = v_id_map['dia{}utt{}'.format(d_id, i + 1)]

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

    x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_id, diag_len = map(
        np.array, [x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_id, diag_len]
    )
    print('Finish loading data from {} etc with time {:.2f}s'.format(paths[0], time.time() - start_time))
    return x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, y_pairs, diag_id, diag_len, v_emb, a_emb


def load_data_step2(paths,
                    word2idx: dict,
                    max_utt_num: int = 35,
                    max_sent_len: int = 35,
                    real_time: bool = True,
                    choose_emo_cat: bool = False,
                    task_type: str = ''
                    ):
    """
    Load the data from the given paths: [set_path, pred_path, video_id_mapping_filename, video_emb_filename, audio_emb_filename]
    :param word2idx:
    :param paths:
    :param max_utt_num:
    :param max_sent_len:
    :param real_time:
    :param choose_emo_cat:
    :param task_type:
    :return:
    """
    start_time = time.time()
    print('Loading data from {} ...'.format(paths[0]))

    set_path, pred_path, video_id_mapping_file, video_emb_file, audio_emb_file = paths
    v_id_map, v_emb, a_emb = get_video_and_audio(video_id_mapping_file, video_emb_file, audio_emb_file)

    y, x, x_video, diag_id, diag_len, sent_len = [[] for _ in range(6)]
    x_pair, y_pair = [], []
    distance, pairs = [], []
    emo_list, cause_list = [], []
    if choose_emo_cat:
        x_emo_cat = []
        pred_emo_cat = []

    with open(pred_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line == '':
                break

            line = line.strip().split(' ')
            d_id, d_len = int(line[0]), int(line[1])
            emo_list_tmp, cause_list_tmp = [], []
            if choose_emo_cat:
                pred_emo_cat_tmp = np.zeros(max_utt_num)
            for i in range(d_len):
                line = f.readline().strip().split(' | ')
                emo_tmp, cause_tmp = int(line[1]), int(line[2])
                if choose_emo_cat:
                    pred_emo_cat_tmp[i] = emo_tmp
                if emo_tmp > 0:
                    emo_list_tmp.append(i + 1)
                if cause_tmp > 0:
                    cause_list_tmp.append(i + 1)
            emo_list.append(emo_list_tmp)
            cause_list.append(cause_list_tmp)
            if choose_emo_cat:
                pred_emo_cat.append(pred_emo_cat_tmp)

    with open(set_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line == '':
                break

            line = line.strip().split(' ')
            d_id, d_len = int(line[0]), int(line[1])
            diag_id.append(d_id)
            diag_len.append(d_len)

            x_tmp = np.zeros((max_utt_num, max_sent_len), dtype=np.int32)
            x_video_tmp, sent_len_tmp = [np.zeros(max_utt_num, dtype=np.int32) for _ in range(2)]

            line = f.readline().strip()
            pairs_tmp = eval('[' + line + ']')
            pairs_tmp = [list(i) for i in pairs_tmp]
            if real_time:
                for p in pairs_tmp:
                    if p[1] - p[0] > 0:
                        pairs_tmp.remove(p)
            pairs.append(pairs_tmp)

            true_cause_list_tmp = list(set([pairs_tmp[i][1] for i in range(len(pairs_tmp))]))
            true_emo_list_tmp = list(set([pairs_tmp[i][0] for i in range(len(pairs_tmp))]))
            x_emo_cat_tmp = np.zeros(max_utt_num)
            for i in range(d_len):
                line = f.readline().strip().split(' | ')
                utt = line[3].split(' ')
                if len(utt) > max_sent_len:
                    utt = utt[:max_sent_len]
                sent_len_tmp[i] = len(utt)
                for j in range(len(utt)):
                    x_tmp[i, j] = word2idx[utt[j]]
                x_video_tmp[i] = v_id_map['dia{}utt{}'.format(d_id, i + 1)]
                if choose_emo_cat:
                    x_emo_cat_tmp[i] = emotion_idx[line[2]]
                else:
                    x_emo_cat_tmp[i] = 1 if emotion_idx[line[2]] > 0 else 0

            if task_type == 'EC':
                emo_list[d_id - 1] = true_emo_list_tmp
            elif task_type == 'CE':
                cause_list[d_id - 1] = true_cause_list_tmp

            for p in pairs_tmp:
                y_pair.append([d_id, p[0], p[1], x_emo_cat_tmp[p[0] - 1]])
            for i in emo_list[d_id - 1]:
                for j in cause_list[d_id - 1]:
                    valid = False
                    if real_time and i >= j:
                        valid = True
                    elif not real_time:
                        valid = True

                    if valid:
                        if choose_emo_cat:
                            x_pair.append([d_id, i, j, x_emo_cat_tmp[i - 1]] if task_type == 'EC'
                                          else [d_id, i, j, pred_emo_cat[d_id - 1][i - 1]])
                            x_emo_cat.append(
                                x_emo_cat_tmp[i - 1] if task_type == 'EC' else pred_emo_cat[d_id - 1][i - 1])
                        else:
                            x_pair.append([d_id, i, j, x_emo_cat_tmp[i - 1]])
                        distance.append(j - i + 100)
                        x.append([x_tmp[i - 1], x_tmp[j - 1]])
                        x_video.append([x_video_tmp[i - 1], x_video_tmp[j - 1]])
                        sent_len.append([sent_len_tmp[i - 1], sent_len_tmp[j - 1]])
                        y.append([0, 1] if x_pair[-1] in y_pair else [1, 0])

    x, x_video, y, diag_id, diag_len, sent_len, distance = map(np.array,
                                                               [x, x_video, y, diag_id, diag_len, sent_len, distance])
    print('Finish loading data from {} etc with time {:.2f}s'.format(paths[0], time.time() - start_time))
    if choose_emo_cat:
        x_emo_cat = np.array(x_emo_cat)
        pred_emo_cat = np.array(pred_emo_cat)
        return x, x_video, y, diag_id, diag_len, sent_len, distance, x_pair, y_pair, x_emo_cat, pred_emo_cat
    else:
        return x, x_video, y, diag_id, diag_len, sent_len, distance, x_pair, y_pair


class DataSet(object):
    def __init__(self, paths,
                 var_list=None,
                 max_utt_num=35, max_sent_len=35,
                 bert_path=bert_path,
                 choose_emo_cat=False,
                 do_lower_case=True,
                 real_time=True,
                 task_type='',
                 step=1
                 ):
        # paths:
        # step1: [word_path, video_id_mapping_filename, video_emb_filename, audio_emb_filename]
        # step2: [set_path, pred_path, video_id_mapping_filename, video_emb_filename, audio_emb_filename]
        #
        # var_list:
        # step2: [word2idx, word_ebd, pos_ebd]

        self.choose_emo_cat = choose_emo_cat

        if step == 1:
            self.x_video, self.x_bert_sent, self.x_bert_sent_mask, \
                self.y_emotion, self.y_cause, self.y_pairs, \
                self.diag_id, self.diag_len, \
                self.video_embedding, self.audio_embedding = \
                load_data_step1(paths, max_utt_num, max_sent_len,
                                bert_path, choose_emo_cat,
                                do_lower_case, real_time)
        elif step == 2:
            v_id_mapping, self.video_embedding, self.audio_embedding = get_video_and_audio(*paths[1:])
            self.word_embedding, self.pos_embedding, word2idx = var_list
            if choose_emo_cat:
                self.x, self.x_video, self.y, self.diag_id, self.diag_len, self.sent_len, \
                    self.distance, self.x_pair, self.y_pair, self.x_emo_cat, self.pred_emo_cat = \
                    load_data_step2(paths, word2idx, max_utt_num, max_sent_len, real_time, choose_emo_cat, task_type)
            else:
                self.x, self.x_video, self.y, self.diag_id, self.diag_len, self.sent_len, \
                    self.distance, self.x_pair, self.y_pair = \
                    load_data_step2(paths, word2idx, max_utt_num, max_sent_len, real_time, choose_emo_cat, task_type)

    def __len__(self):
        return len(self.x_video)


def get_batch(dataset: DataSet, step=1, batch_size=8, choose_emo_cat: bool = False):
    batch_num = int(np.ceil(len(dataset) / batch_size))
    if step == 1:
        x_video, x_bert_sent, x_bert_sent_mask, y_emotion, y_cause, diag_len = \
            dataset.x_video, dataset.x_bert_sent, dataset.x_bert_sent_mask, dataset.y_emotion, dataset.y_cause, dataset.diag_len
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(x_video))
            yield x_video[start:end], x_bert_sent[start:end], x_bert_sent_mask[start:end], \
                y_emotion[start:end], y_cause[start:end], diag_len[start:end]
    elif step == 2:
        if choose_emo_cat:
            x, x_video, y, sent_len, distance, x_emo_cat = \
                dataset.x, dataset.x_video, dataset.y, dataset.sent_len, dataset.distance, dataset.x_emo_cat
            for i in range(batch_num):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(x))
                yield x[start:end], x_video[start:end], y[start:end], sent_len[start:end], \
                    distance[start:end], x_emo_cat[start:end]
        else:
            x, x_video, y, sent_len, distance = \
                dataset.x, dataset.x_video, dataset.y, dataset.sent_len, dataset.distance
            for i in range(batch_num):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(x))
                yield x[start:end], x_video[start:end], y[start:end], sent_len[start:end], distance[start:end]
    else:
        ValueError('The step must be 1 or 2.')


def cal_prf(pred_y, y, diag_len, num_class):
    """
    Calculate the precision, recall and f1 score.
    :param diag_len: The length of each dialogue.
    :param pred_y: The prediction of model.
    :param y: The ground truth.
    :param num_class: The number of classes.
    :return: The precision, recall and f1 score.
    """

    tp = np.zeros(num_class)
    fp = np.zeros(num_class)
    fn = np.zeros(num_class)

    for i in range(len(diag_len)):
        for j in range(diag_len[i]):
            if pred_y[i, j] == y[i, j]:
                tp[pred_y[i, j]] += 1
            else:
                fp[pred_y[i, j]] += 1
                fn[y[i, j]] += 1
    if num_class > 2:
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        precision = tp[1] / (tp[1] + fp[0] + 1e-8)
        recall = tp[1] / (tp[1] + fn[0] + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def cal_prf_pairs():
    pass
