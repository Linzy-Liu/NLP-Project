import numpy as np
video_id_mapping = np.load('data/video_id_mapping.npy', allow_pickle=True)


def get_text_and_label(path):
    with open(path, 'r', encoding='utf-8') as f:
        utterance = []
        diag_utt_mapping = {}
        diag_label = {}
        line = f.readline()
        while line:
            tmp = line.split(' ')
            diag_num = int(tmp[0])
            utt_sum = int(tmp[1])
            diag_utt_mapping[diag_num] = (len(utterance), len(utterance) + utt_sum - 1)
            labels = f.readline()
            if labels == '\n':
                diag_label[diag_num] = {
                    'has_emotion': [],
                    'is_cause': [],
                    'pair': []
                }
            else:
                labels = labels.replace('(', '').replace(')', '').replace('\n', '').split(',')
                tuple_string = [(int(labels[i]), int(labels[i + 1])) for i in range(0, len(labels), 2)]
                diag_label[diag_num] = {
                    'has_emotion': list(set([i[0] for i in tuple_string])),
                    'is_cause': list(set([i[1] for i in tuple_string])),
                    'pair': tuple_string
                }
            emotion = []
            for i in range(utt_sum):
                line = f.readline()
                parts = line.split(' | ')
                emotion.append(parts[2])
                utterance.append(parts[3])
            diag_label[diag_num]['emotion'] = emotion
            line = f.readline()
    return utterance, diag_utt_mapping, diag_label

def get_vocab(path):

