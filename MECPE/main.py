import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data

if __name__ == '__main__':
    # Get text data and label
    # And diag_label is a dict with key as diag_num and value as a dict with key as 'has_emotion', 'is_cause', 'pair',
    # 'emotion'.
    utterance, diag_utt_mapping, diag_label = data.get_text_and_label('data/all_data_pair.txt')
    video_id_mapping = np.load('data/video_id_mapping.npy', allow_pickle=True)
    audio = np.load('data/audio_embedding_6373.npy')
    video = np.load('data/video_embedding_4096.npy')

