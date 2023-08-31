# Reproduction of "Multimodal Emotion-Cause Pair Extraction"

## Introduction
The code in this repository is a reproduction of the paper *"Multimodal Emotion-Cause Pair Extraction in Conversations"* by [Xia et al.](https://arxiv.org/abs/2110.08020) whose code has been written in Tensorflow. The code in this repository is written in PyTorch.

According to the paper, the MECPE task here will use the following features with Real-time Setting:
* **Text Embedding**: BERT
* **Audio Embedding**: openSMILE
* **Video Embedding**: 3D-CNN(C3D)
* **CSKB**: COMET-ATOMIC_2020, xReact+oReact
* **Model fusion method**: Add

## Dataset

The code in this repository is based on the dataset [ECF](https://github.com/NUSTM/MECPE/tree/main/data) provided by the authors of the paper.
Given that the dataset from the link above has been processed, the code here is not designed to process the raw data with videos and audios.
In addition, the dataset is not included in this repository. Please download it from the link above and put it in the `data` folder.
The dataset contains the following files:
* `all_data_pair.txt`, `train.txt`, `dev.txt`, `test.txt`
* `audio_embedding_6373.npy`
* `video_embedding_4096.npy`
* `video_id_mapping.npy`

## Requirements

* Python 3.6+ (Tested on 3.9)
* PyTorch 2.0.1
