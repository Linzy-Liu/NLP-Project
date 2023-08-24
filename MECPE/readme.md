# Reproduction of "Multimodal Emotion-Cause Pair Extraction"

## Introduction
The code in this repository is a reproduction of the paper *"Multimodal Emotion-Cause Pair Extraction in Conversations"* by [Xia et al.](https://arxiv.org/abs/2110.08020) whose code has been written in Tensorflow. The code in this repository is written in PyTorch.

According to the paper, the MECPE task here will use the following features with Real-time Setting:
* **Text Embedding**: BERT
* **Audio Embedding**: openSMILE
* **Video Embedding**: 3D-CNN(C3D) But probably not implemented
* **CSKB**: COMET-ATOMIC_2020, xReact+oReact
* **Model fusion method**: Add

And the code in this repository is based on the dataset [ECF](https://github.com/NUSTM/MECPE/tree/main/data) provided by the authors of the paper.

## Requirements

* Python 3.6+ (Tested on 3.9)
* PyTorch 2.0.1
