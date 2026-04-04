# Captionomaly: End-to-End Anomaly Detection and Captioning

This is my 6th-semester Computer Vision project for my B.Tech in Computer Science. The goal of this project is to take raw CCTV surveillance footage, automatically detect if a crime or anomaly is happening, isolate the exact timestamp, and then generate a natural English sentence describing what is going on.

This codebase is a modernized, from-scratch PyTorch reimplementation of older TensorFlow 1.x architectures (specifically based on the Sultani anomaly detection paper and the UCFC-VD captioning decoder).

## How It Works

The architecture is split into two main pipelines that run sequentially during inference:

### Pipeline 1: Anomaly Detection

- **Feature Extraction:** Slices the video into 32 segments and passes them through a C3D network to extract spatial-temporal features.
- **MIL Classifier:** Uses a Multiple Instance Learning (MIL) ranking framework to score each segment between 0.0 and 1.0.
- **Cropping:** If a segment crosses the 0.5 threshold, the script identifies the exact start and end timestamps of the crime and crops the video to that specific window.

### Pipeline 2: Anomaly Captioning

- **Spatial/Semantic Extraction:** Passes the cropped anomaly frames through a pretrained ResNeXt-101 model. We extract both the 2048-D visual features (from the avgpool layer) and the 1000-D semantic features (from the classification head).
- **Tagging Network:** An MLP that maps the 1000-D semantic features to a 300-D probability distribution over specific crime keywords.
- **VNS-GRU Decoder:** A custom Visual-Net-Semantic Gated Recurrent Unit. It uses a 64-D bilinear bottleneck to fuse the visual features, the semantic tags, and fixed GloVe embeddings to generate the final English caption word by word.

## Setup and Installation

1. Clone the repo and install the dependencies:
   ```bash
   pip install -r requirements.txt
   Download the GloVe embeddings (glove.840B.300d.txt) and place it in the data/ folder.
   ```

Place your raw .mp4 videos in data/Train/ and the reference caption CSVs in data/Captions/.

Running Inference
You can run the full end-to-end pipeline on a single video using the main script.

Bash

```
python -m src.main --video data/Train/Burglary/Burglary001_x264A.mp4 --device cuda
```

The script will output the duration, the detection threshold scores, the exact time segment the anomaly was found, and the generated caption.

the model weights are uploaded on google drive - https://drive.google.com/drive/folders/1W1DnKulq3IIWqB47sjqSEmbCnEzrpl4K?usp=sharing
