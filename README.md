# Captionomaly: End-to-End Anomaly Detection and Captioning

This is my 6th-semester Computer Vision project for my B.Tech in Computer Science. The goal of this project is to take raw CCTV surveillance footage, automatically detect if a crime or anomaly is happening, isolate the exact timestamp, and then generate a natural English sentence describing what is going on.

## 🚀 Project Evolution & Incremental Upgrades

This project was built iteratively, continuously upgrading components to achieve state-of-the-art results:

1. **Baseline Implementation:** 
Initially, the project was a from-scratch PyTorch reimplementation of older TensorFlow 1.x architectures (the Sultani anomaly detection paper and the UCFC-VD captioning decoder). It utilized C3D (4096-D) video features paired with a VNS-GRU text decoder.

2. **I3D Feature Backbone:**
We replaced the legacy C3D feature extractor with **I3D (Inflated 3D ConvNet)**. I3D provides vastly superior spatial-temporal feature representations (1024-D) compared to C3D, leading to more accurate anomaly detection and significantly reduced noise.

3. **Temporal Graph Convolutional Network (T-GCN):**
We added a T-GCN layer to the Multiple Instance Learning (MIL) neural network. Traditional MIL often suffers from "temporal flickering," where anomaly scores jump erratically between adjacent frames. The T-GCN solves this by explicitly modeling the temporal dependencies between adjacent video segments, resulting in much smoother, stabler, and more reliable detection boundaries.

4. **Unsupervised Predictive Transformer (Experimental / Scrapped):**
We attempted to replace the supervised MIL classifier entirely with an unsupervised Predictive Transformer approach (which learns to reconstruct normal frames and flags high-reconstruction-error frames as anomalies). However, this approach did not achieve competitive results against our supervised MIL + TGCN model, so it was ultimately scrapped to maintain high accuracy.

5. **Pipeline 2 Upgrade (Captioning Transformer):**
Finally, we replaced the legacy VNS-GRU text decoder with a modern, from-scratch **Transformer Decoder** for the captioning phase. The Transformer leverages multi-head causal self-attention to generate much more fluent and context-aware English sentences describing the detected crimes.

---

## ⚙️ How It Works Today

The architecture is split into two main pipelines that run sequentially during inference:

### Phase 1: Anomaly Detection
- **Feature Extraction:** Slices the video into 32 temporal segments and passes them through an I3D network to extract 1024-D spatial-temporal features.
- **MIL + T-GCN Classifier:** Uses a Multiple Instance Learning ranking framework combined with Temporal Graph Convolutions to score each segment between 0.0 and 1.0. 
- **Windowing:** If a segment crosses the threshold, the script identifies the exact start and end timestamps of the crime and isolates that specific context window.

### Phase 2: Anomaly Captioning
- **Caption Transformer:** The anomalous I3D features from the isolated window are passed into our custom PyTorch Transformer Decoder. The model autoregressively generates a descriptive English sentence token by token, translating the visual crime into text.

---

## 🛠️ Setup and Installation

1. Clone the repo and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your raw `.mp4` videos in `data/Test/` 

*(Note: The model weights are hosted externally due to size limitations. You can download them [here on Google Drive](https://drive.google.com/drive/folders/1W1DnKulq3IIWqB47sjqSEmbCnEzrpl4K?usp=sharing). Place `.pth` files in `data/weights/`)*

## 🏃 Running Inference

You can run the full end-to-end pipeline (Anomaly Detection -> Transformer Captioning) on the testing folder using the main inference script:

```bash
python run_test_pipeline.py
```

The script will automatically auto-extract I3D features, run the MIL detection, evaluate threshold scores, locate the anomaly, and print the generated Transformer caption to the console.
