

# VIDEO-SUMMARIZATION-WITH-TRANSFORMER-MODELS

## Driven Video Summarization with Transformer Model

This project aims to achieve a highly advanced video summarizing system to create a meaningful narrative based on visual content. It is designed to build an efficient automated tool that can analyze, summarize, and retrieve video information effectively.

## Preface

Welcome to VIDEO-SUMMARIZATION-WITH-TRANSFORMER-MODELS! In this repository, we introduce the "Driven Video Summarization with Transformer Model" project. This project focuses on developing a sophisticated system to summarize video content effectively using BLIP-2 for precise visual scene descriptions. It converts videos into frames, identifies keyframes through sampling techniques, and generates contextually aligned summaries. Fine-tuning BLIP-2 ensures coherent and accurate content representation, supports standard formats (MP4, AVI, MOV), and leverages GPU acceleration for efficiency.

A more personalized summarization model, trained on SAMSUM, further refines the output of BLIP-2 by processing frame-level descriptions and consolidating these into comprehensive summaries that enhance text understandability. This technique can improve the quality of video content analysis while reducing the workload for industries in media, overcoming traditional summarization techniques that often miss important details.

## Model Summary

| Model Name                      | Model Link                                                            | Dataset Link                                                  |
|---------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------|
| CSM (Custom Summarization Model) | [CSM Model](https://huggingface.co/Chandans01/custom-chandan-samsum) | [SAMSUM Dataset](https://huggingface.co/datasets/Samsung/samsum/tree/main) |

## Overview

### Abstract

While vision-and-language models have grown exponentially in capabilities, the end-to-end training cost of large-scale models has become extremely prohibitive. This project takes advantage of BLIP-2, a highly efficient pre-training strategy that evades the need for end-to-end training by bootstrapping from pre-trained, off-the-shelf image encoders and large language models. This paper fills the modality gap through a lightweight Querying Transformer, pre-trained in two stages: first, to learn vision-language representation from a frozen image encoder, and second, to facilitate vision-to-language generative learning using a frozen language model. BLIP-2 demonstrates state-of-the-art performance with significantly fewer parameters and outperforms larger models such as Flamingo-80B in tasks like zero-shot VQAv2.

The approach ensures that BLIP-2 produces initial visual descriptions of content for video frames, which are then refined by a custom summarization model to obtain a well-structured summary that captures the essential elements of the video. The solutions implemented address challenges such as large-scale video processing, memory constraints, and the readability of text output. Techniques include memory-efficient processing, model fine-tuning, and post-processing of the resultant text for high-quality output. This system is scalable and adaptive, leveraging advanced data handling to transform video data into a usable format through iterative training methods.

**Index Terms:** Video summarization, Transformer models, BLIP-2, Content description, Video data processing, Model fine-tuning, Memory optimization, Scalable solution, Data management, Iterative training.

## Model Architecture

![image](https://github.com/user-attachments/assets/06bbaf36-02cb-41b2-9796-0174ea02e2f3)

## Video Summarization Evaluation

Video summarization models can be evaluated both qualitatively and quantitatively. Some of the metrics used for quantification include BLEU, ROUGE, CIDEr, METEOR, and even precision, recall, and F1 scores that gauge overlap and correctness between the reference summaries and the ones generated by the model. Such metrics provide a benchmark for performance evaluation.

![image](https://github.com/user-attachments/assets/bcbbf3ae-a9b8-41e9-8b07-3adfdd3064f7)

## Usage

This section provides guidance on how to run, evaluate, and deploy the models.

### Setup

All operations are running under the environment of Python 3.9. If you are not using Python 3.9, you can create a virtual environment with:

```bash
conda create -n video-summarization python=3.9
```

Then run the setup script:

```bash
git clone https://github.com/chandanschandu/VIDEO-SUMMARIZATION-WITH-TRANSFORMER-MODELS.git
cd VIDEO-SUMMARIZATION-WITH-TRANSFORMER-MODELS
```

### Model Preparation

Download the model checkpoints from Hugging Face: CSM.

### Load Models

To load the models, create a script named `load_model.py` and use the following code:

```python
import os

# Change directory to where your preprocessing script is located
%cd /kaggle/input/videoprocessor

# Install required packages
!pip install -r /kaggle/input/videoprocessor/requirements.txt

# Run the preprocessing script
!python load_model.py
```

You also need to run your summarization model to get summaries:

```bash
!python https://github.com/chandanschandu/VIDEO-SUMMARIZATION-WITH-TRANSFORMER-MODELS/blob/main/summarization%20model.py
```

This code will load both the custom summarization model and the preprocessing model, allowing you to process videos for summarization.

## Results and Snapshots

Here are some results obtained from the model, including a snapshot of the video summarization output:

![Snapshot of Video Summarization Output](<URL_TO_YOUR_IMAGE>)  <!-- Replace with the actual URL of your image -->

[Watch the Summarization Video](<URL_TO_YOUR_VIDEO>)  <!-- Replace with the actual URL of your video -->
```

### Notes:
- Remember to replace `<URL_TO_YOUR_IMAGE>` and `<URL_TO_YOUR_VIDEO>` with the actual URLs for your snapshot image and video.
- Make sure that all code blocks and scripts mentioned are present in your repository to facilitate users following the instructions smoothly.
