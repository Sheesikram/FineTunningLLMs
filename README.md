# Fine-Tuning GPT-2 and LLaMA BERT Models for Text Summarization

This project demonstrates the process of fine-tuning GPT-2 and LLaMA BERT models for the task of text summarization. The goal is to use pre-trained language models and adapt them to produce coherent and concise summaries for different types of input texts.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Training the Models](#training-the-models)
- [Evaluation](#evaluation)
- [Fine-Tuning Steps](#fine-tuning-steps)
- [Results](#results)
- [Licensing](#licensing)

## Overview

Text summarization is a critical task in natural language processing that involves condensing long documents into shorter summaries while retaining key information. This project utilizes two state-of-the-art models—GPT-2 and LLaMA BERT—fine-tuned on a summarization dataset to achieve better performance in this task.

### Models Used
- **GPT-2**: A transformer-based model trained for natural language generation tasks, fine-tuned for abstractive summarization.
- **LLaMA BERT**: A variant of BERT (Bidirectional Encoder Representations from Transformers) fine-tuned for extractive summarization.

## Requirements

Before you can use this repository, you must install the necessary dependencies.

### Required Libraries
- Python 3.7+
- PyTorch 1.8.1 or higher
- Hugging Face Transformers Library
- TensorFlow (for LLaMA BERT if required)
- Datasets Library (Hugging Face)
- tqdm (for progress bar)

To install the requirements, run:

```bash
pip install -r requirements.txt
