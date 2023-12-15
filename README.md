# Cross-Lingual NLI Model

## Overview
This repository contains the implementation of a Cross-Lingual Natural Language Inference (NLI) model. Our model is trained on the MultiNLI dataset and validated and tested using the XNLI dataset. We leverage MUSE (Multilingual Unsupervised and Supervised Embeddings) and FASTTEXT for effective cross-lingual word embeddings alignment.

## Features
- **MultiNLI Training**: Utilizes the Multi-Genre Natural Language Inference (MultiNLI) corpus for training, offering a diverse range of text genres and styles.
- **XNLI Validation and Testing**: Employs the Cross-Lingual NLI (XNLI) corpus for robust validation and testing across multiple languages.
- **Embeddings Alignment**: Incorporates MUSE and FASTTEXT for aligning word embeddings across languages, enhancing the model's cross-lingual capabilities.

## Prerequisites
- Python 3.x
- PyTorch
- FastText
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```
   git clone [repository-url]
   ```
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset Preparation
- Download and preprocess the MultiNLI and XNLI datasets. 
- Instructions for preprocessing and format requirements are provided in the `data_preparation` directory.

## Training the Model
To train the model, run the following command:
```
python train.py --config=config/train_config.json
```
Adjust the training configurations in `train_config.json` as needed.

## Evaluating the Model
Evaluate the trained model on the XNLI test set using:
```
python evaluate.py --model_checkpoint=path/to/model_checkpoint
```
