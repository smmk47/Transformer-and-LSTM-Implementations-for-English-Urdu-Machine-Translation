# Transformer and LSTM Implementations for English-Urdu Machine Translation

This repository presents the implementation of **Transformer** and **Long Short-Term Memory (LSTM)** models for **machine translation** from **English** to **Urdu** using the **UMC005 Parallel Corpus**. The project includes the complete workflow from data preprocessing, model architecture design, hyperparameter tuning, training, and evaluation using **BLEU** and **ROUGE** metrics. The comparative analysis highlights performance differences in translation accuracy, training time, memory usage, and inference speed between the two models.


## Introduction
Machine Translation (MT) is a key task in Natural Language Processing (NLP), enabling automated translation between different languages. This project compares two prominent neural machine translation models, **Transformer** and **LSTM**, for English-to-Urdu translation. The **UMC005 Parallel Corpus**, containing parallel English and Urdu sentences, serves as the dataset for training and evaluation. The goal of this project is to evaluate the translation accuracy and efficiency of both models.

## Dataset
The **UMC005 Parallel Corpus** is used for this project, providing pairs of sentences in English and Urdu. This dataset is preprocessed to:
- Tokenize the sentences.
- Remove unnecessary characters.
- Pad sequences for uniform input size.

## Model Architectures
### Transformer Model
The Transformer model is based on the self-attention mechanism, allowing the model to capture global dependencies between words in a sentence. The key components of the Transformer model used in this project include:
- **Encoder-Decoder Architecture**: The encoder processes the input sentence, and the decoder generates the translated output.
- **Multi-Head Attention**: Enables the model to focus on different parts of the sentence simultaneously.
- **Positional Encoding**: Helps the model understand the order of words in the sentence.

### LSTM Model
The LSTM model is a type of Recurrent Neural Network (RNN) designed to handle long-range dependencies in sequences. The architecture includes:
- **Bidirectional LSTM**: Processes input sequences in both forward and reverse directions.
- **Attention Mechanism**: Enhances translation quality by focusing on relevant parts of the sentence during decoding.

## Training Details
- **Epochs**: 20
- **Batch Size**: 64
- **Learning Rate**: 0.001 (Adam Optimizer)
- **Loss Functions**: Cross-entropy loss for both models.
- **Metrics**: BLEU score and ROUGE score to evaluate translation quality.

## Evaluation
The models were evaluated using the following metrics:
- **BLEU (Bilingual Evaluation Understudy)**: Measures the precision of n-grams in the translated text.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Evaluates recall by comparing n-grams in the machine-generated text with reference translations.

## Results
- **Translation Accuracy**: The Transformer model achieved a higher BLEU score, indicating better translation quality.
- **Training Time**: The LSTM model required longer training times due to its sequential nature.
- **Memory Usage**: The Transformer model consumed more memory due to the larger number of parameters and attention mechanism.
- **Inference Speed**: The Transformer model exhibited faster inference due to parallelized processing.
