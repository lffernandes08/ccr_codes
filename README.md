# Boosting Transformers: Recognizing Textual Entailment for Classification of Vaccine News Coverage

This repository contains the codes used for the fine-tuning of BERT models as tested in the paper:  
**"Boosting Transformers: Recognizing Textual Entailment for Classification of Vaccine News Coverage"**

## Fine-tuning Approaches

Two fine-tuning approaches were implemented:

- **fine_tuning_bert**: This approach fine-tunes pre-trained models derived from BERT, using only the class labels.
  
- **fine_tuning_bert_rte**: This approach fine-tunes pre-trained models derived from BERT specifically for the Recognizing Textual Entailment (RTE) task.

The fine-tuning protocol follows the methods and models proposed by [Laurer and colleagues (2023)](https://doi.org/10.1017/pan.2023.20).

## Dataset

The dataset used for fine-tuning consists of 1,000 vaccine-related headlines published by newspapers in the United States, the United Kingdom, and Brazil. The dataset is provided in this repository for reproducibility and further research.
