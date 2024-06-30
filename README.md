# DNN based Speech Recognition and Sentiment Analysis Application (in production)

## Overview
This application utilizes Deep Neural Networks (DNNs) for speech recognition and sentiment analysis. It integrates models for converting speech to text and performing sentiment analysis on movie reviews.

## Features
- **Speech Recognition:**
  - Utilizes DNNs (CNN and bi directional LSTM) to transcribe speech into text.
  - Based on fine-tuning models using LibriSpeech dataset.
  
- **Sentiment Analysis:**
  - Applies ANN models to analyze sentiment in movie reviews.
  - Uses pre-trained models on IMDB movie review data for sentiment classification.

## Requirements
- Python 3.11.5
- PyTorch
- NumPy
- Scikit-learn
- Other dependencies listed in requirements.txt (conda dependencies listed).

## Setup Instructions
1. **Clone Repository:**
   ```bash
   git clone https://github.com/mainakm7/DNN_SpeechRecognition_with_SentimentAnalysis.git
   cd repository-name

2. **Install dependencies**
   Conda packages are currently added to requirements.txt.

This project is licensed under the MIT License.
