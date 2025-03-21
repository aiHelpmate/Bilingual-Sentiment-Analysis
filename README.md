#  Machine Translation and Sentiment Analysis

Hi, I am a freshman at XJTLU. This is my first attempt at NLP related learning. This repository contains a comprehensive implementation of a bilingual sentiment analysis and machine translation system, leveraging deep learning techniques, specifically Sequence-to-Sequence (Seq2Seq) models for machine translation and Long Short-Term Memory (LSTM) networks for sentiment analysis. The system is capable of translating text between English and Chinese and analyzing the sentiment of the translated text.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Machine Translation](#machine-translation)
  - [Sentiment Analysis](#sentiment-analysis)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project demonstrates my expertise in natural language processing (NLP) by combining machine translation and sentiment analysis into a single pipeline. The machine translation component uses a Seq2Seq model with attention mechanisms to translate text between English and Chinese. The sentiment analysis component uses an LSTM model to classify the sentiment of the translated text into one of five categories: joy, fear, anger, sadness, or neutral.

## Features

- **Machine Translation**: Translates text between English and Chinese using a Seq2Seq model with attention mechanisms.
- **Sentiment Analysis**: Classifies the sentiment of the translated text using an LSTM model.
- **Bilingual Support**: Handles both English and Chinese text for translation and sentiment analysis.
- **Deep Learning**: Utilizes TensorFlow and Keras for building and training deep learning models.
- **Preprocessing**: Includes text cleaning, tokenization, and sequence padding for both translation and sentiment analysis tasks.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aiHelpmate/Bilingual-Sentiment-Analysis.git
   cd Bilingual-Sentiment-Analysis
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Place the `data_train.csv` and `data_test.csv` files in the `data/` directory.

## Usage

### Machine Translation

To translate a sentence from Chinese to English:

```python
text = "对不起。"
translation_result = translate(text)
print("Translated Text:", translation_result)
```

### Sentiment Analysis

To analyze the sentiment of the translated text:

```python
emotion = analyze_sentiment(translation_result)
print("Emotion:", emotion)
```

## Methodology

### Machine Translation

The machine translation model is based on a Seq2Seq architecture with attention mechanisms. The model is trained on a bilingual corpus to translate text between English and Chinese. The key components of the model include:

- **Encoder**: A GRU-based encoder that processes the input sequence and generates a context vector.
- **Decoder**: A GRU-based decoder that generates the output sequence using the context vector and attention mechanisms.
- **Attention Mechanism**: Helps the model focus on relevant parts of the input sequence during translation.

### Sentiment Analysis

The sentiment analysis model uses an LSTM network to classify the sentiment of the translated text. The model is trained on a labeled dataset with five sentiment categories. The key components of the model include:

- **Embedding Layer**: Converts input text into dense vectors.
- **LSTM Layers**: Capture sequential information in the text.
- **Dropout Layers**: Prevent overfitting by randomly dropping units during training.
- **Dense Layer**: Outputs the probability distribution over the sentiment categories.

## Results

The machine translation model achieved high accuracy in English-Chinese text translation, while only 68% accuracy was achieved in the sentiment analysis model, mainly because it is difficult to find suitable content for the five sentiment analysis data. When only conducting positive and negative sentiment analysis, the accuracy can reach around 90%.

### Example Output

```plaintext
Input: 对不起。
Translated Text: i m sorry .
Emotion: fear
```

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.
