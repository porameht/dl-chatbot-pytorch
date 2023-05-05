# Chatbot

A simple chatbot implementation using PyTorch, NLTK, and JSON.

## Overview

This project contains a chatbot implementation that uses a feedforward neural network with two hidden layers and a ReLU activation function. The model is trained on a dataset of intents and their corresponding responses, which are stored in a JSON file.

The main components of the project are:

1. Data processing and model training.
2. Helper functions for tokenizing, stemming, and creating bag of words representations.
3. Chat loop for user interaction.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- NLTK

### Installation

1. Clone the repository:
```
git clone https://github.com/porameht/dl-chatbot-pytorch.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```

### Usage

1. Train the model:
```
python train.py
```
2. Interact with the chatbot:
```
python chat.py
```

## Project Structure

- `train.py`: Processes the data, trains the model, and saves the trained model to a file.
- `nltk_utils.py`: Contains helper functions for tokenizing, stemming, and creating bag of words representations.
- `model.py`: Defines the `NeuralNet` class, which is a simple feedforward neural network.
- `chat.py`: Loads the trained model and allows users to interact with the chatbot.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



