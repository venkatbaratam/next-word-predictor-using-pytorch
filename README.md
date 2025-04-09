# LSTM-based Next Word Predictor

## Overview
This project implements a Next Word Prediction system using Long Short-Term Memory (LSTM) networks in PyTorch. The model is trained to predict the next word in a sequence based on previous words, demonstrating natural language processing capabilities and text generation.

## Features
- Custom text tokenization using NLTK
- Vocabulary building from corpus
- Text-to-numerical conversion pipeline
- Custom PyTorch Dataset and DataLoader implementation for efficient batch processing
- LSTM neural network architecture for sequence modeling
- Interactive prediction functionality that can generate text continuations

## Model Architecture
- **Embedding Layer**: Converts token indices to dense vectors (dimension: 100)
- **LSTM Layer**: Processes the sequence of embeddings (hidden size: 150)
- **Fully Connected Layer**: Maps LSTM output to vocabulary size for classification

## Performance
- The model achieves good accuracy in predicting the next word in sequences from the training data
- After 50 epochs of training, the model demonstrates the ability to generate contextually relevant continuations

## Setup and Usage
1. **Install Dependencies**: pip install torch numpy nltk
2. **Prepare Dataset**:
- The model is trained on a Q&A document about a Data Science Mentorship Program
- You can substitute this with your own text corpus by modifying the `document` variable

3. **Training**:
- Run the script to train the model on your dataset
- Default parameters: learning rate = 0.001, epochs = 50, embedding dimension = 100, hidden size = 150

4. **Prediction**:
- Use the `prediction()` function to generate the next word for a given input text
- Example: `prediction(model, vocab, "The course follows a monthly")`
- For generating multiple words in sequence, use the iterative prediction approach shown in the code

## Future Improvements
- Implement more sophisticated text preprocessing techniques
- Explore bidirectional LSTM and attention mechanisms
- Add temperature parameter for controlling randomness in predictions
- Implement beam search for better text generation
- Train on larger and more diverse datasets for improved generalization

## About the Author
I'm a B.Tech student in Artificial Intelligence & Data Science at IIITDM with experience in Machine Learning and NLP. This project demonstrates my skills in implementing neural network architectures for natural language processing tasks.
