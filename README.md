# Clickbait Spoiler Classification and Generation

This repository contains the code and resources for the project **"Clickbait Spoiler Classification and Generation,"** which aims to classify spoiler types and generate spoiler texts for clickbait articles. The project builds on the tasks presented at SemEval 2023 and leverages state-of-the-art transformer models such as BERT, RoBERTa, and BART.

## Project Overview

Clickbait articles are designed to capture user attention with sensational headlines, often lacking substantial information. This project addresses the challenge of **spoiler classification** and **spoiler generation** by identifying the type of spoiler (phrase, passage, or multi-line) and generating spoiler text that fulfills the curiosity sparked by clickbait posts. The tasks are treated as classification and text retrieval problems using pre-trained language models fine-tuned on custom datasets.

### Key Features

- **Spoiler Type Classification:** Classifies spoilers into different types: phrase, passage, or multi-line.
- **Spoiler Generation:** Extracts or generates spoiler texts from clickbait articles.
- **Utilizes Pre-trained Transformer Models:** Fine-tunes models such as BERT, RoBERTa, and BART for enhanced performance.
- **Evaluation Metrics:** Uses F1-score, ROUGE, and METEOR metrics to evaluate model performance.

## Methodology

### Spoiler Classification

- **Model Used:** BERT (Bidirectional Encoder Representations from Transformers) model is fine-tuned for classifying spoiler types.
- **Approach:** Adds a fully connected layer on top of the transformer output to categorize spoiler types based on the textual input.

### Spoiler Generation

- **Models Used:** RoBERTa and BART models fine-tuned using the `AutoModelForQuestionAnswering` class from HuggingFace.
- **Approach:** Treats the spoiler generation task as a text extraction problem similar to extractive question-answering, where models predict the start and end positions of the spoiler in the text.

### Combined Approach

- A combined model is trained for both spoiler classification and generation, leveraging shared representations to improve overall performance. The loss function is a weighted sum of the classification and generation losses.

## Dataset

- The dataset consists of 3200 training, 400 validation, and 400 test samples, each containing clickbait headlines and their corresponding article content.
- The dataset includes three types of spoilers: phrase, passage, and multi-line, which are used to train and validate the models.

## Experiments

- **Training Parameters:** Models were fine-tuned over 10 epochs using the AdamW optimizer with a learning rate of 1e-5 and a batch size of 16.
- **Evaluation:** The performance of each model was evaluated using F1-score for classification and METEOR for generation.

## Results

- **Spoiler Classification:** The RoBERTa model fine-tuned on the SQuaD dataset achieved the highest F1-score of 0.666.
- **Spoiler Generation:** The RoBERTa model also demonstrated the best performance for spoiler generation, achieving a METEOR score of 0.3892.

## Installation

To run this project, clone the repository and install the required dependencies using the following commands:

## Usage

1. Prepare the dataset in the required format.
2. Train the model by running the provided training scripts.
3. Use the trained model to classify and generate spoilers for new clickbait articles.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions for improvements or find any bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to Professor Olga Vechtomova and TA Gaurav Sahu for their guidance and support throughout this project.
