# Topic Modelling with LDA

This repository contains a Python implementation of Latent Dirichlet Allocation (LDA) for topic modeling using the NLTK and Gensim libraries. The project aims to analyze text documents and identify prevalent topics using a statistical model.

## Features

- **Text Preprocessing:** Removes stopwords and punctuation, tokenizes, and lemmatizes the text.
- **Corpus Preparation:** Converts a collection of text documents into a format that can be used by the LDA model.
- **LDA Model Training:** Trains an LDA model to discover topics in a corpus.
- **Topic Prediction:** Predicts topic distribution for new documents.
- **Model Evaluation:** Evaluates the model using coherence scores.
- **Topic Visualization:** Visualizes the topics using `pyLDAvis`.

## Prerequisites

Before running this project, you will need the following:

- Python 3.x
- NLTK
- Gensim
- pyLDAvis

You can install the required packages using the following command:

```bash
pip install nltk gensim pyLDAvis
```

## Usage

To use this project:

1. Ensure all dependencies are installed.
2. Download necessary NLTK data:

```bash
python -m nltk.downloader stopwords wordnet punkt
```

3. Run the `main.py` script:

```bash
python main.py
```

This will process the predefined documents, train the LDA model, and save the topic visualization as an HTML file.

## Visualization

After running the script, you can open the `lda_visualization.html` file generated in the root directory to view the topic distribution visualization.

## Contributing

Contributions to this project are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is open-source and available under the MIT License.
