import os
import nltk
import gensim
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models


class TopicModel:
    def __init__(self):
        self.dictionary = None
        self.ldamodel = None

    def clean(self, doc):
        stop = set(stopwords.words('english'))
        stop.update(list(string.punctuation))
        lemma = WordNetLemmatizer()
        tokenized = word_tokenize(doc.lower())
        stop_free = " ".join([i for i in tokenized if i not in stop])
        normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
        return normalized

    def prepare_corpus(self, documents):
        doc_clean = [self.clean(doc).split() for doc in documents]
        self.dictionary = corpora.Dictionary(doc_clean)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in doc_clean]
        return doc_term_matrix

    def train_lda_model(self, doc_term_matrix, num_topics=5, passes=10000):
        self.ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=self.dictionary, passes=passes)

    def predict_topic(self, text):
        preprocessed_text = self.clean(text)
        bow = self.dictionary.doc2bow(preprocessed_text.split())
        topic_distribution = self.ldamodel.get_document_topics(bow)
        return topic_distribution

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def load_documents_from_csv(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select 'Review Text' column
    documents = df['Review Text'].dropna().tolist()
    
    # Split into training and testing sets (80% train, 20% test)
    train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)
    
    return train_docs, test_docs

if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('wordnet') 
    nltk.download('punkt')

    # Path to your CSV file
    csv_file_path = 'yelp_reviews.csv'  # Update the path to your CSV file

    # Load documents from CSV file
    train_docs, test_docs = load_documents_from_csv(csv_file_path)

    # Create an instance of TopicModel
    topic_model = TopicModel()

    # Prepare the corpus and train the LDA model with train_docs
    doc_term_matrix = topic_model.prepare_corpus(train_docs)
    topic_model.train_lda_model(doc_term_matrix)

    # Print the topics
    for idx, topic in topic_model.ldamodel.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # Evaluate model coherence
    coherence_model_lda = CoherenceModel(model=topic_model.ldamodel, texts=[doc.split() for doc in train_docs], dictionary=topic_model.dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Visualizing the topics and saving to an HTML file
    vis = pyLDAvis.gensim_models.prepare(topic_model.ldamodel, doc_term_matrix, topic_model.dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')

    # Predict topics for a sample document from test_docs
    sample_test_doc = test_docs[0]  # Take the first document from the test set
    topic_distribution = topic_model.predict_topic(sample_test_doc)
    print('New Document Topic Distribution:', topic_distribution)
