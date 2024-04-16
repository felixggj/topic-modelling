import os
import nltk
import gensim
import string
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 


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

    def train_lda_model(self, doc_term_matrix, num_topics=3, passes=10000):
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

if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('wordnet') 
    nltk.download('punkt')

    # Load documents from .txt files for training
    training_directory = 'train/'  # Update the path to your training files directory
    documents = load_documents(training_directory)

    # Create an instance of TopicModel
    topic_model = TopicModel()

    # Prepare the corpus and train the LDA model
    doc_term_matrix = topic_model.prepare_corpus(documents)
    topic_model.train_lda_model(doc_term_matrix)

    # Print the topics
    for idx, topic in topic_model.ldamodel.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # Evaluate model coherence
    coherence_model_lda = CoherenceModel(model=topic_model.ldamodel, texts=[doc.split() for doc in documents], dictionary=topic_model.dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Visualizing the topics and saving to an HTML file
    vis = pyLDAvis.gensim_models.prepare(topic_model.ldamodel, doc_term_matrix, topic_model.dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')

    # Load and predict topics for a new document from a .txt file
    test_document_path = 'test/american_salaries.txt'  # Update the path to your test document
    with open(test_document_path, 'r', encoding='utf-8') as file:
        new_doc = file.read()
    topic_distribution = topic_model.predict_topic(new_doc)
    print('New Document Topic Distribution:', topic_distribution)