import nltk
import gensim
import string
import pandas as pd  # Import pandas for handling CSV files
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

    def train_lda_model(self, doc_term_matrix, num_topics=3, passes=50):
        self.ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=self.dictionary, passes=passes)

    def predict_topic(self, text):
        preprocessed_text = self.clean(text)
        bow = self.dictionary.doc2bow(preprocessed_text.split())
        topic_distribution = self.ldamodel.get_document_topics(bow)
        return topic_distribution

if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('wordnet') 
    nltk.download('punkt')

    # Load documents from a CSV file
    df = pd.read_csv('path/to/yourfile.csv')  # Update the path to your CSV file
    documents = df['column_name'].tolist()  # Replace 'column_name' with the name of your column that contains the text

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

    # Predict topics for a new document
    new_doc = "such as automotive vehicles and medical devices are indeed regulated in this way. But it is less clear what forms the regulations should take and who should implement them. See, for example, proposals in the EU compared to the USA on the amount of restrictions that should be placed on AI creators [3, 4]. Moreover, there is the key question of who should create and enforce the regulations? Should the regulations be public or private [5]? That is, should they be created by government bodies, or by industrial organisations and accrediting bodies such as the IEEE? And once they have been created, who will enforce them? Enforcement could involve actions such as auditing source code, training runs, training data, and monitoring for consumer complaints. This could be done by government sector organisations, but it could also be done by third-party auditors [6]. To help governments choose the most suitable kind of regulatory framework, we need to be able to predict the effects of different regulatory systems. Most of the discourse around this at the moment is qualitative and does not lead to formal predictions [7–10]. This limits the ability of governments, technology companies, and citizens to foresee what the effects of different regulatory systems might be [11]. There is a small amount of literature on AI race modelling [12–14], including from the perspective of (evolutionary) game theory [15–18], but this has not considered how different regulatory mechanisms influence both user trust and the compliance of companies with AI safety considerations. To address this, we propose that evolutionary game theory [19] can be used to formally model the effects of different regulatory systems in terms of their incentives on tech companies, end users, and regulators. We present a framework for formalising the strategic interaction between users, AI system creators, and regulators as a game (Fig. 1). This game captures three key decisions that these actors face: 1. Users: do they trust and hence use an AI system or not? 2. AI creators: do they follow a safety optimal development path in compliance with regulations, or do they pursue a competitive development path that violates regulations in a race to the market? 3. Regulators: do they invest in monitoring AI system creators and enforcing the regulations effectively, or do they cut corners to save the costs of doing this? 3 This highlights the dilemmas facing users, creators, and regulators. Users can benefit from using an AI system, but also run the risk that the system may not act in their best interest, i.e. may not be trustworthy [20]. This follows from the fact that AI creators are themselves in competition with each other, as highlighted by the current “AI race” to develop artificial general intelligence [8, 12, 21–23]. Consequently, we cannot assume that creators will always act in the best interests of their users by complying with regulations and developing systems worthy of user trust. Finally, regulators themselves may be self-interested. This may occur when governments delegate the enforcement of regulations to other actors, such as private audit firms [6, 9, 24]. This kind of delegation may reduce government costs, but it also introduces a principal agent problem [25]: regulators are themselves agents with their own profit maximising goals. To analyse the model, we use the methods of evolutionary game theory. Evolutionary game theory is based on the idea that agents can learn behaviours that benefit them from social learning, i.e. by copying the behaviour of other agents in their population that are doing better than themselves. This avoids the need to assume that the agents are fully rational and have complete information. In our model, we consider three populations corresponding to the three actors: users, creators, and regulators. Our analysis demonstrates how incentives for regulators are important. Governments desire that all creators produce trustworthy AI systems, and all users trust these systems. Such a state cannot be reached if regulators that do their job properly cannot be distinguished from regulators that cut corners. This holds regardless of the severity of punishment for defecting creators. We consider two possible institutional solutions to this problem. First, we show that if governments can provide rewards to regulators that do a good job, and use of the AI system is not too risky for users, then some level of trustworthy development and trust by users occurs. We then consider an alternative solution, where users may condition their trust decision on the effectiveness of the regulators, for example, where information about the past performance of the regulators is available. This leads to effective regulation, and consequently the development of trustworthy AI and user trust, provided that the cost of implementing regulations is not too high."
    topic_distribution = topic_model.predict_topic(new_doc)
    print(topic_distribution)
