from typing import List
import numpy as np
from gensim import corpora
from gensim.models import LdaModel , CoherenceModel
import scipy.sparse as ss
from corextopic import corextopic as ct
from novelties_detection.Experience.data_utils import DocumentsWordsCounter, LabelsWordsCounter


class Engine:
    """
    engine model we will use two get topic in corpus data
    topic is a list of revelant words .
    We will use different core method like CoreX , Latent dirichlet allocation and TFIDF
    and different training way : no supervised , supervised and Guided (semi-supervised)
    """

    def __init__(self, texts: List[List], nb_topics : int = 5, dictionnary: corpora.Dictionary = None,  random_state : int = 42):
        """

        @param texts: texts ,data
        @param nb_topics: number of relevant topics
        @param random_state:
        """
        self.dictionnary = dictionnary
        self.random_state = random_state
        self.texts = texts
        self.nb_topics = nb_topics
        self.core = None

    @property
    def coherence(self):
        return None

    def __len__(self):
        return self.nb_topics

    def get_topic_terms(self,topic_id : int, topn=100):
        pass

class SupervisedEngine(Engine):
    def __init__(self , labels: List , labels_idx: List  , **kwargs):
        """

        @param labels: labels of text data
        @param labels_idx: list of labels (the idx is the list idx of the label)
        @param kwargs:
        """
        super(SupervisedEngine, self).__init__(**kwargs)
        self.labels_idx = labels_idx
        self.labels = labels


class GuidedEngine(Engine):
    def __init__(self , seed : dict  , **kwargs):
        """

        @param seed: dictionnary of list seed words with label as key (this words are specific of the kay label)
        @param kwargs:
        """
        super(GuidedEngine, self).__init__(**kwargs)
        self.seed = seed


class CoreX(Engine):

    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        self.document_words_counter = DocumentsWordsCounter(self.texts)
        self.words = [ word for word in self.document_words_counter.columns]
        self.documents_matrix = self.document_words_counter.to_numpy(dtype = bool)
        self.documents_matrix = ss.csc_matrix(self.documents_matrix)
        self.core = ct.Corex(n_hidden=self.nb_topics, max_iter=200, verbose=False, seed=1)
        self.core.fit(self.documents_matrix, words=self.words)

    @property
    def coherence(self):
        return self.core.tcs

    def get_topic_terms(self,topic_id : int, topn=100):
        res = self.core.get_topics(n_words=topn , topic=topic_id)
        return {word[0] : word[1] for word in res}

# not usable
class SupervisedCoreX(SupervisedEngine , CoreX ):
    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        self.core.fit(self.documents_matrix , y=self.labels)


class GuidedCoreX(GuidedEngine,CoreX):
    def __init__(self , anchor_strength = 3 , **kwargs ):
        super(GuidedCoreX, self).__init__(**kwargs)
        self.anchors = [[word for word in words if word in self.words] for _ , words in self.seed.items()]
        self.core.fit(self.documents_matrix , words=self.words, anchors=self.anchors , anchor_strength=anchor_strength)



class LFIDF(SupervisedEngine):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.n_docs = len(self.texts)
        self.labels_words_counter = LabelsWordsCounter(self.texts , self.labels)
        self.labels_words_counter = (self.labels_words_counter.reindex(self.labels_idx)).fillna(0)
        self.idx_words = {i : word for i , word in enumerate(self.labels_words_counter.columns)}
        self.lfidf_matrix = self.process_lfidf_matrix()


    def process_lfidf_matrix(self):

        lfidf_matrix = self.labels_words_counter.to_numpy()
        shadow = np.zeros_like(lfidf_matrix)
        for i in range(lfidf_matrix.shape[0]):
            if np.sum(lfidf_matrix[i]) == 0:
                #return random value when the topic is inexistant in the texts input
                shadow[i] = np.random.random(lfidf_matrix.shape[1])/1000
                continue
            for j in range(lfidf_matrix.shape[1]):
                lf = (lfidf_matrix[i][j]/np.sum(lfidf_matrix[i]))
                idf = np.log(self.n_docs/np.sum(lfidf_matrix[:,j]))
                shadow[i][j] = lf*idf
        return shadow


    def get_topic_terms(self, topic_id : int, topn=100):

        topic = self.labels_idx[topic_id]
        label_serie = self.lfidf_matrix[topic_id]
        words_idx = np.flipud((np.argsort(label_serie)[-topn:]))
        #[(word , score) , ...]
        return {self.idx_words[word_id] : label_serie[word_id] for word_id in words_idx}



class LDA(Engine):

    def __init__(self, random_state=42, passes : int = 1 , **kwargs):
        """

        @param dictionnary: specific gensim dictionnary linked to the data corpus
        @param random_state:
        @param passes: number of time that we passe the corpus during training (like epoch)
        @param kwargs:
        """
        super().__init__(**kwargs)
        self.passes = passes
        self.random_state = random_state
        self.corpus_bow = [self.dictionnary.doc2bow(text) for text in self.texts]
        self.ldaargs = {
            "corpus" : self.corpus_bow,
            "num_topics" : self.nb_topics,
            "id2word" : self.dictionnary ,
            "random_state" : self.random_state,
            "passes" : self.passes
        }
        self.core  = LdaModel(**self.ldaargs)

    @property
    def coherence(self):
        coherence_model = CoherenceModel(model=self.core, topics=self.nb_topics,
                                         corpus=self.corpus_bow, dictionary=self.dictionnary)
        return coherence_model.get_coherence()



    def get_topic_terms(self ,topic_id : int , topn = 100 , **kwargs):

        res = self.core.get_topic_terms(topicid=topic_id , topn = topn, **kwargs)
        return {self.core.id2word[word_id] : score for word_id , score in res}


class GuidedLDA(GuidedEngine , LDA):


    def __init__(self, overrate, **kwargs):

        super().__init__(**kwargs)
        self.ldaargs['eta'] = self.generate_eta(self.seed, self.dictionnary, overratte=overrate)
        self.core = LdaModel(**self.ldaargs)

    @staticmethod
    def generate_eta(seed , dictionnary, normalize_eta=False, overratte=1e7):
        """
        for LDA , eta is the matrix of weigth parameters with labels as row and words as column
        @param seed:
        @param dictionnary:
        @param normalize_eta:
        @param overratte: overratte eta weigth
        @return:
        """
        ntopics = len(seed)

        eta = np.full(shape=(ntopics, len(dictionnary)),
                      fill_value=1)  # create a (ntopics, nterms) matrix and fill with 1
        for topic_id, topic in enumerate(seed.keys()):
            for word in seed[topic]:
                keyindex = [index for index, term in dictionnary.items() if
                            term == word]  # look up the word in the dictionary
                if (len(keyindex) > 0):  # if it's in the dictionary
                    try:
                        eta[topic_id, keyindex[0]] = overratte  # put a large number in there
                    except Exception as e:
                        print(e)
                        print(topic)
                        print(keyindex[0])
        if normalize_eta:
            eta = np.divide(eta, eta.sum(axis=0))  # normalize so that the probabilities sum to 1 over all topics
        # we can remove this line for other tmp_test_obj
        return eta