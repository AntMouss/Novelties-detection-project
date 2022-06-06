import copy
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
import scipy.sparse as ss
from corextopic import corextopic as ct
from data_utils import DocumentsWordsCounter, LabelsWordsCounter


class Engine:

    def __init__(self, texts: list = None, n_topics : int = 5 , random_state : int = 42):

        self.random_state = random_state
        self.texts = texts
        self.n_topics = n_topics
        self.core = None

    def get_topic_terms(self,topic_id= 0, topn=100):
        pass

class SupervisedEngine(Engine):
    def __init__(self , labels: list = None  , **kwargs):
        super(SupervisedEngine, self).__init__(**kwargs)
        self.labels = labels

class GuidedEngine(Engine):
    def __init__(self , seed : dict = None  , **kwargs):
        super(GuidedEngine, self).__init__(**kwargs)
        self.seed = seed


class CoreX(Engine):

    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        self.document_words_counter = DocumentsWordsCounter(self.texts)
        self.words = [ word for word in self.document_words_counter.columns]
        self.documents_matrix = self.document_words_counter.to_numpy(dype = bool)
        self.documents_matrix = ss.csc_matrix(self.documents_matrix)
        self.core = ct.Corex(n_hidden=self.n_topics, words=self.words, max_iter=200, verbose=False, seed=1)

    def get_topic_terms(self,topic_id= 0, topn=100):

        return self.core.get_topics(n_words=topn , topic=topic_id)


class SupervisedCoreX(SupervisedEngine , CoreX ):
    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        self.core.fit(self.documents_matrix , y=self.labels)


class GuidedCoreX(GuidedEngine,CoreX):
    def __init__(self , anchor_strength = 3 , **kwargs ):
        super(GuidedCoreX, self).__init__(**kwargs)
        self.anchors = [words for lab , words in self.seed.items()]
        self.core.fit(self.documents_matrix , anchors=self.anchors , anchor_strength=anchor_strength)



class LFIDF(SupervisedEngine):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.n_docs = len(self.texts)
        self.labels_words_counter = LabelsWordsCounter(self.texts , self.labels)
        self.labels_idx = {label : i for i , label in enumerate(self.labels_words_counter.index)}
        self.idx_words = {i : word for i , word in enumerate(self.labels_words_counter.columns)}
        self.lfidf_matrix = self.process_lfidf_matrix()


    def process_lfidf_matrix(self):
        lfidf_matrix = copy.deepcopy(self.labels_words_counter).to_numpy()
        for i in range(lfidf_matrix.shape[0]):
            for j in range(lfidf_matrix.shape[1]):
                lf = (lfidf_matrix[i][j]/np.sum(lfidf_matrix[i]))
                idf = np.log(self.n_docs/np.sum(lfidf_matrix[j]))
                lfidf_matrix[i][j] = lf*idf

    def get_topic_terms(self,topic= None, topn=100):
        idx = self.labels_idx[topic]
        label_serie = self.lfidf_matrix[idx]
        words_idx = reversed(np.argsort(label_serie))[:topn]
        #[(word , score) , ...]
        return [(self.idx_words[word_id] , label_serie[word_id]) for word_id in words_idx]



class LDA(Engine):

    def __init__(self, dictionnary: corpora.Dictionary = None,
                 random_state=42,**kwargs):

        super().__init__(**kwargs)
        self.random_state = random_state
        self.dictionnary = dictionnary
        corpus_bow = [self.dictionnary.doc2bow(text) for text in self.texts]
        self.ldaargs = {
            "corpus" : corpus_bow,
            "num_topics" : self.n_topics,
            "id2word" : self.dictionnary ,
            "random_state" : self.random_state
        }
        self.core  = LdaModel(**self.ldaargs)


    def get_topic_terms(self , **kwargs):

        res = self.core.get_topic_terms(**kwargs)
        return [(self.core.id2word(word_id) , score) for word_id , score in res]


class GuidedLDA(GuidedEngine , LDA):


    def __init__(self , table : dict = None , **kwargs):

        super().__init__(**kwargs)
        self.table = table
        self.ldaargs['eta'] = self.generate_eta(self.seed , self.dictionnary , self.table)
        self.core = LdaModel(**self.ldaargs)

    @staticmethod
    def generate_eta(seed , dictionnary , table, normalize_eta=False, overratte=1e7):

        ntopics = len(seed)

        eta = np.full(shape=(ntopics, len(dictionnary)),
                      fill_value=1)  # create a (ntopics, nterms) matrix and fill with 1
        for topic in seed.keys():
            topic_id = table[topic]
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
        # we can remove this line for other test
        return eta