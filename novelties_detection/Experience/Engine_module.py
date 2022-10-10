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

    def _train_core(self):
        pass

class SupervisedEngine(Engine):
    def __init__(self ,texts: List[List],labels: List , labels_idx: List, nb_topics : int = 5, dictionnary: corpora.Dictionary = None,  random_state : int = 42  , **kwargs):
        """

        @param labels: labels of text data
        @param labels_idx: list of labels (the idx is the list idx of the label)
        @param kwargs:
        """
        super(SupervisedEngine, self).__init__(texts , nb_topics , dictionnary , random_state)
        self.labels_idx = labels_idx
        self.labels = labels


class GuidedEngine(Engine):

    def __init__(self , seed : dict ,texts: List[List], nb_topics : int = 5, dictionnary: corpora.Dictionary = None,  random_state : int = 42  , **kwargs):
            """

            @param seed: dictionnary of list seed words with label as key (this words are specific of the kay label)
            @param kwargs:
            """
            super(GuidedEngine, self).__init__(texts , nb_topics , dictionnary , random_state)
            self.seed = seed


class CoreX(Engine):

    def __init__(self , texts: List[List] ,  nb_topics : int = 5, dictionnary: corpora.Dictionary = None,  random_state : int = 42  ,  **kwargs):
        super().__init__(texts , nb_topics , dictionnary , random_state)
        self.document_words_counter = DocumentsWordsCounter(self.texts)
        self.words = [ word for word in self.document_words_counter.columns]
        self.documents_matrix = self.document_words_counter.to_numpy(dtype = bool)
        self.documents_matrix = ss.csc_matrix(self.documents_matrix)
        self.core = ct.Corex(n_hidden=self.nb_topics , **kwargs)
        self.core = self._train_core()

    def _train_core(self):
        self.core.fit(self.documents_matrix, words=self.words)
        return self.core

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
    def _train_core(self):
        self.core.fit(self.documents_matrix , y=self.labels)
        return self.core


class GuidedCoreX(GuidedEngine,CoreX):
    def __init__(self, seed : dict ,texts: List[List], nb_topics : int = 5, dictionnary: corpora.Dictionary = None,  random_state : int = 42 ,  seed_strength  = 3, **kwargs):
        self.seed_strength = seed_strength
        base_words = [word for word in DocumentsWordsCounter(texts).columns]
        self.anchors = [[word for word in topic_words if word in base_words] for _, topic_words in seed.items()]
        super(GuidedCoreX, self).__init__(seed , texts , nb_topics , dictionnary , random_state , **kwargs)

    def _train_core(self):
        self.core.fit(self.documents_matrix, words=self.words, anchors=self.anchors, anchor_strength=self.seed_strength)
        return self.core


class TFIDF(SupervisedEngine):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.n_docs = len(self.texts)
        self.words_idx = list(self.dictionnary.token2id.keys())
        documents_words_binar_counter = DocumentsWordsCounter(self.texts , binary=True)
        self.documents_words_binar_counter = documents_words_binar_counter.reindex(columns = self.words_idx)
        self.corpus_words_counter = self.documents_words_binar_counter.to_numpy()
        self.corpus_words_counter = np.sum(self.corpus_words_counter , axis=0)

        labels_words_counter = LabelsWordsCounter(self.texts , self.labels)

        # reindex columns to keep same words order between self.documents_words_binar_counter and self.labels_words_counter
        # and add label row for label that not existent in the self.labels (list of label)
        self.labels_words_counter = labels_words_counter.reindex(self.labels_idx, columns = self.words_idx)

        # replace 'nan' value by zero
        self.labels_words_counter.fillna(0)

        self.core = self._train_core()


    def _train_core(self):

        tfidf_matrix = self.labels_words_counter.to_numpy()
        shadow = np.zeros_like(tfidf_matrix)
        for i in range(tfidf_matrix.shape[0]):
            if np.sum(tfidf_matrix[i]) == 0:
                #return random value when the topic is inexistant in the texts input
                shadow[i] = np.random.random(tfidf_matrix.shape[1])/1000
            else:
                for j in range(tfidf_matrix.shape[1]):
                    tf = (tfidf_matrix[i][j]/np.sum(tfidf_matrix[i]))
                    idf = np.log(self.n_docs/np.sum(self.corpus_words_counter[j]))
                    shadow[i][j] = tf*idf
        return shadow


    def get_topic_terms(self, topic_id : int, topn=100):

        topic = self.labels_idx[topic_id]
        label_serie = self.core[topic_id]
        word_idxs = np.flipud((np.argsort(label_serie)[-topn:]))

        # output format --> [
        #                       {
        #                           'word1' : score1
        #                        }
        #                        ,
        #                        ...
        #                        ]
        return {self.words_idx[word_idx] : label_serie[word_idx] for word_idx in word_idxs}



class LDA(Engine):

    def __init__(self,texts: List[List], nb_topics : int = 5, dictionnary: corpora.Dictionary = None,  random_state=42 , **kwargs):
        """

        @param dictionnary: specific gensim dictionnary linked to the data corpus
        @param random_state:
        @param passes: number of time that we passe the corpus during training (like epoch)
        @param kwargs:
        """
        super().__init__(texts , nb_topics , dictionnary , random_state)
        self.corpus_bow = [self.dictionnary.doc2bow(text) for text in self.texts]
        self.lda_args = {
            "corpus" : self.corpus_bow,
            "num_topics" : self.nb_topics,
            "id2word" : self.dictionnary ,
            "random_state" : self.random_state,
        }
        self.core  = self._train_core(**kwargs)

    def _train_core(self , **kwargs):
        return LdaModel(
            **self.lda_args,
            **kwargs
        )

    @property
    def coherence(self):
        coherence_model = CoherenceModel(model=self.core, topics=self.nb_topics,
                                         corpus=self.corpus_bow, dictionary=self.dictionnary)
        return coherence_model.get_coherence()



    def get_topic_terms(self ,topic_id : int , topn = 100 , **kwargs):

        res = self.core.get_topic_terms(topicid=topic_id , topn = topn, **kwargs)
        return {self.core.id2word[word_id] : score for word_id , score in res}


class GuidedLDA(GuidedEngine , LDA):


    def __init__(self, seed : dict ,texts: List[List], nb_topics : int = 5, dictionnary: corpora.Dictionary = None,  random_state : int = 42  , seed_strength = 100, **kwargs):

        kwargs["eta"] = self.generate_eta(seed, dictionnary, overratte=seed_strength)
        super().__init__(seed , texts , nb_topics , dictionnary , random_state)
        self.core = LdaModel(**self.lda_args, **kwargs)


    @staticmethod
    def generate_eta(seed , dictionnary, normalize_eta=False, overratte=1e4):
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