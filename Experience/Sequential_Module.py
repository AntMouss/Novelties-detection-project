import random
from typing import List, Callable, Dict
from Experience.data_utils import TimeLineArticlesDataset
from gensim import corpora
from Experience.data_processing import filterDictionnary
from Experience import Engine
import numpy as np
from collections import Counter
import functools


def check_size(func):
    def wrapper(*args, **kwargs):
        if len(args[1]) == 0:
            raise Exception('documents empty , we can not process the sequence')
        return func(*args, **kwargs)
    return wrapper



class MetaSequencialLangageSimilarityCalculator:

    def __init__(self, nb_topics: int, thresholding_fct_above: Callable,
                 thresholding_fct_bellow: Callable, kwargs_above: Dict, kwargs_bellow: Dict):

        self.kwargs_bellow = kwargs_bellow
        self.kwargs_above = kwargs_above
        self.thresholding_fct_above = thresholding_fct_above
        self.thresholding_fct_bellow = thresholding_fct_bellow
        self.engine = Engine.Engine
        self.semi_filtred_dictionnary = corpora.Dictionary()
        self.nb_topics = nb_topics
        self.seedFileName = '_seed.json'
        self.bad_words = []
        self.info = {"engine_type": self.engine.__name__, 'nb_topics': self.nb_topics}
        self.res = {}
        self.models = []
        self.info_file = 'info.json'
        self.resFileName = 'res.json'
        self.semi_dictionnaryFileName = '_semiDict'
        self.nb_windows = 0
        self.dateFile = 'date.json'
        self.date_window_idx = {}
        self.predefinedBadWords = ['...', 'commenter', 'réagir', 'envoyer', 'mail', 'partager', 'publier', 'lire',
                                   'journal', "abonnez-vous", "d'une", "d'un", "mars", "avril", "mai",
                                   "juin", "juillet", "an", "soir", "mois", "lundi", "mardi", "mercredi"
            , "jeudi", "vendredi", "samedi", "dimanche"]


    def __len__(self):
        return self.nb_windows


    @check_size
    def treat_Window(self, data_window, **kwargs):
        pass

    def add_windows(self, data: TimeLineArticlesDataset, lookback=10, update_res=False, **kwargs):

        self.info['lookback'] = lookback
        rValue = random.Random()
        rValue.seed(37)
        for i, (end_date_window, data_windows) in (enumerate(data)):
            random_state = rValue.randint(1, 14340)
            kwargs["random_state"] = random_state
            print(f"numero of window: {i} -- random state: {random_state}")
            model, window_dictionnary = self.treat_Window(data_windows, **kwargs)
            # for bound window to the right glda model we use no_window
            no_window = i
            if update_res:
                self.updateResults(end_date_window, window_dictionnary, model, no_window)
            self.date_window_idx[end_date_window] = no_window
            self.models.append(model)
            self.nb_windows += 1


    def updateResults(self, end_date, dictionnary_window: corpora.Dictionary, model: Engine.Engine, no_window: int,
                      ntop: int = 100):

        topWordsTopics = self.getTopWordsTopics(model, ntop=ntop, exclusive=False)
        for word, word_id in dictionnary_window.token2id.items():
            if word not in self.res.keys():
                self.res[word] = {}
                self.res[word]['first'] = {
                    'date': end_date}  # we use end date of the window as date of the first appearance to the current world
                self.res[word]['appearances'] = []
            appearance = {}
            appearance['date_end_window'] = end_date
            appearance['no_window'] = no_window
            appearance['isBadWord'] = (word in self.bad_words)
            appearance['df_in_window'] = dictionnary_window.dfs[word_id]
            appearance['cf_in_window'] = dictionnary_window.cfs[word_id]
            self.res[word]['appearances'].append(appearance)
            for topic_id in range(self.nb_topics):
                try:
                    score = topWordsTopics[topic_id][word]
                except KeyError as ke:
                    continue
                if 'keyword' not in appearance.keys():
                    appearance['keyword'] = {}
                if topic_id not in appearance['keyword'].keys():
                    appearance['keyword'][topic_id] = {}
                appearance['keyword'][topic_id]['score'] = str(score)
                # appearance['keyword'][topic_id]['relative_score'] = str(score/average_score_topic)


    def get_res(self):
        return self.res


    def updateBadwords(self):

        nb_docs = self.semi_filtred_dictionnary.num_docs
        abs_no_above = self.thresholding_fct_above(nb_docs=nb_docs, **self.kwargs_above)
        abs_no_bellow = self.thresholding_fct_bellow(nb_docs=nb_docs, **self.kwargs_bellow)
        if abs_no_bellow >= abs_no_above:
            raise Exception("abs_no_bellow should be inferior to abs_no_above")
        if abs_no_above <= 0:
            raise Exception("abs_no_above should be superior to zero")
        self.bad_words = [word for id, word in self.semi_filtred_dictionnary.items() if
                          self.semi_filtred_dictionnary.dfs[id] < abs_no_bellow or self.semi_filtred_dictionnary.dfs[
                              id] > abs_no_above]
        self.bad_words += self.predefinedBadWords


    def getTopWordsTopics(self, model: Engine.Engine = None, ntop: int = 100, exclusive=False, **kwargs):
        """
        :param ntop: number of keywords that the model return by topic
        :param exclusive: if we want that the keywors being exclusive to the topic
        return: list of list id of words in the dictionnary, one list by gldaModel so one list by time intervall
        """
        topWordsTopics = []
        for topic_id in range(model.nb_topics):
            topWordsTopic = self.getTopWordsTopic(topic_id, model, ntop, **kwargs)
            topWordsTopics.append(topWordsTopic)

        if exclusive == False:

            return topWordsTopics
        else:
            return self.exclusiveWordsPerTopics(topWordsTopics)


    def getTopWordsTopic(self, topic_id, model: Engine.Engine = None, ntop: int = 100, **kwargs):

        # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
        topWords = model.get_topic_terms(topic_id=topic_id, topn=ntop)
        topWordsTopic = {topWord[0]: topWord[1] for topWord in topWords.items()}
        return topWordsTopic

    @staticmethod
    def exclusiveWordsPerTopics(topWordsTopics: List[dict]):

        topWordsTopics_tmp = [set(topWordsTopic.keys()) for topWordsTopic in topWordsTopics]
        for i in range(len(topWordsTopics)):
            for j in range(i, len(topWordsTopics)):
                topWordsTopics_tmp[i] = topWordsTopics_tmp[i].difference(topWordsTopics_tmp[j])
                topWordsTopics_tmp[j] = topWordsTopics_tmp[j].difference(topWordsTopics_tmp[i])
        return [{word: topWordsTopics[i][word] for word in topWordsTopics_tmp[i]} for i in range(len(topWordsTopics))]


    def compute_similarity(self , cluster1 : Dict , cluster2 : Dict , soft = False):

        intersection = set(cluster1).intersection(set(cluster2))
        difference = set(cluster1).difference(set(cluster2))
        disappearance = set(cluster2).difference(set(cluster1))
        if soft:
            # to normalize output result because score depend of the engine
            total = sum([score for _ , score in cluster1.items() ])
            similarity_score = sum([cluster1[word] for word in intersection])/total
        else:
            similarity_score = len(intersection) / len(cluster1)
        return similarity_score,difference ,  intersection , disappearance

    def compare_Windows_Sequentialy(self, first_w=0, last_w=0, ntop=100, back=3, **kwargs):

        # we use thi condition to set the numero of the last window because by
        # default we want to compute similarity until the last window
        if last_w == 0:
            last_w = len(self.models)
        res = []
        mode = ''
        for i in range(first_w + 1, last_w):
            window_res = []
            for j in range(back):
                try:
                    similarities, _  = self.calcule_similarity_topics_W_W(
                        ntop=ntop, previous_window=i - 1 - j, new_window=i, **kwargs)
                    if np.isnan(np.sum(similarities)):
                        print('fix this')
                        raise Exception
                    window_res.append(similarities)
                except Exception as e:
                    break
            window_res = np.array(window_res)
            res.append(np.mean(window_res , axis=0))
        return res


    def calcule_similarity_topics_W_W(self, reproduction_threshold, ntop=100, previous_window=0, new_window=1,
                                      soft=False, **kwargs):
        return (np.nan , np.nan)


    def print_novelties(self, n_to_print=10, **kwargs):
        pass



class NoSupervisedSequantialLangageSimilarityCalculator(MetaSequencialLangageSimilarityCalculator):

    @check_size
    def treat_Window(self, texts: List[List], **kwargs):
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, **kwargs)
        return model, window_dictionnary_f

    @functools.lru_cache(maxsize=3)
    def calcule_similarity_topics_W_W(self, reproduction_threshold, ntop=100, previous_window=0, new_window=1,
                                      soft=False, **kwargs):
        """

        @param reproduction_threshold: threshold from which we considerate that the 2 topics from two differents
        windows match
        @param ntop: number of top revelant words from the window engine
        @param previous_window: number of the first window to compare
        @param new_window: number of the second window to compare
        @param soft: take the brut score (1pt for one commun word) or the soft score (score delivered by the engine)
        @param kwargs:
        @return:
        """
        if previous_window < 0 or new_window < 0:
            raise Exception("index should be positives")
        total_similarity_score = 0
        links = [[] for _ in range(self.nb_topics)]
        # 'no_relationship' because this novelties are from "new topic" that aren't reproducte itself (similarity_score < reproduction_threshold)
        no_relationship_novelties = np.zeros(self.nb_topics, dtype=list)
        no_relationship_disappearances = np.zeros(self.nb_topics, dtype=list)
        # 'relationship' because this novelties are from "new topic" that reproducte itself (similarity_score >= reproduction_threshold)
        relationship_novelties_matrix = np.zeros((self.nb_topics , self.nb_topics) , dtype=list)
        relationship_habbits_matrix = np.zeros((self.nb_topics, self.nb_topics), dtype=list)
        relationship_disappearances_matrix = np.zeros((self.nb_topics, self.nb_topics), dtype=list)
        previousTopWordsTopics = self.getTopWordsTopics(self.models[previous_window], ntop=ntop, **kwargs)
        # list of sets of top words per topics in jth window
        newTopWordsTopics = self.getTopWordsTopics(self.models[new_window], ntop=ntop, **kwargs)
        # the number of topics is static so we can use self.nb_topics for iterate
        for new_topic in range(self.nb_topics):
            for previous_topic in range(self.nb_topics):
                similarity_score , novelties , habbits , disappearances = self.compute_similarity(newTopWordsTopics[new_topic] , previousTopWordsTopics[previous_topic] , soft=soft)
                if similarity_score >= reproduction_threshold:
                    total_similarity_score += similarity_score
                    links[new_topic].append(previous_topic)
                    relationship_novelties_matrix[new_topic][previous_topic] = novelties
                    relationship_habbits_matrix[new_topic][previous_topic] = habbits
                    relationship_disappearances_matrix[new_topic][previous_topic] = disappearances
                else:
                    no_relationship_novelties[new_topic] = novelties
                    no_relationship_disappearances[previous_topic] = disappearances
        # finaly we don't need malus cause the malus already exist because if nothing are added to the
        #total_score when the threshold isn't exceed
        # note that we compute a total_score for all the window in the no supervised case
        #malus = np.sum(persist)
        return np.array([total_similarity_score]) , (links , relationship_novelties_matrix , relationship_habbits_matrix , relationship_disappearances_matrix)

    def print_novelties(self, n_to_print=10, **kwargs):
        """

        @param topic_id:
        @param habbits: set of words intersection between 2 windows.
        @param novelties: set of words difference from the current window different to the previous window.
        @param n_to_print: number of maximum words to print
        @param kwargs:
        """
        last_window_idx = len(self)
        similarity, (links , novelties_matrix, habbits_matrix) = self.calcule_similarity_topics_W_W(
            previous_window=last_window_idx - 1, new_window=last_window_idx, **kwargs)
        print(f"Information from new window calculate by the Sequantial Calculator : {id(self)}")
        print(f"Similarity score with the last window = {similarity}")
        for topic_id in range(self.nb_topics):
            print(f" the topic {topic_id} have {len(links[topic_id])} parents from last windows :")
            for parent_topic_id in links[topic_id]:
                novelties = novelties_matrix[topic_id][parent_topic_id]
                habbits = habbits_matrix[topic_id][parent_topic_id]
                print(f"parent id {parent_topic_id}")
                print("-" * 30)
                print(f" Novelties for the topic :")
                print(*novelties[:n_to_print], sep='\n')
                print("-" * 30)
                print(f" Habbits for the topic :")
                print(*habbits[:n_to_print], sep='\n')
                print("*" * 30)
            print("#" * 30)
            print("#" * 30)



class SupervisedSequantialLangageSimilarityCalculator(MetaSequencialLangageSimilarityCalculator):

    def __init__(self, labels_idx, **kwargs):
        super(SupervisedSequantialLangageSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.SupervisedEngine
        self.labels_idx = labels_idx
        self.label_articles_counter = []


    def updateLabelCounter(self , labels):
        window_counter = dict(Counter(labels))
        window_counter = {self.labels_idx.index(k) : v for k , v in window_counter.items()}
        self.label_articles_counter.append(window_counter)


    @check_size
    def treat_Window(self, data_windows: tuple, **kwargs):
        texts, labels = data_windows
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        self.updateLabelCounter(labels)
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, labels=labels, labels_idx=self.labels_idx, **kwargs)
        return model, window_dictionnary_f


    # use lru cache for avoid useless call method in compareTopicSequentialy method . lazy way to avoid code refaction of the compareTopicSequentialy method
    @functools.lru_cache(maxsize=3)
    def calcule_similarity_topics_W_W(self, ntop=100, previous_window=0, new_window=1, soft=False,
                                      **kwargs):
        """

        @param ntop: number of top revelant words from the window engine
        @param previous_window: number of the first window to compare
        @param new_window: number of the second window to compare
        @param soft: take the brut score (1pt for one commun word) or the soft score (score delivered by the engine)
        @param kwargs:
        @return: List of percentage of similarity between the two windows (one value by topic)
        we decide to make the method compute similarity for all topics because of exclusive option that let us compute
         the similarity just for the words that are exclusive to the topic
         (no exclusive words are removed and not used during the similarity computing).
        """
        if previous_window < 0 or new_window < 0:
            raise Exception("index should be positives")
        previousTopWordsTopics = self.getTopWordsTopics(self.models[previous_window], ntop=ntop, **kwargs)
        # list of sets of top words per topics in jth window
        newTopWordsTopics = self.getTopWordsTopics(self.models[new_window], ntop=ntop, **kwargs)
        similarities = []
        novelties = []
        habbits = []
        disappearances = []
        for topic_id in range(self.nb_topics):
            similarity_score , novelties_topic , habbits_topic , disappearances_topic = self.compute_similarity(
                newTopWordsTopics[topic_id], previousTopWordsTopics[topic_id], soft=soft)
            similarities.append(similarity_score)
            novelties.append(novelties_topic)
            habbits.append(habbits_topic)
            disappearances.append(disappearances_topic)
        return np.array(similarities) , (novelties , habbits , disappearances)


    def print_novelties(self , n_to_print = 10, **kwargs):
        """

        @param topic_id:
        @param habbits: set of words intersection between 2 windows.
        @param novelties: set of words difference from the current window different to the previous window.
        @param n_to_print: number of maximum words to print
        @param kwargs:
        """
        last_window_idx = len(self)
        res = self.calcule_similarity_topics_W_W(
            previous_window= last_window_idx - 1 ,new_window= last_window_idx , **kwargs)
        print(f"Information from new window calculate by the Sequantial Calculator : {id(self)}")
        for topic_id , (similarity , novelties , habbits) in zip(range(self.nb_topics),res):

            print(f"similarity score for the topic {self.labels_idx[topic_id]}  : {similarity} ")
            print("-"*30)
            print(f" Novelties for the topic {self.labels_idx[topic_id]}  :")
            print(*novelties[:n_to_print] , sep='\n')
            print("-"*30)
            print(f" Habbits for the topic {self.labels_idx[topic_id]}  :")
            print(*habbits[:n_to_print], sep='\n')
            print("*" * 30)
        print("#" * 30)
        print("#" * 30)


class GuidedSequantialLangageSimilarityCalculator(SupervisedSequantialLangageSimilarityCalculator):

    def __init__(self, seed: dict, **kwargs):
        super(GuidedSequantialLangageSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.GuidedEngine
        self.seed = seed

    @check_size
    def treat_Window(self, data_windows: tuple, **kwargs):
        # i use labels for Guided calculator because of labels_counter that let us to
        # understand the relation between number of articles and similarity score
        # but we don't need label for revelant words computing (just the seed words).
        texts, labels = data_windows
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        self.updateLabelCounter(labels)
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, seed=self.seed, **kwargs)
        return model, window_dictionnary_f


    def getTopWordsTopic(self, topic_id, model: Engine = None, ntop: int = 100, remove_seed_words: bool = True):

        # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
        topWordsTopic = model.get_topic_terms(topic_id=topic_id, topn=ntop)
        topic = self.labels_idx[topic_id]
        if remove_seed_words:
            words2keep = set(topWordsTopic.keys()).intersection(self.seed[topic])
            for word in words2keep:
                del (topWordsTopic[word])
        return topWordsTopic


class LDASequentialSimilarityCalculator(NoSupervisedSequantialLangageSimilarityCalculator):

    def __init__(self, **kwargs):
        super(LDASequentialSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.LDA

    @check_size
    def treat_Window(self, texts, **kwargs):
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, dictionnary=window_dictionnary_f, **kwargs)
        return model, window_dictionnary_f


class GuidedLDASequentialSimilarityCalculator(GuidedSequantialLangageSimilarityCalculator):

    def __init__(self, **kwargs):
        super(GuidedLDASequentialSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.GuidedLDA

    @check_size
    def treat_Window(self, data_window, **kwargs):
        texts, labels = data_window
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        window_dictionnary = corpora.Dictionary(texts)
        self.updateLabelCounter(labels)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts,seed=self.seed, dictionnary=window_dictionnary_f, **kwargs)
        return model, window_dictionnary_f


class GuidedCoreXSequentialSimilarityCalculator(GuidedSequantialLangageSimilarityCalculator):

    def __init__(self, **kwargs):
        super(GuidedCoreXSequentialSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.GuidedCoreX


class CoreXSequentialSimilarityCalculator(NoSupervisedSequantialLangageSimilarityCalculator):
    def __init__(self, **kwargs):
        super(CoreXSequentialSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.CoreX


class SupervisedCoreXSequentialSimilarityCalculator(SupervisedSequantialLangageSimilarityCalculator):

    def __init__(self, **kwargs):
        super(SupervisedCoreXSequentialSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.SupervisedCoreX


class LFIDFSequentialSimilarityCalculator(SupervisedSequantialLangageSimilarityCalculator):

    def __init__(self, **kwargs):
        super(LFIDFSequentialSimilarityCalculator, self).__init__(**kwargs)
        self.engine = Engine.LFIDF
