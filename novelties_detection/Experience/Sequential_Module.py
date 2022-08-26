import copy
import random
from typing import List, Dict
from novelties_detection.Experience.data_utils import TimeLineArticlesDataset
from gensim import corpora
from novelties_detection.Collection.data_processing import cleanDictionnary
from novelties_detection.Experience import Engine_module
from novelties_detection.Experience.Exception_utils import CompareWindowsException, NoWordsClusterException
import numpy as np
from collections import Counter
import functools
from collections import ChainMap
from collections import deque


def check_size(func):
    def wrapper(*args, **kwargs):
        if len(args[1]) == 0:
            raise Exception('documents empty , we can not process the sequence')
        return func(*args, **kwargs)

    return wrapper


class MetaSequencialLangageSimilarityCalculator:
    semi_filtred_dictionnary = corpora.Dictionary()
    seedFileName = '_seed.json'
    bad_words = []
    res = {}
    info_file = 'info.json'
    resFileName = 'res.json'
    semi_dictionnaryFileName = '_semiDict'
    dateFile = 'date.json'
    date_window_idx = {}
    predefinedBadWords = ['...', 'commenter', 'réagir', 'envoyer', 'mail', 'partager', 'publier', 'lire',
                          'journal', "abonnez-vous", "d'une", "d'un", "mars", "avril", "mai",
                          "juin", "juillet", "an", "soir", "mois", "lundi", "mardi", "mercredi"
        , "jeudi", "vendredi", "samedi", "dimanche"]

    def __init__(self, bad_words_args: dict, memory_length: int = None):
        """

        @param bad_words_args: args for bad words removing
        @param memory_length: number of engine models we keep in memory
        """
        self.bad_words_args = bad_words_args
        if memory_length is None:
            self.models = []
        else:
            self.models = deque(maxlen=memory_length)
        self.engine = Engine_module.Engine
        self.info = {"engine_type": self.engine.__name__}

    def __len__(self):
        return len(self.models)

    @check_size
    def treat_Window(self, data_window, **kwargs):
        pass

    def add_windows(self, data: TimeLineArticlesDataset, lookback=10, update_res=False, **kwargs):

        self.info['lookback'] = lookback
        rValue = random.Random()
        rValue.seed(37)
        for window_idx, (end_date_window, data_windows) in (enumerate(data)):
            random_state = rValue.randint(1, 14340)
            kwargs["random_state"] = random_state
            print(f"numero of window: {window_idx} -- random state: {random_state}")
            model, window_dictionnary = self.treat_Window(data_windows, **kwargs)
            # for bound window to the right glda model we use no_window
            if update_res:
                self.updateResults(end_date_window, window_dictionnary, window_idx)
            self.date_window_idx[end_date_window] = window_idx

    @staticmethod
    def compute_similarity(cluster1: Dict, cluster2: Dict, soft=False):

        if len(cluster1) == 0 or len(cluster2) == 0:
            raise NoWordsClusterException("cluster contain no words... impossible to compute similarity ")
        intersection = list(set(cluster1).intersection(set(cluster2)))
        difference = list(set(cluster1).difference(set(cluster2)))
        disappearance = list(set(cluster2).difference(set(cluster1)))
        if soft:
            # to normalize output result because score depend of the engine
            total = sum([score for _, score in cluster1.items()])
            similarity_score = sum([cluster1[word] for word in intersection]) / total
        else:
            similarity_score = len(intersection) / len(cluster1)
        return similarity_score, (difference, intersection, disappearance)

    @functools.lru_cache(maxsize=3)
    def compare_Windows_Sequentialy(self, ntop=100, back=1, **kwargs):

        # we use thi condition to set the numero of the last window because by
        # default we want to compute similarity until the last window
        if back < 1:
            raise Exception("back should be superior to one because one is the minimum difference between index"
                            "of the current window and index of the previous windows")
        res = []
        for i in range(1, len(self.models)):
            window_res = []
            j = 1
            while j <= back and i  - j >= 0:
                similarities, _ = self.calcule_similarity_topics_W_W(
                    ntop=ntop, previous_window_idx=i  - j, new_window_idx=i, **kwargs)
                similarities = np.array(similarities)
                window_res.append(similarities)
                j += 1
            window_res = np.array(window_res)
            res.append(np.mean(window_res, axis=0))
        return np.transpose(np.array(res))

    def calcule_similarity_topics_W_W(self, previous_window: int, new_window: int, reproduction_threshold, ntop=100,
                                      soft=False, **kwargs):
        return (np.nan, np.nan)

    def print_novelties(self, n_words_to_print=10, **kwargs):
        pass

    def updateBadwords(self):

        thresholding_fct_above = self.bad_words_args["thresholding_fct_above"]
        thresholding_fct_bellow = self.bad_words_args["thresholding_fct_bellow"]
        kwargs_above = self.bad_words_args["kwargs_above"]
        kwargs_bellow = self.bad_words_args["kwargs_bellow"]
        nb_docs = self.semi_filtred_dictionnary.num_docs
        abs_no_above = thresholding_fct_above(nb_docs=nb_docs, **kwargs_above)
        abs_no_bellow = thresholding_fct_bellow(nb_docs=nb_docs, **kwargs_bellow)
        if abs_no_bellow >= abs_no_above:
            raise Exception("abs_no_bellow should be inferior to abs_no_above")
        if abs_no_above <= 0:
            raise Exception("abs_no_above should be superior to zero")
        self.bad_words = [word for id, word in self.semi_filtred_dictionnary.items() if
                          self.semi_filtred_dictionnary.dfs[id] < abs_no_bellow or self.semi_filtred_dictionnary.dfs[
                              id] > abs_no_above]
        self.bad_words += self.predefinedBadWords

    def updateResults(self, end_date, dictionnary_window: corpora.Dictionary, window_idx: int,
                      ntop: int = 100):

        # for comprehension numero of window
        topWordsTopics = self.getTopWordsTopics(window_idx, ntop=ntop, exclusive=False)
        nb_topics = len(topWordsTopics)
        for word, word_id in dictionnary_window.token2id.items():
            if word not in self.res.keys():
                self.res[word] = {}
                self.res[word]['first'] = {
                    'date': end_date}  # we use end date of the window as date of the first appearance to the current world
                self.res[word]['appearances'] = []
            appearance = {}
            appearance['date_end_window'] = end_date
            appearance['no_window'] = window_idx
            appearance['isBadWord'] = (word in self.bad_words)
            appearance['df_in_window'] = dictionnary_window.dfs[word_id]
            appearance['cf_in_window'] = dictionnary_window.cfs[word_id]
            self.res[word]['appearances'].append(appearance)
            for topic_id in range(nb_topics):
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

    def getTopWordsTopics(self, model_idx: int, ntop: int = 100, exclusive=False, **kwargs):
        """
        :param ntop: number of keywords that the model return by topic
        :param exclusive: if we want that the keywors being exclusive to the topic
        return: list of list id of words in the dictionnary, one list by gldaModel so one list by time intervall
        """
        topWordsTopics = []
        nb_topics = self.models[model_idx].nb_topics
        for topic_id in range(nb_topics):
            topWordsTopic = self.getTopWordsTopic(topic_id, model_idx, ntop, )
            topWordsTopics.append(topWordsTopic)

        if exclusive == False:

            return topWordsTopics
        else:
            return self.exclusiveWordsPerTopics(topWordsTopics)

    @staticmethod
    def exclusiveWordsPerTopics(topWordsTopics: List[dict]):

        topWordsTopics_tmp = [set(topWordsTopic.keys()) for topWordsTopic in topWordsTopics]
        topWordsTopics_tmp_ref = copy.deepcopy(topWordsTopics_tmp)
        for i in range(len(topWordsTopics) - 1):
            for j in range(i + 1, len(topWordsTopics)):
                intersection = topWordsTopics_tmp_ref[i].intersection(topWordsTopics_tmp_ref[j])
                topWordsTopics_tmp[i] = topWordsTopics_tmp[i].difference(intersection)
                topWordsTopics_tmp[j] = topWordsTopics_tmp[j].difference(intersection)
        return [{word: topWordsTopics[i][word] for word in topWordsTopics_tmp[i]} for i in range(len(topWordsTopics))]

    def getTopWordsTopic(self, topic_id, model_idx: int, ntop: int = 100 , **kwargs):
        model = self.models[model_idx]
        # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
        topWords = model.get_topic_terms(topic_id=topic_id, topn=ntop)
        topWordsTopic = {topWord[0]: topWord[1] for topWord in topWords.items()}
        return topWordsTopic


class MetaDynamicalSequencialLangageSimilarityCalculator(MetaSequencialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, nb_max_topics: int, nb_min_topics: int,
                 minimum_ratio_selection: float = 0.10, memory_length: int = None):
        """

        @param nb_min_topics:
        @param bad_words_args:
        @param nb_max_topics: maximum number of topics that we can choose
        @param minimum_ratio_selection: minimum ratio between 2 derivate coherence scores from which the number of topics is chosen
        """
        super().__init__(bad_words_args, memory_length)
        self.nb_min_topics = nb_min_topics
        self.minimum_ratio_selection = minimum_ratio_selection
        self.nb_max_topics = nb_max_topics

    @check_size
    def treat_Window(self, texts: List[List], **kwargs):

        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = cleanDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        models = []
        coherences = []
        for nb_topics in range(self.nb_min_topics, self.nb_max_topics):
            model = self.engine(texts=texts, nb_topics=nb_topics, dictionnary=window_dictionnary_f,
                                **kwargs)
            models.append(model)
            coherence = model.coherence
            coherences.append(coherence)
        nb_topics, choosen_model_idx = self.select_optimal_topic_number(coherences)
        model = models[choosen_model_idx]
        self.models.append(model)
        return model, window_dictionnary_f

    def select_optimal_topic_number(self, coherences):
        coherences_derivate = np.gradient(coherences, len(coherences))
        for idx, (nb_topics, coherence_derivate) in enumerate(
                zip(coherences_derivate, range(self.nb_min_topics, self.nb_max_topics))):
            if coherence_derivate < self.minimum_ratio_selection:
                return nb_topics - 1, idx


class MetaFixedSequencialLangageSimilarityCalculator(MetaSequencialLangageSimilarityCalculator):
    """
    Meta class to calculate relation , similarity and novelties between sequential bunch of temporal data window.

    """

    def __init__(self, nb_topics: int, bad_words_args: dict, memory_length: int = None):
        """

        @param nb_topics: number of topics we fixed a priori
        @param bad_words_args: args for bad words removing
        @param memory_length: number of engine models we keep in memory
        """
        super().__init__(bad_words_args, memory_length)
        self.nb_topics = nb_topics
        self.info = {"engine_type": self.engine.__name__, 'nb_topics': self.nb_topics}


class NoSupervisedSequantialLangageSimilarityCalculator(MetaSequencialLangageSimilarityCalculator):

    @functools.lru_cache(maxsize=3)
    def calcule_similarity_topics_W_W(self, previous_window_idx: int, new_window_idx: int, reproduction_threshold,
                                      ntop=100, soft=False, **kwargs):
        """

        @param reproduction_threshold: threshold from which we considerate that the 2 topics from two differents
        windows match
        @param ntop: number of top revelant words from the window engine
        @param previous_window_idx: number of the first window to compare
        @param new_window_idx: number of the second window to compare
        @param soft: take the brut score (1pt for one commun word) or the soft score (score delivered by the engine)
        @param kwargs:
        @return:
        """
        if previous_window_idx < 0 or new_window_idx < 0:
            raise CompareWindowsException("index should be positives")
        previous_top_words_topics = self.getTopWordsTopics(previous_window_idx, ntop=ntop, **kwargs)
        # list of sets of top words per topics in jth window
        new_top_words_topics = self.getTopWordsTopics(new_window_idx, ntop=ntop, **kwargs)
        previous_nb_topics = len(previous_top_words_topics)
        new_nb_topics = len(new_top_words_topics)
        links = [[] for _ in range(new_nb_topics)]
        # 'no_relationship' because this novelties are from "new topic" that aren't reproducte itself (similarity_score < reproduction_threshold)
        no_relationship_novelties = np.zeros(new_nb_topics, dtype=list)
        no_relationship_disappearances = np.zeros(new_nb_topics, dtype=list)
        # 'relationship' because this novelties are from "new topic" that reproducte itself (similarity_score >= reproduction_threshold)
        relationship_novelties_matrix = np.zeros((new_nb_topics, previous_nb_topics), dtype=list)
        relationship_habbits_matrix = np.zeros((new_nb_topics, previous_nb_topics), dtype=list)
        relationship_disappearances_matrix = np.zeros((new_nb_topics, previous_nb_topics), dtype=list)
        # the number of topics is static so we can use self.nb_topics for iterate
        total_previous_Top_words = dict(ChainMap(*previous_top_words_topics))
        total_new_Top_words = dict(ChainMap(*new_top_words_topics))
        try:
            total_similarity_score, _ = self.compute_similarity(total_new_Top_words, total_previous_Top_words,
                                                                soft=soft)
        except NoWordsClusterException:
            total_similarity_score = np.nan
            pass
        for new_topic in range(new_nb_topics):
            for previous_topic in range(previous_nb_topics):
                try:
                    similarity_score, (novelties, habbits, disappearances) = self.compute_similarity(
                        new_top_words_topics[new_topic], previous_top_words_topics[previous_topic], soft=soft)
                except NoWordsClusterException:
                    similarity_score = 0
                    novelties = []
                    habbits = []
                    disappearances = []
                    pass
                if similarity_score >= reproduction_threshold:
                    links[new_topic].append(previous_topic)
                    relationship_novelties_matrix[new_topic][previous_topic] = novelties
                    relationship_habbits_matrix[new_topic][previous_topic] = habbits
                    relationship_disappearances_matrix[new_topic][previous_topic] = disappearances
                else:
                    no_relationship_novelties[new_topic] = novelties
                    no_relationship_disappearances[previous_topic] = disappearances
        # finaly we don't need malus cause the malus already exist because if nothing are added to the
        # total_score when the threshold isn't exceed
        # note that we compute a total_score for all the window in the no supervised case
        # malus = np.sum(persist)
        return [total_similarity_score], (
            links, relationship_novelties_matrix, relationship_habbits_matrix, relationship_disappearances_matrix)


class NoSupervisedFixedSequantialLangageSimilarityCalculator(MetaFixedSequencialLangageSimilarityCalculator,
                                                             NoSupervisedSequantialLangageSimilarityCalculator):

    @check_size
    def treat_Window(self, texts: List[List], **kwargs):
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = cleanDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, nb_topics=self.nb_topics, dictionnary=window_dictionnary_f, **kwargs)
        self.models.append(model)
        return model, window_dictionnary_f


class NoSupervisedDynamicalSequantialLangageSimilarityCalculator(MetaDynamicalSequencialLangageSimilarityCalculator,
                                                                 NoSupervisedSequantialLangageSimilarityCalculator):
    pass


class SupervisedSequantialLangageSimilarityCalculator(MetaSequencialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, labels_idx: list, memory_length: int = None):

        super().__init__(bad_words_args, memory_length)
        self.engine = Engine_module.SupervisedEngine
        self.labels_idx = labels_idx
        self.nb_topics = len(self.labels_idx)
        if memory_length is None:
            self.label_articles_counters = []
        else:
            self.label_articles_counters = deque(maxlen=memory_length)

    def updateLabelCounter(self, labels):
        window_counter = dict(Counter(labels))
        window_counter = {self.labels_idx.index(k): v for k, v in window_counter.items()}
        for label_idx in range(self.nb_topics):
            if label_idx not in window_counter.keys():
                window_counter[label_idx] = 0
        window_counter = {label_idx : label_count for label_idx, label_count in sorted(window_counter.items(), key=lambda item: item[1])}
        self.label_articles_counters.append(window_counter)

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
        window_dictionnary_f = cleanDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(
            texts=texts, labels=labels,
            dictionnary=window_dictionnary_f,
            nb_topics=self.nb_topics,
            labels_idx=self.labels_idx,
            **kwargs)
        self.models.append(model)
        return model, window_dictionnary_f

    # use lru cache for avoid useless call method in compareTopicSequentialy method . lazy way to avoid code refaction of the compareTopicSequentialy method
    @functools.lru_cache(maxsize=3)
    def calcule_similarity_topics_W_W(self, previous_window_idx: int, new_window_idx: int, ntop=100, soft=False,
                                      **kwargs):
        """

        @param ntop: number of top revelant words from the window engine
        @param previous_window_idx: number of the first window to compare
        @param new_window_idx: number of the second window to compare
        @param soft: take the brut score (1pt for one commun word) or the soft score (score delivered by the engine)
        @param kwargs:
        @return: List of percentage of similarity between the two windows (one value by topic)
        we decide to make the method compute similarity for all topics because of exclusive option that let us compute
         the similarity just for the words that are exclusive to the topic
         (no exclusive words are removed and not used during the similarity computing).
        """
        if previous_window_idx < 0 or new_window_idx < 0:
            raise CompareWindowsException("index should be positives")
        previousTopWordsTopics = self.getTopWordsTopics(previous_window_idx, ntop=ntop, **kwargs)
        # list of sets of top words per topics in jth window
        newTopWordsTopics = self.getTopWordsTopics(new_window_idx, ntop=ntop, **kwargs)
        similarities = []
        novelties = []
        habbits = []
        disappearances = []
        for topic_id in range(self.nb_topics):
            try:
                similarity_score, (novelties_topic, habbits_topic, disappearances_topic) = self.compute_similarity(
                    newTopWordsTopics[topic_id], previousTopWordsTopics[topic_id], soft=soft)
            except NoWordsClusterException:
                similarity_score = np.nan
                novelties_topic = []
                habbits_topic = []
                disappearances_topic = []
            similarities.append(similarity_score)
            novelties.append(novelties_topic)
            habbits.append(habbits_topic)
            disappearances.append(disappearances_topic)
        return similarities, (novelties, habbits, disappearances)

    def print_novelties(self, n_words_to_print=10, **kwargs):
        """
        print novleties (revelant words by labels ) from the last windows data linked to the last engine model
        @param n_words_to_print: number of maximum words to print
        @param kwargs:
        """
        if len(self) == 0:
            raise Exception("there is no engine load in self.models")
        elif len(self) == 1:
            print("no novelties yet...")
        else:
            last_window_idx = len(self) - 1
            similarities, (novelties, habbits, disapearrances) = self.calcule_similarity_topics_W_W(
                previous_window_idx=last_window_idx - 1, new_window_idx=last_window_idx, **kwargs)
            print(f"Information from new window calculate by the Sequantial Calculator : {id(self)}")
            for topic_id in range(self.nb_topics):
                similarity = similarities[topic_id]
                nov = list(novelties[topic_id])
                habb = list(habbits[topic_id])
                disap = list(disapearrances[topic_id])
                print(f"similarity score for the topic {self.labels_idx[topic_id]}  : {similarity} ")
                print("-" * 30)
                print(f" Novelties for the topic {self.labels_idx[topic_id]}  :")
                print(*nov[:n_words_to_print], sep='\n')
                print("-" * 30)
                print(f" Habbits for the topic {self.labels_idx[topic_id]}  :")
                print(*habb[:n_words_to_print], sep='\n')
                print("*" * 30)
                print(f" Disapearrances for the topic {self.labels_idx[topic_id]}  :")
                print(*disap[:n_words_to_print], sep='\n')
                print("*" * 30)
            print("#" * 30)
            print("#" * 30)


class GuidedSequantialLangageSimilarityCalculator(SupervisedSequantialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, labels_idx: list, seed: dict, memory_length: int = None):
        super().__init__(bad_words_args, labels_idx, memory_length)
        self.engine = Engine_module.GuidedEngine
        self.seed = seed
        if len(seed) != self.nb_topics:
            raise Exception("the number of topics in the seed need to be same as the number of topics that we"
                            "declared in the init function ")

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
        window_dictionnary_f = cleanDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, dictionnary=window_dictionnary_f, nb_topics=self.nb_topics, seed=self.seed, **kwargs)
        self.models.append(model)
        return model, window_dictionnary_f

    def getTopWordsTopic(self, topic_id, model_idx: int, ntop: int = 100 , remove_seed_words = True):

        # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
        model = self.models[model_idx]
        topWordsTopic = []
        mult = 0
        while len(topWordsTopic) < ntop:
            topWordsTopic = model.get_topic_terms(topic_id=topic_id, topn=ntop * mult)
            topic = self.labels_idx[topic_id]
            if remove_seed_words:
                words2keep = set(topWordsTopic.keys()).intersection(self.seed[topic])
                for word in words2keep:
                    del (topWordsTopic[word])
            mult += 1
        return {word: score for word, score in list(topWordsTopic.items())[:ntop]}


class LDASequentialSimilarityCalculatorFixed(NoSupervisedFixedSequantialLangageSimilarityCalculator):

    def __init__(self, nb_topics: int, bad_words_args: dict, memory_length: int = None):
        super().__init__(nb_topics, bad_words_args, memory_length)
        self.engine = Engine_module.LDA


class GuidedLDASequentialSimilarityCalculator(GuidedSequantialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, labels_idx: list, seed: dict, memory_length: int = None):
        super().__init__(bad_words_args, labels_idx, seed, memory_length)
        self.engine = Engine_module.GuidedLDA


class GuidedCoreXSequentialSimilarityCalculator(GuidedSequantialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, labels_idx: list, seed: dict, memory_length: int = None):
        super().__init__(bad_words_args, labels_idx, seed, memory_length)
        self.engine = Engine_module.GuidedCoreX


class CoreXSequentialSimilarityCalculatorFixed(NoSupervisedFixedSequantialLangageSimilarityCalculator):

    def __init__(self, nb_topics: int, bad_words_args: dict, memory_length: int = None):
        super().__init__(nb_topics, bad_words_args, memory_length)
        self.engine = Engine_module.CoreX


class LFIDFSequentialSimilarityCalculator(SupervisedSequantialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, labels_idx: list, memory_length: int = None):
        super().__init__(bad_words_args, labels_idx, memory_length)
        self.engine = Engine_module.LFIDF


class DynamicalCoreXSequentialSimilarityCalculator(MetaDynamicalSequencialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, nb_max_topics: int, nb_min_topics: int,
                 minimum_ratio_selection: float = 0.10, memory_length: int = None):
        super().__init__(bad_words_args, nb_max_topics, nb_min_topics,
                                                                     minimum_ratio_selection, memory_length)
        self.engine = Engine_module.CoreX


class DynamicalLDASequentialSimilarityCalculator(MetaDynamicalSequencialLangageSimilarityCalculator):

    def __init__(self, bad_words_args: dict, nb_max_topics: int, nb_min_topics: int,
                 minimum_ratio_selection: float = 0.10, memory_length: int = None):
        super().__init__(bad_words_args, nb_max_topics, nb_min_topics,
                                                                   minimum_ratio_selection, memory_length)
        self.engine = Engine_module.LDA
