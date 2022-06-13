import math
import random
from typing import List
from data_utils import TimeLineArticlesDataset
from gensim import corpora
from data_processing import filterDictionnary
import Engine
import numpy as np
from collections import Counter



def check_size(func):

    def wrapper(*args , **kwargs):
        if len(args[1]) == 0:
            raise Exception('documents empty , we can not process the sequence')
        return func(*args , **kwargs)
    return wrapper



class MetaSequencialLangageModeling:



    def __init__(self  , nb_topics=5):

        self.engine = Engine.Engine
        self.semi_filtred_dictionnary = corpora.Dictionary()
        self.nb_topics = nb_topics
        self.seedFileName = '_seed.json'
        self.bad_words = []
        self.info = {"engine_type" : self.engine.__name__ , 'nb_topics': self.nb_topics}
        self.res = {}
        self.models = []
        self .info_file  = 'info.json'
        self.resFileName = 'res.json'
        self.semi_dictionnaryFileName = '_semiDict'
        self.nb_windows = 0
        self.dateFile = 'date.json'
        self.date_window_idx = {}
        self.predefinedBadWords = ['...','commenter','réagir','envoyer','mail','partager' , 'publier' , 'lire' ,
                                   'journal' , "abonnez-vous" , "d'une" , "d'un" ,"mars" , "avril" , "mai" ,
                                   "juin" , "juillet" , "an" , "soir" , "mois", "lundi" , "mardi" , "mercredi"
            , "jeudi" , "vendredi" , "samedi" , "dimanche"]


    @check_size
    def treat_Window(self , data_window , **kwargs):
        pass


    def add_windows(self, data: TimeLineArticlesDataset, lookback=10, update_res=False, **kwargs):

        self.info['lookback'] = lookback
        rValue = random.Random()
        rValue.seed(37)
        for i, (end_date_window, data_windows) in (enumerate(data)):
            random_state = rValue.randint(1, 14340)
            kwargs["random_state"] = random_state
            print(f"numero of window: {i} -- random state: {random_state}")
            try:
                model, window_dictionnary = self.treat_Window(data_windows, **kwargs)
                # for bound window to the right glda model we use no_window
                no_window = i
                self.updateBadwords()
                if update_res:
                    self.updateResults(end_date_window, window_dictionnary, model, no_window)
                self.date_window_idx[end_date_window] = no_window
                self.models.append(model)
                self.nb_windows += 1

            except Exception as e:
                print(e)
                pass


    def updateResults(self, end_date ,  dictionnary_window : corpora.Dictionary, model : Engine, no_window: int , ntop : int = 100):


        topWordsTopics = self.getTopWordsTopics(model, ntop=ntop, exclusive=False)
        for word , word_id in dictionnary_window.token2id.items():
            if word not in self.res.keys():
                self.res[word] = {}
                self.res[word]['first'] = {'date': end_date} #we use end date of the window as date of the first appearance to the current world
                self.res[word]['appearances'] = []
            appearance={}
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
                #appearance['keyword'][topic_id]['relative_score'] = str(score/average_score_topic)


    def updateBadwords(self):

        no_above = 1 - 0.5 * (1 - (1 /( 1 + math.log10(self.semi_filtred_dictionnary.num_docs / 100))))
        abs_no_above = no_above * self.semi_filtred_dictionnary.num_docs
        rel_no_bellow = 0.00005
        abs_no_bellow = rel_no_bellow * self.semi_filtred_dictionnary.num_docs

        self.bad_words = [ word for id , word in self.semi_filtred_dictionnary.items() if abs_no_bellow > self.semi_filtred_dictionnary.dfs[id] or self.semi_filtred_dictionnary.dfs[id] > abs_no_above ]
        self.bad_words += self.predefinedBadWords



    def getTopWordsTopics(self, model : Engine = None, ntop : int  = 100, exclusive=False , **kwargs):
        """
        :param ntop: number of keywords that the model return by topic
        :param exclusive: if we want that the keywors being exclusive to the topic
        return: list of list id of words in the dictionnary, one list by gldaModel so one list by time intervall
        """
        topWordsTopics=[]
        for topic_id in range(model.nb_topics):
            topWordsTopic = self.getTopWordsTopic(topic_id, model, ntop , **kwargs)
            topWordsTopics.append(topWordsTopic)

        if exclusive == False:

            return topWordsTopics
        else:
            return self.exclusiveWordsPerTopics(topWordsTopics)



    def getTopWordsTopic(self, topic_id, model : Engine = None, ntop : int = 100 , **kwargs):

       # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
        topWords = model.get_topic_terms(topicid= topic_id, topn=ntop)
        topWordsTopic = {topWord[0] : topWord[1] for topWord in topWords}
        return topWordsTopic


    @staticmethod
    def exclusiveWordsPerTopics(topWordsTopics : List[dict]):
        topWordsTopics_tmp = [set(topWordsTopic.keys()) for topWordsTopic in topWordsTopics ]
        for i in range (len(topWordsTopics)):
            for j in range (i , len(topWordsTopics)):
                topWordsTopics_tmp[i] = topWordsTopics_tmp[i].difference(topWordsTopics_tmp[j])
                topWordsTopics_tmp[j] = topWordsTopics_tmp[j].difference(topWordsTopics_tmp[i])
        return [{word : topWordsTopics[i][word] for word in topWordsTopics_tmp[i]} for i in range (len(topWordsTopics))]



    def get_res(self):
        return self.res



class NoSupervisedSequantialLangagemodeling(MetaSequencialLangageModeling):

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

    def no_supervised_stuff(self):
        pass


class SupervisedSequantialLangagemodeling(MetaSequencialLangageModeling):

    def __init__(self ,labels_idx , **kwargs):
        super(SupervisedSequantialLangagemodeling, self).__init__(**kwargs)
        self.engine = Engine.SupervisedEngine
        self.labels_idx = labels_idx
        self.label_articles_counter = []

    @check_size
    def treat_Window(self, data_windows : tuple, **kwargs):
        texts , labels = data_windows
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        self.label_articles_counter.append(Counter(labels))
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts , labels=labels, **kwargs)
        return model, window_dictionnary_f


    def compareTopicSequentialy(self, topic_id, first_w=0, last_w=0, ntop=100, fixeWindow=False, back = 3 ,**kwargs):

        # we use thi condition to set the numero of the last window because by
        # default we want to compute similarity until the last window
        if last_w == 0:
            last_w = len(self.models)
        res = []
        if fixeWindow == True:
            for i in range(first_w + 1, last_w):
                res.append(self.calcule_similarity_topics_W_W('jaccard', ntop, first_w, i, **kwargs)[topic_id])
        else:
            for i in range(first_w+1, last_w):
                window_res = []
                for j in range(back):
                    try:
                        window_res.append(self.calcule_similarity_topics_W_W('jaccard', ntop, i - 1 - j, i , **kwargs)[topic_id])
                    except Exception as e:
                        break
                res.append(np.mean(window_res))
        return res


    def compareTopicsSequentialy(self , **kwargs):

        return [self.compareTopicSequentialy(topic_id , **kwargs) for topic_id in range(self.nb_topics)]


    def calcule_similarity_topics_W_W(self, distance='jaccard', ntop=100, ith_window=0, jth_window=1, soft=False,
                                      **kwargs):
        if ith_window < 0 or jth_window < 0:
            raise Exception("index should be positives")
        if distance == 'jaccard':
            ithTopWordsTopics = self.getTopWordsTopics(self.models[ith_window], ntop=ntop, **kwargs)
            # list of sets of top words per topics in jth window
            jthTopWordsTopics = self.getTopWordsTopics(self.models[jth_window], ntop=ntop, **kwargs)
            if soft == False:
                return [len(set(ithTopWordsTopics[topic_id].keys()).difference(set(jthTopWordsTopics[topic_id]))) / len(
                    jthTopWordsTopics) for topic_id in range(len(ithTopWordsTopics))]
            else:
                intersections = [(set(ithTopWordsTopics[topic_id].keys()).difference(set(jthTopWordsTopics[topic_id])))
                                 for topic_id in range(len(ithTopWordsTopics))]
                return [sum([jthTopWordsTopics[word]] for word in intersection) / len(jthTopWordsTopics) for
                        intersection in intersections]
        else:
            raise Exception('for the moment there is just jaccard distance')



class GuidedSequantialLangagemodeling(SupervisedSequantialLangagemodeling):

    def __init__(self , seed : dict ,  **kwargs):
        super(GuidedSequantialLangagemodeling, self).__init__(**kwargs)
        self.engine = Engine.GuidedEngine
        self.seed = seed
        self.table = {idx : label for idx , label in enumerate(seed.keys())}

    @check_size
    def treat_Window(self, data_windows: tuple, **kwargs):
        texts, labels = data_windows
        print(f"size documents: {len(texts)} ")
        print("-" * 30)
        self.label_articles_counter.append(Counter(labels))
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, **kwargs)

        return model, window_dictionnary_f


    def getTopWordsTopic(self, topic_id, model : Engine = None, ntop : int = 100 , remove_seed_words : bool = True):

        # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
        topWordsTopic = model.get_topic_terms(topicid= topic_id, topn=ntop)
        topic = self.table[topic_id]
        if remove_seed_words:
            words2keep = set(topWordsTopic.keys()).intersection(self.seed[topic])
            for word in words2keep:
                del (topWordsTopic[word])
        return topWordsTopic


class LDASequantialModeling(NoSupervisedSequantialLangagemodeling):

    def __init__(self , **kwargs):
        super(LDASequantialModeling, self).__init__(**kwargs)
        self.engine = Engine.LDA

    @check_size
    def treat_Window(self, texts, **kwargs):
        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts, dictionnary=window_dictionnary_f, **kwargs)

        return model, window_dictionnary_f



class GuidedLDASequentialModeling(GuidedSequantialLangagemodeling):

    def __init__(self , **kwargs):
        super(GuidedLDASequentialModeling, self).__init__(**kwargs)
        self.engine = Engine.GuidedLDA

    @check_size
    def treat_Window(self, data_window, **kwargs):
        texts , labels = data_window
        window_dictionnary = corpora.Dictionary(texts)
        self.label_articles_counter.append(Counter(labels))
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        # we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary, bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts , dictionnary=window_dictionnary_f, **kwargs)

        return model, window_dictionnary_f


class GuidedCoreXSequentialModeling(GuidedSequantialLangagemodeling):

    def __init__(self , **kwargs):
        super(GuidedCoreXSequentialModeling, self).__init__(**kwargs)
        self.engine = Engine.GuidedCoreX


class NoSuperviedCoreXSequentialModeling(NoSupervisedSequantialLangagemodeling):
    def __init__(self, **kwargs):
        super(NoSuperviedCoreXSequentialModeling, self).__init__(**kwargs)
        self.engine = Engine.CoreX


class SupervisedCoreXSequentialModeling(SupervisedSequantialLangagemodeling):

    def __init__(self , **kwargs):
        super(SupervisedCoreXSequentialModeling, self).__init__(**kwargs)
        self.engine = Engine.SupervisedCoreX


class LFIDFSequentialModeling(SupervisedSequantialLangagemodeling):

    def __init__(self , **kwargs):
        super(LFIDFSequentialModeling, self).__init__(**kwargs)
        self.engine = Engine.LFIDF




