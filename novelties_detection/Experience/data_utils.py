import pickle
from datetime import datetime
from typing import List, Tuple, Dict
import math
import ijson
import numpy as np
from novelties_detection.Collection.data_processing import FrenchTextPreProcessor
import pandas as pd
from collections import Counter


class Data:

    def save(self , path):
        with open(path , 'wb') as f:
            f.write(pickle.dumps(self.__dict__))

    @staticmethod
    def load(path):
        with open(path , 'rb') as f:
            data = pickle.load(f)
        return Data(**data)



class Thematic(Data):
    def __init__(self , name : str , label : str  ,date : datetime ,
                 article_ids : List , lifetime : str , window_idx_begin : int = None):
        self.window_idx_begin = window_idx_begin
        self.lifetime = lifetime
        self.article_ids = article_ids
        self.date = date
        self.label = label
        self.name = name

    def __len__(self):
        return len(self.article_ids)

class MacroThematic(Thematic):
    def __init__(self, name: str, label: str, date: datetime, article_ids: List, thematic_relative_count : List = None,
                 threshold : float = 0 ):
        """


        @param thematic_relative_count: percent of thematic articles belongs to the thematic in the window
        @param threshold: threshold that define if a window is injectable or not
        injectable means that we can inject this thematics at this window idx during the experience processs.
        """
        super().__init__(name, label, date, article_ids, "long")
        self.threshold = threshold
        self.thematic_relative_count = thematic_relative_count

    @property
    def good_windows(self):
        return [window_idx for window_idx ,  percent in enumerate(self.thematic_relative_count) if percent > self.threshold]

    def set_threshold(self , threshold):
        self.threshold = threshold



class MicroThematic(Thematic):
    def __init__(self, name: str, label: str, date: datetime, article_ids: List , window_idx_begin : int = None):
        super().__init__(name, label, date, article_ids, "short" , window_idx_begin)


class Alerte(Data):
    
    def __init__(self, topic_id, mean_distance : float , energy_distance : float, risk : float, windows : List, pvalue : float, is_normal : bool):
        self.energy_distance = energy_distance
        self.mean_distance = mean_distance
        self.is_normal = is_normal
        self.pvalue = pvalue
        self.windows = windows
        self.risk = risk
        self.topic_id = topic_id
        
        
    


class ExperiencesMetadata(Data):

    def __init__(self, name : str = None, ranges : List[List[Tuple]] = None, nb_windows : int = None , begin_window_idx : int = None,
                 cheat : bool = False , boost = 0):
        self.begin_window_idx = begin_window_idx
        self.boost = boost
        self.cheat = cheat
        self.nb_windows = nb_windows
        self.ranges = ranges
        self.name = name


class ExperiencesResult(Data):

    def __init__(self, metadata : ExperiencesMetadata, similarities_score : np.array, label_counter_w : List[dict] = None
                 , label_counter_ref : List[dict] = None):
        """
        
        @param metadata: metadata about experience in the timeline using for the instanciation ExperiencesResult
        contain : ranges boundaries index  , thematic name ...
        @param similarities_score: one array for similarities score of the original timeline
        one array for the edited timeline with thematic articles removed outside the experiences ranges
        @param label_counter_w: 
        @param label_counter_ref: 
        """
        self.label_counter_ref = label_counter_ref
        self.label_counter_w = label_counter_w
        self.similarities_score = similarities_score
        self.metadata = metadata


class ExperiencesResults(Data):

    def __init__(self , results : List[ExperiencesResult], info : dict):
        self.info = info
        self.results = results

    def __len__(self):
        """
        self.results contain the number of result object generate during experience process but one result object
        can contain many experiences.
        @return:
        """
        return len(self.results)



class ArticlesDataset:

    def __init__(self , path ,start = 1615105271 , end = 1630999271, is_sorted = True, lang = 'fr'):

        self.lang = lang
        self.path = path
        if start >= end:
            raise Exception("start should be inferior of end")
        else:
            self.start_date  = start
            self.end_date = end
        if is_sorted:
            if self.check_sorted() == False:
                raise Exception("dataset is not sorted !!")

    @property
    def articles(self):
        f = open(self.path , 'r')
        return ijson.items(f , 'item')


    def verif(self , article):
        if article["lang"] != self.lang:
            return False
        else:
            return True


    def __iter__(self):

         for article in self.articles:
            if article['timeStamp'] < self.start_date:
                 continue
            if article['timeStamp'] > self.end_date:
                break
            if self.verif(article):
                 yield article


    def check_sorted(self):

        tmsp_ref = 0
        for article in self.articles:
            if article['timeStamp'] >= tmsp_ref:
                tmsp_ref = article['timeStamp']
            else:
                return False
        return True


class TimeLineArticlesDataset(ArticlesDataset):


    def __init__(self, path, start = 1615105271, end = 1630999271, lang = 'fr', delta = 1, lookback = 0,
                 processor : FrenchTextPreProcessor = None, transform_fct: callable = None):
        """

        @param delta: duration of each window
        @param look_back: number of article of the previous window that we agregate to the current window if look_back < 1
        the look back is relative (percentage of the last window)
        @param kwargs:
        """
        super().__init__(path , start , end , lang=lang , is_sorted=True)
        self.transform = transform_fct
        self.processor = processor
        self.lookback = lookback
        self.lookback_articles = []
        self.delta = delta * 3600
        self.window_idx = 0
        self.label_articles_counter = []
        self.default_stop_window = math.ceil((self.end_date - self.start_date)/self.delta)
        self.stop_window = self.default_stop_window
        self.begin_window = 0

    def __len__(self):

        return self.stop_window - self.begin_window


    def stop_timeline(self, article_date):
            return (article_date > self.end_date or self.window_idx >= self.stop_window)

    def begin_timeline(self , article_date):
        return (article_date < self.start_date or self.window_idx >= self.stop_window)

    def set_stop_window(self , window_idx):
        if window_idx < 1:
            raise Exception("stop window index can't be inferior to one because we need at least one window")
        elif window_idx > self.default_stop_window :
            raise Exception("the stop windows index set is too large and exceed size of the dataset")
        self.stop_window = window_idx

    def set_begin_window(self , window_idx):
        if window_idx < 0:
            raise Exception("begin window index can't be negative")
        if window_idx > self.default_stop_window - 1:
            raise Exception("the begin windows index set is too large and need to be inferior of dataset size -1 because"
                            "we need at least one window for iteration")
        else:
            self.begin_window = window_idx


    def update_lookback_articles(self, window_articles):
        #transform relative look_back to absolute look_back
        if self.lookback< 0:
            raise Exception("lookback need to be superior to 0")
        elif self.lookback < 1:
            self.lookback = math.ceil(self.lookback * len(window_articles))

        if self.lookback > len(window_articles):
            self.lookback_articles = self.lookback_articles[(self.lookback - len(window_articles)):] + window_articles
        elif self.lookback == 0:
            self.lookback_articles = []
        else:
            self.lookback_articles = window_articles[-self.lookback:]


    def set_transform_function(self , transform_fct):
        self.transform = transform_fct

    def __iter__(self):

        ref_date_tmsp = int(self.start_date + self.begin_window  * self.delta)
        timeline_start_date = ref_date_tmsp
        self.window_idx = self.begin_window
        window_articles = []
        for _ , article in enumerate(self.articles):
            if article['timeStamp'] < timeline_start_date:
                 continue
            elif self.stop_timeline(article["timeStamp"]):
                break
            if article['timeStamp'] >= ref_date_tmsp + self.delta :

                #generation
                if self.transform is None:
                    yield ref_date_tmsp, window_articles
                else:
                    yield ref_date_tmsp, self.transform(window_articles, processor=self.processor)

                #incrementation
                self.window_idx += 1
                ref_date_tmsp = ref_date_tmsp + self.delta
                self.update_lookback_articles(window_articles)

                #re-initialization
                del window_articles
                window_articles = []
                window_articles += (self.lookback_articles)

            if self.verif(article):
                window_articles.append(article)
            else:
                continue
        if len(window_articles) != 0:
            if self.transform is None:
                yield ref_date_tmsp , window_articles
            else:
                yield ref_date_tmsp, self.transform(window_articles, processor=self.processor)


# i decide to make checkout for update_lookback fuction because
# it's not fair to remove thematic article of the look_back_articles (it's a methodology bias)
class EditedTimeLineArticlesDataset(TimeLineArticlesDataset):

    gap_between_first_range_entry_and_begin_window = 2
    gap_between_last_range_exit_and_stop_window = 3

    def __init__(self, thematics : List[Thematic], metadata : ExperiencesMetadata,optimize_mode : bool = True, **kwargs):
        super(EditedTimeLineArticlesDataset, self).__init__(**kwargs)
        self.thematics = thematics
        self.metadata = metadata
        self.thematics_ids_to_remove = [thematic.article_ids for thematic in self.thematics]
        tmp_last_window_ranges = []
        if optimize_mode:
            for ranges_thematic in self.metadata.ranges:
                for ranges in ranges_thematic:
                    tmp_last_window_ranges.append(ranges[1])
            stop_window_idx = int(np.max(tmp_last_window_ranges) + self.gap_between_last_range_exit_and_stop_window)
            self.set_stop_window(stop_window_idx)
            begin_window_idx = self.metadata.begin_window_idx
            self.set_begin_window(begin_window_idx)


    def set_ids_to_remove(self, ids_to_remove: List):
        self.thematics_ids_to_remove = ids_to_remove


    def update_lookback_articles(self, window_articles):
        #transform relative look_back to absolute look_back
        if self.lookback< 0:
            raise Exception("lookback need to be superior to 0")
        elif self.lookback < 1:
            self.lookback = math.ceil(self.lookback * len(window_articles))

        if self.lookback > len(window_articles):
            self.lookback_articles = self.lookback_articles[(self.lookback - len(window_articles)):] + window_articles
        elif self.lookback == 0:
            self.lookback_articles = []
        else:
            self.lookback_articles = window_articles[-self.lookback:]
        #check if we are at the left boundaries to remove thematics articles from the lookback_articles
        for i , ranges_thematic in enumerate(self.metadata.ranges):
            for range in ranges_thematic:
                if self.window_idx == range[1]:
                    self.clean_lookback_articles(i)


    def verif(self , article):

        if article["lang"] != self.lang:
            return False
        for thematic_ranges , thematic_ids_to_rem in zip(self.metadata.ranges , self.thematics_ids_to_remove):
            if not self.between_boundaries(thematic_ranges)  and article['id'] in thematic_ids_to_rem:
                # remove id for optimization
                thematic_ids_to_rem.remove(article["id"])
                return False
        return True


    def between_boundaries(self , ranges):

        for range in ranges:
            if range[0] <= self.window_idx < range[1]:
                return True
        return False


    def clean_lookback_articles(self , ranges_thematic_idx):
        for article in self.lookback_articles:
            if article["id"] in self.thematics_ids_to_remove[ranges_thematic_idx]:
                self.lookback_articles.remove(article)
                # for optimization we remove the id to the _ids_to_remove
                # list because we will not met this article anymore
                self.thematics_ids_to_remove[ranges_thematic_idx].remove(article["id"])


class WordsCounter:

    def __new__(cls, words : list , binary = False):
        """

        @param words: list of words
        @param binary: if True --> set 1 for each word is words else set number of word occurence in words
        """
        if binary:
            return {word : 1 for word in set(words)}
        else:
            counter = dict(Counter(words))
            return counter


class DocumentsWordsCounter:

    def __new__(cls, documents , binary = False):
        documents_words_counter = []
        for document in documents:
            counter = WordsCounter(document , binary=binary)
            documents_words_counter.append(counter)
        return pd.DataFrame(documents_words_counter).fillna(0)


class LabelsWordsCounter:

    def __new__(cls, documents , labels , binary = False):
        labels = pd.Series(labels)
        doc_words_counter = DocumentsWordsCounter(documents , binary=binary)
        return doc_words_counter.groupby(labels).sum()


class LabelsDictionnary:

    def __init__(self, labels : List[str] = None, texts  : List[list] = None):

        self.index = {}
        self.update_index(texts=texts , labels=labels)


    def update_index (self, texts, labels):

        for text  , label in zip(texts , labels):
            if label not in self.index.keys():
                self.index[label] = {}
            for word in text:
                if word not in self.index[label].keys():
                    self.index[label][word] = 0
                self.index[label][word] += 1

class LabelisedSample(Data):

    def __init__(self , labels : List[str] , samples_window : List[Dict] ):
        self.samples_window = samples_window
        self.labels = labels
        if len(samples_window) != len(labels):
            raise Exception("labels and samples_window haven't the same size")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.samples_window[item]

    @property
    def to_dataframe(self):
        dataframe = []
        for label  , samples_window_label in zip(self.labels , self.samples_window):
            for window_type , samples in samples_window_label.items():
                for value in samples:
                    dataframe.append({"label" : label , "type" : window_type , "similarity_value" : value})
        return pd.DataFrame(dataframe)



