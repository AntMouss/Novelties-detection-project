from typing import List
import math
import ijson
from data_processing import ProcessorText
import pandas as pd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ExperienceGen import ExperiencesMetadata, Thematic


class ArticlesDataset:

    def __init__(self , path ,start = 1615105271 , end = 1630999271, is_sorted = True, lang = 'fr'):

        self.lang = lang
        self.path = path
        self.article_keys_idx = {article['id'] : i for i , article in enumerate(self.articles)}
        self.article_keys_list = list(self.article_keys_idx.keys())
        if end is not None:
            self.end_date = end
        else:
            self.end_date = len(self)
        if start is not None:
            self.start_date = start
        else:
            self.start_date = 0
        if is_sorted:
            if self.check_sorted() == False:
                raise Exception("dataset is not sorted !!")

    @property
    def articles(self):
        f = open(self.path , 'r')
        return ijson.items(f , 'item')

    def __len__(self):

        return len(self.article_keys_idx)

    def verif(self , article):
        if ProcessorText.detectLanguage(article['title']) == self.lang:
            return True
        else:
            return False

    def __iter__(self):

         for article in self.articles:
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

    def __init__(self, delta = 1 , lookback = 100 , mode : chr = 's' , **kwargs):
        """

        @param delta: duration of each window
        @param look_back: number of article of the previous window that we agregate to the current window if look_back < 1
        the look back is relative (percentage of the last window)
        @param kwargs:
        """
        super().__init__(**kwargs)
        if mode == 's':
            self.features = ['text' , 'labels']
        else:
            self.features = ['text']
        self.lookback = lookback
        self.lookback_articles = []
        self.delta = delta * 3600
        self.window_idx = 0
        self.label_articles_counter = []


    def update_lookback_articles(self, window_articles):
        #transform relative look_back to absolute look_back
        if self.lookback < 1:
            self.lookback = math.ceil(self.lookback * len(window_articles))
        if self.lookback > len(window_articles):
            self.lookback_articles = self.lookback_articles[(self.lookback - len(window_articles)):] + window_articles
        else:
            self.lookback_articles = window_articles[-self.lookback:]


    def transform(self , articles):

        res = []
        for article in articles:
            res.append(article[feature] for feature in self.features)
        return zip(*res)


    def __iter__(self):

        ref_date_tmsp = self.start_date
        self.window_idx = 1
        window_articles = []
        for i , article in enumerate(self.articles):

            while ref_date_tmsp + self.delta < article['timeStamp'] and article['timeStamp'] < self.end_date:
                self.window_idx += 1
                ref_date_tmsp = ref_date_tmsp + self.delta
                window_articles.append(self.lookback_articles)
                yield  ref_date_tmsp , self.transform(window_articles)
                self.update_lookback_articles(window_articles)
                del window_articles
                window_articles = []
            if self.verif(article):
                continue
            else:
                window_articles.append(article)
        yield ref_date_tmsp , self.transform(window_articles)


class EditedTimeLineArticlesDataset(TimeLineArticlesDataset):
    def __init__(self , thematic : 'Thematic',metadata : 'ExperiencesMetadata',**kwargs ):
        super(EditedTimeLineArticlesDataset, self).__init__(**kwargs)
        self.thematic = thematic
        self.metadata = metadata
        self._ids_to_remove = self.thematic.article_ids

    def set_ids_to_remove(self, ids_to_remove: List):
        self._ids_to_remove = ids_to_remove


    def verif(self , article):

        if article['timeStamp'] < self.start_date:
            return False
        elif ProcessorText.detectLanguage(article['title']) != self.lang:
            return False
        elif self.between_boundaries()  and article['id'] in self._ids_to_remove:
            return False
        else:
            return True

    def between_boundaries(self):

        for range in self.metadata.ranges:
            if range[0] <= self.window_idx <= range[1]:
                return False
        return True




class WordsCounter:

    def __new__(cls, words , binary = False):
        counter = {}
        if binary:
            return {word : 1 for word in set(words)}
        else:
            for word in words:
                if word not in counter.keys():
                    counter[word] = 0
                counter[word] += 1
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




if __name__ == '__main__':

    path = '/home/mouss/data/final_database.json'
    test = TimeLineArticlesDataset(path, delta=24)
    see = []
    for el in test:
        see.append(len(el))

    test_it = iter(test)
    print(next(test_it))
    print(next(test_it))