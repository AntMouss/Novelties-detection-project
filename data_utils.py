from typing import List

from data_processing import addTimestampField , dateToDateTime
import math
import ijson
from data_processing import ProcessorText
import pandas as pd


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


    def __iter__(self):

         for article in self.articles:
             if ProcessorText.detectLanguage(article['title']) == 'fr':
                 yield article



    def check_sorted(self):

        tmsp_ref = 0
        for article in self.articles:
            if article['timeStamp'] >= tmsp_ref:
                tmsp_ref = article['timeStamp']
            else:
                return False
        return True


class SplitedArticlesDataset(ArticlesDataset):

    def __init__(self, delta = 1 , look_back = 100 , **kwargs):
        """

        @param delta: duration of each window
        @param look_back: number of article of the previous window that we agregate to the current window if look_back < 1
        the look back is relative (percentage of the last window)
        @param kwargs:
        """
        super().__init__(**kwargs)
        self.look_back = look_back
        self.look_back_articles = []
        self.delta = delta * 3600



    def __iter__(self):

        ref_date_tmsp = self.start_date
        n_window = 1
        window_articles = []
        for i ,  article in enumerate(self.articles):
            if article['timeStamp'] < self.start_date or ProcessorText.detectLanguage(article['title']) != 'fr':
                continue
            while ref_date_tmsp + n_window * self.delta < article['timeStamp'] and article['timeStamp'] < self.end_date:
                n_window += 1
                window_articles.append(self.look_back_articles)
                yield window_articles
                self.look_back_articles = window_articles[-self.look_back:]
                del window_articles
                window_articles = []
            window_articles.append(article)
        yield window_articles



class TimeWindows:

    def __init__(self,deltaTime,startTime=None,endTime=None) :
        """

        @param startTime: start of the time window (timestamp format suggered but can handdle string date)
        @param endTime: end of the time window (timestamp format suggered but can handdle string date)
        @param deltaTime: time intervall to split the window in many windows (timestamp format) in hour

        """
        if startTime is not None and endTime is not None:
            if isinstance(startTime , str):
                startTime = dateToDateTime(startTime , timeStamp=True)
            if isinstance(endTime , str):
                endTime = dateToDateTime(startTime , timeStamp=True)
            if startTime-endTime>0:
                change=startTime
                startTime=endTime
                endTime=change

        self.startTime= startTime
        self.endTime = endTime
        self.deltaTime = deltaTime * 3600



    def defineEndTimeAndNumberOfWindows (self):

        self.n_windows = abs(math.ceil((self.endTime - self.startTime) / self.deltaTime))
        self.endTime = self.startTime + self.deltaTime * self.n_windows



    def splittArticlesPerWindows(self, data):
        """

        :param data: data from dataBase with a 'date' field as timeStamp format but can handdle string format the data need to be sorted by date (ascending)
        return: list of list of id articles per time window sorted by time
        """

        # allArticleInWindow = [sorted(article_id , key=data[article_id]['date']) for article_id in data.keys() if self.startTime <= data[article_id]['date'] <= self.endTime]
        if self.startTime == None:
            self.startTime = data[0]['timeStamp']
        if self.endTime == None:
            self.endTime = data[-1]['timeStamp']
        self.defineEndTimeAndNumberOfWindows()

        splittedData = []
        g=0
        jref=0
        for i in range (self.n_windows-1):
            crossing = False
            start_date_window = self.startTime + i * self.deltaTime
            end_date_window = self.startTime + (i + 1) * self.deltaTime
            window_articles=[]
            for j in range(jref , len(data)):
                try:
                    if 'timeStamp' not in data[j].keys():
                        data[j] = addTimestampField(data[j]['id'])
                    date_article = data[j]['timeStamp']

                    if start_date_window <= date_article < end_date_window:
                        window_articles.append(data[j]['id'])
                        crossing = True
                    else:
                        if crossing:
                            jref = j
                            splittedData.append((end_date_window, window_articles))

                            break

                except TypeError as e:
                    print('empty field for date stimestamp')
                    g += 1
                    continue
                except Exception as er:
                    print('erere')
                        # note the ids lists are not sorted by date
                        # but in theory it's not necessary to sorted by date articles in the same window time

        return splittedData



    # we don't need this function anymore because data had to be sorted (ascending way)
    def findStartAndEndTime(self, data):
        """
        we can use this fonction if the database isn't sorted to found start article and end article in the data
        but this function is unuseful because we choose a more optimize process that needed sorted data in asceneding way
        @param data: data from dataBase with 'date' field
        """
        if self.endTime==0:
            maxTime=0
            for article_id in data.keys():
                if 'timeStamp' not in data[article_id].keys():
                    if len(data[article_id]['date'])==0:
                        continue
                    data[article_id] = addTimestampField(data[article_id])
                if data[article_id]['timeStamp'] > maxTime:
                    maxTime = data[article_id]['timeStamp']
            self. endTime=maxTime

        if self.startTime==0:
            minTime=1000000000000 #big value to be sure that the initial min value is superior to the real min value in the data
            for article_id in data.keys():
                if 'timeStamp' not in data[article_id].keys():
                    if len(data[article_id]['date'])==0:
                        continue
                    data[article_id] = addTimestampField(data[article_id])
                if data[article_id]['timeStamp'] < maxTime:
                    minTime = data[article_id]['timeStamp']
            self.startTime=minTime


class Formater:

    def __init__(self , texts : List[List[str]] = None , labels : List[str] = None , with_label = True):
        pass


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
    test = SplitedArticlesDataset(path , delta=24)
    see = []
    for el in test:
        see.append(len(el))

    test_it = iter(test)
    print(next(test_it))
    print(next(test_it))