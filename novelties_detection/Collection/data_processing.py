import math
import os
import json
from datetime import datetime
import locale
from urllib.parse import urlparse
import requests
from nltk.tokenize import word_tokenize
from langdetect import detect
import simplemma
import stopwordsiso
import copy
from bs4 import BeautifulSoup
import functools
import signal
from contextlib import contextmanager
import wrapt_timeout_decorator

@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError



def fileToObject(filepath):
    with open(filepath, "r") as f:
        JsonFile = json.load(f)

    return JsonFile




def dateToDateTime(stringDate , timeStamp=True):

    format1 = {}
    format1['fmt'] = '%a, %d %B %Y %X %Z'
    format1['locate'] = 'C'
    format2 = {}
    format2['fmt'] = '%a, %d %B %Y %X %z'
    format2['locate'] = 'C'
    format3 = {}
    format3['fmt'] = '%A %d %B %Y, %H:%M'
    format3['locate'] = 'fr_FR.UTF-8'
    format4 = {}
    format4['fmt'] = '%Y-%m-%d %X'
    format4['locate'] = 'C'
    format5={}
    format5['fmt'] = '%a, %d %b %Y %X %z'
    format5['locate'] = 'C'
    format6 = {}
    format6['fmt'] = '%a, %d %b %Y %X %Z'
    format6['locate'] = 'C'
    listOfFormat = [format1, format2, format3, format4, format5, format6]

    for forma in listOfFormat:
        try:
            locale.setlocale(locale.LC_ALL, forma['locate'])
            date = datetime.strptime(stringDate, forma['fmt'])
            if timeStamp:
                tstpdate = datetime.timestamp(date)
                return tstpdate
            else:
                return date
        except ValueError:
            pass

def addTimestampField(article):

    article['timeStamp'] = dateToDateTime(article['date'])
    return article


def getDataPathList(ROOT_FOLDER, targetFileName):
    """


    :param ROOT_FOLDER:
    :param targetFileName:
    :return:
    """
    listpath = []

    try:

        f = os.walk(ROOT_FOLDER, topdown=False)
        for root, dirs, files in f:
            # listpath.append(os.path.join(root,name) for name in files if name=="data.json")
            for name in files:
                if name == targetFileName:
                    listpath.append(os.path.join(root, name))
    except OSError as e:
        print(f"le fichier ou répertoire n'existe pas: {e}")

    return listpath




class MergerDataJson:
    """
    we use this class for merging all the "data.json" file that contain information about articles inside our
     Root Folder(folder that receive the collected articles)

    note: i think it was not really useless to create a class just for merge data, a function is enough

    """

    def __init__(self, targetFileName, ROOT_FOLDER_DESTINATION, fileDatabaseName, ROOT_FOLDER_SOURCE):
        """

        :param targetFileName: the file that we want to merge, in our case  "data.json"
        :param ROOT_FOLDER_DESTINATION: the root Folder that contain our data
        :param fileDatabaseName: name of our json file that contain all our 'data.json'
        :param ROOT_FOLDER_SOURCE: the folder where we want save the database.json
        """
        self.ROOT_FOLDER = ROOT_FOLDER_SOURCE
        self.ROOT_FOLDER_DESTINATION = ROOT_FOLDER_DESTINATION
        self.fileDatabaseName = fileDatabaseName
        self.pathDatabase = os.path.join(self.ROOT_FOLDER_DESTINATION, self.fileDatabaseName)
        self.targetFileName = targetFileName
        self.mainKeysData = ['title', 'url', 'label', 'date', 'rss_media', 'images', 'domainName', 'folderPath']
        self.listpath = getDataPathList(self.ROOT_FOLDER, self.targetFileName)
        print(f"{len(self.listpath)} articles")
        self.total_count = len(self.listpath) // 2000

    def mergeData(self, allData=False):

        if not os.path.exists(self.ROOT_FOLDER_DESTINATION):
            os.makedirs(self.ROOT_FOLDER_DESTINATION)

        dataArticles = []
        nb_articles = 0
        count = 0
        for path in self.listpath:
            with open(path, "r") as fi:
                data = json.load(fi)
                # the id corresponds to the article folder name that is the hash of the article url
                id = os.path.split(os.path.dirname(path))[1]
                # for generate a dataBase with the useful informations for the interface
                if allData:
                    # dataArticles.append(data)
                    # we decide to save our data as a dictionnary with one key by article
                    # because it's easyer for the interface
                    data['id'] = id
                    dataArticles.append(data)
                    nb_articles += 1
                    if nb_articles % 2000 == 0 :
                        count += 1
                        print(f"{count}/{self.total_count}" )


                else:
                    try:
                        data2 = {key: data[key] for key in self.mainKeysData}
                        # dataArticles.append(data2)
                        dataArticles[id] = data2
                    except Exception as e:
                        pass

        with open(self.pathDatabase, "w") as fo:
            fo.write(json.dumps(dataArticles))


class imputerData:
    """
    we can use this class for fetch data that we didn't catch during the collect
    in particular html page because of enconding error
    also we can use this class to add useful feed for the interface like domainName or folderPath
    we can use folderPath for return html and image from our root data source to our front interface

    """

    HTML_HEADER = "<html><body>"
    HTML_TAIL = "</body></html>"

    def __init__(self, ROOT_FOLDER_SOURCE, targetFileName, htmlFileName, fileFullDataName):
        """

        :param ROOT_FOLDER_SOURCE: the Root folder of our data
        :param targetFileName: name of the file that we want to process in our case it's 'data.json'
        :param htmlFileName: name of the html file in our case it's 'news.html'
        :param fileFullDataName: file name that we want save in the same folder that the original 'data.json' with other
        name and new fields (we give new name avoid confusion and we make new file to have a copy)
        """
        self.ROOT_FOLDER_SOURCE = ROOT_FOLDER_SOURCE
        self.fileFullDataName = fileFullDataName
        self.pathFullData = os.path.join(self.ROOT_FOLDER_SOURCE, self.fileFullDataName)
        self.targetFileName = targetFileName
        self.htmlFileName = htmlFileName
        self.listpath = getDataPathList(self.ROOT_FOLDER_SOURCE, self.targetFileName)
        print("bonjour")

    def imputeHTML(self):
        """

        fetch and save html in the right folder
        if the html file is empty len(html)==0
        """

        fullFileList = getDataPathList(self.ROOT_FOLDER_SOURCE, self.fileFullDataName)
        for path in fullFileList:
            with open(path, "r") as fi:
                data = json.load(fi)
            htmlPath = os.path.join(data['folderPath'], self.htmlFileName)
            try:
                with open(htmlPath, 'r') as fh:
                    html = fh.read()
            except:
                # html file does not exist
                html = ''
                pass

            if len(html) == 0:
                r = requests.get(data["url"])
                soup = BeautifulSoup(r.text, "lxml")  # Parse HTML
                article = soup.find("article")
                # on va remplacer le chemin dans notre javascript
                # images=article.findAll('img')
                # if replaceImagePath:
                #     for image in images
                #     image['src']=os.path.join(data['folderPath'],artic['src'])
                try:
                    with open(os.path.join(data['folderPath'], self.htmlFileName), "w") as f:
                        html = self.HTML_HEADER + str(article) + self.HTML_TAIL
                        f.write(html)
                except Exception as e:
                    #when we caugth encoding exception we write the html file with 'utf-8' encoding
                    # problem: we don't replace the img path , we need to had this features
                    with open(os.path.join(data['folderPath'], self.htmlFileName), "w", encoding="utf-8") as f:
                        f.write(html)
                    pass

    def addDomainAndPath(self, replaceImagePath=True):
        """
        we use this function for adding path and domain Name
        and replace the labels "actuality , no label" by the label 'general'

        """

        for path in self.listpath:
            with open(path, "r") as fi:
                data = json.load(fi)
                if len(data["label"]) == 0:
                    data["label"].append('general')
                if data["label"][0] in ['no label', 'actuality']:
                    data["label"][0] = 'general'

                data["folderPath"] = os.path.dirname(path)
                data["domainName"] = urlparse(data["url"]).netloc
                # dataFile["domainImagePath"]=
                # on vérifie que le html n'est pas manquant ou vide
                despath = os.path.join(data['folderPath'], self.fileFullDataName)
                with open(despath, 'w') as fo:
                    fo.write(json.dumps(data))



class MetaTextPreProcessor:

    default_undesirable_characters = ['/', '=', '#', '&']
    default_undesirable_words = []

    def __init__(self, lang  : str = "" , long_lang  : str = "",undesirable_words : list = None,
                 undesirable_characters : list = None, max_word_size : int = 18,
                 min_word_size : int = 3 , lemmatize : bool = True , remove_stop_words : bool = True ,
                 remove_small_words : bool = True , remove_numbers : bool = True):
        self.remove_numbers = remove_numbers
        self.remove_small_words = remove_small_words
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize
        self.long_lang = long_lang
        self.lang = lang
        self.min_word_size = min_word_size
        self.max_word_size = max_word_size
        if undesirable_words is None:
            undesirable_words = []
        if undesirable_characters is None:
            undesirable_characters = []
        self.undesirable_words = undesirable_words + self.default_undesirable_words
        self.undesirable_characters  = undesirable_characters + self.default_undesirable_characters

    @property
    @functools.lru_cache(maxsize=1)
    def langData(self):
        return simplemma.load_data(self.lang)

    @property
    @functools.lru_cache(maxsize=1)
    def stop_words(self):
        return stopwordsiso.stopwords(self.lang)


    @staticmethod
    def detectLanguage(text):
        '''
        Returns the most probable language in which the text was written
        :param text:
        :return:
        '''
        lang = ''
        try:
            if text.strip():
                try:
                    lang = detect(text.strip())
                except:
                    pass
            else:
                lang = 'en'
            return lang
        except:
            pass

    def tokenizeText(self, text: str):
        return word_tokenize(text, language=self.long_lang)
        # return [word for word in text.lower().split()]

    def lemmatize_word(self, word):
        return simplemma.lemmatize(word, self.langData)

    @staticmethod
    def check_lang(text : str , lang : str):
        tmp_lang = MetaTextPreProcessor.detectLanguage(text)
        if tmp_lang == lang:
            return True
        else:
            return False


    def processWord(self, word):

        if self.remove_small_words:
            if len(word) < self.min_word_size:
                return None
        if self.remove_numbers and word.isdigit():
            return None
            # 'aux' is lemmatize as 'à les' but 'à les' isn't a stop words so we did filterstopwords twice before and after lemmatization
        if self.remove_stop_words:
            if word in self.stop_words:
                return None
        if self.lemmatize:
            word= self.lemmatize_word(word)
        if self.remove_stop_words:
            if word in self.stop_words:
                return None
        if len(word)> self.max_word_size:
            return None
        if word.lower() in self.undesirable_words:
            return None
        for char in self.undesirable_characters:
            if char in word:
                return None

        return word.lower()

    @functools.lru_cache(maxsize=100)
    @wrapt_timeout_decorator.timeout(120)
    def preprocessText(self, text: str):

        textProcessed = []
        if MetaTextPreProcessor.check_lang(text , self.lang):
            text = self.tokenizeText(text)
            for word in text:
                word = self.processWord(word)
                # to remove None type words
                if isinstance(word, str):
                    textProcessed.append(word)
            return textProcessed
        else:
            return None


    def preprocessTexts(self, texts):

        textsProcessed = []
        for i, text in enumerate(texts):
            textprocessed = self.preprocessText(text)
            # to remove None type text
            if isinstance(textprocessed, list):
                textsProcessed.append(textprocessed)
        return textsProcessed


class FrenchTextPreProcessor(MetaTextPreProcessor):
    """
    we use this class for process french text
    """
    # we load the data for french lemmatization just one time because we will just treat french text

    default_undesirable_words = ['commenter', 'réagir', 'envoyer', 'mail', 'partager', 'facebook', 'twitter', 'commenter',
                              'cliquer', 'publier', 'commentaire', 'lire', 'journal']
    def __init__(self , **kwargs):
        super().__init__("fr" , "french" , **kwargs )



class EnglishTextPreProcessor(MetaTextPreProcessor):

    default_undesirable_words = ['subscribe', 'reaction', 'send', 'mail', 'partager', 'facebook', 'twitter',
                                 'like','click', 'publish', 'comment', 'read', 'paper']
    def __init__(self , **kwargs):
        super().__init__("en" , "english" , **kwargs )



def absoluteThresholding(absolute_value , **kwargs):
    return absolute_value

def linearThresholding(relative_value , nb_docs):
    return relative_value * nb_docs

def exponentialThresholding(nb_docs, limit = 0.6, pente = 100):
    if limit > 1:
        raise Exception("limit can't be superior to 1")
    limit = 1 - limit
    # i do this while method to avoid relative_value negative
    while pente > nb_docs:
        pente = pente // 2
    relative_value = 1 - limit * (1 - (1 / (1 + math.log10(nb_docs / pente))))
    return relative_value * nb_docs



def cleanDictionnary(dictionnary, filterStopwords=False, filterSmallWord=False, bad_ids=None, bad_words=None):
    """
    clean dictionnary of words that we don't want in the treatment dictionnary
    @param dictionnary: dictionnary to clean
    @param filterStopwords: stop words to remove
    @param filterSmallWord: remove words with length < 3
    @param bad_ids: remove words by id
    @param bad_words: remove words by value
    @return: clean dictionnary


    """
    dictionnary_output = copy.deepcopy(dictionnary)
    if filterStopwords:
        stop_ids = []
        setStopwords = stopwordsiso.stopwords('fr')
        for stopword in setStopwords:
            try:
                stop_id = dictionnary_output.token2id[stopword]
                stop_ids.append(stop_id)
            except KeyError as e:
                pass

                # other option to add remove number and do the loops on the dictionnary element instead of stopwordsiso list elements

        dictionnary_output.filter_tokens(bad_ids=stop_ids)


    if filterSmallWord:
        small_ids = [dictionnary_output.token2id[word] for word in dictionnary_output.token2id if len(word) < 3]
        dictionnary_output.filter_tokens(bad_ids=small_ids)


    if bad_words is not None:
        bad_ids=[]
        for word in bad_words:
            try:
                bad_ids.append(dictionnary_output.token2id[word])
            except KeyError:
                pass

    if bad_ids is not None:
        dictionnary_output.filter_tokens(bad_ids=bad_ids)

    return dictionnary_output



def transformU(articles, processor : FrenchTextPreProcessor = None, process_already_done = True):
    """
    change format of the articles list to being ready to use
    @param articles: articles in list of dictionnary format
    @param processor:
    @param process_already_done:
    @return:
    """
    texts = []

    for article in articles:
        if process_already_done:
            text = article['process_text']
        else:
            text =processor.preprocessText(article['text'])
        texts.append(text)
    return texts


def transformS(articles, processor : FrenchTextPreProcessor = None, process_already_done = True):

    res = []
    for article in articles:
        if process_already_done:
            res.append((article['process_text'] , article['label'][0]))
        else:
            res.append((processor.preprocessText(article['text']) , article['label'][0]))
    texts , labels = list(zip(*res))
    labels = list(labels)
    return texts , labels


