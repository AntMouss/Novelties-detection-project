import math
import os
import json
from datetime import datetime
import locale
from urllib.parse import urlparse
from tqdm import tqdm
import requests
from nltk.tokenize import word_tokenize
from langdetect import detect
import simplemma
import stopwordsiso
import copy
import re
from bs4 import BeautifulSoup
from html2text import HTML2Text
import sys
import signal
from threading import Thread
import functools



def timeout(timeout):
    """
    use this function for time out certain function like"process , because signal.SIGALRM doesn't work on window
    @param timeout:
    @return:
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco



def fileToObject(filepath):
    with open(filepath, "r") as f:
        JsonFile = json.load(f)

    return JsonFile


def stringToList(string):
    return string.split()


def addID(articles):
    dictArticlesWithID = {}  # dictionnary that contain articles with the added ID
    for i in range(len(articles)):
        dictArticlesWithID["article_" + str(i)] = articles[i]
    return dictArticlesWithID


def addDomainName(articles):
    for article in articles:
        article['domainName'] = urlparse(article['url']).netloc

    return articles





def findArticle(listOfWord, JsonOfArticles):
    foundArticles = []
    for word in listOfWord:
        for Article in JsonOfArticles:
            if word in Article["title"]:
                foundArticles.append(Article)

    sorted_Articles = sorted(foundArticles, key=lambda x: x['tstpdate'])

    listete = []
    for article in sorted_Articles:
        listete.append(article['tstpdate'] - 1620290000.0)

    return sorted_Articles


def dateToDatetimeInArticles(articles):


    return [dateToDateTime(article['date']) for article in articles]




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


def FileToArticleList(pathname, string):
    jsonFile = fileToObject(pathname)
    jsonFile = dateToDatetimeInArticles(jsonFile)
    listOfWords = stringToList(string)

    return findArticle(listOfWords, jsonFile)


def imputeEmptyLabel(filePath):
    data = fileToObject(filePath)
    for key in data.keys():
        if len(data[key]['label']) == 0:
            data[key]['label'] += ['no label']
    with open(filePath, 'w') as f:
        f.write(json.dumps(data))


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



def makeTable(label, domain, dataBaseFile, tableFile):
    # take dataBase file json with id index and domain name include in keys
    tabDomain = []
    tabLabel = {}
    tableId = {}
    with open(dataBaseFile, "r") as fi:
        data = json.load(fi)
        a = 0
        try:
            for id in data.keys():

                domainKey = data[id][domain]
                labelKey = data[id][label]
                if len(labelKey) == 0:
                    labelKey.append('no label')
                # this domain Name never pasted
                if domainKey not in tabDomain:
                    tableId[domainKey] = {}
                    tabLabel[domainKey] = []
                    tabDomain.append(domainKey)
                if labelKey[0] not in tabLabel[domainKey]:
                    tableId[domainKey][labelKey[0]] = []
                    tabLabel[domainKey].append(labelKey[0])

                tableId[domainKey][labelKey[0]].append(id)
                a = a + 1


        except Exception as e:
            print(e)

    with open(tableFile, 'w') as fo:
        fo.write(json.dumps(tableId))


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




class ProcessorText:
    """
    we use this class for text processing
    """
    def __init__(self):
        # we load the data for french lemmatization just one time because we will just treat french text
        self.frenchData  = simplemma.load_data('fr')
        self.setStopWords  = stopwordsiso.stopwords('fr')
        self.specialWords1 = ['commenter','réagir','envoyer','mail','partager','facebook','twitter','commenter' , 'cliquer' , 'publier' ,'commentaire' , 'lire' , 'journal']
        self.specialWords2 = ['commenter','réagir','envoyer','mail','partager' , 'publier' , 'lire' , 'journal' , "abonnez-vous"]
        self.specialWords3 = []
        self.undesirableChar = ['/' , '=' , '#' , '&']

    def tokenizeText(self, text):
        return word_tokenize(text , language='french')
        # return [word for word in text.lower().split()]

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

    def processWord(self, word, langData , lemmatize=True , filtreStopWords=True , filtreSmallWords=True , filtreNumber = True):


        if filtreSmallWords:
            if len(word)<3:
                return None
        if filtreNumber:
            try:
                int(word[0])
                return  None
            except:
                pass
            try:
                int(word[1])
                return None
            except:
                pass
            # 'aux' is lemmatize as 'à les' but 'à les' isn't a stop words so we did filterstopwords twice before and after lemmatization
        if filtreStopWords:
            if word in self.setStopWords:
                return None
        if lemmatize:
            word=simplemma.lemmatize(word, langData)
        if filtreStopWords:
            if word in self.setStopWords:
                return None
        if len(word)>18:
            return None
        if word.lower() in self.specialWords2:
            return None
        for char in self.undesirableChar:
            if char in word:
                return None

        return word.lower()

    @functools.lru_cache(maxsize=100)
    def processText(self, text, lematize=True ,filtreStopWords=True , filtreSmallWords=True , filtreNumber = True ):


        lang = ProcessorText.detectLanguage(text)
        text = self.tokenizeText(text)
        textProcessed=[]

        if lang == 'fr':
            # return [self.processWord(word, self.frenchData, lemmatize=lematize, filtreStopWords=filtreStopWords,
            #                          filtreSmallWords=filtreSmallWords) for word in text]
            for word in text:
                word = self.processWord(word, self.frenchData, lemmatize=lematize, filtreStopWords=filtreStopWords,
                                     filtreSmallWords=filtreSmallWords , filtreNumber = filtreNumber)
                # to remove None type words
                if isinstance(word , str):
                    textProcessed.append(word)
            return textProcessed
        else:
            return []



    def processTexts(self, texts, lematize=True , filtreStopWords=True , filtreSmallWords=True , filtreNumber = True ):

        textsProcessed = []
        for i, text in enumerate(texts):
            textprocessed = self.processText(text , lematize=lematize , filtreStopWords=filtreStopWords , filtreSmallWords=filtreSmallWords , filtreNumber=filtreNumber)
            # to remove None type text
            if isinstance(textprocessed , list):
                textsProcessed.append(textprocessed)

        return textsProcessed



def absoluteThresholding(absolute_value , **kwargs):
    return absolute_value

def linearThresholding(relative_value , nb_docs):
    return relative_value * nb_docs

def exponentialThresholding(nb_docs, limit = 0.5, pente = 100):
    if limit > 1:
        raise Exception("limit can't be superior to 1")
    # i do this while method to avoid relative_value negative
    while pente > nb_docs:
        pente = pente // 2
    relative_value = 1 - limit * (1 - (1 / (1 + math.log10(nb_docs / pente))))
    return relative_value * nb_docs



def filterDictionnary(dictionnary, filterStopwords=False, total_filter=False, filterNumber=False, filterSmallWord=False, keep_tokens=None, bad_ids=None, bad_words=None):
    """

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

    # keep the default argument
    if total_filter == True:
        # we filtre frequent and infrequent words

        no_above=1-(0.5 * (dictionnary_output.num_docs / 10000))
        rel_no_bellow=0.001
        abs_no_bellow= rel_no_bellow * dictionnary_output.num_docs
        dictionnary_output.filter_extremes(no_below=abs_no_bellow, no_above=no_above, keep_tokens=keep_tokens)


    if filterSmallWord:
        small_ids = [dictionnary_output.token2id[word] for word in dictionnary_output.token2id if len(word) < 3]
        # for word in dictionnary.token2id:
        #     if len(word)<3:
        #         idToRmv.append()
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




def extract_text( articlePage, remove_list  , clean = False):

    h = HTML2Text()
    h.ignore_links = True
    if clean:
        for tag in remove_list:
            try:
                if len(tag["id"]) == 0:
                    if len(tag["class"]) == 0:
                        for element in articlePage.find_all(tag["tag"]):
                            element.decompose()
                    else:
                        for i, element in enumerate(articlePage.find_all(tag["tag"], class_=tag["class"])):
                            element.decompose()
                else:
                    articlePage.find(id=tag["id"]).decompose()
            except AttributeError:
                pass
            except Exception as e:
                pass
    text = h.handle(str(articlePage))  # Get only the text in the tag article
    text = re.sub("\n|#", " ", text)  # Clean the text, cette ligne ne fonctionne plus et détruit le code
    text = re.sub("_|>|<|!\[.*?\]|\(https:.*?\)", "", text)
    return text



def imputeFieldAfterCollect (rootDatabase, configPath,fileName = 'data.json', htmlFileName ='news.html'):

    er=0
    HTML_HEADER = '<html><body>'
    HTML_TAIL = "</body></html>"
    #open config
    with open (configPath , 'r') as fc:
        config = json.load(fc)
    with open (config['rss_feed_config_file'] , 'r') as frss:
        rss = json.load(frss)

    #search all data.json path
    listOfPath = getDataPathList(rootDatabase , fileName)
    total_count = len(listOfPath) // 2000
    nb_articles = 0
    count = 0

    for path in listOfPath:
        with open(path , 'r') as f:
            article = json.load(f)
        if len(article['label']) != 0:
            for label in article['label']:
                if label in ['actuality' , 'no label']:
                    label = 'general'
        else:
            article['label'].append('general')
        try:
            if 'cleansed_text' not in article.keys():

                link = article['url']
                dirPath = os.path.dirname(path)
                htmlPath = os.path.join(dirPath, htmlFileName)
                # page are encoded in utf-8 or default encoding so we need to handle the 2 encoding when we open page
                try:
                    with open(htmlPath, 'r') as f:
                        article_html = f.read()
                        article_html = BeautifulSoup(article_html, 'lxml')
                        if article_html.text != 'None  ':
                            article['htmlCollected'] = True
                            remove_list_rss = []
                            for j in range(len(rss['rss_feed_url'])):
                                if rss['rss_feed_url'][j]['url'] == article['rss']:
                                    remove_list_rss += rss['rss_feed_url'][j]['toRemove']
                                    break
                            remove_list = rss['global'] + remove_list_rss
                            article['cleansed_text'] = extract_text(article_html, remove_list, clean=True)
                            with open(path, 'w') as f:
                                f.write(json.dumps(article))

                        else:
                            article['htmlCollected'] = False
                            article['cleansed_text'] = ''
                            article['text'] = ''

                    nb_articles += 1
                    if nb_articles % 2000 == 0:
                        count += 1
                        print((f"{count}/{total_count}"))



                except UnicodeDecodeError:
                    with open(htmlPath, 'r' , encoding='UTF-8') as f:
                        article_html = f.read()
                    article_html = BeautifulSoup(article_html, 'lxml')
                    article['htmlCollected'] = True
                    remove_list_rss = []
                    for j in range(len(rss['rss_feed_url'])):
                        if rss['rss_feed_url'][j]['url'] == article['rss']:
                            remove_list_rss += rss['rss_feed_url'][j]['toRemove']
                            break
                    remove_list = rss['global'] + remove_list_rss
                    article['cleansed_text'] = extract_text(article_html, remove_list, clean=True)
                    with open(path, 'w') as f:
                        f.write(json.dumps(article))
                    nb_articles += 1

                    if nb_articles % 2000 == 0:
                        count += 1
                        print((f"{count}/{total_count}"))


                except FileNotFoundError as ef:
                    r = requests.get(link, timeout=4)
                    if r.status_code != 200:
                        continue
                    soup = BeautifulSoup(r.text, 'lxml')
                    article_html = soup.find("article")

                    if article_html is None:
                        article['htmlCollected'] = False
                        article['cleansed_text'] = ''
                        article['text'] = ''
                        pass
                    else:
                        article['htmlCollected'] = True
                        with open(htmlPath, 'w', encoding='UTF-8') as fh:
                            page = HTML_HEADER + str(article_html) + HTML_TAIL
                            fh.write(page)
                        remove_list_rss = []
                        for j in range(len(rss['rss_feed_url'])):
                            if rss['rss_feed_url'][j]['url'] == article['rss']:
                                remove_list_rss += rss['rss_feed_url'][j]['toRemove']
                                break
                        remove_list = rss['global'] + remove_list_rss
                        article['cleansed_text'] = extract_text(article_html, remove_list, clean=True)
                    with open(path, 'w') as f:
                        f.write(json.dumps(article))
                    print(article['url'])
                    nb_articles += 1
                    if nb_articles % 2000 == 0:
                        count += 1
                        print((f"{count}/{total_count}"))



            #we find the according rss url by this way that is not conventionnal




        except Exception as e:
            exc_tb = sys.exc_info()[2]
            exc_line = exc_tb.tb_lineno
            er+=1
            pass


def remove_empty_articles(timeWindow_data):
    """
        use this function for remove the text that we ignore (english article texts for example)
        @param timeWindow_data: data on timeWindow format -->  for each articles {label :  text processed } sorted by time and splitted according to the size of the time-window
        """

    for window in timeWindow_data:
        for article in window[1]:
            if len(article[1]) == 0 :
                window[1].remove(article)
    return timeWindow_data



def handler(signum, frame):
    """"
    use this function for raise exception when time out is reached just on linux
    """
    raise Exception


def addProcessedText(  dataBasePath_source , databasePath_destination   ):


    data2 = []
    # data_for_dictionnary = []
    # dictionnary_test = gensim.corpora.Dictionary()
    with open(dataBasePath_source , 'r') as f:
        data = json.load(f)
    count = 0
    nb_articles = 0
    processor = ProcessorText()
    func = timeout(timeout=10)(ProcessorText.processText)

    for article in tqdm(data):
        if len(article['cleansed_text']) > 1    and article['cleansed_text'] != 'None ':
            textToProcess = article['cleansed_text']
        elif len(article['summary']) != 0:
            textToProcess = article['summary']
            textToProcess = re.sub("\n|#", " ", textToProcess)  # Clean the text, cette ligne ne fonctionne plus et détruit le code
            textToProcess = re.sub("_|>|<|!\[.*?\]|\(https:.*?\)", "", textToProcess)
        elif len(article['content']) != 0:
            textToProcess = article['content']
            textToProcess = re.sub("\n|#", " ",textToProcess)  # Clean the text, cette ligne ne fonctionne plus et détruit le code
            textToProcess = re.sub("_|>|<|!\[.*?\]|\(https:.*?\)", "", textToProcess)
        elif len(article['title']) != 0:
            textToProcess = article['title']
            textToProcess = re.sub("\n|#", " ",textToProcess)  # Clean the text, cette ligne ne fonctionne plus et détruit le code
            textToProcess = re.sub("_|>|<|!\[.*?\]|\(https:.*?\)", "", textToProcess)

        # #linux
        # signal.signal(signal.SIGALRM, handler)
        # signal.alarm(10)

        #window
        try:

            process_text = processor.processText(textToProcess)
        except Exception as e:
            nb_articles += 1
            continue
        article['process_text'] = process_text
        data2.append(article)
        #data_for_dictionnary.append(process_text)
        nb_articles += 1
        if nb_articles % 50000:
            with open(databasePath_destination, 'w') as f:
                f.write(json.dumps(data2))
