import os
import json
import time
from typing import List, Dict
import feedparser
from bs4 import BeautifulSoup
import hashlib
import requests
import shutil
from urllib.parse import urlparse
from datetime import datetime
import logging
import threading
from novelties_detection.Collection.data_processing import MetaTextPreProcessor , FrenchTextPreProcessor , dateToDateTime
from novelties_detection.Collection.data_cleaning import extract_text
import sys
import validators
import pathlib

lock = threading.Lock()


def urlId(url):
    '''
    From one URL string, calculates a unique ID for it
    :param url:
    :type url:
    :return:
    :rtype:
    '''
    return hashlib.md5(url.encode()).hexdigest()


def initialize_hashs(sourcesList : str) -> list:

    hashs = []

    with open(sourcesList, "r") as f:
        rss_config_urls = json.load(f)["rss_feed_urls"]
    rss_urls = [feed["url"] for feed in rss_config_urls]

    for rss_url in rss_urls:
        try:
            rss_feed = feedparser.parse(rss_url)
            for entry in rss_feed.entries:
                article_id = urlId(entry["link"])
                hashs.append(article_id)
        except Exception as e:
            print(e)
            continue
    return hashs



class RSSCollector:
    '''
    Main class to collect rss feeds
    '''
    sourcesList=''
    hashs=[]
    dayOutputFolder= ''
    rootHashFolder=''
    log_error_file = 'log_error.txt'
    log_performance_file = 'log_performance.txt'
    log_folder = 'logFolder'
    dateNow = datetime.now()
    dayFolderName = "rss" + format(dateNow.year, '04d') + format(dateNow.month, '02d') + format(dateNow.day, '02d')
    hashFolderName = "hashFolder"
    targetDataFile = "data.json"
    standardHashsFile = "hashsDataset.json"
    htmlFileName="news.html"
    fullDatasetFile="fullDataSet.json"
    HTML_HEADER = '<html><body>'
    HTML_TAIL="</body></html>"
    listOfFields = ["title", "url", "text", "label", "rss", "updated", "date", "summary", "content"]

    def __init__(self, sourcesList, preprocessor : MetaTextPreProcessor = None, rootOutputFolder = None):
        '''
        Constructor
        :param sourcesList: File where the sources are stored
        :param rootOutputFolder: Folder where we will save the results
        '''
        self.preprocessor = preprocessor
        self.sourcesList = sourcesList
        self.rootOutputFolder = rootOutputFolder

        if rootOutputFolder is not None:
            self.dayOutputFolder = os.path.join(rootOutputFolder, self.dayFolderName)
            self.rootHashFolder=os.path.join(rootOutputFolder, self.hashFolderName)
            self.logFolderPath = os.path.join(rootOutputFolder , self.log_folder)
            self.log_performance_path = os.path.join(self.logFolderPath , self.log_performance_file)
            self.log_error_path = os.path.join(self.logFolderPath , self.log_error_file)
            self.hashs = self.getProcessedNews()

        else:
            self.hashs = []
        self.global_remove_tags = []


    def update_hashs(self, hashs : list):
        if len(self.hashs) == 0:
            self.hashs = hashs
        else:
            for hash_ in hashs:
                if hash_ not in self.hashs:
                    self.hashs.append(hash_)


    def treatNewsFeedList(self, print_log = True,**kwargs):
        '''
        Collects the information from the news feeds
        :param collectFullHtml: If the full
        :param collectRssImage:
        :param collectArticleImages:
        :return:
        '''
        global lock
        print("RSS news extraction start")

        # to avoid call two threads to treat the feeds at the same time. The second thread, stops.
        # no thread will start while the previous is still "inUse"  state
        lock.acquire()

        if self.rootOutputFolder is not None:
            # To guarantee that the folder date will be respected for this run
            dateNow = datetime.now()
            dayFolderName = "rss" + format(dateNow.year, '04d') + format(dateNow.month, '02d') + format(dateNow.day, '02d')
            self.dayOutputFolder = os.path.join(self.rootOutputFolder, dayFolderName)

        if not os.path.exists(self.sourcesList):
            logging.warning(f"ERROR: RSS sources list file not found!{self.sourcesList} ")
            raise Exception("ERROR: RSS sources list file not found! ", self.sourcesList)


        with open(self.sourcesList, "r") as f:
            rss_config = json.load(f)
        url_rss = rss_config["rss_feed_urls"]
        self.global_remove_tags = rss_config["global_remove_tags"]
        totalListOfReadEntries = []

        for i in range(len(url_rss)):
            #print(i)
            labels = url_rss[i]["label"]
            url = url_rss[i]["url"]
            if "remove_tags" in url_rss[i].keys():
                remove_tag_list = url_rss[i]["remove_tags"] + self.global_remove_tags
            else:
                remove_tag_list = self.global_remove_tags
            try:
                # Get information from the rss config file
                rssListOfReadEntries =self.treatRSSEntry(labels, url ,
                                                         remove_tag_list, print_log=print_log,**kwargs)
                totalListOfReadEntries += rssListOfReadEntries
                self.evaluateCollect(
                    rssListOfReadEntries, url, print_log=print_log, **kwargs)
                if self.rootOutputFolder is not None:
                    self.saveProcessedNewsHashes(self.hashs)

            except Exception as e:
                self.writeLogError(e, 1, url)
                pass
        # Add the news information (if there's new news) in the database
        print("RSS news extraction end")
        lock.release()

        return totalListOfReadEntries


    def treatRSSEntry(self, labels : list, rss_url : str, remove_tags_list : list,
                      collectFullHtml=True, collectRssImage=True, collectArticleImages=True, print_log = True):
        '''
        Treats a given Rss entry
        :param labels:
        :param rss_url:
        :return: list of readed feed information
        '''
        try:
            # Get global information from the rss feed
            rss_feed = feedparser.parse(rss_url)

            # get default timestamp date to use it if the datetodatetime function return None (i.e if the format of the string date is not registered)
            default_timestamp = time.time()

            if "updated" in rss_feed.keys():
                feed_date = rss_feed["updated"]
            else:
                feed_date= ""
            feedList=[]
        except Exception as e:
            self.writeLogError(e, 1, rss_url , print_log=print_log)
            return []
        # For each news in the rss feed
        j=0
        for entry in rss_feed.entries:
            j+=1
            try:
                article_id = urlId(entry["link"])
                if article_id in self.hashs:
                    continue
                domain = urlparse(rss_url).netloc
                r = requests.get(entry["link"], timeout=10)  # Get HTML from links
                if r.status_code != 200:
                    continue
            except Exception as e:
                self.writeLogError(e, 2, entry['link'] , print_log=print_log)
                continue

            try:
                # Complete the news information for the database
                feed = {"title": entry["title"],
                        "url": entry["link"],
                        "label": labels,
                        "rss": rss_url,
                        "updated": False,
                        "id": article_id
                        }
                feed["lang"] = self.preprocessor.detectLanguage(feed["title"])
                if "published" in entry.keys():
                    feed["date"] = entry["published"]
                else:
                    feed["date"] = feed_date
                timestamp_date = dateToDateTime(feed["date"], timeStamp=True)
                if timestamp_date is not None:
                    feed["timeStamp"] = timestamp_date
                else:
                    feed["timeStamp"] = default_timestamp
                if "summary" in entry.keys():
                    feed["summary"] = entry["summary"]
                else:
                    feed["summary"] = None
                if "content" in entry.keys():
                    if "value" in entry["content"][0].keys():
                        feed["content"] = entry["content"][0]["value"]
                else:
                    feed["content"] = ""

                feed['domainName'] = domain
                # Check if the article is already in the database
                # Extract information from the website
                soup = BeautifulSoup(r.text, "lxml")  # Parse HTML
                article = soup.find("article")# Get article in HTML
                text_fields , article_copy = self.treatArticle(article , remove_tags_list , feed["title"] , feed["summary"])
                feed.update(text_fields)

            except Exception as e:
                self.writeLogError(e, 2, entry['link'] , print_log=print_log)
                continue

            feedList.append(feed)
            self.hashs.append(article_id)


            if self.rootOutputFolder is not None:

                folderName = os.path.join(domain, article_id[:5], article_id)
                targetDir = os.path.join(self.dayOutputFolder, folderName)
                os.makedirs(targetDir, exist_ok=True)

                try:
                    if collectRssImage:
                        feed["rss_media"]=[]
                        collectedImagesNames=self.treatRssImages(entry, targetDir, entry['link'] , print_log = print_log)
                        feed["rss_media"]=collectedImagesNames
                except Exception as e:
                    pass


                try: #on gère pas les erreurs de treatArticleImages
                    if collectArticleImages:
                        feed["images"]=[]
                        article,collectedImagesNames=self.treatGetArticleImages(article,targetDir , entry['link'] , print_log = print_log)
                        feed["images"]=collectedImagesNames
                except Exception as e:
                    exc_tb = sys.exc_info()[2]
                    exc_line = exc_tb.tb_lineno
                    pass
                    # Saves the full html text


                if article is not None:
                    try: #on gère pas les erreurs html
                        if collectFullHtml:
                            with open(os.path.join(targetDir, self.htmlFileName), "w" , encoding='UTF-8') as f:
                                page=self.HTML_HEADER + str(article_copy) + self.HTML_TAIL
                                f.write(page)
                    # erreur corrigée avec le bon encodage
                    except UnicodeError as e:
                        print(e)
                        pass
                    except:
                        pass

                with open(os.path.join(targetDir, self.targetDataFile), "w") as f:
                    json.dump(feed, f)

        return feedList


    def treatArticle(self , article , remove_tags_list , article_title , article_summary):
        """
        this function treat the text content of the requests we want to fetch the text inside the article tag
        and clean the useless tag and process the final text to have text already processed and stocked
        note that we made copy because during the cleaning we modified the html object and we want to save the original
        html so we make copy to save original html latter in the top level function
        @param article_title:
        @param article:
        @param remove_tags_list:
        @return: dict with the text field , article copy
        """
        if article is not None:
            article_copy = article.__copy__()
            text = extract_text(article, remove_tags_list, clean=False)
            cleansed_text = extract_text(article, remove_tags_list, clean=True)
            try:
                if self.preprocessor is not None:
                    if cleansed_text is not None:
                        process_text = self.preprocessor.preprocessText(cleansed_text)
                    elif article_summary is not None:
                        process_text = self.preprocessor.preprocessText(article_summary)
                    elif article_title is not None:
                        process_text = self.preprocessor.preprocessText(article_title)
                    elif text is not None:
                        process_text = self.preprocessor.preprocessText(text)
                    else:
                        process_text = []
                else:
                    process_text = []
            except TimeoutError:
                process_text = []
                pass
            htmlCollected = True
        else:
            article_copy = article
            text = None
            cleansed_text = None
            process_text = []
            htmlCollected = False
        return {
            "text" : text,
            "cleansed_text" : cleansed_text,
            "process_text" : process_text,
            "htmlCollected" : htmlCollected

        } , article_copy


    def treatRssImages (self, entry_rss, targetDir, link , **kwargs) :
        """
        we want to
        :param entry_rss: the entry corresponding to the article in the rssfeed Parser
        :param targetDir: the output directory for the entry
        :return: list of file image names that we can use for catch duplicate files
        """
        collectedImagesNames=[]

        if "media_content" in entry_rss.keys():
            for eMedia in entry_rss["media_content"]:

                if "url" in eMedia:
                    mediaName = self.downloadMedia(eMedia["url"], targetDir,link , **kwargs)
                    if mediaName != '':
                        collectedImagesNames.append(mediaName)


        if "links" in entry_rss.keys():
            for link in entry_rss["links"]:

                if "type" in link.keys():
                    if "image" in link["type"] or "jpeg" in link["type"]:
                        mediaName = self.downloadMedia(link["href"],targetDir,link , **kwargs)
                        if mediaName != '':
                            collectedImagesNames.append(mediaName)

                    else:
                        if "image" in link['href']:
                            mediaName = self.downloadMedia(link["href"], targetDir,link , **kwargs)
                            if mediaName != '':
                                collectedImagesNames.append(mediaName)
        return collectedImagesNames


    def treatGetArticleImages(self,article,targetDir , link , **kwargs):
        '''
        Treats the case where we need to download a media content from the RSS feed
        :param targetDir:
        :return:
        '''
        collectedImagesNames = []

        try:
            images = article.findAll('img')
        except AttributeError:
            return article , collectedImagesNames

        for img in images:
            try:
                src=img['src']
                if src:
                    mediaName=self.downloadMedia(src, targetDir, link , **kwargs )
                    if mediaName != '':
                        img['src']=mediaName
                        collectedImagesNames.append(mediaName)

            except KeyError as e:
                continue
        return article,collectedImagesNames



    def downloadMedia(self, url_media, targetDir , link , print_log = True):
        '''
        Downloads a media content from the main page
        :param url_media:
        :param targetDir:
        :return: name of the saved media in the target dir
        '''
        # check if url is complete
        valid = validators.url(url_media)
        if valid != True:
            # we can certainly find a better solution
            # we took the three first element of the article link --> '.' , 'https:' , Domain
            #and append the incomplete url_media
            p = pathlib.Path(link).parents
            i = len(p) - 3
            url_media = str(p[i]).replace("\\" , "//") + url_media
        try:
            mediaContent = requests.get(url_media, timeout=3)
            if mediaContent.status_code != 200:
                return ''

            name = os.path.basename(url_media)
            if len(name)>100:
                name=name[:20]
                # # if "jpg" not in name or "jpeg" not in name:
                # #     name=name+".jpg"
            characters_to_remove=["?","-","/","|","%" , "=" , "&"]
            for char in characters_to_remove:
                name=name.replace(char,"")
            mediaName = os.path.join(targetDir, name)
            mediaName = self.treatAlreadyUsedFileName(mediaName)
            with open(mediaName, 'wb') as fMedia:
                fMedia.write(mediaContent.content)
        except Exception as e:
            self.writeLogError(e, 3, url_media , print_log=print_log)

            return ''

        return os.path.basename(mediaName)



    def treatAlreadyUsedFileName(self, fileName):
        '''
        Treats if the file already exists in the system
        :param fileName: name of the file
        :return:
        '''
        if not os.path.exists(fileName):
            return fileName

        tmpFileName = fileName
        count = 1
        while os.path.exists(tmpFileName):
            tmpFileName = fileName + "_" + str(count)
            count += 1
        return tmpFileName


    def saveProcessedNewsHashes(self, hashs):
        '''
        Save news hashes
        :return:
        '''
        hashFile = os.path.join(self.rootHashFolder, self.standardHashsFile)
        if not os.path.exists(self.rootHashFolder):
            os.makedirs(self.rootHashFolder)
        if hashs:
            try:
                with open(hashFile+"_tmp", "w") as f:
                    f.write(json.dumps(hashs))
                if os.path.exists(hashFile):
                    os.remove(hashFile)
                shutil.move(hashFile+"_tmp", hashFile)
            except Exception as e:
                logging.warning(f"Error saving hash file: type 4 :  {str(e)}")


    def getProcessedNews(self):
        '''
        Reads the hashs of the already read news
        :return:
        '''
        hashSetFile = os.path.join(self.rootHashFolder, self.standardHashsFile)
        if os.path.exists(hashSetFile):
            with open(hashSetFile, "r") as f:
                hashs = json.load(f)
        else:
            hashs = []
        return hashs


    def writeLogError(self, exception, nu_type, link , print_log : bool = True):

            exc_tb = sys.exc_info()[2]
            exc_line = exc_tb.tb_lineno
            msg_err = f" {str(datetime.now())}-----New Error type {str(nu_type)}: {str(exception)} for link: {link} at line {exc_line}"

            if self.rootOutputFolder is None:
                if not os.path.exists(self.logFolderPath):
                    os.makedirs(self.logFolderPath)

                with open(self.log_error_path , 'a') as f:
                    f.write(msg_err + "\n")

            if print_log:
                print(msg_err)


    def writeLogPerf(self, msg_perf , print_log : bool = True):


        if self.rootOutputFolder is not None:
            if not os.path.exists(self.logFolderPath):
                os.makedirs(self.logFolderPath)

            with open(self.log_performance_path, 'a') as f:
                f.write(msg_perf + "\n")

        if print_log:
            print(msg_perf)

    def evaluateCollect(self, listOfReadEntry : List[Dict], url_rss : str , collectArticleImages : bool,collectRssImage : bool , print_log : bool = True , **kwargs):
        """
        read the output of our feedlist and evaluate the percentage of field empty and no collected rss
        and give the average of collected images per articles for each feed

        """
        date = datetime.now()
        msg_perf = f"{str(date)} ---- {url_rss} \n"
        # print(f"we received {nb_feed} articles from {url_rss}")

        msg_perf = msg_perf + f"{len(listOfReadEntry)} articles collected \n"
        if len(listOfReadEntry) != 0:

            countField=0
            countImage=0
            try:
                for feed in listOfReadEntry:
                    if collectRssImage:
                        countImage+=len(feed["rss_media"])
                    if collectArticleImages:
                        countImage+=len(feed["images"])

                    for field in self.listOfFields:
                        if not feed[field]:#the field is empty
                            countField+=1

                msg_perf = msg_perf + f" {countField / (len(self.listOfFields) * len(listOfReadEntry)) * 100} % of field empty \n"
                msg_perf = msg_perf + f"{countImage} images collected \n"
            except KeyError as e:
                pass
            except Exception as e:
                pass

        self.writeLogPerf(msg_perf , print_log=print_log)



# rootOutputFolder="/home/mouss/tmpTest18"
#
# if __name__ == '__main__':
#
#     preprocessor = FrenchTextPreProcessor()
#     RSS_URL_file= "../../config/RSSfeeds_test.json"
#     rssc = RSSCollector(RSS_URL_file, preprocessor = preprocessor , rootOutputFolder=None)
#     test = rssc.treatNewsFeedList(collectFullHtml=False, collectRssImage=False,collectArticleImages=False , print_log=True)




