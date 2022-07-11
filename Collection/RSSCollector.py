import os
import json
from html2text import HTML2Text
import feedparser
from bs4 import BeautifulSoup
import hashlib
import requests
import shutil
from urllib.parse import urlparse
from datetime import datetime
import logging
import threading
from Collection.data_processing import ProcessorText , dateToDateTime
from Collection.data_cleaning import extract_text
import sys
import validators
import pathlib


def urlId(url):
    '''
    From one URL string, calculates a unique ID for it
    :param url:
    :type url:
    :return:
    :rtype:
    '''
    return hashlib.md5(url.encode()).hexdigest()

lock = threading.Lock()


class RSSCollect():
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

    def __init__(self, sourcesList, rootOutputFolder , processor : ProcessorText):
        '''
        Constructor
        :param sourcesList: File where the sources are stored
        :param rootOutputFolder: Folder where we will save the results
        '''
        self.sourcesList = sourcesList
        self.dayOutputFolder = os.path.join(rootOutputFolder, self.dayFolderName)
        self.rootHashFolder=os.path.join(rootOutputFolder, self.hashFolderName)
        self.hashs = self.getProcessedNews()
        self.rootOutputFolder=rootOutputFolder
        self.logFolderPath = os.path.join(rootOutputFolder , self.log_folder)
        self.log_performance_path = os.path.join(self.logFolderPath , self.log_performance_file)
        self.log_error_path = os.path.join(self.logFolderPath , self.log_error_file)
        self.global_remove_tags = []


    def treatNewsFeedList(self, **kwargs):
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

        # To guarantee that the folder date will be respected for this run
        dateNow = datetime.now()
        dayFolderName = "rss" + format(dateNow.year, '04d') + format(dateNow.month, '02d') + format(dateNow.day, '02d')
        self.dayOutputFolder = os.path.join(self.rootOutputFolder, dayFolderName)

        if not os.path.exists(self.sourcesList):
            logging.warning(f"ERROR: RSS sources list file not found!{self.sourcesList} ")
            raise Exception("ERROR: RSS sources list file not found! ", self.sourcesList)


        with open(self.sourcesList, "r") as f:
            rss_config = json.load(f)
        url_rss = rss_config["rss_feed_url"]
        self.global_remove_tags = rss_config["global_remove_tags"]
        listOfReadEntries = []

        for i in range(len(url_rss)):
            print(i)
            label = url_rss[i]["label"]
            url = url_rss[i]["url"]
            if "remove_tags" in url_rss[i].keys():
                remove_tag_list = url_rss[i]["remove_tags"] + self.global_remove_tags
            else:
                remove_tag_list = self.global_remove_tags
            try:
                # Get information from the rss config file
                listOfReadEntries += self.treatRSSEntry(label, url , remove_tag_list,**kwargs)
                self.saveProcessedNewsHashes(self.hashs)
                self.evaluateCollect(listOfReadEntries,url,**kwargs)

            except Exception as e:

                self.updateLogError(e , 1 , url)
                pass
        # Add the news information (if there's new news) in the database
        print("RSS news extraction end")
        lock.release()
        return listOfReadEntries


    def treatRSSEntry(self, label : list, rss_url : str, remove_tags_list : list,
                      collectFullHtml=True, collectRssImage=True, collectArticleImages=True):
        '''
        Treats a given Rss entry
        :param label:
        :param rss_url:
        :return: list of readed feed information
        '''
        try:
            h = HTML2Text()
            h.ignore_links = True  # Ignore links like <a...>
            # Get global information from the rss feed
            rss_feed = feedparser.parse(rss_url)
            if "updated" in rss_feed.keys():
                feed_date = rss_feed["updated"]
            else:
                feed_date= ""
            feedList=[]
        except Exception as e:
            self.updateLogError(e, 1, rss_url)
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
                folderName = os.path.join(domain, article_id[:5], article_id)
                targetDir = os.path.join(self.dayOutputFolder, folderName)
                os.makedirs(targetDir, exist_ok=True)
            except Exception as e:
                self.updateLogError(e, 2, entry['link'])
                continue

            try:
                # Check if the article is already in the database
                # Extract information from the website
                r = requests.get(entry["link"], timeout=10)  # Get HTML from links
                if r.status_code!=200:
                    continue
                soup = BeautifulSoup(r.text, "lxml")  # Parse HTML
                article = soup.find("article")  # Get article in HTML
                article_copy = article.__copy__()
                if article is not None:
                    text = extract_text(article, remove_tags_list, clean=False)
                    cleansed_text = extract_text(article, remove_tags_list, clean=True)
                    process_text = ProcessorText.processText(cleansed_text)
                    htmlCollected = True
                else:
                    text = None
                    cleansed_text = None
                    process_text = None
                    htmlCollected = False

                # Complete the news information for the database
                feed = {"title": entry["title"],
                        "url": entry["link"],
                        "text": text,
                        "cleansed_text": cleansed_text,
                        "label": label,
                        "rss": rss_url,
                        "process_text" : process_text,
                        "updated": False ,
                        "htmlCollected": htmlCollected,
                        "id" : article_id
                        }

                if "published" in entry.keys():
                    feed["date"] = entry["published"]
                else:
                    feed["date"] = feed_date

                feed["timeStamp"] = dateToDateTime(feed["date"] , timeStamp=True)

                if "summary" in entry.keys():
                    feed["summary"] = entry["summary"]
                else:
                    feed["summary"] = ""

                if "content" in entry.keys():
                    if "value" in entry["content"][0].keys():
                        feed["content"] = entry["content"][0]["value"]
                else:
                    feed["content"] = ""

                feed['domainName'] = urlparse(feed['url']).netloc


            except Exception as e:

                self.updateLogError(e, 2, entry['link'])
                continue

            try:
                if collectRssImage:
                    feed["rss_media"]=[]
                    collectedImagesNames=self.treatRssImages(entry, targetDir, entry['link'])
                    feed["rss_media"]=collectedImagesNames
            except Exception as e:
                pass



            try: #on gère pas les erreurs de treatArticleImages

                if collectArticleImages:
                    feed["images"]=[]
                    article,collectedImagesNames=self.treatGetArticleImages(article,targetDir , entry['link'])
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
                json.dump(article, f)

            feedList.append(feed)
            self.hashs.append(article_id)

        return feedList


    def treatRssImages (self, entry_rss, targetDir, link) :
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
                    mediaName = self.downloadMedia(eMedia["url"], targetDir,link)
                    if mediaName != '':
                        collectedImagesNames.append(mediaName)


        if "links" in entry_rss.keys():
            for link in entry_rss["links"]:

                if "type" in link.keys():
                    if "image" in link["type"] or "jpeg" in link["type"]:
                        mediaName = self.downloadMedia(link["href"],targetDir,link)
                        if mediaName != '':
                            collectedImagesNames.append(mediaName)

                    else:
                        if "image" in link['href']:
                            mediaName = self.downloadMedia(link["href"], targetDir,link)
                            if mediaName != '':
                                collectedImagesNames.append(mediaName)
        return collectedImagesNames


    def treatGetArticleImages(self,article,targetDir , link):
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
                    mediaName=self.downloadMedia(src, targetDir, link )
                    if mediaName != '':
                        img['src']=mediaName
                        collectedImagesNames.append(mediaName)

            except KeyError as e:
                continue
        return article,collectedImagesNames



    def downloadMedia(self, url_media, targetDir , link):
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
            self.updateLogError(e , 3 , url_media)

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


    def updateLogError(self , exception , nu_type , link):

            exc_tb = sys.exc_info()[2]
            exc_line = exc_tb.tb_lineno
            msg_err = f" {str(datetime.now())}-----New Error type {str(nu_type)}: {str(exception)} for link: {link} at line {exc_line}"

            if not os.path.exists(self.logFolderPath):
                os.makedirs(self.logFolderPath)

            with open(self.log_error_path , 'a') as f:
                f.write(msg_err + "\n")



    def updateLogPerf(self, msg_perf):

        if not os.path.exists(self.logFolderPath):
            os.makedirs(self.logFolderPath)

        with open(self.log_performance_path, 'a') as f:
            f.write(msg_perf + "\n")


    def evaluateCollect(self, listOfReadEntry, url_rss,collectImageArticle,collectRssImage):
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
                for feed_id in listOfReadEntry:
                    feed=listOfReadEntry[feed_id]
                    if collectImageArticle and collectRssImage:
                        countImage+=len(feed["rss_media"])+len(feed["images"])
                    elif collectRssImage:
                        countImage+=len(feed["rss_media"])
                    elif collectImageArticle:
                        countImage+=len(feed["images"])

                    for field in self.listOfFields:
                        if not feed[field]:#the field is empty
                            countField+=1

                msg_perf = msg_perf + f" {countField / (len(self.listOfFields) * len(listOfReadEntry)) * 100} % of field empty \n"
                msg_perf = msg_perf + f"{countImage} images \n"
            except KeyError as e:
                # msg_err = f"{str(date) : }"
                # self.updateLog(f"New Key Error type 3:  {str(e),url_rss}")
                pass
            except Exception as e:
                # logging.warning(f"New Error Evaluation type 3: {str(e), url_rss}")
                pass
        self.updateLogPerf(msg_perf )



rootOutputFolder="/home/mouss/tmpTest18"

if __name__ == '__main__':

    processor = ProcessorText()
    RSS_URL_file= "../tmp_test_obj/rssfeed_news_test.json"
    rssc = RSSCollect(RSS_URL_file,rootOutputFolder , processor = processor)
    rssc.treatNewsFeedList(collectFullHtml=True, collectRssImage=False,collectArticleImages=False)




