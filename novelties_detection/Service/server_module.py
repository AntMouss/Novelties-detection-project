import json
import time
from threading import Thread , Lock , get_ident
from typing import List , Dict
from novelties_detection.Collection.RSSCollect import RSSCollector , initialize_hashs
from novelties_detection.Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator , NoSupervisedFixedSequantialLangageSimilarityCalculator
from novelties_detection.Experience.WindowClassification import WindowClassifierModel
from novelties_detection.Collection.data_processing import transformS , MetaTextPreProcessor
from novelties_detection.Experience.Exception_utils import CompareWindowsException
import logging
from novelties_detection.Experience.utils import timer_func

logging.basicConfig(level=logging.INFO)


WINDOW_DATA = [] # contain the data of one temporal window
COLLECT_IN_PROGRESS = False
PROCESS_IN_PROGRESS = False
COLLECT_LOCKER = Lock()
PROCESS_LOCKER = Lock()




class CollectThread(Thread):
    """
    Service to periodically collect information from the selected sources
    """
    default_collect_kwargs = {
        "collectFullHtml" : True,
        "collectRssImage" : True,
        "collectArticleImages" : True,
        "print_log" : True
    }
    def __init__(
            self, rss_feed_source_path : str, loop_delay : int ,
            preprocessor : MetaTextPreProcessor  = None,  output_path : str = None , collect_kwargs : dict = None  ):
        """

        @param rss_feed_source_path: path to json file that contain rss feed source like url , label and removing tags
        @param output_path: path from Root Directory where we save information about articles (images , html , metadata)
        @param preprocessor: engine that process (preprocess) text for being ready to use as Bag of Words.
        @param loop_delay: delay between two collect process
        """
        Thread.__init__(self)
        if collect_kwargs is None:
            self.collect_kwargs = self.default_collect_kwargs
        else:
            self.collect_kwargs = collect_kwargs
        self.loop_delay = loop_delay
        self.rss_feed_config_path=rss_feed_source_path
        self.output_path=output_path
        self.rssCollector=RSSCollector(self.rss_feed_config_path,
                                       preprocessor=preprocessor, rootOutputFolder=self.output_path)
        hashs = initialize_hashs(self.rss_feed_config_path)
        self.rssCollector.update_hashs(hashs)
        self.lang = self.rssCollector.preprocessor.lang

    @timer_func
    def update_window_data(self):
        global WINDOW_DATA
        global COLLECT_LOCKER
        global PROCESS_LOCKER
        global COLLECT_IN_PROGRESS
        global PROCESS_IN_PROGRESS
        COLLECT_LOCKER.acquire()
        if PROCESS_IN_PROGRESS == False:
            COLLECT_IN_PROGRESS = True
            new_data = self.rssCollector.treatNewsFeedList(**self.collect_kwargs)
            new_data = self.clean_lang(new_data)
            WINDOW_DATA += new_data
            logging.info(f"the thread with id : {get_ident()} collect {len(new_data)} articles")
            COLLECT_IN_PROGRESS = False
        COLLECT_LOCKER.release()

    def clean_lang(self , articles):
        articles_idx_to_remove = []
        for idx ,  article in enumerate(articles):
            if article["lang"] != self.lang:
                articles_idx_to_remove.append(idx)
        for idx in articles_idx_to_remove:
            del articles[idx]
        return articles

    def run(self):
        logging.info("Collect will start")
        time.sleep(self.loop_delay * 60)
        while True:
            self.update_window_data()
            time.sleep(self.loop_delay * 60)


class NoveltiesDetectionThread(Thread):
    """
    Service to detect and return novelties in the collect information flow
    """
    def __init__(self, supervised_calculator : SupervisedSequantialLangageSimilarityCalculator
                 , micro_calculator : NoSupervisedFixedSequantialLangageSimilarityCalculator, training_args : Dict,
                 results_args : Dict, micro_training_args : Dict, loop_delay : int, classifier_models :List[WindowClassifierModel]):
        """

        @param supervised_calculator: supervised calculator to get novelties on the labels (topics) in the flow
        @param micro_calculator: no supervised calculator to get more detail information about "micro-topic" in the flow
        @param training_args: args use to treat new window
        @param results_args: args use to compute similarity between windows recursively
        @param micro_training_args: training arguments of mirco calculator
        @param loop_delay: delay between two process
        @param classifier_models: window classifier model to get the rarety level of the window (rarety of the similarity score with the previous one)
        """
        Thread.__init__(self)
        self.micro_training_args = micro_training_args
        self.results_args = results_args
        self.training_args = training_args
        self.supervised_reference_calculator = supervised_calculator
        self.micro_calculator = micro_calculator
        self.classifier_models = classifier_models
        self.loop_delay = loop_delay


    @staticmethod
    def log_error():
        logging.warning("no articles collected during his windows")


    def process(self , window_data):
        supervised_data = transformS(window_data, process_already_done=True)
        no_supervised_data = supervised_data[0] #just take the text
        # we train the micro calculator but we don't print the result
        _,_  = self.micro_calculator.treat_Window(no_supervised_data , **self.micro_training_args)
        _,_ = self.supervised_reference_calculator.treat_Window(supervised_data, **self.training_args)
        self.print_res()



    def print_res(self):

        try:
            self.supervised_reference_calculator.print_novelties(n_words_to_print=10, **self.results_args)
            # we call the function calcul_similarity_topics_W_W in the function print_novelties above but we keep the
            # result in cache so it's more comprehensive to recall the function another time bellow
            new_window_idx = len(self.supervised_reference_calculator)  -1
            previous_window_idx = new_window_idx - 1
            similarities, _ = self.supervised_reference_calculator.calcule_similarity_topics_W_W(previous_window_idx, new_window_idx, **self.results_args)
            print("@" * 30)
            for topic_id, (similarity_score, classifier) in enumerate(zip(similarities, self.classifier_models)):
                classifier.print(similarity_score)
                group_id = classifier.predict(similarity_score)
                classifier.update(similarity_score)
                topic = self.supervised_reference_calculator.labels_idx[topic_id]
                print(f"Rarety level : {group_id} / {len(classifier)} for the topic: {topic}.  ")
        except CompareWindowsException:
            logging.warning("no comparaison possible yet because there are less than 2 windows treated")
            pass
        except Exception as e:
            pass

    @timer_func
    def do_process(self):
        global WINDOW_DATA
        global PROCESS_LOCKER
        global PROCESS_IN_PROGRESS
        global COLLECT_IN_PROGRESS
        PROCESS_LOCKER.acquire()
        if COLLECT_IN_PROGRESS == False:
            PROCESS_IN_PROGRESS = True
            if len(WINDOW_DATA) != 0:
                with open("/window_data.json" , "w") as f:
                    f.write(json.dumps(WINDOW_DATA))
                self.process(WINDOW_DATA)
                # empty and reinitialization of WINDOW_DATA
                del WINDOW_DATA
                WINDOW_DATA = []
            else:
                NoveltiesDetectionThread.log_error()
            PROCESS_IN_PROGRESS = False
        PROCESS_LOCKER.release()


    def run(self):
        logging.info("Process will start")
        time.sleep(self.loop_delay * 60)
        while True:
            self.do_process()
            time.sleep(self.loop_delay * 60)