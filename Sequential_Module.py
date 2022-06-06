import math
import random
from typing import List
from data_utils import TimeLineArticlesDataset
from gensim import corpora
from data_processing import filterDictionnary
import os
from corextopic import corextopic as ct
import Engine


# class SequencialLangageModeling:
#
#     """
#     the purpose of this class is to train a new guidedLda model by temporal windows
#     and compare the keywords by topic and follow the comportement of this keywords in the differents topics
#     """
#
#     def __init__(self,guided = None , nb_topics=0 , semi_filtred_dictionnary=None , seed=None):
#         """
#         this class need one specific dictionnary and one specific dataBase
#         as the dataBase is so important we don't want to load the dataBase with this class
#         the better way is to connect a class objet to the database and build a new guided lda model (object) by defined time intervall
#         but for the begining we just load the data for the 6 last days and we make the operation on different defined time intervall
#         (1 hour, 6 hour , 1 day) or not constant time intervall but constant entry data (all 100, 500 , 1000 articles)
#         We had to have a model repository for save the different model by bunch
#         We need a specific eta too (for the guided LDA).
#         note: we don't process a eta for each intervall but for all the data bunch because the dictionnary is fixed for each intervall so
#         we don't need to process a new eta (we process a new eta just when the dictionnary is updated or if the seed change)
#
#         we decide to remove the filtred_dictionnary to the class because this dictionnary change after each time-windows
#         data are added so we change our method , we keep the semi_filtred dictionnary and we train the glda window with
#         the specific dictionnary of the window that we previously filtred we filtred_id gensim method.
#         the word that we are removing come from the semi-dictionnary of our class.
#
#         """
#
#         self.engine = Engine.Engine
#         self.semi_filtred_dictionnary = corpora.Dictionary()
#         if guided == False and seed is not None:
#             raise Exception("no guided lda can't have seed")
#
#
#
#         if guided == True and seed is None:
#             raise Exception("guided lda need seed")
#         if guided == True and nb_topics != 0:
#             raise Exception("guided lda can't have predefined number of topics , "
#                             "the number of topics is defined by the seed")
#
#         if guided == False and nb_topics == 0:
#             raise Exception("no guided lda need to have defined number of topics")
#
#
#         if guided == True:
#             self.seed = seed
#             self.nb_topics = len(self.seed.keys())
#             seed_documents = [self.seed[topic_id] for topic_id in seed.keys()]
#             self.semi_filtred_dictionnary.add_documents(seed_documents)
#             self.seed_words_ids = [i for i in range (len(self.semi_filtred_dictionnary))]
#             self.table_topics_keys  = { key : i for i , key in enumerate(self.seed) }
#             self.table_keys_topics = { i : key for i , key in enumerate(self.seed) }
#
#
#
#         else:
#             self.nb_topics = nb_topics
#
#
#
#         if semi_filtred_dictionnary is not None:
#             self.semi_filtred_dictionnary.merge_with(semi_filtred_dictionnary)
#
#         # we don't filter small and stop words because we already did in the corpus using for generate this dictionnary
#         # we want keep the token from the seed even they are relatively frequent or infrequent
#
#         # words to remove for each the window dictionnary according to the referring semi_dictionnary
#         # this word are the relative infrequent or frequent word according to our semi_filtred dictionnary
#         self.seedFileName = '_seed.json'
#         self.guided = guided
#         self.bad_words = []
#         self.info = {'guided':self.guided , 'nb_topics': self.nb_topics}
#         self .info_file  = 'info.json'
#         self.LDAs = []
#         self.ldaModelFolder = 'LDAs'
#         self.ldaFileName = '_lda_'
#         self.listLDAFile = []
#         self.listLDAName = 'listLDA'
#         self.resFileName = 'res.json'
#         self.semi_dictionnaryFileName = '_semiDict'
#         self.res = {}
#         self.nb_windows = 0
#         self.dateFile = 'date.json'
#         self.dates = {}
#         self.predefinedBadWords = ['...','commenter','réagir','envoyer','mail','partager' , 'publier' , 'lire' ,
#                                    'journal' , "abonnez-vous" , "d'une" , "d'un" ,"mars" , "avril" , "mai" ,
#                                    "juin" , "juillet" , "an" , "soir" , "mois", "lundi" , "mardi" , "mercredi"
#             , "jeudi" , "vendredi" , "samedi" , "dimanche"]
#         self.coherence_score_file = 'coherence.json'
#         self.coherence_scores = []
#         self.counterLabels_file = 'counterLabels.json'
#         self.counterLabels = []
#
#         # self.data=data#amount of data for the 6 last days just for testing but in the futur we had to connect to our dataBase and doing continuous treatment
#         # data= bow , date we just need this two field for this class. Maybe it'is not to this class to manage that
#
#
#
#     def treat_Window(self, documents_window, random_state):
#         """
#
#         """
#
#         if len(documents_window) != 0:
#             if self.guided == True:
#                 texts_window = [ text for text , label in documents_window]
#             else:
#                 texts_window = documents_window
#
#             window_dictionnary = corpora.Dictionary(texts_window)
#
#             #update semi-filtred dictionnary
#             self.semi_filtred_dictionnary.merge_with(window_dictionnary)
#
#             #we filtre bad words from window_dictionnary
#             self.updateBadwords()
#             window_dictionnary_f = filterDictionnary(window_dictionnary , bad_words=self.bad_words)
#
#             # fourth transform text corpus in bow corpus with the filtred window dictionnary
#             corpus_bow = [window_dictionnary_f.doc2bow(document) for document in texts_window]
#
#             if self.guided == True:
#                 # update eta
#                 eta = GuidedLDA.generate_eta(self.seed, window_dictionnary_f , self.table_topics_keys)
#
#                 # train specific GLDA model correlated to the window
#                 model = self.engine(corpus_bow, num_topics=self.nb_topics, id2word=window_dictionnary_f , eta=eta , random_state=random_state)
#
#             else:
#                 model = self.engine(corpus_bow, num_topics=self.nb_topics, id2word=window_dictionnary_f , random_state=random_state)
#                 cm = CoherenceModel(model=model, topn=150, coherence='c_v')
#                 scores = cm.get_coherence_per_topic()
#                 self.coherence_scores.append(scores)
#
#             return model, window_dictionnary_f
#
#
#     def add_windows(self, data , lookback = 10 , updateRes = True   ):
#
#         rValue= random.Random()
#
#         self.info['lookback'] = lookback
#         rValue.seed(37)
#
#         for i  , (end_date_window , documents_window) in tqdm(list(enumerate(data))):
#             random_state = rValue.randint(1, 14340)
#             print(f"numero of window: {i}")
#             print(f"random state: {random_state} ")
#             print(f"size documents: {len(documents_window)} ")
#             if self.guided == True:
#                 labels_window = [ label for text , label in documents_window]
#                 # count labels
#                 self.counterLabels.append(dict(Counter(labels_window)))
#             try:
#                 doc_to_add = []
#                 j= i -1
#                 if lookback < 1:
#                     abs_lookback = math.ceil(lookback * len(data[i-1][1]))
#                 else:
#                    abs_lookback = lookback
#
#                 while len(doc_to_add) !=abs_lookback and j >= 0:
#                     doc_to_add += data[j][1][-(abs_lookback - len(doc_to_add)):]
#                     j -= 1
#                 documents_window += doc_to_add
#
#
#                 lda , window_dictionnary = self.treat_Window(documents_window , random_state=random_state)
#
#             except ValueError as e:
#                 print(e)
#                 pass
#             # for bound window to the right glda model we use no_window
#             no_window = i
#             self.LDAs.append(lda)
#             self.updateBadwords()
#             if updateRes:
#                 self.updateResults(end_date_window , window_dictionnary , lda , no_window)
#             self.dates[end_date_window] = no_window
#             self.nb_windows += 1
#
#
#
#     def updateResults(self, end_date, dictionnary_window , lda , no_window):
#
#         # check the seed words during the loops for calculate seed score and have the relative score of words that aren't in the seed
#         dictOfTopWordsAllTopics = self.topWordsPerTopics(lda , len(dictionnary_window) ,exclusive=False , removeSeedWords=False )[1]
#         average_score = {}
#         try:
#             for topic_id in range(self.nb_topics):
#                 seedScore_topic = 0
#                 nb_seed_word_topic = 0
#                 topic = self.table_keys_topics[topic_id]
#                 for item in dictOfTopWordsAllTopics[topic_id].items():
#                     if item[0] in self.seed[topic]:
#                         seedScore_topic += item[1]
#                         nb_seed_word_topic += 1
#                         #for optimize we break after the first no seed word found in the loop
#                     else:
#                         break
#                 average_score_topic = (1 - seedScore_topic) / (len(dictionnary_window) - nb_seed_word_topic)
#                 average_score[topic] = average_score_topic
#
#
#
#
#             for word , word_id in dictionnary_window.token2id.items():
#                 if word not in self.res.keys():
#                     self.res[word] = {}
#                     self.res[word]['first'] = {'date': end_date} #we use end date of the window as date of the first appearance to the current world
#                     self.res[word]['appearances'] = []
#                 appearance={}
#                 appearance['date_end_window'] = end_date
#                 appearance['no_window'] = no_window
#                 appearance['isBadWord'] = (word in self.bad_words)
#                 appearance['df_in_window'] = dictionnary_window.dfs[word_id]
#                 appearance['cf_in_window'] = dictionnary_window.cfs[word_id]
#                 self.res[word]['appearances'].append(appearance)
#                 for topic_id in range(self.nb_topics):
#                     topic = self.table_keys_topics[topic_id]
#                     average_score_topic = average_score[topic]
#                     # don't forget it's the word id and not the real word
#
#                     # for j, item in enumerate(
#                     #         dictOfTopWordsAllTopics[topic_id].items()):
#                     #     if word == item[0]:
#                     score = dictOfTopWordsAllTopics[topic_id][word]
#                     if 'keyword' not in appearance.keys():
#                         appearance['keyword'] = {}
#                     if topic not in appearance['keyword'].keys():
#                         appearance['keyword'][topic] = {}
#                     if word in self.seed[topic]:
#                         appearance['keyword'][topic]['isSeed'] = True
#                         break
#                     else:
#                         # appearance['keyword'][topic]['ranking'] = j
#                         appearance['keyword'][topic]['score'] = str(score)
#                         appearance['keyword'][topic]['relative_score'] = str(score/average_score_topic)
#
#
#
#         except Exception as e:
#             exc_tb = sys.exc_info()[2]
#             exc_line = exc_tb.tb_lineno
#             print(e)
#
#
#
#     def updateBadwords(self):
#
#         no_above = 1 - 0.5 * (1 - (1 /( 1 + math.log10(self.semi_filtred_dictionnary.num_docs / 100))))
#         abs_no_above = no_above * self.semi_filtred_dictionnary.num_docs
#         rel_no_bellow = 0.00005
#         abs_no_bellow = rel_no_bellow * self.semi_filtred_dictionnary.num_docs
#
#         self.bad_words = [ word for id , word in self.semi_filtred_dictionnary.items() if abs_no_bellow > self.semi_filtred_dictionnary.dfs[id] or self.semi_filtred_dictionnary.dfs[id] > abs_no_above ]
#         self.bad_words += self.predefinedBadWords
#
#
#
#
#     def compareTopicSequentialy (self , topic , first_w = 0 , last_w = 0,ntop = 100,fixeWindow = False, exclusive=False, soft=False, removeSeedWords=True):
#         if self.guided != True:
#             raise Exception("you can't use this function for normal lda but just for guided lda because it return similarity temporal vector also normal lda return matrix similarity ")
#
#         if last_w == 0:
#             last_w = len(self.LDAs)
#
#         if len(self.LDAs) != 0:
#             topic, topic_id = self.getTopicNTopicID(topic)
#
#             if fixeWindow == True:
#                 return [
#                     self.compareWordsTopicsW_W('jaccard' , ntop, first_w,i , exclusive=exclusive, soft=soft,
#                                                removeSeedWords=removeSeedWords)[
#                         topic_id] for i in range(first_w + 1, last_w)]
#             else:
#                 return [self.compareWordsTopicsW_W('jaccard' , ntop, i, i + 1, exclusive=exclusive, soft=soft, removeSeedWords=removeSeedWords)[topic_id] for i in range ( first_w , last_w - 1)]
#         else:
#             raise Exception('the model is empty please add window')
#
#
#
#     def visualizeWordOccurenceEvolution(self , word, df=False):
#
#         x = [i for i in range (len(self.LDAs))]
#         y=[0 for _ in range (len(self.LDAs))]
#         word_appearances = self.res[word]['appearances']
#         if df:
#             for appearance in word_appearances :
#                 y[appearance['no_window']] = appearance['df_in_window']
#         else:
#             for appearance in word_appearances :
#                 y[appearance['no_window']] = appearance['cf_in_window']
#         plt.plot(x , y)
#         plt.show()
#
#
#     def visualizeWordEvolutionInTopic (self , word , topic_id , relative = True):
#
#         x = [i for i in range (len(self.LDAs))]
#         y=[0 for _ in range (len(self.LDAs))]
#         if relative:
#             for appearance in self.res[word]['appearances']:
#                 y[appearance['no_window']] = appearance['keyword'][topic_id]['relative_score']
#         else:
#             for appearance in self.res[word]['appearances']:
#                 y[appearance['no_window']] = appearance['keyword'][topic_id]['score']
#
#         plt.plot(x, y)
#         plt.show()
#
#
#
#     def visualizeTopicEvolution(self, topic_id,first_w = 0 , last_w = 0, ntop = 100, fixeWindow = False, num_window=None, exclusive=False, soft=False, removeSeedWords=True):
#
#
#         y = self.compareTopicSequentialy(topic_id , first_w=first_w , last_w=last_w ,  ntop = ntop,fixeWindow = fixeWindow, exclusive=exclusive,
#                                          soft=soft, removeSeedWords=removeSeedWords)
#         self.dates = {no_w : date for date , no_w in self.dates.items()}
#         dates = [datetime.fromtimestamp(float(self.dates[i])) for i in range(first_w + 1, last_w)]
#         topic = self.table_keys_topics[topic_id]
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval= (last_w - first_w)//10))
#         plt.plot(dates, y)
#         plt.title(f'evolution of the topic : {topic}')
#         plt.gcf().autofmt_xdate()
#         plt.show()
#
#     #def compareTopics
#
#     def compareWordsTopicsW_W(self, distance = 'jaccard', ntop = 100, ith_window = 0, jth_window = 1, exclusive=False, soft=False, removeSeedWords=True , normed=True):
#
#
#         if self.guided == True and distance != 'jaccard':
#             raise Exception(f"you can't use {distance} distance for guided Lda . please use 'jaccard' distance ")
#
#         if self.guided == True:
#         # list of sets of top words per topics in ith window
#
#             stwptiw = self.topWordsPerTopics(self.LDAs[ith_window], ntop=ntop, exclusive=exclusive,
#                                              removeSeedWords=removeSeedWords)
#             # list of sets of top words per topics in jth window
#             stwptjw = self.topWordsPerTopics(self.LDAs[jth_window], ntop=ntop, exclusive=exclusive,
#                                              removeSeedWords=removeSeedWords)
#             if soft == False:
#                 ridgeScoreIW = [
#                     len(stwptiw[0][topic_id].intersection(stwptjw[0][topic_id])) / len(stwptiw[0][topic_id]) if len(stwptiw[0][topic_id]) != 0 else 0
#                     for topic_id in range(len(stwptiw[0])) ]
#                 return ridgeScoreIW
#             else:
#                 softScoreIW = []
#                 W_W_Score = []
#
#                 for topic_id in range(len(stwptiw[0])):
#                     softScoreTopic = 0
#                     W_W_ScoreTopic = 0
#                     total = 0
#                     set_intersection = stwptiw[0][topic_id].intersection(stwptjw[0][topic_id])
#                     for id_word in stwptiw[1][topic_id].keys():
#                         # we accept that we take the probability score given by the glda model affected for the word in the ith window and not in the jth window
#                         if id_word in set_intersection:
#                             softScoreTopic += stwptiw[1][topic_id][id_word]
#                             W_W_ScoreTopic += abs(stwptiw[1][topic_id][id_word] - stwptjw[1][topic_id][id_word])
#                         else:
#                             W_W_ScoreTopic += stwptiw[1][topic_id][id_word]
#                         total += stwptiw[1][topic_id][id_word]
#                     if total == 0:
#                         softScoreTopic=0
#                     else:
#                         softScoreTopic = (softScoreTopic / total)
#                     softScoreIW.append(softScoreTopic)
#                     W_W_Score.append(W_W_ScoreTopic)
#                 return softScoreIW
#
#         if self.guided != True:
#
#             self.LDAs[ith_window].diff(self.LDAs[jth_window] , distance=distance , num_words=ntop , normed=normed , annotation=False)
#
#
#
#
#     def getTopWordsForTopicForWindow(self, topic, ntop, nu_window, exclusive=False, removeSeedWords=True):
#         """
#
#         @param ntop: number of top words to return
#         @param nu_window: numero of window
#         @param topic: numero of topic
#         @param exclusive: if the word is exclusive to one topic
#         @param removeSeedWords: remove word that belong to topic seed word
#         """
#         if exclusive==False:
#             return self.topWordsForCurrentTopic(topic, self.LDAs[nu_window], ntop, removeSeedWords=removeSeedWords)[0]
#         else:
#             return self.topWordsPerTopics(self.LDAs[nu_window], ntop, exclusive=exclusive, removeSeedWords=removeSeedWords)[0][topic]
#
#
#
#
#     def topWordsPerTopics(self, lda, ntop, exclusive=False , removeSeedWords=True):
#         """
#         :param ntop: number of keywords that the model return by topic
#         :param exclusive: if we want that the keywors being exclusive to the topic
#         return: list of list id of words in the dictionnary, one list by gldaModel so one list by time intervall
#         """
#         ListSetOfTopWordsPerTopic=[]
#         ListDictOfTopWordsPerTopic=[]
#         for topic in range(lda.num_topics):
#             setOfTopWordsCurrentTopic , dictOfTopWordsCurrentTopic = self.topWordsForCurrentTopic(topic, lda, ntop , removeSeedWords=removeSeedWords)
#             ListSetOfTopWordsPerTopic.append(setOfTopWordsCurrentTopic)
#             ListDictOfTopWordsPerTopic.append(dictOfTopWordsCurrentTopic)
#         #setOfTopWordsPerTopic = [set(glda.get_topic_terms(topic_id, topn=ntop)) for topic_id in range(glda.num_topics)]
#
#         if exclusive == False:
#
#             return ListSetOfTopWordsPerTopic , ListDictOfTopWordsPerTopic
#         else:
#             return self.exclusiveWordsPerTopics(ListSetOfTopWordsPerTopic) , ListDictOfTopWordsPerTopic
#
#
#
#     def topWordsForCurrentTopic(self, topic, lda_window, ntop, removeSeedWords=True):
#         """
#         :param ntop: number of keywords that the model return by topic
#         :param exclusive: if we want that the keywors being exclusive to the topic
#         return: list of list id of words in the dictionnary, one list by gldaModel so one list by time intervall
#         """
#         #get the the value e.g: 'crime' and the key e.g: 0 of the topic
#         topic , topic_id  = self.getTopicNTopicID(topic)
#
#         setOfTopWordsCurrentTopic=set()
#         dictOfTopWordsCurrentTopic={}
#         # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
#         listOfTopWords = lda_window.get_topic_terms(topicid= topic_id, topn=ntop)
#         for word_id , score in listOfTopWords:
#             word = lda_window.id2word[word_id]
#             if removeSeedWords:
#                 if word not in self.seed[topic]:
#                     setOfTopWordsCurrentTopic.add(word)
#                     dictOfTopWordsCurrentTopic[word] = score
#                     if len(setOfTopWordsCurrentTopic) == ntop:
#                         break
#
#             else:
#                 setOfTopWordsCurrentTopic.add(word)
#                 dictOfTopWordsCurrentTopic[word] = score
#                 if len(setOfTopWordsCurrentTopic) == ntop:
#                     break
#
#         return setOfTopWordsCurrentTopic , dictOfTopWordsCurrentTopic
#
#
#
#     def exclusiveWordsPerTopics(self, listOfSetOfWords):
#
#         listOfSetOfExclusiveWords = []
#         for i, setOfWords in enumerate(listOfSetOfWords):
#             listOfNoExclusiveWords = set()
#             for j in range(len(listOfSetOfWords)):
#                 if i == j:
#                     pass
#                 else:
#                     listOfNoExclusiveWords = listOfNoExclusiveWords.union(listOfSetOfWords[j])
#             listOfSetOfExclusiveWords.append(setOfWords.difference(listOfNoExclusiveWords))
#
#         return listOfSetOfExclusiveWords
#
#
#
#     def getRes(self):
#         return  self.res
#
#
#     def getTopicNTopicID(self , topic):
#
#         try:
#             if isinstance(topic, str):
#                 topic_id = self.table_topics_keys[topic]
#             if isinstance(topic, int):
#                 topic_id = topic
#                 topic = self.table_keys_topics[topic]
#
#             return topic , topic_id
#
#         except KeyError:
#             print("this key doesn't exist in your seed")
#
#
#     def save(self ,FolderPath):
#
#         modelFolderPath = os.path.join(FolderPath, self.ldaModelFolder)
#         #check if the folder exist
#         if not os.path.exists(modelFolderPath):
#             os.makedirs(modelFolderPath)
#
#         #save info
#         infoPath = os.path.join(FolderPath, self.info_file)
#         with open (infoPath,'w') as finfo:
#             finfo.write(json.dumps(self.info))
#
#         # save lda model one by one
#
#         for i, glda in enumerate(self.LDAs):
#             fileName = self.ldaFileName + str(i)
#             filePath = os.path.join (modelFolderPath , fileName)
#             glda.save(filePath)
#             self.listLDAFile.append(fileName)
#
#         #save the list of lda files name for make the loading easier
#         listPath = os.path.join(FolderPath, self.listLDAName)
#         with open (listPath,'w') as flist:
#             flist.write(json.dumps(self.listLDAFile))
#
#
#         #save semi-dictionnary correlated to the model above
#         semiDictionnaryPath = os.path.join(FolderPath, self.semi_dictionnaryFileName)
#         self.semi_filtred_dictionnary.save(semiDictionnaryPath)
#
#         if self.guided == True:
#
#             # save seed correlated to the model above
#             seedPath = os.path.join(FolderPath, self.seedFileName)
#             with open (seedPath,'w') as fseed:
#                 fseed.write(json.dumps(self.seed))
#
#         # save res
#         resPath = os.path.join(FolderPath, self.resFileName)
#         with open(resPath, 'w') as fres:
#             fres.write(json.dumps(self.res))
#
#         #save end date of window
#         datePath = os.path.join(FolderPath, self.dateFile)
#         with open(datePath, 'w') as fdate:
#             fdate.write(json.dumps(self.dates))
#
#         # save coherence scores
#         if self.guided != True:
#             coherencePath = os.path.join(FolderPath, self.coherence_score_file)
#             with open(coherencePath, 'w') as fcohe:
#                 fcohe.write(json.dumps(self.coherence_scores))
#         else:
#             countPath = os.path.join(FolderPath, self.counterLabels_file)
#             with open(countPath, 'w') as fcount:
#                 fcount.write(json.dumps(self.counterLabels))
#
#
#     def load(self , FolderPath):
#
#         modelFolderPath = os.path.join(FolderPath, self.ldaModelFolder)
#         # load lda model one by one
#         with open(os.path.join(FolderPath , self.listLDAName), 'r') as flist:
#             self.listLDAFile = json.load(flist)
#         self.LDAs = [LdaModel.load(os.path.join(modelFolderPath, file)) for file in self.listLDAFile]
#         self.nb_windows = len(self.LDAs)
#
#         #load info
#         infoPath = os.path.join(FolderPath, self.info_file)
#         with open(infoPath, 'r') as finfo:
#             self.info = json.load(finfo)
#         self.guided = self.info['guided']
#         self.nb_topics = self.info['nb_topics']
#
#         # self.guided = True
#         # self.nb_topics = 4
#
#         # load semi-dictionnary correlated to the model above
#         semiDictionnaryPath = os.path.join(FolderPath , self.semi_dictionnaryFileName)
#         self.semi_filtred_dictionnary = self.semi_filtred_dictionnary.load(semiDictionnaryPath)
#
#         if self.guided == True:
#             # load seed correlated to the model above
#             seedPath = os.path.join(FolderPath, self.seedFileName)
#             with open(seedPath, 'r') as fseed:
#                 self.seed = json.load(fseed)
#             #put key of seed as integer because json serilization transform it as a char e.g : '0'
#             self.table_topics_keys  = { key : i for i , key in enumerate(self.seed) }
#             self.table_keys_topics  = { i : key for i , key in enumerate(self.seed) }
#
#
#
#         # load res
#         resPath = os.path.join(FolderPath, self.resFileName)
#         with open(resPath, 'r') as fres:
#             self.res = json.load(fres)
#
#         # load end date of window
#         datePath = os.path.join(FolderPath, self.dateFile)
#         with open(datePath, 'r') as fdate:
#             self.dates = json.load(fdate)
#
#         # load coherence scores or count
#         if self.guided != True:
#             coherencePath = os.path.join(FolderPath, self.coherence_score_file)
#             with open(coherencePath, 'r') as fcohe:
#                 self.coherence_scores = json.load(fcohe)
#         else:
#             countPath = os.path.join(FolderPath, self.counterLabels_file)
#             with open(countPath, 'r') as fcount:
#                 self.counterLabels = json.load(fcount)























def check_size(func):
    def wrapper(*args):
        if len(args[0]) == 0:
            raise Exception('documents empty , we can not process the sequence')
        func(*args)
    return wrapper



class SequencialLangageModeling:



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
    def treat_Window(self,texts : List[List] ,  **kwargs):

        window_dictionnary = corpora.Dictionary(texts)
        # update semi-filtred dictionnary
        self.semi_filtred_dictionnary.merge_with(window_dictionnary)
        #we filtre bad words from window_dictionnary
        self.updateBadwords()
        window_dictionnary_f = filterDictionnary(window_dictionnary , bad_words=self.bad_words)
        # train specific Engine model correlated to the window
        model = self.engine(texts=texts , n_topics=self.nb_topics ,**kwargs )

        return model, window_dictionnary_f


    def add_windows(self, data : TimeLineArticlesDataset, lookback = 10, update_res = True):

        self.info['lookback'] = lookback
        rValue= random.Random()
        rValue.seed(37)
        for i  , (end_date_window , texts_window) in (enumerate(data)):
            random_state = rValue.randint(1, 14340)
            print(f"numero of window: {i} -- random state: {random_state}")
            print(f"size documents: {len(texts_window)} ")
            print("-"*30)
            kwargs = {"random_state" : random_state}
            try:
                model , window_dictionnary = self.treat_Window(texts_window , **kwargs)
                # for bound window to the right glda model we use no_window
                no_window = i
                self.updateBadwords()
                if update_res:
                    self.updateResults(end_date_window , window_dictionnary , model , no_window)
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
        for topic_id in range(model.n_topics):
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



class NoSupervisedSequantiallangageModeling(SequencialLangageModeling):

    def compareTopicSequentialy(self):
        pass

    def calcule_similarity_topics_W_W(self):
        # find a way to compare no unknown topic from 2 differents windows
        pass


class SupervisedSequantialLangagemodeling(SequencialLangageModeling):

    def __init__(self , **kwargs):
        super(SupervisedSequantialLangagemodeling, self).__init__(**kwargs)
        self.engine = Engine.SupervisedEngine


    def add_windows(self, data : TimeLineArticlesDataset, lookback = 10, update_res = True):

        self.info['lookback'] = lookback
        rValue = random.Random()
        rValue.seed(37)
        for i, (end_date_window, (texts_window , labels)) in (enumerate(data)):
            random_state = rValue.randint(1, 14340)
            print(f"numero of window: {i} -- random state: {random_state}")
            print(f"size documents: {len(texts_window)} ")
            print("-" * 30)
            kwargs = {"random_state": random_state , "labels" : labels}
            try:
                model, window_dictionnary = self.treat_Window(texts_window, **kwargs)
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

    def compareTopicSequentialy(self, topic_id, first_w=0, last_w=0, ntop=100, fixeWindow=False, **kwargs):

        # we use thi condition to set the numero of the last window because by
        # default we want to compute similarity until the last window
        if last_w == 0:
            last_w = len(self.models)
        if fixeWindow == True:
            return [self.calcule_similarity_topics_W_W('jaccard', ntop, first_w, i, **kwargs)[topic_id] for i in
                    range(first_w + 1, last_w)]
        else:
            return [self.calcule_similarity_topics_W_W('jaccard', ntop, i, i + 1, **kwargs)[topic_id] for i in
                    range(first_w, last_w - 1)]

    def calcule_similarity_topics_W_W(self, distance='jaccard', ntop=100, ith_window=0, jth_window=1, soft=False,
                                      **kwargs):

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

    def __init__(self , seed ,  **kwargs):
        super(GuidedSequantialLangagemodeling, self).__init__(**kwargs)
        self.engine = Engine.GuidedEngine
        self.seed = seed


    def getTopWordsTopic(self, topic_id, model : Engine = None, ntop : int = 100 , remove_seed_words : bool = True):

        # implement new technic to remove seed words before generate list of ntop words to have a output list with the exact number of words asking by the users
        topWords = model.get_topic_terms(topicid= topic_id, topn=ntop)
        if remove_seed_words:
            for word in topWords.keys():
                if word in self.seed[topic_id]:
                    del (topWords[word])
        topWordsTopic = {topWord[0] : topWord[1] for topWord in topWords}
        return topWordsTopic


class LDASequantialModeling(SequencialLangageModeling):

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
        model = self.engine(texts=texts, n_topics=self.nb_topics, dictionnary=window_dictionnary_f, **kwargs)

        return model, window_dictionnary_f



class GuidedLDASenquentialModeling( LDASequantialModeling , GuidedSequantialLangagemodeling ):

    def __init__(self , **kwargs):
        super(GuidedLDASenquentialModeling, self).__init__(**kwargs)
        self.engine = Engine.GuidedLDA







if __name__ == '__main__':

    #load database , define Time-windows , splitt data
    start_time = 1620637617 # 10 may 9 hour A.M
    end_time = 1626340017 #15 jully 9 hour A.M
    deltaTime=24 # in hour
    folderPath='/home/mouss/data'

    dataBaseName1 = 'final_database_50000_100000.json_2'
    dataBaseOriginalName = 'final_database.json'
    dataSplitName = 'text_per_window_data_50000.json'
    seedName = 'seed_test_200'
    seedPath = os.path.join(folderPath , seedName)
    dataBasePath1 = os.path.join(folderPath, dataBaseName1)
    dataSplitPath = os.path.join(folderPath, dataSplitName)
    model1Name  = 'model_1'
    model1Path = os.path.join(folderPath , model1Name)

    topic_model = ct.Corex(n_hidden=2)
    topic_model.get_topics()


    # with open(dataBasePath1 , 'r') as f:
    #      data = json.load(f)
    #
    # data = splittDataPipeline(data , delta_time=deltaTime )
    # print('f')
    # # #
    # # # # # # #
    # #save the windows with processed text without remake the process
    # #put data in json format (because tuple is not json serializable):
    # data = [[window[0] ,window[1] ] for window in data ]
    # with open (dataSplitPath,'w') as f:
    #     f.write(json.dumps(data))
    #
    #
    # #load the windows with processed text
    # with open (dataSplitPath,'r') as f:
    #     data = json.load(f)
    # data=[(window[0] , window[1]) for window in data]
    #
    # # #fake_data = [(data[100][0] , data[100][1]+data[101][1]+data[102][1]) for _ in range (10)]
    # # #analyseStreamCollect(data[:100] , perLabel= True)
    #
    #
    # #generate seed randomly
    # random_seed = SeedGenerator(databasePath=dataBasePath1, k=200).generateSeed(exclusive=True, doLFIDF=True, nbWordsByLabel=200)
    #
    # with open(seedPath , 'r') as fs:
    #     seed = json.load(fs)
    #
    # MTGLDA=SequencialLangageModeling(guided=True, seed=random_seed)
    # start_time = time.time()
    # MTGLDA.add_windows(data , lookback=0.5)
    # #MTGLDA.visualizeTopicEvolution(0 , ntop= 100)
    #
    # MTGLDA.getRes()
    # MTGLDA.save(model1Path)
    #
    # MTGLDA = SequencialLangageModeling()
    # MTGLDA.load(model1Path)
    # MTGLDA.visualizeTopicEvolution(1 , ntop=400 , soft=True)
    # print('f')

















