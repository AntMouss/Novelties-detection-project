import json
import math
from novelties_detection.Collection.data_processing import fileToObject
import random


class SeedGenerator:

    """
    use this class to generate seed object.
    In our case seed objet is a set of words correlated to a specific label in our database.
    Because each aticle is labelized we can use this class to return one word set by label in the database
    """

    def __init__(self, seedArticlePath=None, databasePath=None, k=10):
        """
        we can use pre-selected
        @param seedArticlePath:
        @param databasePath:
        @param k:
        """
        self.labels=['crime' , 'politique' , 'economy' , 'justice' , 'general']
        self.seedPath = seedArticlePath
        self.databasePath = databasePath
        self.database_with_keys = fileToObject(self.databasePath)
        self.k = k
        self.labelDictionnary = {}


        if self.seedPath is None:
           self.seedArticlesId = self.selectArticlesSeedRandomly()
        else:
            self.seedArticlesId = fileToObject(self.seedPath)



    def selectArticlesSeedRandomly(self):


        articlesRandomlySelected = []
        articlesRandomlySelectedId = []
        articlesIdPerlabel = {}
        for article_id in self.database_with_keys:
            try:
                label_article = self.database_with_keys[article_id]['label'][0]
                if label_article in self.labels:
                    if label_article not in articlesIdPerlabel.keys():
                        articlesIdPerlabel[label_article] = []
                    articlesIdPerlabel[label_article].append(article_id)
            except Exception as e:
                continue
        for label in articlesIdPerlabel.keys():
            count = 0
            if self.k > math.floor(5*(len(articlesIdPerlabel[label]) / 100)):
                k_real = math.floor(5*(len(articlesIdPerlabel[label]) / 100))
            else:
                k_real = self.k
            while(count != k_real):
                article_id_randomly_selected = random.choice(articlesIdPerlabel[label])
                if len(self.database_with_keys[article_id_randomly_selected]['process_text']) != 0 and article_id_randomly_selected not in articlesRandomlySelectedId:
                    count += 1
                    articlesRandomlySelectedId.append(article_id_randomly_selected)

        return articlesRandomlySelectedId



    def updateLabelDictionnary (self , processedText , label):

        first = []
        try:
            for word in processedText:
                if word not in self.labelDictionnary.keys():
                    self.labelDictionnary[word] = {}
                    self.labelDictionnary[word]['dfs'] = 0
                    self.labelDictionnary[word]['cfs'] = 0

                if label not in self.labelDictionnary[word].keys():
                    self.labelDictionnary[word][label] = {}
                    self.labelDictionnary[word][label]['dfs_label'] = 0
                    self.labelDictionnary[word][label]['cfs_label'] = 0

                self.labelDictionnary[word]['cfs'] += 1
                self.labelDictionnary[word][label]['cfs_label'] += 1

                if word not in first:
                    self.labelDictionnary[word][label]['dfs_label'] += 1
                    self.labelDictionnary[word]['dfs'] += 1
                    first.append(word)
        except Exception as e:
            print(e)



    def fetchLabelsTexts(self):
        """
        for each label process and append all text in the according list (according to the label)

        """

        labelsTexts = {}
        for article_id in self.seedArticlesId:
            article_label = self.database_with_keys[article_id]['label'][0]
            article_text = self.database_with_keys[article_id]['process_text']
            if article_label not in labelsTexts.keys():
                labelsTexts[article_label] = []
            self.updateLabelDictionnary(article_text , article_label)
            labelsTexts[article_label] += article_text
        for label in labelsTexts.keys():
            labelsTexts[label] = set(labelsTexts[label])

        return labelsTexts

        # return [fileToJson(path)['text']for path in seedPath]


    def generateSeed(self , exclusive =True , doLFIDF = False , nbWordsByLabel = 0):
        """"
        for each topic remove world that are not exclusive to a topic
        and return a dictionnary with labels as keys and set of words as values

        """
        labelsSets = self.fetchLabelsTexts()
        filtredLabelsSets = {}
        # we can keep the same dictionnary during the iteration because when we modified a set from a label we lost the words that are not exclusive to the current label and we can use this word for eliminate no exclusive words to the others label so we had to make a new dictionnary
        if exclusive:
            for i , targetLabel in enumerate(labelsSets.keys()):

                filtredLabelsSets[targetLabel] = set()
                setOthersLabel = set()
                for label in labelsSets.keys():
                    if label != targetLabel:
                        setOthersLabel = setOthersLabel.union(labelsSets[label])

                    filtredLabelsSets[targetLabel] = list(labelsSets[targetLabel].difference(setOthersLabel))
        else:
            # we don't filter the set of words Label
            filtredLabelsSets = labelsSets

        # make ranking for select more significant seed words

        if doLFIDF:
            for label in filtredLabelsSets:
                lfidf_liste = []
                for word in filtredLabelsSets[label]:
                    lfidf_score = self.labelDictionnary[word][label]['dfs_label']
                    # if self.labelDictionnary[word][label]['dfs_label']>1:
                    #     print('erer')
                    # lfidf_score = (self.labelDictionnary[word][label]['dfs_label']/self.k )/ (self.labelDictionnary[word]['dfs']/(len(self.labels)*self.k))
                    lfidf_liste.append((word , lfidf_score))
                lfidf_liste_sorted = sorted(lfidf_liste, key=lambda tup: tup[1] , reverse=True)
                if nbWordsByLabel > len(lfidf_liste_sorted):
                    filtredLabelsSets[label] = [tpl[0] for tpl in lfidf_liste_sorted]
                else:
                    filtredLabelsSets[label] = [tpl[0] for tpl in lfidf_liste_sorted[:nbWordsByLabel]]
        return filtredLabelsSets




    def generatePriors(self):
        """

        :return:dictionnary of priors with words as keys and label as value
        """
        priors = {}
        labelSet = self.generateSeed()
        # priors={ word : label for word in labelSet[label] for label in labelSet.keys() }

        for label in labelSet.keys():
            for word in labelSet[label]:
                priors[word] = label
        return priors

    @staticmethod
    def filterPriors(priors, dictionnary):
        """

        :param priors: dictionnary with words as keys and label as value
        :param dictionnary: gensim reference dictionnary based on our all corpus
        :return: other priors that is a copy of the input priors without the word that is not in the reference dictionnary
        """
        priorsToReturn = priors.copy()
        for word in priors.keys():

            try:
                dictionnary.token2id[word]
            except KeyError as e:
                del priorsToReturn[word]
        return priorsToReturn

    @staticmethod
    def savePriors(priors, filePath):

        with open(filePath, 'w') as f:
            f.write(json.dumps(priors))

    @staticmethod
    def loadPriors(filePath):

        with open(filePath, 'r') as f:
            return json.load(f)
