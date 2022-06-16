import numpy as np
import copy
from data_utils import ExperiencesResults
import statistics
from matplotlib import pyplot as plt
from scipy.stats import pearsonr , ttest_ind



class Sampler:

    def __init__(self , results : ExperiencesResults ):

        self.results = results.results
        self.info = results.info

    @property
    def samples(self):

        topic_samples = {
            "before": [],
            "after": [],
            "middle": [],
            "in": [],
            "out": [],
            "inside": []
        }
        samples = [topic_samples for _ in range(self.info["nb_topics"])]
        for result in self.results:
            similarity = result.similarity
            for i , (topic_res_w , topic_res_wout) in enumerate(zip(similarity['with'] , similarity['without'])):
                tmp = abs(topic_res_w - topic_res_wout)
                for j , value in enumerate(tmp):
                    key = Sampler.choose_key(j, result.metadata.ranges)
                    samples[i][key].append(value)
        return samples


    @staticmethod
    def choose_key(idx_window , ranges):

        if idx_window < ranges[0][0]:
            return 'before'
        elif idx_window > ranges[-1][1]:
            return 'after'
        else:
            for entry , out in ranges:
                if idx_window == out:
                    return 'out'
                elif idx_window == entry:
                    return 'in'
                elif entry < idx_window < out:
                    return 'inside'
            return 'middle'


class Analyser:

    def __init__(self , samples , risk = 0.05):
        self.risk = risk
        self.samples = samples
        self.nb_topics = len(samples)

    def topic_pvalue_matrix(self , topic_id , trim = 0  , verbose = True):

        topic_samples = self.samples[topic_id]
        types_window = list(topic_samples.keys())
        nb_windows = len(types_window)
        pvalue_matrix = np.zeros((nb_windows , nb_windows))
        for i in range(len(types_window)):
            a = topic_samples[types_window[i]]
            for j in range(i , len(types_window)):
                b = topic_samples[types_window[j]]
                _ , pvalue = ttest_ind(a , b , trim=trim)
                if pvalue < self.risk and verbose:
                    print('H0 refused')
                pvalue_matrix[i][j] = pvalue
                pvalue_matrix[j][i] = pvalue
        return pvalue_matrix
















def testModel(corpus, model, id2topic, topic2id):
    scoreStrict = 0
    scoreSoft = 0
    for bow, label in corpus:
        res = model.get_document_topics([tuple(token) for token in bow])
        # i'd to check if the index in the table label accord to the index for topics in the model using function model
        try:
            scoreSoft += res[topic2id[label]][1]
            if topic2id[label] == res.index(max(res)):
                scoreStrict += 1
        except Exception as e:
            print(e)

    return scoreStrict / len(corpus), scoreSoft / len(corpus)


# we don't see priors words

def getDistributionTopics(model, i_topic, topn, n_points):
    # the model don't give inferior to 10^-8 probability

    # top word score for the current topic
    topWord = (model.get_topic_terms(i_topic, topn=1))[0][1]
    linearIntervall = np.linspace(0, topWord, n_points)
    linearScore = np.zeros(len(linearIntervall))
    # logspace doesn't work like the exemple bellow so i use other solution that is not perfect
    # logIntervall = np.logspace(0, topWord, int(n_points / 10))
    logIntervall = np.logspace(-3, -7, 15)
    logScore = np.zeros(len(logIntervall))
    words = model.get_topic_terms(i_topic, topn=topn)
    for i, word in enumerate(words):
        for j in range(len(linearIntervall) - 1):
            if linearIntervall[j] >= word[1] > linearIntervall[j + 1]:
                linearScore[j] += 1
        # for j in range (len(logIntervall)-1):
        #     if logIntervall[j]>=word[1]>logIntervall[j+1]:
        #         logScore[j] += 1
    return linearScore, linearIntervall, logScore, logIntervall


# histogram to visualize the distribution
def vizDistributionTopics(model, topic, topn, n_points):
    linearScore, linearIntervall, logScore, logIntervall = getDistributionTopics(model, topic, topn, n_points)
    plt.hist(linearScore)
    plt.show()


def analyseStreamCollect(timeWindow_data , perLabel = True):
    """
    use this function for visualisation as time-line of the numbers of articles according to their label
    @param timeWindow_data: data on timeWindow format -->  label of each articles sorted by time and splitted according to the size of the time-window
    @perLabel : plot evolution of numbers of articles for all label
    """
    #evolution of number of articles per time-window
    global id_label
    plt.figure()
    if perLabel:
        plt.title("number of articles per window per label")
        y = []
        label_to_key = {}
        b = 0
        for window in timeWindow_data:
            y_window = {}
            for label in window[1]:
                label_article = label[0]
                if label_article not in label_to_key.keys():
                    label_to_key[label_article] = b
                    b += 1
                if label_to_key[label_article] not in y_window.keys():
                    id_label = label_to_key[label_article]
                    y_window[id_label] = 0
                y_window[id_label] += 1
            y.append(y_window)
        y_np = np.zeros((len(timeWindow_data) , b))
        key_to_label = { id : label for label , id in label_to_key.items()}
        for l in range (y_np.shape[0]):
            for c in range (y_np.shape[1]):
                try:
                    y_np[l , c] = y[l][c]
                except KeyError:
                    y_np[l  , c] = 0
                    pass
        for i in range (y_np.shape[1]):
            plt.plot( y_np[: , i] , label=key_to_label[i])
            plt.show()

    else:
        plt.title("number of articles per window")
        y = np.zeros(len(timeWindow_data))
        for i , window in enumerate(timeWindow_data):
            y[i] += (len(window[1]))
        plt.plot( y )
        plt.show()




def calculeCaracteristics(car ):

    car_stat = copy.deepcopy(car)

    return {label : {type_window : {'mean' : statistics.mean(res) , 'std' : statistics.stdev(res) } for type_window , res in data.items()} for label , data in car_stat.items()}



def vizualiseCaracteristics(car):

    for label in car.keys():
        plt.boxplot([car[label][window_type] for window_type in car[label].keys()] , labels=[key for key in car[label]])
        plt.title(f'caracterisation of different types of window for the categorie : {label}')
        plt.show()



