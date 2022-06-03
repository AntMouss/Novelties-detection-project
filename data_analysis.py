
import numpy as np
import copy
import os
import json
import statistics
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


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


def caracterisationExperiences(resultats):

    # assume that all the timeline in the experiences have the same size
    timeline_size = len(resultats[0]['serie'][0]['general'][0]['data']) + 1
    car_experiences = {}

    experiences = [ res['serie'][2]['with'] for res in resultats ]

    #assume that it's same lookback for all experiences
    lookback = experiences[0]['lookback']



    res_experiences= []
    for resultat_experience in resultats:
        tmp_res = {}
        #label means categorie: 'general' , 'justice' , 'politique' ...
        for label , data_s in resultat_experience['serie'][0].items():

            diffBetween_W_n_Wout = [ abs(data_s[0]['data'][i][1] - data_s[1]['data'][i][1]) for i in range (len(data_s[0]['data']))]
            percent_articles_thematic = []

            for i in range(len(data_s[0]['data'])):
                if (data_s[0]['data'][i][2] - lookback) != 0:
                    percent_articles_thematic.append(((data_s[1]['data'][i][2] - lookback) - (data_s[0]['data'][i][2] - lookback)) / (data_s[0]['data'][i][2] - lookback))
                else:
                    percent_articles_thematic.append(0)

            tmp_res[label] = (diffBetween_W_n_Wout , percent_articles_thematic)
        res_experiences.append(tmp_res)

    #calcul te correlation between variation of number of thematics articles and difference of similarity between w model and wout model
    diff_vector = []
    var_articles_percent_vector = []
    for label , res in tmp_res.items():
        var_articles_percent_vector += [0]
        diff_vector += res[0]
        var_articles_percent_vector += [abs((res[1][i-1] - res[1][i])) for i in range (len(res[1])) if i != 0]
    print("correlation pearson scipy : "+str(pearsonr(diff_vector , var_articles_percent_vector)[0]))
    #print("correlation coefficient numpy : "+str(np.corrcoef(diff_vector , var_articles_percent_vector)))




    #better way for caracterisation {label: { type_window : [] , ...} , ...}
    for i , experience in enumerate(experiences):

        tmp_res = res_experiences[i]
        first_entry = experience['ranges'][0][0]
        last_out = experience['ranges'][-1][1]

        for label in tmp_res.keys():
            if label not in car_experiences:
                car_experiences[label] = {}
            for j in range(1, timeline_size):
                if j < first_entry:
                    if 'before' not in car_experiences[label].keys():
                        car_experiences[label]['before'] = []
                    car_experiences[label]['before'].append(tmp_res[label][0][j - 1])
                    continue
                if j > last_out+1:
                    if 'after' not in car_experiences[label].keys():
                        car_experiences[label]['after'] = []
                    car_experiences[label]['after'].append(tmp_res[label][0][j - 1])
                    continue
                passage = False
                for n_range in experience['ranges']:
                    if j == n_range[0]:
                        if 'entry' not in car_experiences[label].keys():
                            car_experiences[label]['entry'] = []
                        car_experiences[label]['entry'].append(tmp_res[label][0][j - 1])
                        passage = True
                        break
                    if j == n_range[1] + 1:
                        if 'out' not in car_experiences[label].keys():
                            car_experiences[label]['out'] = []
                        car_experiences[label]['out'].append(tmp_res[label][0][j - 1])
                        passage = True
                        break
                    if n_range[0] < j <= n_range[1]:
                        if 'inside' not in car_experiences[label].keys():
                            car_experiences[label]['inside'] = []
                        car_experiences[label]['inside'].append(tmp_res[label][0][j - 1])
                        passage = True
                        break
                if passage:
                    continue
                else:
                    if 'outside' not in car_experiences[label].keys():
                        car_experiences[label]['outside'] = []
                    car_experiences[label]['outside'].append(tmp_res[label][0][j - 1])
                    continue


    return car_experiences




def calculeCaracteristics(car ):

    car_stat = copy.deepcopy(car)

    return {label : {type_window : {'mean' : statistics.mean(res) , 'std' : statistics.stdev(res) } for type_window , res in data.items()} for label , data in car_stat.items()}



def vizualiseCaracteristics(car):

    for label in car.keys():
        plt.boxplot([car[label][window_type] for window_type in car[label].keys()] , labels=[key for key in car[label]])
        plt.title(f'caracterisation of different types of window for the categorie : {label}')
        plt.show()





if __name__ == '__main__':

    #experiences path and root path
    root = '/home/mouss/data'
    experiences_file_path = os.path.join(root , 'myExperiences10.json')

    #load experiences results
    with open(experiences_file_path , 'r') as f:
        data = json.load(f)

    #format experiences data as caracterisation data
    caracterisation_data = caracterisationExperiences(data)
    vizualiseCaracteristics(caracterisation_data)
    car_statistique = calculeCaracteristics(caracterisation_data)
    print('f')