import math
import random
import copy

import json

from matplotlib import pyplot as plt
import os
from NLP_Module import SequencialLangageModeling , TimeWindows , SeedGenerator
from tqdm import tqdm





def createArtificialChange (timeline, thematics, experience):
    """
    @param timeline: original timeline with this format [(end_timeStamp,[id_article1 , id_article2...]) , (...)]
    @param thematics: specific thematics from the timeline
    @param modes: specify the thematic to inject and the numero of the window of the injection
    @rtype: original time line without the targeted thematics , time line with targeted thematics injected from a specific mode
    """
    artificialTimeline = copy.deepcopy(timeline)

    # set up the parameter of the mode
    thematic_name = experience['name']
    articles_ids_thematic = thematics[thematic_name]['articles']
    intervall_to_delete = []
    ranges = experience['ranges']
    for i, n_range in enumerate(ranges):
        if i==0:
            intervall_to_delete.append([-1 , n_range[0]])
            continue
        intervall_to_delete.append([ranges[i-1][1],n_range[0]])
        if i==len(ranges) - 1:
            intervall_to_delete.append([n_range[1] , len(timeline)])

    #remove article that belong to the specific thematic except for the window that belong to the 'mode intervall' for the timeline with thematic
    for i in range (len(timeline)):
        for article_id in timeline[i][1]:
            if article_id in articles_ids_thematic:
                timeline[i][1].remove(article_id)
                passage = False
                for intervall in intervall_to_delete:
                        if intervall[0]< i < intervall[1]:
                            artificialTimeline[i][1].remove(article_id)
                            passage = True
                            break
                if passage == False:
                    if experience['cheat']:
                        for _ in range(experience['boost']):
                            artificialTimeline[i][1].append(article_id)


                articles_ids_thematic.remove(article_id)



    return timeline , artificialTimeline


# def getNewWords(topWordsith , topWordsjth):
#
#     return [ word for word in topWordsjth if word not in topWordsith]



def generateResultsGLDAForInterface (model , mode , window_size):
    """
    generate result objet that contain timeline of the evolution of each topic for different window that we passsed in our sequencialy LDA objet
    @param model:
    @param mode:
    @param window_size:
    @return:
    """

    #format the data of the model for the timeline
    res = {}
    for topic_id in range(model.nb_topics):
        topic = model.table_keys_topics[topic_id]
        res[topic] = {}
        res[topic]['series'] = []
        res[topic]['words'] = {}
        cor_rates = model.compareTopicSequentialy(topic_id, soft=True)
        for nu_window in range(1, model.nb_windows):
            buble_id = topic + str(nu_window)
            top_words_last = model.getTopWordsForTopicForWindow(topic_id, 100, nu_window-1)
            top_words_current = model.getTopWordsForTopicForWindow(topic_id, 100, nu_window)
            new_words_current = top_words_current.difference(top_words_last)
            top_words_current = list(top_words_current)
            new_words_current = list(new_words_current)
            try:
                count_score = model.counterLabels[nu_window][topic]
            except KeyError:
                count_score = 0
            serie = [nu_window, round(cor_rates[nu_window-1] * 100), count_score]
            res[topic]['series'].append(serie)
            res[topic]['words'][buble_id] = [top_words_current , new_words_current]

    # format the data like this to be more easy to use in the trends interface and generate view with apexChart
    res_series = []
    for topic in res:
        dic = {}
        dic['name'] = topic
        dic['data'] = res[topic]['series']
        res_series.append(dic)
    res_words = {}
    for label in res.keys():
        res_words.update(res[label]['words'])

    # specific information about the time line and the model
    res_info = mode
    res_info['lookback'] = model.info['lookback']
    res_info['window_size'] = window_size
    res_final = [res_series , res_words , res_info]

    return res_final





def formatResultForTest(resW , resWout):
    """
    use the results of the topics evolution of the experience to compare the topic evolution with thematics injection and without thematics injection
    we format the data specificly for our interface. For each experience there are one data for series topics evolution(with and without injection) , one data for top and new words for a window,
    and informations related to the experience (information about the thematic injection , the loockback for the training model etc ...)
    @param resW: result obtained for timeline with injection
    @param resWout: result obtained for the timeline without thematic injection
    @return: new result objet with one timeline for the topic with injection and one timeline for the topic without injection , and others data related two the top and new words and information of the experience
    """
    res_series = {}
    for topic_series_w in resW[0]:
        topic_name = topic_series_w['name']

        # replace the key name by new name contain 'w' or 'wout' to be affected to the good serie.
        keyw = []
        for key in resW[1].keys():
            if topic_name in key:
                new_key = key.replace(topic_name , topic_name+'w')
                keyw.append([key , new_key])
        for el in keyw:
            resW[1][el[1]] = resW[1].pop(el[0])

        keywout = []
        for key in resWout[1].keys():
            if topic_name in key:
                new_key = key.replace(topic_name, topic_name + 'wout')
                keywout.append([key, new_key])
        for el in keywout:
            resWout[1][el[1]] = resWout[1].pop(el[0])

    word_data = {}
    word_data.update(resW[1])
    word_data.update(resWout[1])

    #change the key of the set of the top , new words to not confuse set with the thematic injected and set without the injection
    for topic_series_w in resW[0]:
        topic_name = topic_series_w['name']
        res_series[topic_name] = []
        for topic_series_wout in resWout[0]:
            if topic_series_w['name'] == topic_series_wout['name']:
                topic_series_wout['name'] = topic_series_wout['name'] + 'wout'
                topic_series_w['name'] = topic_series_w['name'] + 'w'
                res_series[topic_name].append(topic_series_wout)
                res_series[topic_name].append(topic_series_w)
                break

    # information about the experience for the timeline with and without injection
    res_info = {}
    res_info['with'] = resW[2]
    res_info['without'] = resWout[2]
    res_test = [res_series ,  word_data , res_info]
    return res_test






def generateWordTimeLineForTheInterface (res_data_model, nb_windows):
    """
    use the resultats file from the model to return the time line data of each word and
    plot the timeline in the interface.
    @param res_data_model: res from the model
    @param nb_windows: number of windows of the model
    @return: new res file with words as keys and timelines as values
    """
    res_interface = {}
    for word in res_data_model.keys():
        word_values = [0 for _ in range(nb_windows)]
        for appearance in res_data_model[word]['appearances']:
            word_values[appearance['no_window']] = appearance['df_in_window']
        res_interface[word] = word_values
    return res_interface


def generateExperience(database_withKey , database_withoutKey , experience , thematics , seed, experience_folder ,save_words_timeline = True,  lookback = 0 , window_size = 24):


    #define our Path
    finalDataPath = os.path.join(experience_folder , 'experience_final_data.json')
    gldaWFolderPath = os.path.join(experience_folder ,'guidedldaW' )
    gldaWoutFolderPath = os.path.join(experience_folder , 'guidedldaWout')

    # generate original timeline
    timeSplitter = TimeWindows(window_size)
    timeline = timeSplitter.splittArticlesPerWindows(database_withoutKey)

    #make copy for experience
    thematics_copy = copy.deepcopy(thematics)

    #create two time line : one with the injection another without
    tlWithout, tlWith = createArtificialChange(timeline, thematics_copy, experience)


    # return processed_text for train guided lda
    tlwout = [(end_date,
               [(database_withKey[article_id]['process_text'], database_withKey[article_id]['label'][0]) for article_id in window])
              for end_date, window in tlWithout]
    tlw = [(end_date,
            [(database_withKey[article_id]['process_text'], database_withKey[article_id]['label'][0]) for article_id in window]) for
           end_date, window in tlWith]

    # init guided lda for timeline with changement and without
    glda_w = SequencialLangageModeling(guided=True, seed=seed)
    glda_wout = SequencialLangageModeling(guided=True, seed=seed)

    # train guidedlda for the 2 timelines with same hyperparameter (seed , lookback...)
    glda_w.add_windows(tlw, lookback=lookback , updateRes = save_words_timeline)
    glda_wout.add_windows(tlwout, lookback=lookback ,  updateRes = save_words_timeline)

    # save model
    glda_w.save(gldaWFolderPath)
    glda_wout.save(gldaWoutFolderPath)

    # generate result and put it in the right format for each topic for each window
    res_w = generateResultsGLDAForInterface(glda_w, mode=experience, window_size=window_size)
    res_wout = generateResultsGLDAForInterface(glda_wout, mode=experience, window_size=window_size)

    # format results for test
    res_test = formatResultForTest(res_w, res_wout)

    # add words evolution data just with the 'glda_w' because it contain all the words needed for the analysis in contrary to glda_wout that not contain thematics articles words
    if save_words_timeline:
        word_evolution_data = generateWordTimeLineForTheInterface(glda_w.res, glda_w.nb_windows)
    else:
        word_evolution_data = {}
    final_data = {'serie': res_test, 'words': word_evolution_data}

    #save final_data
    with open(finalDataPath, 'w') as f:
        f.write(json.dumps(final_data))

    return final_data



def generateExperienceData(thematics  , timeline_size , nb_experience = 32 , cheat = False , boost = 0):

    experiences = {}
    experiences['info'] = {'timeline_size' : timeline_size}
    experiences['experiences'] = []
    min_size_experience = 2
    max_size_experience = timeline_size // 4
    thematics_experience = [thematic_name for thematic_name in thematics.keys() if len(thematics[thematic_name]['articles']) > 4000]
    count = 0
    while count < nb_experience:
        experience = {}
        thematic = random.choice(thematics_experience)
        experience['name'] = thematic
        experience['ranges'] = []
        fail = 0
        while count < nb_experience and fail < 15:
            size = random.randrange(min_size_experience , max_size_experience)
            window_start = random.randrange(timeline_size)
            ver = verifSide(window_start , size , timeline_size , experience['ranges'])
            if ver:
                experience['ranges'].append([window_start , window_start+size])
                count += 1
                print(f"count : {count}")
                fail = 0
            else:
                fail += 1
                print(f"fail : {fail}")
        #sort ranges
        experience['ranges'].sort(key=lambda tup:tup[0])

        experience['cheat'] = cheat
        experience['boost'] = boost

        experiences['experiences'].append(experience)


    return experiences




def verifSide(start , size , total_size , ranges):

    end  = start + size
    if start < 3 or end > total_size - 3:
        return False
    for range in ranges:
        if range[0] - 3 < start < range[1] + 3 or  range[0] - 3 < end < range[1] + 3:
            return False
    return  True


def doExperience(path  ,database_withKey ,  database_withoutKey , thematics ,
                           seed , lookback ,window_size , save_words_timeline):


    # add number of window in timeline
    timeline_size = len(TimeWindows(window_size).splittArticlesPerWindows(database_withoutKey))
    # generate timeline with targeted thematic and without trageted thematic

    experiences = generateExperienceData(thematics=thematics, timeline_size=timeline_size, nb_experience=50,
                                         cheat=False)


    all_experiences_data = []
    for i, experience in tqdm(list(enumerate(experiences['experiences'] , start=2))):

        experience_folder = os.path.join(path, 'experience' + str(i))
        # check if the folder exist
        if not os.path.exists(experience_folder):
            os.makedirs(experience_folder)
        final_data = generateExperience(database_withKey=database_withKey, database_withoutKey=database_withoutKey,
                                        experience=experience, thematics=thematics,
                                        seed=seed, experience_folder=experience_folder, lookback=lookback, window_size=window_size,
                                        save_words_timeline=save_words_timeline)
        all_experiences_data.append(final_data)

    # save all experiences results
    with open(os.path.join(path , 'interface.json'), 'w') as f:
        f.write(json.dumps(all_experiences_data))



if __name__ == '__main__':


    #file for data and res tot load and save
    #please do not confuse about withoutKey and WithoutChange the data form withoutKey file
    root = '/home/mouss/data'
    dataWithKeyFilePath = '/home/mouss/data/final_database_50000_100000_process_with_key.json'
    dataWithOutKeyFilePath = '/home/mouss/data/final_database_50000_100000_process_without_key.json'
    seedPath = '/home/mouss/data/mySeed.json'
    all_experiences_file = '/home/mouss/data/myExperiences_with_random_state.json'
    # resWithChangePath = '/home/mouss/data/resW5.json'
    # resWithOutChangePath = '/home/mouss/data/resWout5.json'
    # resTestPath = '/home/mouss/data/resTest5.json'
    # finalDataPath = '/home/mouss/data/final_data_interface.json'
    # gldaWFolderPath = '/home/mouss/data/guidedldaW_test5'
    # gldaWoutFolderPath = '/home/mouss/data/guidedldaWout_test5'


    #load data with key and without key
    with open(dataWithOutKeyFilePath , 'r') as f:
        data_withoutk = json.load(f)
    with open( dataWithKeyFilePath , 'r') as f:
        data_withk = json.load(f)

    #load thematics
    with open('/home/mouss/data/thematics.json', 'r') as f:
        thematics = json.load(f)
        thematics = thematics['thematics']

    # #add number of window in timeline
    # timeline_size = len(TimeWindows(12).splittArticlesPerWindows(data_withoutk))
    # #generate timeline with targeted thematic and without trageted thematic
    #
    # experiences = generateExperienceData(thematics=thematics , timeline_size=timeline_size ,nb_experience=50 , cheat=False)



    #load seed
    with open(seedPath , 'r') as f:
        seed = json.load(f)


    doExperience( path=os.path.join(root,'Experience_lookback0_window_size24_2')  ,database_withKey=data_withk, database_withoutKey=data_withoutk , seed=seed , thematics=thematics , lookback=0, window_size=24 , save_words_timeline=False)
    # doExperience( path=os.path.join(root,'Experience_lookback200_window_size24')  ,database_withKey=data_withk, database_withoutKey=data_withoutk , seed=seed , thematics=thematics , lookback=200, window_size=24, save_words_timeline=False)
    # doExperience( path=os.path.join(root,'Experience_lookback0_window_size1')  ,database_withKey=data_withk ,  database_withoutKey=data_withoutk , seed=seed , thematics=thematics , lookback=0, window_size=1, save_words_timeline=False)
    # doExperience( path=os.path.join(root,'Experience_lookback200_window_size1')  , database_withKey=data_withk ,database_withoutKey=data_withoutk , seed=seed , thematics=thematics , lookback=200, window_size=1, save_words_timeline=False)


    # all_experiences_data = []
    # for i , experience in tqdm(list(enumerate(experiences['experiences']))):
    #
    #     experience_folder = os.path.join( root,'experience'+str(i))
    #     # check if the folder exist
    #     if not os.path.exists(experience_folder):
    #         os.makedirs(experience_folder)
    #     final_data = generateExperience(database_withKey=data_withk , database_withoutKey=data_withoutk , experience=experience , thematics=thematics ,
    #                        seed=seed , experience_folder=experience_folder, lookback=0 ,window_size=12 , save_words_timeline=False )
    #     all_experiences_data.append(final_data)
    #
    #
    # #save all experiences results
    # with open(all_experiences_file , 'w') as f:
    #     f.write(json.dumps(all_experiences_data))











