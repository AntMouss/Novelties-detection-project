import math
import random
import json
from typing import List, Tuple
import Sequential_Module
from data_utils import (TimeLineArticlesDataset,
                        EditedTimeLineArticlesDataset,
                        Thematic,
                        ExperiencesMetadata,
                        ExperiencesResults,
                        ExperiencesResult)
from data_processing import ProcessorText
from data_analysis import Sampler , Analyser




class ExperiencesMetadataGenerator:

    def __init__(self, thematics : List[Thematic] = None, timeline_size : int = None, nb_experiences : int = 32
                 , max_size_exp_rel = 0.25 , min_size_exp = 3,cheat : bool = False, boost : int = 0 ,
                 min_thematic_size : int = 1000):
        self.boost = boost
        self.cheat = cheat
        self.nb_experiences = nb_experiences
        self.timeline_size = timeline_size
        self.thematics = thematics
        self.min_thematic_size = min_thematic_size
        if max_size_exp_rel >= 1:
            raise Exception("max_size_exp_rel need to be inferior to 1")
        else:
            self.max_size_exp_abs = int(self.timeline_size * max_size_exp_rel)
        self.min_size_exp = min_size_exp
        if self.min_size_exp >= self.max_size_exp_abs or self.min_thematic_size < 2:
            raise Exception("min_size_exp should be superior to 1 and inferior to max_size_exp_abs")

    @staticmethod
    def verifSide(start, size, total_size, ranges):

        end = start + size
        if start < 3 or end > total_size - 3:
            return False
        for range in ranges:
            if range[0] - 3 < start < range[1] + 3 or range[0] - 3 < end < range[1] + 3:
                return False
            if start -3 < range[0]  < end + 3 and start -3 < range[1] < end + 3:
                return False
        return True

    def __iter__(self):

        thematics_name = [(i , thematic.name) for i , thematic in enumerate(self.thematics) if
                                len(thematic.article_ids) > self.min_thematic_size]
        count = 0
        while count < self.nb_experiences:
            experience = {}
            thematic_idx , thematic_name = random.choice(thematics_name)
            experience['name'] = thematic_name
            thematic = self.thematics[thematic_idx]
            experience['ranges'] = []
            fail = 0
            while count < self.nb_experiences and fail < 15:
                size = random.randrange(self.min_size_exp, self.max_size_exp_abs)
                window_start = random.randrange(self.timeline_size)
                ver = ExperiencesMetadataGenerator.verifSide(window_start, size, self.timeline_size, experience['ranges'])
                if ver:
                    experience['ranges'].append((window_start, window_start + size))
                    count += 1
                    print(f"count : {count}")
                    fail = 0
                else:
                    fail += 1
                    print(f"fail : {fail}")
            # sort ranges
            experience['ranges'].sort(key=lambda tup: tup[0])

            experience['cheat'] = self.cheat
            experience['boost'] = self.boost

            yield ExperiencesMetadata(**experience) , thematic





class ExperiencesGenerator:

    def __init__(self):
        self.experiences_res = []
        self.new_experience = {}
        self.info = {}


    def generate_timelines(self , **kwargs) -> Tuple[TimeLineArticlesDataset]:

        for metadata , thematic in ExperiencesMetadataGenerator(**kwargs['experience']):

            self.new_experience["metadata"] = metadata
            timelinewout = TimeLineArticlesDataset(**kwargs["initialize_dataset"])
            timelinew = EditedTimeLineArticlesDataset(thematic=thematic , metadata=metadata , **kwargs["initialize_dataset"])
            yield timelinewout , timelinew


    def generate_model(self,**kwargs) -> Tuple[Sequential_Module.MetaSequencialLangageModeling]:

        for timeline_wout , timeline_w in self.generate_timelines(**kwargs):
            self.info["nb_topics"] = kwargs['initialize_engine']['nb_topics']
            sequential_model = kwargs['initialize_engine']['model_type']
            training_args = kwargs['initialize_engine']['training_args']
            del kwargs['initialize_engine']['model_type']
            del kwargs['initialize_engine']['training_args']
            sq_model_w = sequential_model(**kwargs["initialize_engine"])
            sq_model_w.add_windows(timeline_w , kwargs["initialize_dataset"]['lookback'] , **training_args)
            sq_model_wout = sequential_model(**kwargs['initialize_engine'])
            sq_model_wout.add_windows(timeline_wout , kwargs["initialize_dataset"]['lookback'] , **training_args)
            yield sq_model_wout , sq_model_w


    def generate_results(self , **kwargs):

        for model_wout , model_w in self.generate_model(**kwargs):
            res_w = model_w.compareTopicSequentialy(**kwargs["generate_result"])
            res_wout = model_w.compareTopicSequentialy(**kwargs["generate_result"])
            similarity = (res_w , res_wout)
            self.new_experience['similarity'] = similarity
            try:
                self.new_experience['label_counter_w'] = model_w.label_articles_counter
                self.new_experience['label_counter_wout'] = model_wout.label_articles_counter

            except AttributeError:
                pass
            self.experiences_res.append(ExperiencesResult(**self.new_experience))
            del self.new_experience
            self.new_experience = {}


    def analyse_results(self , risk = 0.05 , trim = 0):

        experiences_results = ExperiencesResults(self.experiences_res , self.info)
        samples = Sampler(experiences_results).samples
        analyser = Analyser(samples , risk = risk)
        for topic_id in range (len(samples)):
            analyser.topic_pvalue_matrix(topic_id , trim=trim)




if __name__ == '__main__':


    #initialize all arguments

    dataPath = '/home/mouss/data/final_database.json'
    dataprocessedPath = '/home/mouss/data/final_database_50000_100000_process_without_key.json'
    seedPath = '/home/mouss/data/mySeed.json'
    all_experiences_file = '/home/mouss/data/myExperiences_with_random_state.json'
    thematics_path = '/home/mouss/data/thematics.json'
    nb_jours = 1
    start_date = 1622376100.0
    end_date = start_date + nb_jours * 24 * 3600
    lookback = 10
    delta = 1
    timeline_size = math.ceil((end_date - start_date)/delta)

    # load seed
    with open(seedPath, 'r') as f:
        seed = json.load(f)

    labels_idx = list(seed.keys())

    #load thematics
    with open(thematics_path, 'r') as f:
        thematics = json.load(f)
        thematics = [Thematic(**thematic) for thematic in thematics]

    processor  = ProcessorText()
    model_type = Sequential_Module.LFIDFSequentialModeling

    nb_experiences = 32
    min_thematic_size = 4000
    min_size = 2
    max_size = 0.25
    ntop = 100
    topic_id = 0
    nb_topics = len(labels_idx)


    kwargs = {
        "experience" : {"nb_experiences" : nb_experiences , "timeline_size" : timeline_size , "thematics" : thematics ,
                        "min_thematic_size" : min_thematic_size , "min_size" : min_size
                        , "max_size" : max_size, "cheat" : False , "boost" : 0 },

        "initialize_dataset":{"start" : start_date, "end" : end_date , "path": dataprocessedPath,
                              "lookback" : lookback , "delta" : delta  , "processor" : processor},

        "initialize_engine" : {"model_type" : model_type,"nb_topics" : nb_topics ,
                               "labels_idx" : labels_idx},

        "generate_result": {"topic_id": topic_id, "first_w": 0, "last_w": 0, "ntop": ntop,
                            "fixeWindow": False, "remove_seed_words": True}
    }

    #generate experiences
    experienceGenerator = ExperiencesGenerator()
    experienceGenerator.generate_results(**kwargs)

