import math
import random
import json
from typing import List, Tuple , Type
from Sequential_Module import SequencialLangageModeling , GuidedLDASenquentialModeling
from datetime import datetime
from data_utils import TimeLineArticlesDataset, EditedTimeLineArticlesDataset
from data_processing import ProcessorText


class Thematic:
    def __init__(self , name : str , label : str  ,date : datetime , article_ids : List , lifetime : str):
        self.lifetime = lifetime
        self.article_ids = article_ids
        self.date = date
        self.label = label
        self.name = name

    def save(self , path):
        with open(path , 'w') as f:
            to_dumps = {
                "name" : self.name,
                "date" : self.date,
                "label" : self.label,
                "articles" : self.article_ids
            }
            f.write(json.dumps(to_dumps))


    def load(self , path):
        with open(path , 'r') as f:
            thematic = json.load(f)
        return Thematic(**thematic)




class ExperiencesMetadata:

    def __init__(self, name : str = None, ranges : List[Tuple] = None, nb_windows : int = None , cheat : bool = False , 
                 boost = 0):
        self.boost = boost
        self.cheat = cheat
        self.nb_windows = nb_windows
        self.ranges = ranges
        self.name = name



class ExperiencesMetadataGenerator:

    def __init__(self, thematics : List[Thematic] = None, timeline_size : int = None, nb_experiences : int = 32
                 , max_size_exp_rel = 0.25 , min_size_exp = 3,cheat : bool = False, boost : int = 0):
        self.boost = boost
        self.cheat = cheat
        self.nb_experiences = nb_experiences
        self.timeline_size = timeline_size
        self.thematics = thematics
        self.min_thematic_size = 100
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

    def __iter__(self ):

        thematics_name = [(i , thematic.name) for i , thematic in enumerate(self.thematics) if
                                len(thematic.article_ids) > self.min_thematic_size]
        count = 0
        while count < self.nb_experiences:
            experience = {}
            thematic_idx , thematic_name = random.choice(thematics_name)
            experience['name'] = thematic_name
            thematic = thematics[thematic_idx]
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


    def __call__(self, min_thematic_size = 4000 , min_size = 2 , max_size = 0.25):
        self.min_thematic_size = min_thematic_size
        self.min_size_exp = min_size
        self.max_size_exp_abs = max_size



class ExperiencesGenerator:

    def __init__(self, experiencesMetadataGenerator : ExperiencesMetadataGenerator , sequential_model : Type[SequencialLangageModeling]):
        self.sequential_model = sequential_model
        self.experiencesMetadataGenerator = experiencesMetadataGenerator


    def generate_timelines(self , **kwargs) -> Tuple[TimeLineArticlesDataset]:

        for metadata , thematic in self.experiencesMetadataGenerator:
            timelinewout = TimeLineArticlesDataset(**kwargs["initialize_dataset"])
            timelinew = EditedTimeLineArticlesDataset(thematic=thematic , metadata=metadata , **kwargs["initialize_dataset"])
            yield timelinew , timelinewout


    def generate_model(self,**kwargs) -> Tuple[SequencialLangageModeling]:

        for timeline_w , timeline_wout in self.generate_timelines(**kwargs):
            sq_model_w = self.sequential_model(**kwargs["initialize_engine"])
            sq_model_w.add_windows(timeline_w , kwargs["initialize_dataset"]['lookback'] , **kwargs['initialize_engine'])
            sq_model_wout = self.sequential_model(**kwargs['initialize_engine'])
            sq_model_wout.add_windows(timeline_wout , kwargs["initialize_dataset"]['lookback'] , **kwargs['initialize_engine'])
            yield sq_model_w , sq_model_wout

    def generate_results(self , **kwargs):

        for model_w , model_wout in self.generate_model(**kwargs):
            res_w = model_w.compareTopicSequentialy(**kwargs["comparaison"])
            res_wout = model_w.compareTopicSequentialy(**kwargs["comparaison"])
            yield res_w , res_wout





if __name__ == '__main__':


    #file for data and res tot load and save
    dataPath = '/home/mouss/data/final_database.json'
    dataprocessedPath = '/home/mouss/data/final_database_50000_100000_process_without_key.json'
    seedPath = '/home/mouss/data/mySeed.json'
    all_experiences_file = '/home/mouss/data/myExperiences_with_random_state.json'
    thematics_path = '/home/mouss/data/thematics.json'
    start_time = 1622376100.0
    nb_jours = 1
    end_time = start_time + nb_jours*24*3600

    # load seed
    with open(seedPath, 'r') as f:
        seed = json.load(f)


    #load thematics
    with open(thematics_path, 'r') as f:
        thematics = json.load(f)
        thematics = [Thematic(**thematic) for thematic in thematics]

    processor  = ProcessorText()

    kwargs = {
        "experience" : {"nb_experiences" : 32 , "cheat" : False , "boost" : 0},
        "comparaison" : {"topic_id" : 0, "first_w" :0, "last_w" :0, "ntop" :100, "fixeWindow" :False , "remove_seed_words" : True},
        "initialize_dataset":{"start" : start_time,"end" : end_time , "path":dataprocessedPath, "lookback" : 0.2 , "delta" : 1  , "processor" : processor},
        "initialize_engine" : {"nb_topics" : len(seed) , "seed" : seed}
    }
    data = TimeLineArticlesDataset(**kwargs["initialize_dataset"])
    kwargs["experience"]["timeline_size"] = len(data)
    res = []
    metadataGenerator = ExperiencesMetadataGenerator(thematics=thematics , **kwargs['experience'])
    model_type = GuidedLDASenquentialModeling
    experienceGenerator = ExperiencesGenerator(experiencesMetadataGenerator=metadataGenerator , sequential_model=model_type)
    for res_w , res_wout in experienceGenerator.generate_results(**kwargs):
        res.append((res_w , res_wout))



