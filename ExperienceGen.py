import math
import random
import json
from typing import List, Tuple , Type
from Sequential_Module import SequencialLangageModeling , GuidedLDASenquentialModeling
from datetime import datetime
from data_utils import TimeLineArticlesDataset, EditedTimeLineArticlesDataset


class Thematic:
    def __init__(self , name : str , label : str  ,date : datetime , article_ids : List):
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

    def __init__(self, name : str = None, ranges : List[Tuple] = None, nb_windows : int = None):
        self.nb_windows = nb_windows
        self.ranges = ranges
        self.name = name



class ExperiencesMetadataGenerator:

    def __init__(self ,thematics : List[Thematic] = None  , timeline_size : int = None , nb_experience : int = 32
                 , cheat : bool = False , boost : int = 0):
        self.boost = boost
        self.cheat = cheat
        self.nb_experience = nb_experience
        self.timeline_size = timeline_size
        self.thematics = thematics
        self.min_thematic_size = 100
        self.max_size = self.timeline_size//4
        self.min_size = 2

    @staticmethod
    def verifSide(start, size, total_size, ranges):

        end = start + size
        if start < 3 or end > total_size - 3:
            return False
        for range in ranges:
            if range[0] - 3 < start < range[1] + 3 or range[0] - 3 < end < range[1] + 3:
                return False
        return True

    def __iter__(self ):

        if self.max_size >= 1:
            raise Exception("max_size need to be inferior to 1")
        max_size = math.ceil(self.timeline_size * self.max_size)
        thematics_name = [(i , thematic.name) for i , thematic in enumerate(self.thematics) if
                                len(thematic.article_ids) > self.min_thematic_size]
        count = 0
        while count < self.nb_experience:
            experience = {}
            thematic_idx , thematic_name = random.choice(thematics_name)
            experience['name'] = thematic_name
            thematic = thematics[thematic_idx]
            experience['ranges'] = []
            fail = 0
            while count < self.nb_experience and fail < 15:
                size = random.randrange(self.min_size, max_size)
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
        self.min_size = min_size
        self.max_size = max_size



class ExperiencesGenerator:

    def __init__(self, experiencesMetadataGenerator : ExperiencesMetadataGenerator , sequential_model : Type[SequencialLangageModeling]):
        self.sequential_model = sequential_model
        self.experiencesMetadataGenerator = experiencesMetadataGenerator


    def generate_timelines(self , **kwargs) -> Tuple[TimeLineArticlesDataset]:

        for metadata , thematic in self.experiencesMetadataGenerator:
            timelinewout = TimeLineArticlesDataset(**kwargs["preprocessing"])
            timelinew = EditedTimeLineArticlesDataset(thematic=thematic , metadata=metadata , **kwargs)
            yield timelinew , timelinewout


    def generate_model(self,**kwargs) -> Tuple[SequencialLangageModeling]:

        for timeline_w , timeline_wout in self.generate_timelines(**kwargs):
            sq_model_w = self.sequential_model(kwargs["processing"]['nb_topics']).add_windows(timeline_w , kwargs["preprocessing"]['lookback'])
            sq_model_wout = self.sequential_model(kwargs["processing"]['nb_topics']).add_windows(timeline_wout , kwargs["preprocessing"]['lookback'])
            yield sq_model_w , sq_model_wout

    def generate_results(self , **kwargs):

        for model_w , model_wout in self.generate_model(**kwargs):
            res_w = model_w.compareTopicSequentialy(**kwargs["comparaison"])
            res_wout = model_w.compareTopicSequentialy(**kwargs["comparaison"])
            yield res_w , res_wout





if __name__ == '__main__':


    #file for data and res tot load and save
    #please do not confuse about withoutKey and WithoutChange the data form withoutKey file
    root = '/home/mouss/data'
    dataPath = '/home/mouss/data/final_database.json'
    seedPath = '/home/mouss/data/mySeed.json'
    all_experiences_file = '/home/mouss/data/myExperiences_with_random_state.json'

    # load seed
    with open(seedPath, 'r') as f:
        seed = json.load(f)


    #load thematics
    with open('/home/mouss/data/thematics.json', 'r') as f:
        thematics = json.load(f)
        thematics = [Thematic(**thematic) for thematic in thematics]

    kwargs = {
        "experience" : {"nb_experiences" : 32 , "cheat" : False , "boost" : 0},
        "comparaison" : {"topic_id" : 0, "first_w" :0, "last_w" :0, "ntop" :100, "fixeWindow" :False , "remove_seed_words" : True},
        "preprocessing":{"lookback" : 100 , "delta" : 1 , "mode" : 's'},
        "processing" : {"nb_topics" : 5}
    }
    res = []
    metadataGenerator = ExperiencesMetadataGenerator(thematics=thematics , **kwargs['experience'])
    model_type = GuidedLDASenquentialModeling
    experienceGenerator = ExperiencesGenerator(experiencesMetadataGenerator=metadataGenerator , sequential_model=model_type)
    for res_w , res_wout in experienceGenerator.generate_results(**kwargs):
        res.append((res_w , res_wout))



