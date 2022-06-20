import math
import random
import json
from config_arguments import LOG_PATH
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
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , filename=LOG_PATH , level=logging.INFO)
logger = logging.getLogger(__name__)



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
            if len(experience['ranges']) == 0:
                raise Exception("metadata have no ranges in 'ranges' attributes "
                                ", probably because timeline_size is too small")
            experience['ranges'].sort(key=lambda tup: tup[0])

            experience['cheat'] = self.cheat
            experience['boost'] = self.boost

            yield ExperiencesMetadata(**experience) , thematic



class ExperiencesGenerator:

    def __init__(self):
        self.experiences_res = []
        self.new_experience = {}
        self.info = {}
        self.reference_timeline = None
        self.reference_model = None
        self.model_type = None
        self.training_args = None


    def generate_timelines(self , **kwargs) -> Tuple[TimeLineArticlesDataset]:
        try:
            for metadata , thematic in ExperiencesMetadataGenerator(**kwargs['experience']):

                self.new_experience["metadata"] = metadata
                if self.reference_timeline is None:
                    self.reference_timeline = TimeLineArticlesDataset(**kwargs["initialize_dataset"])
                timelinew = EditedTimeLineArticlesDataset(thematic=thematic , metadata=metadata , **kwargs["initialize_dataset"])
                yield self.reference_timeline , timelinew

        except Exception as e:
            logger.debug(f"Exception occurred in Timeline Generation: {e}", exc_info=True)


    def generate_model(self,**kwargs) -> Tuple[Sequential_Module.MetaSequencialLangageModeling]:

        try:
            self.info["nb_topics"] = kwargs['initialize_engine']['nb_topics']
            self.model_type = kwargs['initialize_engine']['model_type']
            self.training_args = kwargs['initialize_engine']['training_args']
            del kwargs['initialize_engine']['model_type']
            del kwargs['initialize_engine']['training_args']
            for reference_timeline , timeline_w in self.generate_timelines(**kwargs):
                sequential_model = self.model_type
                if self.reference_model is None:
                    self.reference_model = sequential_model(**kwargs['initialize_engine'])
                    self.reference_model.add_windows(reference_timeline, kwargs["initialize_dataset"]['lookback'],
                                                     **self.training_args)
                sq_model_w = sequential_model(**kwargs["initialize_engine"])
                sq_model_w.add_windows(timeline_w , kwargs["initialize_dataset"]['lookback'] , **self.training_args)
                yield self.reference_model , sq_model_w

        except Exception as e:
            logger.debug(f"Exception occurred in Model Generation: {e}", exc_info=True)


    def generate_results(self , **kwargs):

        try:
            # delete topic_id key for use compareTopicsSequentialy that is like compareTopicSequentialy for all topics
            del kwargs["generate_result"]["topic_id"]
            for model_ref , model_w in self.generate_model(**kwargs):
                res_w = model_w.compareTopicsSequentialy(**kwargs["generate_result"])
                res_wout = model_ref.compareTopicsSequentialy(**kwargs["generate_result"])
                similarity = (res_w , res_wout)
                self.new_experience['similarity'] = similarity
                self.new_experience['label_counter_w'] = model_w.label_articles_counter
                self.new_experience['label_counter_ref'] = model_ref.label_articles_counter
                self.experiences_res.append(ExperiencesResult(**self.new_experience))
                del self.new_experience
                self.new_experience = {}
        except Exception as e:
            logger.debug(f"Exception occurred in Results Generation: {e}", exc_info=True)


    @staticmethod
    def analyse_results(experiences_results : ExperiencesResults , risk = 0.05 , trim = 0):

        try:
            samples = Sampler(experiences_results).samples
            analyser = Analyser(samples , risk = risk , trim=trim)
            return [alert for alert in analyser.test_hypothesis()]
        except Exception as e:
            logger.debug(f"Exception occurred in Alert Generation: {e}", exc_info=True)



