import logging
import random
from typing import List, Tuple
from novelties_detection.Experience.data_utils import (TimeLineArticlesDataset,
                        EditedTimeLineArticlesDataset,
                        Thematic,
                        ExperiencesMetadata,
                        ExperiencesResults,
                        ExperiencesResult)
from novelties_detection.Experience.Sequential_Module import MetaSequencialLangageSimilarityCalculator, \
    NoSupervisedSequantialLangageSimilarityCalculator
from novelties_detection.Experience.config_arguments import LOG_PATH
from novelties_detection.Experience.data_analysis import Analyser
from novelties_detection.Experience.Exception_utils import *

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
            raise MetadataGenerationException("max_size_exp_rel need to be inferior to 1")
        else:
            self.max_size_exp_abs = int(self.timeline_size * max_size_exp_rel)
        self.min_size_exp = min_size_exp
        if self.min_size_exp >= self.max_size_exp_abs or self.min_thematic_size < 2:
            raise MetadataGenerationException("min_size_exp should be superior to 1 and inferior to max_size_exp_abs")

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
                raise MetadataGenerationException("metadata have no ranges in 'ranges' attributes "
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
        self.reference_calculator = None
        self.calculator_type = None
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
            raise TimelinesGenerationException("Exception occurred in Timeline Generation" , e)


    def generate_calculator(self, **kwargs) -> Tuple[MetaSequencialLangageSimilarityCalculator]:

        try:
            self.info["nb_topics"] = kwargs['initialize_engine']['nb_topics']
            self.calculator_type = kwargs['initialize_engine']['calculator_type']
            if issubclass(type(self.calculator_type), NoSupervisedSequantialLangageSimilarityCalculator):
                self.info["mode"] = "u"
            else:
                self.info["mode"] = "s"
            self.training_args = kwargs['initialize_engine']['training_args']
            del kwargs['initialize_engine']['calculator_type']
            del kwargs['initialize_engine']['training_args']
            for reference_timeline , timeline_w in self.generate_timelines(**kwargs):
                sequential_calculator = self.calculator_type
                if self.reference_calculator is None:
                    self.reference_calculator = sequential_calculator(**kwargs['initialize_engine'])
                    self.reference_calculator.add_windows(reference_timeline, kwargs["initialize_dataset"]['lookback'],
                                                          **self.training_args)
                sq_calculator_w = sequential_calculator(**kwargs["initialize_engine"])
                sq_calculator_w.add_windows(timeline_w , kwargs["initialize_dataset"]['lookback'] , **self.training_args)
                yield self.reference_calculator , sq_calculator_w

        except Exception as e:
            logger.debug(f"Exception occurred in Calculator Generation: {e}", exc_info=True)
            raise CalculatorGenerationException("Exception occurred in Calculator Generation")


    def generate_results(self , **kwargs):

        try:
            # delete topic_id key for use compareTopicsSequentialy that is like compareTopicSequentialy for all topics
            for calculator_ref , calculator_with in self.generate_calculator(**kwargs):
                res_w = calculator_with.compare_Windows_Sequentialy(**kwargs["generate_result"])
                res_wout = calculator_ref.compare_Windows_Sequentialy(**kwargs["generate_result"])
                similarities_score = (res_w , res_wout)
                self.new_experience['similarity'] = similarities_score
                self.new_experience['label_counter_w'] = calculator_with.label_articles_counter
                self.new_experience['label_counter_ref'] = calculator_ref.label_articles_counter
                self.experiences_res.append(ExperiencesResult(**self.new_experience))
                del self.new_experience
                self.new_experience = {}
        except Exception as e:
            logger.debug(f"Exception occurred in Results Generation: {e}", exc_info=True)
            raise ResultsGenerationException("Exception occurred in Results Generation")


    @staticmethod
    def analyse_results(experiences_results : ExperiencesResults , risk = 0.05 , trim = 0):
        alerts = []
        try:
            analyser = Analyser(experiences_results , risk = risk , trim=trim)
            for alert in analyser.multi_test_hypothesis_topic_injection():
                alerts.append(alert)
            return alerts
        except Exception as e:
            logger.debug(f"Exception occurred during results analyse: {e}", exc_info=True)
            raise AnalyseException("Exception occurred during results analyse")



