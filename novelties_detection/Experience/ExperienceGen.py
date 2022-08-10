"""
this module isn't made to be used.
used to select best macro calculator with MACRO_THEMATICS data and associated window articles dataset that aren't't available on this repo.
"""
import logging
import random
from typing import List, Tuple, Type, Union
from novelties_detection.Experience.data_utils import (TimeLineArticlesDataset,
                        EditedTimeLineArticlesDataset,
                        MacroThematic,
                        ExperiencesMetadata,
                        ExperiencesResults,
                        ExperiencesResult)
from novelties_detection.Experience.Sequential_Module import MetaSequencialLangageSimilarityCalculator, \
    NoSupervisedSequantialLangageSimilarityCalculator
from novelties_detection.Experience.data_analysis import Analyser
from novelties_detection.Experience.Exception_utils import *

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , level=logging.INFO)
logger = logging.getLogger(__name__)



class ExperiencesMetadataGenerator:

    space_length = 1

    def __init__(self, thematics : List[MacroThematic], timeline_size : int , random_seed_gen : random.Random , nb_experiences : int = 32
                 , max_size_exp_rel = 0.25 , min_size_exp = 3,min_thematic_size : int = 1000 ):
        """

        @param thematics: thematics contain ids article that belong to the thematic
        @param timeline_size:
        @param nb_experiences:
        @param max_size_exp_rel: max size of a experience (thematics injection) . relative number
        @param min_size_exp:
        @param min_thematic_size:
        """

        self.random_seed_gen = random_seed_gen
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

    def verifSide(self, start_idx : int, end_idx:  int, total_size : int, ranges : list):

        if start_idx < self.space_length or end_idx > total_size - self.space_length:
            return False
        for range in ranges:
            if range[0] - self.space_length < start_idx < range[1] + self.space_length or range[0] - self.space_length < end_idx < range[1] + self.space_length:
                return False
            if start_idx -self.space_length < range[0]  < end_idx + self.space_length and start_idx -self.space_length < range[1] < end_idx + self.space_length:
                return False
        return True

    def __iter__(self):

        random_state = self.random_seed_gen.randint(1, 14340)
        random.seed(random_state)
        thematics_name = [(i , thematic.name) for i , thematic in enumerate(self.thematics) if
                                len(thematic.article_ids) > self.min_thematic_size]
        count = 0
        while count < self.nb_experiences:
            experience = {}
            thematic_idx , thematic_name = random.choice(thematics_name)
            experience['name'] = thematic_name
            thematic = self.thematics[thematic_idx]
            #specific case where there is no range possible
            if len(thematic.good_windows) < 2:
                continue
            experience['ranges'] = []
            fail = 0
            while count < self.nb_experiences and fail < 15:
                window_start = random.choice(thematic.good_windows)
                idx_good_window_start = thematic.good_windows.index(window_start)
                try:
                    range_set = self.get_range_set(thematic.good_windows , idx_good_window_start)
                except IndexError:
                    fail += 1
                    continue
                # +1 because we want that the out window being a window after the last window that contain good rate of thematic articles
                window_end = random.choice(range_set) + 1
                ver = self.verifSide(window_start, window_end, self.timeline_size, experience['ranges'])
                if ver:
                    experience['ranges'].append((window_start, window_end))
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

            yield [ExperiencesMetadata(**experience)] , thematic

    def get_range_set(self, good_windows : list, idx_good_window_start : int):
        """
        function to get the idx of the range element that respect the max and min size experience condition that we
        stare. after we can choose randomly the good window idx of the experience end.
        @param good_windows: list of good windows idx for the current thematic we choose
        @param idx_good_window_start: window idx that correspond to the begin of the experience (the first window which we inject thematic articles)
        @return: list of good window idx
        """
        first_range_element_idx = idx_good_window_start
        while good_windows[first_range_element_idx] - good_windows[idx_good_window_start ] < self.min_size_exp:
            first_range_element_idx += 1
        last_range_element_idx = first_range_element_idx
        try:
            while good_windows[last_range_element_idx] - good_windows[idx_good_window_start ] < self.max_size_exp_abs:
                last_range_element_idx += 1
        except IndexError:
            pass
        return good_windows[first_range_element_idx:last_range_element_idx]



class ExperiencesGenerator:
    """
    use this class to generate 2 differents timeline ,one with thematics injection and another one without
    then we compute similarity score for each time line and we analyse the similarity score to see if thematics injection
    have been detected by the calculator
    """

    def __init__(self , experience_metadata_generator : Union[List[Tuple[ExperiencesMetadata , List[MacroThematic]]] , MetadataGenerationException] , dataset : TimeLineArticlesDataset = None):
        self.metadata_generator = experience_metadata_generator
        self.experiences_res = []
        self.new_experience = {}
        self.info = {}
        self.reference_timeline = dataset
        self.reference_calculator = None


    def generate_timelines(self , dataset_args : dict) -> Tuple[TimeLineArticlesDataset]:
        try:
            for metadata , thematics in self.metadata_generator:
                self.new_experience["metadata"] = metadata
                if self.reference_timeline is None:
                    self.reference_timeline = TimeLineArticlesDataset(**dataset_args)
                timelinew = EditedTimeLineArticlesDataset(thematics=thematics, metadata=metadata, **dataset_args)
                stop_window_idx = timelinew.stop_window
                self.reference_timeline.set_stop_window(stop_window_idx)
                yield self.reference_timeline , timelinew

        except MetadataGenerationException:
            raise
        except Exception as e:
            logger.debug(f"Exception occurred in Timeline Generation: {e}", exc_info=True)
            raise TimelinesGenerationException("Exception occurred in Timeline Generation" , e)


    def generate_calculator(self, nb_topics : int, calculator_type : Type[MetaSequencialLangageSimilarityCalculator]
                            , training_args : dict, dataset_args : dict, **kwargs) -> Tuple[MetaSequencialLangageSimilarityCalculator]:

        try:
            self.info["nb_topics"] = nb_topics
            lookback = dataset_args["lookback"]
            if issubclass(calculator_type, NoSupervisedSequantialLangageSimilarityCalculator):
                self.info["mode"] = "u"
            else:
                self.info["mode"] = "s"
            for reference_timeline , timeline_w in self.generate_timelines(dataset_args):
                if self.reference_calculator is None:
                    self.reference_calculator = calculator_type(nb_topics , **kwargs)
                    self.reference_calculator.add_windows(reference_timeline, lookback,
                                                          **training_args)
                sq_calculator_w = calculator_type(nb_topics , **kwargs)
                sq_calculator_w.add_windows(timeline_w , lookback , **training_args)
                yield self.reference_calculator , sq_calculator_w

        except MetadataGenerationException:
            raise
        except TimelinesGenerationException:
            raise
        except Exception as e:
            logger.debug(f"Exception occurred in Calculator Generation: {e}", exc_info=True)
            raise CalculatorGenerationException("Exception occurred in Calculator Generation")


    def generate_results(self, first_w : int, last_w : int, ntop : int, back : int, calculator_args : dict, **kwargs):

        try:
            # delete topic_id key for use compareTopicsSequentialy that is like compareTopicSequentialy for all topics
            for calculator_ref , calculator_with in self.generate_calculator(**calculator_args):
                res_w = calculator_with.compare_Windows_Sequentialy(first_w, last_w, ntop, back, **kwargs)
                res_wout = calculator_ref.compare_Windows_Sequentialy(first_w, last_w, ntop, back, **kwargs)
                similarities_score = (res_w , res_wout)
                self.new_experience['similarities_score'] = similarities_score
                self.new_experience['label_counter_w'] = calculator_with.label_articles_counters
                self.new_experience['label_counter_ref'] = calculator_ref.label_articles_counters
                self.experiences_res.append(ExperiencesResult(**self.new_experience))
                del self.new_experience
                self.new_experience = {}

        except MetadataGenerationException:
            raise
        except TimelinesGenerationException:
            raise
        except CalculatorGenerationException:
            raise
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



