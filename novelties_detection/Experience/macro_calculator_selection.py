"""
this module isn't made to be used.
used to select best macro calculator with MACRO_THEMATICS data and associated window articles dataset that aren't available on this repo.
"""
from itertools import repeat
from multiprocessing import Pool
import bisect
from typing import List , Union , Tuple
from novelties_detection.Experience.kwargsGen import RandomKwargsGenerator , FullKwargs
from novelties_detection.Experience.ExperienceGen import ExperiencesProcessor , MetadataGenerationException
from novelties_detection.Experience.data_utils import ExperiencesResults , Alerte , TimeLineArticlesDataset , MacroThematic , ExperiencesMetadata
import pickle
import logging
from threading import Lock
from novelties_detection.Experience.Exception_utils import SelectionException , AnalyseException , ResultsGenerationException
from novelties_detection.Experience.config_path import RES_HOUR_CALCULATOR_SELECTION_PATH , RES_DAY_CALCULATOR_SELECTION_PATH, LOG_PATH
from novelties_detection.Experience.config_calculator_selection import (
    STATIC_KWARGS_GENERATOR_HOURS ,
    EXPERIENCES_METADATA_GENERATOR_HOURS ,
    STATIC_KWARGS_GENERATOR_DAYS ,
    EXPERIENCES_METADATA_GENERATOR_DAYS ,
    TEST_EXPERIENCES_METADATA_GENERATOR_HOURS,
    TEST_EXPERIENCES_METADATA_GENERATOR_DAYS
)
from novelties_detection.Experience.utils import timer_func

l = Lock()

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , filename=LOG_PATH , level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.propagate = False

NB_BEST_CALCULATORS = 3
NB_CALCULATORS = 15


class CalculatorInfo:

    def __init__(self , calculator_id , score : float , resultats : ExperiencesResults = None , full_kwargs : FullKwargs = None ):
        self.full_kwargs = full_kwargs
        self.resultats = resultats
        self.calculator_id = calculator_id
        self.score = score
    def __lt__(self, other):
        return self.score < other.score
    def __gt__(self, other):
        return self.score > other.score

class MicroCalculatorInfo(CalculatorInfo):
    def __init__(self, calculator_id, score : float , nb_clusters : int , kernel_type : str):
        super().__init__(calculator_id, score)
        self.kernel_type = kernel_type
        self.nb_clusters = nb_clusters




class MetaCalculatorSelector:
    best_calculators = []
    calculators_info = []
    def __init__(self, kwargs_calculator_generator):
        self.kwargs_calculator_generator = kwargs_calculator_generator

    @timer_func
    def process_selection(self, full_kwargs : FullKwargs, max_to_save : int, path = None):
        pass

    def run(self, max_to_save : int, nb_workers : int = 1, path = None):
        if nb_workers == 1:
            for i , kwargs in enumerate(self.kwargs_calculator_generator):
                self.process_selection(kwargs, max_to_save=max_to_save, path=path)
                print(f"-------------"
                    f"calculator numero {i} process over {len(self.kwargs_calculator_generator)} from selector {id(self)}"
                      f"-------------")

        else:
            with Pool(nb_workers) as p:
                p.starmap(self.process_selection, zip(self.kwargs_calculator_generator, repeat(max_to_save), repeat(path)))
        return self.best_calculators

    def save(self,save_path : str ) :
        to_save = {
            "calculators_info" : self.calculators_info,
            "best_calculators" : self.best_calculators
        }
        with open(save_path , "wb") as f:
            f.write(pickle.dumps(to_save))

    def update_best_calculators(self, calculator_id  , max_to_save : int):
        bisect.insort(self.best_calculators, calculator_id)
        if len(self.best_calculators) > max_to_save:
            del self.best_calculators[0]



class MacroCalculatorSelector(MetaCalculatorSelector):

    def __init__(self, kwargs_calculator_generator , experiences_metadata_generator : Union[List[Tuple[ExperiencesMetadata , List[MacroThematic]]] , MetadataGenerationException]):
        super().__init__(kwargs_calculator_generator)
        self.experiences_metadata_generator = experiences_metadata_generator

    @staticmethod
    def compute_calculator_score(alerts : List[Alerte] , average=True):
        total_energy_distance = 0
        try:
            for alert in alerts:
                total_energy_distance += alert.mean_distance
            if average:
                return total_energy_distance / len(alerts)
            else:
                return total_energy_distance
        except ZeroDivisionError:
            return 0

    @timer_func
    def process_selection(self, full_kwargs : FullKwargs, max_to_save : int, path = None):

        global l
        calculator_id = id(full_kwargs)
        energy_distance = 0
        kwargs_dataset = full_kwargs["dataset_args"]
        experiences_results = None
        try:

            dataset = TimeLineArticlesDataset(**kwargs_dataset)
            experience_generator = ExperiencesProcessor(self.experiences_metadata_generator, dataset)
            experience_generator.generate_results(**full_kwargs["results_args"])
            experiences_results = ExperiencesResults(experience_generator.experiences_res , experience_generator.info)
            alerts = ExperiencesProcessor.analyse_results(experiences_results, **full_kwargs["analyse_args"])
            energy_distance = self.compute_calculator_score(alerts)
            print(f"hypothesis mean distance equals to {energy_distance} for calculator {calculator_id}")
        except ResultsGenerationException:
            pass
        except AnalyseException:
            pass
        except Exception as e:
            logger.critical(f"Unknown Problem affect runtime : {e}")
            raise
        finally:
            calculator_info = CalculatorInfo(calculator_id, energy_distance , experiences_results , full_kwargs)
            self.update_best_calculators(calculator_id, max_to_save)
            self.calculators_info.append(calculator_info)
            l.acquire()
            if path is not None:
                self.save(path)
            l.release()
            del experience_generator



class RandomMacroCalculatorSelector(MacroCalculatorSelector):
    """
    Select macro calculator with random kwargs parameters and choose the best according to many criteria
    """

    def __init__(self, kwargs_calculator_generator : RandomKwargsGenerator , experiences_metadata_generator : Union[List[Tuple[ExperiencesMetadata , List[MacroThematic]]] , MetadataGenerationException]):
        super().__init__(kwargs_calculator_generator , experiences_metadata_generator)




def main():

    static_selector_hours = MacroCalculatorSelector(STATIC_KWARGS_GENERATOR_HOURS, TEST_EXPERIENCES_METADATA_GENERATOR_HOURS)
    static_selector_days = MacroCalculatorSelector(STATIC_KWARGS_GENERATOR_DAYS, TEST_EXPERIENCES_METADATA_GENERATOR_DAYS)
    static_selector_days.run(max_to_save=3, nb_workers=1 , path="/home/mouss/PycharmProjects/novelties-detection-git/results/day2.pck")
    static_selector_hours.run( max_to_save=3 , nb_workers=1 , path="/home/mouss/PycharmProjects/novelties-detection-git/results/hour2.pck")


if __name__ == '__main__':

    main()