"""
this module isn't made to be used.
used to select best macro calculator with MACRO_THEMATICS data and associated window articles dataset that aren't available on this repo.
"""
from itertools import repeat
from multiprocessing import Pool
import bisect
from typing import List , Union , Tuple
from novelties_detection.Experience.kwargsGen import RandomKwargsGenerator , FullKwargs
from novelties_detection.Experience.ExperienceGen import ExperiencesGenerator , MetadataGenerationException
from novelties_detection.Experience.data_utils import ExperiencesResults , Alerte , TimeLineArticlesDataset , MacroThematic , ExperiencesMetadata
import pickle
import logging
from threading import Lock
from novelties_detection.Experience.Exception_utils import SelectionException
from novelties_detection.Experience.config_path import SAVE_CALCULATOR_KWARGS_PATH, LOG_PATH
from novelties_detection.Experience.config_calculator_selection import STATIC_KWARGS_GENERATOR , EXPERIENCES_METADATA_GENERATOR

l = Lock()

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , filename=LOG_PATH , level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.propagate = False

NB_BEST_CALCULATORS = 3
SAVE_PATH = SAVE_CALCULATOR_KWARGS_PATH
NB_CALCULATORS = 15


class CalculatorInfo:

    def __init__(self , calculator_id , score : float ):
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
    def __init__(self, kwargs_calculator_generator):
        self.kwargs_calculator_generator = kwargs_calculator_generator
        self.best_calculators = []
        self.res = {}

    def select(self, full_kwargs : FullKwargs, max_to_save : int, path = None):
        pass

    def run(self, max_to_save : int, nb_workers : int = 1, path = None):
        if nb_workers == 1:
            for kwargs in self.kwargs_calculator_generator:
                self.select(kwargs , max_to_save=max_to_save , path=path)
        else:
            with Pool(nb_workers) as p:
                p.starmap(self.select, zip(self.kwargs_calculator_generator, repeat(max_to_save), repeat(path)))
        return self.best_calculators

    def save(self,save_path : str ) :
        to_save = {
            "res" : self.res,
            "best_micro_calculators" : self.best_calculators
        }
        with open(save_path , "wb") as f:
            f.write(pickle.dumps(to_save))

    def update_best_calculators(self, calculator_info : CalculatorInfo , max_to_save):
        bisect.insort(self.best_calculators, calculator_info)
        if len(self.best_calculators) > max_to_save:
            del self.best_calculators[0]


class MacroCalculatorSelector(MetaCalculatorSelector):

    def __init__(self, kwargs_calculator_generator , experiences_metadata_generator : Union[List[Tuple[ExperiencesMetadata , List[MacroThematic]]] , MetadataGenerationException]):
        super().__init__(kwargs_calculator_generator)
        self.experiences_metadata_generator = experiences_metadata_generator
        self.resultats = {
            "calculator" : {},
            "best" : self.best_calculators
        }

    @staticmethod
    def compute_calculator_score(alerts : List[Alerte] , average=True):
        total_energy_distance = 0
        if len(alerts) == 0:
            return 0
        for alert in alerts:
            total_energy_distance += alert.mean_distance
        if average:
            return total_energy_distance / len(alerts)
        else:
            return total_energy_distance

    def select(self, full_kwargs : FullKwargs, max_to_save : int, path = None):

        global l
        calculator_id = id(full_kwargs)
        kwargs_dataset = full_kwargs["dataset_args"]
        dataset = TimeLineArticlesDataset(**kwargs_dataset)
        experience_generator = ExperiencesGenerator(self.experiences_metadata_generator , dataset)
        try:
            experience_generator.generate_results(**full_kwargs["results_args"])
            experiences_results = ExperiencesResults(experience_generator.experiences_res , experience_generator.info)
            alerts = ExperiencesGenerator.analyse_results(experiences_results, **full_kwargs["analyse_args"])
            energy_distance = self.compute_calculator_score(alerts)
            calc_info = CalculatorInfo(calculator_id, energy_distance)
            self.update_best_calculators(calc_info, max_to_save)
            l.acquire()
            self.resultats["calculator"][calculator_id] = {"kwargs": full_kwargs, "alerts": alerts , "experience_resusltas" : experiences_results}
            if path is not None:
                self.save(path)
            l.release()
            if len(alerts) != 0:
                logger.info(f"Hypothesis confirmed for process id : {calculator_id}")
        except SelectionException:
            pass
        except Exception as e:
            logger.critical(f"Unknown Problem affect runtime : {e}")
        finally:
            del experience_generator



class RandomMacroCalculatorSelector(MacroCalculatorSelector):
    """
    Select macro calculator with random kwargs parameters and choose the best according to many criteria
    """

    def __init__(self, kwargs_calculator_generator : RandomKwargsGenerator , experiences_metadata_generator : Union[List[Tuple[ExperiencesMetadata , List[MacroThematic]]] , MetadataGenerationException]):
        super().__init__(kwargs_calculator_generator , experiences_metadata_generator)
        self.resultats = {
            "calculator" : {},
            "best" : self.best_calculators
        }



def main():

    static_selector = MacroCalculatorSelector(STATIC_KWARGS_GENERATOR , EXPERIENCES_METADATA_GENERATOR)
    static_selector.run( max_to_save=3 , nb_workers=1)


if __name__ == '__main__':

    main()