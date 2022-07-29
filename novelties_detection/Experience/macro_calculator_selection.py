"""
this module isn't made to be used.
used to select best macro calculator with MACRO_THEMATICS data and associated window articles dataset that aren't available on this repo.
"""
from multiprocessing import Pool
from bisect import bisect_right
from typing import List
from novelties_detection.Experience.kwargsGen import RandomKwargsGenerator , FullKwargs
from novelties_detection.Experience.ExperienceGen import ExperiencesGenerator
from novelties_detection.Experience.data_utils import ExperiencesResults , Alerte , TimeLineArticlesDataset
import pickle
import logging
from threading import Lock
from novelties_detection.Experience.Exception_utils import SelectionException
from novelties_detection.Experience.config_path import SAVE_CALCULATOR_KWARGS_PATH, LOG_PATH

l = Lock()

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , filename=LOG_PATH , level=logging.INFO)
logger = logging.getLogger(__name__)

NB_BEST_CALCULATORS = 3
SAVE_PATH = SAVE_CALCULATOR_KWARGS_PATH
NB_CALCULATORS = 15


class RandomMacroCalculatorSelector:
    """
    Select macro calculator with random kwargs parameters and choose the best according to many criteria
    """

    def __init__(self, nb_best_to_save : int, save_path : str):
        self.save_path = save_path
        self.best_calculators = [{"id" : "0000" , "score" : 0} for _ in range(nb_best_to_save)]
        self.resultats = {
            "calculator" : {},
            "best" : self.best_calculators
        }


    def save(self,calculator_id, kwargs, alerts):
        self.resultats["calculator"][calculator_id] = {"kwargs" : kwargs, "alerts" : alerts}
        with open(SAVE_PATH , "wb") as f:
            f.write(pickle.dumps(self.resultats))

    def compute_calculator_score(self,alerts : List[Alerte]):
        total_KL_score = 0
        for alert in alerts:
            total_KL_score += alert.score
        return total_KL_score

    def update_best_calculator(self,calculator_id , calculator_KL_score):
        score_list = [calculator["score"] for calculator in self.best_calculators]
        bisect_idx = bisect_right(score_list , calculator_KL_score)
        self.best_calculators[bisect_idx - 1] = {"id" : calculator_id , "score" : calculator_KL_score}


    def process(self, full_kwargs : FullKwargs):

        calculator_id = id(full_kwargs)
        kwargs_dataset = full_kwargs["dataset"]
        dataset = TimeLineArticlesDataset(**kwargs_dataset)
        experienceGenerator = ExperiencesGenerator(dataset)
        try:
            experienceGenerator.generate_results(**full_kwargs["results_args"])
            experiences_results = ExperiencesResults(experienceGenerator.experiences_res , experienceGenerator.info)
            alerts = ExperiencesGenerator.analyse_results(experiences_results, **full_kwargs["analyse"])
            KL_score = self.compute_calculator_score(alerts)
            self.update_best_calculator(calculator_id , KL_score)
            l.acquire()
            self.save(calculator_id, full_kwargs, alerts)
            l.release()
            if len(alerts) != 0:
                logger.info(f"Hypothesis confirmed for process id : {calculator_id}")
        except SelectionException:
            pass
        except Exception as e:
            logger.critical(f"Unknown Problem affect runtime : {e}")
        finally:
            del experienceGenerator


    def run(self , nb_calculators , nb_workers = 1):
        kwargs_generator = RandomKwargsGenerator(nb_calculators)
        with Pool(nb_workers) as p:
            p.map(self.process, kwargs_generator)


class StaticMacroCalculatorSelector(RandomMacroCalculatorSelector):

    def __init__(self, nb_best_to_save: int, save_path: str):
        super().__init__(nb_best_to_save, save_path)

    def run(self , static_macro_calculator_generator : List[FullKwargs] , nb_workers = 1):
        """

        @param static_macro_calculator_generator: in fact it's about a list of FullKwargs that we generate manually in
        config file
        """
        with Pool(nb_workers) as p:
            p.map(self.process, static_macro_calculator_generator)







if __name__ == '__main__':
    selector = RandomMacroCalculatorSelector(
        nb_best_to_save=NB_BEST_CALCULATORS ,
        save_path=SAVE_PATH
    )

    selector.run(NB_CALCULATORS , 3)