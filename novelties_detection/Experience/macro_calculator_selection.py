from multiprocessing import Pool
from bisect import bisect_right
from typing import List
from novelties_detection.Experience.kwargsGen import KwargsGenerator
from novelties_detection.Experience.ExperienceGen import ExperiencesGenerator
from novelties_detection.Experience.config_arguments import LOG_PATH
from novelties_detection.Experience.data_utils import ExperiencesResults , Alerte
import pickle
import logging
from threading import Lock
from novelties_detection.Experience.Exception_utils import SelectionException

l = Lock()

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , filename=LOG_PATH , level=logging.INFO)
logger = logging.getLogger(__name__)
#
# parser = argparse.ArgumentParser(description="pass config_file and save_path",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("dest", help="Destination location")
# args = parser.parse_args()
# config = vars(args)
#
# SAVE_PATH = config["dest"]

NB_BEST_CALCULATORS = 3
SAVE_PATH = "/home/mouss/PycharmProjects/novelties-detection-git/results/wesh.pck"
NB_CALCULATORS = 15


class MacroCalculatorSelector:

    def __init__(self , nb_best_calculators : int  , save_path : str):
        self.save_path = save_path
        self.best_calculators = [{"id" : "0000" , "score" : 0} for _ in range(nb_best_calculators)]
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


    def process(self,kwargs):

        calculator_id = id(kwargs)
        experienceGenerator = ExperiencesGenerator()
        try:
            experienceGenerator.generate_results(**kwargs)
            experiences_results = ExperiencesResults(experienceGenerator.experiences_res , experienceGenerator.info)
            alerts = ExperiencesGenerator.analyse_results(experiences_results , **kwargs["analyse"])
            KL_score = self.compute_calculator_score(alerts)
            self.update_best_calculator(calculator_id , KL_score)
            l.acquire()
            self.save(calculator_id , kwargs , alerts)
            l.release()
            if len(alerts) != 0:
                logger.info(f"Hypothesis confirmed for process id : {calculator_id}")
        except SelectionException:
            pass
        except Exception as e:
            logger.critical(f"Unknown Problem affect runtime : {e}")
        finally:
            del experienceGenerator


    def run(self , nb_calculators):
        kwargs_generator = KwargsGenerator(nb_calculators)
        with Pool(3) as p:
            p.map(self.process, kwargs_generator)


if __name__ == '__main__':
    selector = MacroCalculatorSelector(
        nb_best_calculators=NB_BEST_CALCULATORS ,
        save_path=SAVE_PATH
    )

    selector.run(NB_CALCULATORS)