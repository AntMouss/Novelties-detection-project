from kwargsGen import KwargsGenerator
from ExperienceGen import ExperiencesGenerator
from multiprocessing import Pool
from data_utils import ExperiencesResults
import pickle
import argparse
import logging
from threading import Lock

l = Lock()

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , filename='log.log' , level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="pass config_file and save_path",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("dest", help="Destination location")
args = parser.parse_args()
config = vars(args)

SAVE_PATH = config["dest"]
NB_MODELS = 30
DATA = {}


def save(process_id , kwargs , alerts):
    DATA[process_id] = {"kwargs" : kwargs,"alerts" : alerts}
    with open(SAVE_PATH , "wb") as f:
        f.write(pickle.dumps(DATA))

def process(**kwargs):

        process_id = id(kwargs)
        experienceGenerator = ExperiencesGenerator()
        experienceGenerator.generate_results(**kwargs)
        experiences_results = ExperiencesResults(experienceGenerator.experiences_res , experienceGenerator.info)
        alerts = ExperiencesGenerator.analyse_results(experiences_results , **kwargs["analyse"])
        l.locked()
        save(process_id , kwargs , alerts)
        l.release()
        if len(alerts) != 0:
            logger.info(f"Hypothesis confirmed for process id : {process_id}")


# def main():
#     kwargs_generator = KwargsGenerator(NB_MODELS)
#     with Pool(processes=5) as p:
#         p.map(process, kwargs_generator)


if __name__ == '__main__':
    kwargs_generator = KwargsGenerator(NB_MODELS)
    for kwargs in kwargs_generator:
        process(**kwargs)