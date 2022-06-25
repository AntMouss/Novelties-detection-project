from Experience.kwargsGen import KwargsGenerator
from Experience.ExperienceGen import ExperiencesGenerator
from Experience.config_arguments import LOG_PATH
from Experience.data_utils import ExperiencesResults
import pickle
import argparse
import logging
from threading import Lock

l = Lock()

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , filename=LOG_PATH , level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="pass config_file and save_path",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("dest", help="Destination location")
args = parser.parse_args()
config = vars(args)

SAVE_PATH = config["dest"]
NB_CALCULATORS = 30
RESULTATS = {}


def save(process_id , kwargs , alerts):
    RESULTATS[process_id] = {"kwargs" : kwargs, "alerts" : alerts}
    with open(SAVE_PATH , "wb") as f:
        f.write(pickle.dumps(RESULTATS))

def process(**kwargs):

    try:
        process_id = id(kwargs)
        experienceGenerator = ExperiencesGenerator()
        experienceGenerator.generate_results(**kwargs)
        assert (len(experienceGenerator.experiences_res) != 0)
        experiences_results = ExperiencesResults(experienceGenerator.experiences_res , experienceGenerator.info)
        alerts = ExperiencesGenerator.analyse_results(experiences_results , **kwargs["analyse"])
        assert (alerts is not None)
        l.acquire()
        save(process_id , kwargs , alerts)
        l.release()
        if len(alerts) != 0:
            logger.info(f"Hypothesis confirmed for process id : {process_id}")
    except AssertionError:
        pass


# def main():
#     kwargs_generator = KwargsGenerator(NB_MODELS)
#     with Pool(processes=5) as p:
#         p.map(process, kwargs_generator)


if __name__ == '__main__':
    kwargs_generator = KwargsGenerator(NB_CALCULATORS)
    for kwargs in kwargs_generator:
        process(**kwargs)