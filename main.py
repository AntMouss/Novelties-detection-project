from kwargsGen import KwargsGenerator
from ExperienceGen import ExperiencesGenerator
from multiprocessing import Pool
from data_utils import ExperiencesResults
import pickle
import argparse

parser = argparse.ArgumentParser(description="pass config_file and save_path",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("dest", help="Destination location")
args = parser.parse_args()
config = vars(args)

SAVE_PATH = config["dest"]
NB_MODELS = 30

def save(kwargs, alerts):
    data = {"kwargs" : kwargs,"alerts" : alerts}
    with open(SAVE_PATH , "wb") as f:
        f.write(pickle.dumps(data))

def process(**kwargs):
    try:
        experienceGenerator = ExperiencesGenerator()
        experienceGenerator.generate_results(**kwargs)
        experiences_results = ExperiencesResults(experienceGenerator.experiences_res , experienceGenerator.info)
        alerts = ExperiencesGenerator.analyse_results(experiences_results , **kwargs["analyse"])
        save(kwargs , alerts)
        if len(alerts) != 0:
            print("hypothesis confirmed")


    except Exception as e:
        print(e)
        print("tr")


# def main():
#     kwargs_generator = KwargsGenerator(NB_MODELS)
#     with Pool(processes=5) as p:
#         p.map(process, kwargs_generator)

if __name__ == '__main__':
    kwargs_generator = KwargsGenerator(NB_MODELS)
    for kwargs in kwargs_generator:
        process(**kwargs)