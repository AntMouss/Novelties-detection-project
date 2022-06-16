from kwargsGen import KwargsGenerator
from ExperienceGen import ExperiencesGenerator
from multiprocessing import Pool

NB_MODELS = 30

def process(**kwargs):
    try:
        experienceGenerator = ExperiencesGenerator()
        experienceGenerator.generate_results(**kwargs)
        experienceGenerator.analyse_results(**kwargs["analyse"])
    except Exception as e:
        print(e)
        print("tr")


def main():
    kwargs_generator = KwargsGenerator(NB_MODELS)
    with Pool(processes=5) as p:
        p.map(process, kwargs_generator)

if __name__ == '__main__':
    kwargs_generator = KwargsGenerator(NB_MODELS)
    for kwargs in kwargs_generator:
        process(**kwargs)