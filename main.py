from kwargsGen import Generator
from ExperienceGen import ExperiencesGenerator
from multiprocessing import Pool

NB_MODELS = 30

def process(**kwargs):
    experienceGenerator = ExperiencesGenerator()
    experienceGenerator.generate_results(**kwargs)


def main():
    kwargs_generator = Generator(NB_MODELS)
    with Pool(processes=5) as p:
        p.map(process, kwargs_generator)

if __name__ == '__main__':
    main()