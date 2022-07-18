import pickle
import pytest
from gensim.corpora import Dictionary
from novelties_detection.Collection.data_processing import transformU, transformS
from novelties_detection.Experience.Engine_module import Engine
from novelties_detection.Experience.Sequential_Module import (
    MetaSequencialLangageSimilarityCalculator,
    NoSupervisedSequantialLangageSimilarityCalculator,
    LFIDFSequentialSimilarityCalculator,
    GuidedCoreXSequentialSimilarityCalculator,
    GuidedLDASequentialSimilarityCalculator,
    LDASequentialSimilarityCalculator,
    CoreXSequentialSimilarityCalculator
)
from novelties_detection.Experience.data_utils import TimeLineArticlesDataset
from novelties_detection.Experience.kwargsGen import (
    FullKwargsGenerator,
    GuidedLDACalculatorKwargs,
    GuidedCoreXCalculatorKwargs,
    LFIDFCalculatorKwargs,
    LDACalculatorKwargs, CoreXCalculatorKwargs,

)

sequential_calculator_types = [
    LFIDFSequentialSimilarityCalculator,
    GuidedCoreXSequentialSimilarityCalculator,
    GuidedLDASequentialSimilarityCalculator,
    LDASequentialSimilarityCalculator,
    CoreXSequentialSimilarityCalculator
]

kwargs_calculator_generators = [
    GuidedLDACalculatorKwargs,
    GuidedCoreXCalculatorKwargs,
     LFIDFCalculatorKwargs,
    LDACalculatorKwargs,
    CoreXCalculatorKwargs
]


DATA_PATH = '/home/mouss/data/final_database_50000_100000_process_without_key.json'
NB_HOURS = 10
START_DATE = 1622376100.0
END_DATE = START_DATE + NB_HOURS * 3600


supervised_dataset = TimeLineArticlesDataset(path=DATA_PATH,
                                             end=END_DATE, start=START_DATE, lookback=10, transform_fct=transformS)
unsupervised_dataset = TimeLineArticlesDataset(path=DATA_PATH,
                                               end=END_DATE, start=START_DATE, lookback=10, transform_fct=transformU)

with open("/home/mouss/PycharmProjects/novelties-detection-git/tmp_test_obj/data_window.pck", "rb") as f:
    supervised_data_window = pickle.load(f)
    unsupervised_data_window = (supervised_data_window[0], supervised_data_window[1][0])

@pytest.mark.parametrize([])
def test_kwargs_generator():
    full_kwargs = FullKwargsGenerator()
    assert type(full_kwargs) == dict



@pytest.mark.parametrize(kwargs_calculator_generators)
def test_sequential_calculators_treat_window(kwargs_calculator_type):
    topn = 100
    full_kwargs = FullKwargsGenerator(kwargs_calculator_type)
    assert type(full_kwargs) == dict
    calculator_type = full_kwargs['initialize_engine']['calculator_type']
    training_args = full_kwargs['initialize_engine']['training_args']
    del full_kwargs['initialize_engine']['calculator_type']
    del full_kwargs['initialize_engine']['training_args']
    calculator: MetaSequencialLangageSimilarityCalculator = calculator_type(**full_kwargs['initialize_engine'])
    if issubclass(type(calculator), NoSupervisedSequantialLangageSimilarityCalculator):
        model, dictionnary = calculator.treat_Window(unsupervised_data_window[1], **training_args)
    else:
        model, dictionnary = calculator.treat_Window(supervised_data_window[1], **training_args)
    for topic_id in range(calculator.nb_topics):
        top_words = model.get_topic_terms(topic_id , topn=topn)
        assert type(top_words) == dict
        if len(dictionnary) >= topn:
            assert len(top_words) == topn
    assert issubclass(type(model) , Engine)
    assert isinstance(dictionnary , Dictionary)


# def test_sequential_calculators_add_windows_and_compare_sequentialy():
#     try:
#         for kwargs_calculator_type in kwargs_calculator_generators:
#             full_kwargs = FullKwargsGenerator(kwargs_calculator_type)
#             assert type(full_kwargs) == dict
#             calculator_type = full_kwargs['initialize_engine']['calculator_type']
#             training_args = full_kwargs['initialize_engine']['training_args']
#             del full_kwargs['initialize_engine']['calculator_type']
#             del full_kwargs['initialize_engine']['training_args']
#             calculator: MetaSequencialLangageSimilarityCalculator = calculator_type(**full_kwargs['initialize_engine'])
#             if issubclass(type(calculator), NoSupervisedSequantialLangageSimilarityCalculator):
#                 calculator.add_windows(unsupervised_dataset, **training_args)
#                 res = calculator.compare_Windows_Sequentialy(**full_kwargs['generate_result'])
#                 assert type(res) == np.ndarray
#                 assert res.shape == (len(calculator) - 1 , 1)
#                 assert type(res[0]) == np.ndarray
#                 mode = "u"
#
#             else:
#                 calculator.add_windows(supervised_dataset, **training_args)
#                 res = calculator.compare_Windows_Sequentialy(**full_kwargs['generate_result'])
#                 assert isinstance(res , np.ndarray)
#                 assert isinstance(res[0] , np.ndarray)
#                 assert res.shape == (len(calculator) - 1, calculator.nb_topics)
#                 mode = "s"
#
#             assert len(calculator) == full_kwargs['experience']['timeline_size']
#             assert len(calculator.models) == full_kwargs['experience']['timeline_size']
#             if mode == 'u':
#                 res1: np.ndarray = np.array([np.array([random.uniform(0, 1)]) for _ in range(len(res))])
#             else:
#                 res1: np.ndarray = np.random.random(res.shape)
#             fake_simi = (res, res1)
#             fake_info = {"nb_topics": full_kwargs["initialize_engine"]["nb_topics"], "mode": mode}
#             fake_metadata = {"ranges": [(3,5)]}
#             fake_metadata = ExperiencesMetadata(**fake_metadata)
#             result_exp = ExperiencesResult(fake_metadata, fake_simi)
#             results = ExperiencesResults([result_exp], fake_info)
#
#             analyser = Analyser(results)
#             genera = analyser.multi_test_hypothesis_topic_injection(test_normality=True)
#             alerts = [alert for alert in genera]
#             print(len(alerts))
#     except AssertionError:
#         pass
#     except TypeError as e:
#         pass
#     except Exception as e:
#         pass