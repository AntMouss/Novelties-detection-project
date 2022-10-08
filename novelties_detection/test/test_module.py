from typing import Callable , Type
import numpy as np
import json
from novelties_detection.Experience.Sequential_Module import\
    (
    GuidedSequantialLangageSimilarityCalculator,
    SupervisedSequantialLangageSimilarityCalculator,
    NoSupervisedFixedSequantialLangageSimilarityCalculator,
    LDASequentialSimilarityCalculatorFixed ,
    CoreXSequentialSimilarityCalculatorFixed ,
    GuidedLDASequentialSimilarityCalculator ,
    GuidedCoreXSequentialSimilarityCalculator ,
    TFIDFSequentialSimilarityCalculator
    )
from novelties_detection.Experience.kwargs_utils import UpdateBadWordsKwargs
from novelties_detection.Collection.data_processing import convLogarithmThresholding , linearThresholding
import pytest
from novelties_detection.test.testServer import training_unsupervised_dataset , training_supervised_dataset


with open("novelties_detection/test/testing_data/test_seed.json" , "r") as f:
    seed = json.load(f)
labels_idx = list(seed.keys())


testing_unsupervised_items = [
    {
        "calculator_type" : LDASequentialSimilarityCalculatorFixed ,
        "training_args" :
            {
                "passes" : 2 ,
                "decay" : 0.6 ,
                "offset" : 2.2
            }
    } ,
    {
        "calculator_type" : CoreXSequentialSimilarityCalculatorFixed ,
        "training_args" :
            {
                "max_iter" : 100
            }
    }
]


testing_supervised_items = [
{
        "calculator_type" : TFIDFSequentialSimilarityCalculator ,
        "training_args" : {}
    }
]


testing_guided_items = [
    {
        "calculator_type" : GuidedLDASequentialSimilarityCalculator ,
        "training_args" :
            {
                "decay" : 0.8  ,
                "seed_strength" : 1000
            },
        "dynamic_kwargs":
            {

            }
    },
    {
        "calculator_type" : GuidedCoreXSequentialSimilarityCalculator ,
        "training_args" :
            {
                "max_iter" : 30  ,
                "seed_strength" : 1000
            },
        "dynamic_kwargs" : {

        }
    },
{
        "calculator_type" : GuidedLDASequentialSimilarityCalculator ,
        "training_args" :
            {
                "decay" : 0.8  ,
                "seed_strength" : 1000
            },
        "dynamic_kwargs" :
            {
                "dynamic_seed_mode" : True

            }
    },
    {
        "calculator_type" : GuidedCoreXSequentialSimilarityCalculator ,
        "training_args" :
            {
                "seed_strength" : 1000
            },
        "dynamic_kwargs" :
            {
                "dynamic_seed_mode" : True,
                "static_seed_relative_size" : 0.3,
                "turnover_rate" : 0.7

            }
    }
]
testing_unsupervised_items = [(item["calculator_type"] , item["training_args"]) for item in testing_unsupervised_items]
testing_supervised_items = [(item["calculator_type"] , item["training_args"]) for item in testing_supervised_items]
testing_guided_items = [(item["calculator_type"] , item["training_args"] , item["dynamic_kwargs"]) for item in testing_guided_items]


supervised_kwargs_results : dict = {
    "ntop" : 100,
    "remove_seed_words" : True,
    "back"  :  2
}

unsupervised_kwargs_results : dict = {
    "ntop" : 100,
    "reproduction_threshold" : 0.2,
    "back"  :  2
}

fct_above : Callable = convLogarithmThresholding
fct_below : Callable = linearThresholding
kwargs_above : dict = {
    "limit" : 0.5
}
kwargs_below : dict = {
    "slop" : 0.001
}
bad_words_kwargs = UpdateBadWordsKwargs(
    thresholding_fct_above=fct_above ,
    thresholding_fct_below=fct_below ,
    kwargs_above=kwargs_above  ,
    kwargs_below=kwargs_below
)

nb_micro_topics = 7

@pytest.mark.parametrize( "calculator_type,training_args" , testing_supervised_items )
def test_Supervised_Sequential_module(calculator_type : Type[SupervisedSequantialLangageSimilarityCalculator] , training_args : dict):
    global labels_idx
    global  bad_words_kwargs
    calculator = calculator_type(
        bad_words_args=bad_words_kwargs.__dict__,
        labels_idx=labels_idx,
    )
    calculator.add_windows(training_supervised_dataset , **training_args)
    assert len(calculator) == len(training_supervised_dataset)
    words = calculator.getTopWordsTopic(3 , 3 , 100 )
    assert len(words) != 0
    assert isinstance(words , dict)
    res = calculator.compare_Windows_Sequentialy(**supervised_kwargs_results)
    assert isinstance(res , np.ndarray)
    nb_topics = calculator.nb_topics
    assert res.shape == (nb_topics , len(calculator) - 1)


@pytest.mark.parametrize( "calculator_type,training_args,dynamic_kwargs" , testing_guided_items )
def test_Guided_Sequential_module(
        calculator_type : Type[GuidedSequantialLangageSimilarityCalculator] , training_args : dict , dynamic_kwargs : dict):
    global bad_words_kwargs
    global labels_idx
    global seed
    calculator = calculator_type(
        bad_words_args=bad_words_kwargs.__dict__,
        labels_idx=labels_idx,
        seed = seed,
        **dynamic_kwargs
    )
    calculator.add_windows(training_supervised_dataset , **training_args)
    assert len(calculator) == len(training_supervised_dataset)
    words = calculator.getTopWordsTopic(3 , 3 , 100 )
    assert len(words) != 0
    assert isinstance(words , dict)
    res = calculator.compare_Windows_Sequentialy(**supervised_kwargs_results)
    assert isinstance(res , np.ndarray)
    nb_topics = calculator.nb_topics
    assert res.shape == (nb_topics , len(calculator) - 1)


@pytest.mark.parametrize( "calculator_type,training_args" , testing_unsupervised_items )
def test_Unsupervised_Sequential_module(
        calculator_type : Type[NoSupervisedFixedSequantialLangageSimilarityCalculator] , training_args : dict):
    global nb_micro_topics
    global bad_words_kwargs
    calculator = calculator_type(
        nb_topics=nb_micro_topics,
        bad_words_args=bad_words_kwargs.__dict__
    )
    calculator.add_windows(training_unsupervised_dataset , **training_args)
    assert len(calculator) == len(training_unsupervised_dataset)
    words = calculator.getTopWordsTopic(3 , 3 , 100 )
    assert len(words) != 0
    assert isinstance(words , dict)
    res = calculator.compare_Windows_Sequentialy(**supervised_kwargs_results)
    assert isinstance(res , np.ndarray)
    assert res.shape == (1 , len(calculator) - 1)
