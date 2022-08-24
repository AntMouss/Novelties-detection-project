import json
from novelties_detection.Experience import Sequential_Module
from novelties_detection.Experience.kwargsGen import SupervisedCalculatorKwargs , NoSupervisedCalculatorKwargs , KwargsResults , UpdateBadWordsKwargs
from novelties_detection.Collection.data_processing import exponentialThresholding , linearThresholding , ProcessorText

HOST = "127.0.0.1"
PORT = 5000
LOOP_DELAY_COLLECT = 20#minutes
LOOP_DELAY_PROCESS = 60#minutes
PROCESSOR = ProcessorText()
rss_feeds_path = "config/RSSfeeds_test.json"
model_path = "model/model.pck"
default_seed_path = "config/seed.json"
output_path = None
with open(default_seed_path , "r") as f:
    seed = json.load(f)
labels_idx = list(seed.keys())
macro_calculator_type = Sequential_Module.LFIDFSequentialSimilarityCalculator
micro_calculator_type = Sequential_Module.LDASequentialSimilarityCalculatorFixed
macro_training_args = {
    "passes" : 2 ,
    "overrate" : 1000
}
micro_training_args = {"passes" : 2}
nb_topics = 5
kwargs_above = {"limit" : 0.5 , "pente" : 100}
kwargs_bellow = {"relative_value" : 0.001}
bad_words_kwargs = UpdateBadWordsKwargs(exponentialThresholding , linearThresholding , kwargs_above  , kwargs_bellow)

macro_kwargs_results = KwargsResults(
    ntop=100,
    remove_seed_words=True,
    back = 3
)

micro_kwargs_results = KwargsResults(
    ntop=100,
    back=3
)

macro_calculator_kwargs = SupervisedCalculatorKwargs(
    calculator_type=macro_calculator_type,
    bad_words_args=bad_words_kwargs ,
    labels_idx=labels_idx ,
    training_args=None
)

micro_calculator_kwargs = NoSupervisedCalculatorKwargs(
    calculator_type=micro_calculator_type,
    nb_topics=7,
    bad_words_args=bad_words_kwargs,
    training_args=micro_training_args
)



