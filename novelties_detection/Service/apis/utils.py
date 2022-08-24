import json
from typing import List
import numpy as np
from flask_restx import Namespace

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float("{:.4f}".format(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_ressources(namespace : Namespace, object : dict):
    """
    load needed object for the ressource api
    @param namespace: namespace of the ressource api
    @param object:
    @return: namespace with joined object
    """
    for ressource in namespace.resources:
        ressource.kwargs['resource_class_kwargs'] = object
    return namespace


def build_series(similarities_score : list , labels_counters : List[dict] , labels : list):
    """
    format data in series to being usable in the interface dashboard
    @param similarities_score: np.array of similarity score between 2 adjacent window . one raw by topic
    @param labels_counters: list of counter dictionnary
    @param labels: list of labels
    @return: series format [
                                {
                                    "name" : label
                                    "data" : [
                                                [window_idx , similarity , count],
                                            ]
                                {,
                            ]
    """
    series = []
    for label_idx , (label , similarities_topic_score ) in enumerate (zip(labels ,similarities_score )):
        serie = {
            "name" : label,
            "data" : []
        }
        for i in range (1 , len(labels_counters)):
            serie["data"].append([i - 1 , similarities_topic_score[i - 1] , labels_counters[i][label_idx]])
        series.append(serie)

    return series

