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
    for label in labels:
        serie = {
            "name" : label,
            "data" : []
        }
        for window_id , (similarities_topic_score , labels_counter) in enumerate(zip(similarities_score , labels_counters)):
            serie["data"].append([window_id , similarities_topic_score[window_id] , labels_counter[label]])
        series.append(serie)
    return series

