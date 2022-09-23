import json
from typing import List
import numpy as np
from flask_restx import Namespace
from dateutil import parser


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



class ServiceException(Exception):
    pass

class LabelsException(ServiceException):
    pass


class ElasticSearchQueryBodyBuilder:

    text_fields = ["title" , "text" , "cleansed_text"]
    label_field = "label"
    timestamp_field = "timeStamp"

    def __init__(self , match_string : str = None , label : str = None , end_date : str = None ,
                 start_date : str = None):
        if start_date is not None:
            self.start_date = parser.parse(start_date)
            self.start_date= int(self.start_date.timestamp())
        else:
            self.start_date = None
        if end_date is not None:
            self.end_date = parser.parse(end_date)
            self.end_date= int(self.end_date.timestamp())

        else:
            self.end_date = None
        self.match_string = match_string
        self.label = label

    def build_requests(self):
        query_body = {
            "query" : {
                "filter" : []
            }
        }
        if self.match_string is not None:
            query_body["query"]["multi_match"] = self.build_match_query(self.match_string)
        if self.start_date is not None or self.end_date is not None:
            query_body["query"]["filter"].append(self.build_date_range_filter(self.start_date , self.end_date))
        if self.label is not None:
            query_body["query"]["filter"].append(self.build_label_filter(self.label))
        return query_body



    def build_match_query(self , match_string):
        match_query = {
            "query": match_string,
            "fields": self.text_fields
        }
        return match_query

    def build_date_range_filter(self , start_timestamp : int = None , end_timestamp : int = None):
        date_filtre = {
            "range" : {
                self.timestamp_field : {

                }
            }
        }

        if start_timestamp is not None:
            date_filtre["range"][self.timestamp_field]["gte"] = start_timestamp
        if end_timestamp is not None:
            date_filtre["range"][self.timestamp_field]["gte"] = end_timestamp
        return date_filtre

    def build_label_filter(self , label : str):
        return {"term": {self.label_field: label}}





