import json
import numpy as np
from flask_restx import Namespace
from Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator
from flask import Blueprint, Flask
from flask_restx import Api
from Service.apis import nsp_windows_api , nsp_interface_api , nsp_rss_feed_api


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
    for ressource in namespace.resources:
        ressource.kwargs['resource_class_kwargs'] = object
    return namespace


def initialize_calculator(kwargs_calculator):
    supervised_calculator_type = kwargs_calculator['initialize_engine']['model_type']
    training_args = kwargs_calculator['initialize_engine']['training_args']
    comparaison_args = kwargs_calculator['generate_result']
    del kwargs_calculator['initialize_engine']['calculator_type']
    del kwargs_calculator['initialize_engine']['training_args']
    sequential_model = supervised_calculator_type
    supervised_calculator: SupervisedSequantialLangageSimilarityCalculator = sequential_model(
        **kwargs_calculator['initialize_engine'])
    return {
        "supervised_calculator" : supervised_calculator ,
        "comparaison_args" : comparaison_args ,
        "training_args"  : training_args
    }


def createApp(injected_object_apis : list):
    """

    @param injected_object_apis : list of the specific object i need to initialize the api the order of the
    object in the list matter
    @rtype: Flask app object
    """
    blueprint = Blueprint("api", __name__, url_prefix="/api/v1")
    api = Api(
        blueprint,
        version="1.0",
        validate=False,
    )
    nsp_apis = [nsp_rss_feed_api , nsp_interface_api , nsp_windows_api]
    # inject the objects containing logic here
    for nsp_api , injected_object in zip(nsp_apis , injected_object_apis):
        nsp_api = load_ressources(nsp_api, injected_object)
        # finally add namespace to api
        api.add_namespace(nsp_api)
    app = Flask('test')
    app.json_encoder = NumpyEncoder
    app.register_blueprint(blueprint)
    return app