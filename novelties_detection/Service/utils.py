from novelties_detection.Service.apis.utils import load_ressources , NumpyEncoder
from novelties_detection.Experience.Sequential_Module import MetaSequencialLangageSimilarityCalculator
from flask import Blueprint, Flask
from flask_restx import Api
from novelties_detection.Service.apis import nsp_windows_api , nsp_interface_api , nsp_rss_feed_api




def initialize_calculator(kwargs_calculator , n = 0):
    calculator_type = kwargs_calculator['initialize_engine']['calculator_type']
    training_args = kwargs_calculator['initialize_engine']['training_args']
    #for testing
    if n == 1:
        kwargs_calculator["initialize_engine"]["labels_idx"] = ["sport" , "crime"]
        kwargs_calculator["initialize_engine"]["seed"] = {"sport" : ["cyclisme" , "marathon" , "tour" , "Mbappe" , "football"] , "crime" : ["violence" , "conjuguale" , "arme"]}
        kwargs_calculator["initialize_engine"]["nb_topics"] = 2
    comparaison_args = kwargs_calculator['generate_result']
    del kwargs_calculator['initialize_engine']['calculator_type']
    del kwargs_calculator['initialize_engine']['training_args']
    sequential_model = calculator_type
    supervised_calculator: MetaSequencialLangageSimilarityCalculator = sequential_model(
        **kwargs_calculator['initialize_engine'])
    return {
        "calculator" : supervised_calculator ,
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