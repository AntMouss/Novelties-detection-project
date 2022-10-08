from novelties_detection.Service.endpoints.apis_utils import load_ressources , NumpyEncoder
from flask import Blueprint, Flask
from flask_restx import Api
from novelties_detection.Service.endpoints import NAMESPACE_ENDPOINTS

NAMESPACE_ENDPOINTS = NAMESPACE_ENDPOINTS

def createApp(injected_object_apis : list):
    """
    initialize all the api routes
    @param injected_object_apis : list of the specific object i need to initialize the api the order of the
    object in the list matter
    @rtype: Flask app object
    """
    global NAMESPACE_ENDPOINTS
    blueprint = Blueprint("api", __name__, url_prefix="/api/v1")
    api = Api(
        blueprint,
        version="1.0",
        validate=False,
    )
    # inject the objects containing logic here
    for nsp_endpoint , injected_object in zip(NAMESPACE_ENDPOINTS , injected_object_apis):
        nsp_endpoint = load_ressources(nsp_endpoint, injected_object)
        # finally add namespace to api
        api.add_namespace(nsp_endpoint)

    app = Flask('test')
    app.json_encoder = NumpyEncoder
    app.register_blueprint(blueprint)
    return app

