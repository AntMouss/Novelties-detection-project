from novelties_detection.Service.apis.apis_utils import load_ressources , NumpyEncoder
from flask import Blueprint, Flask
from flask_restx import Api
from novelties_detection.Service.apis import nsp_windows_api , nsp_interface_api , nsp_rss_feed_api



def createApp(injected_object_apis : list):
    """
    initialize all the api routes
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


class ServiceException(Exception):
    pass


class LabelsException(ServiceException):
    pass