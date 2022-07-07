from flask import Blueprint, Flask
from flask_restx import Api
from Server.apis.rss_feed_api.namespaces import namesp
import os.path
import json
import argparse

parser = argparse.ArgumentParser(description="pass config_file with model , kwargs_calculator paths",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config_path", help="paths of the model and kwargs calculator")
parser.add_argument("root_path" , help="root path of the project")
args = parser.parse_args()
args = vars(args)
config_path = args["config_path"]
ROOT = args["root_path"]
config_path = os.path.join(ROOT , config_path)
with open(config_path , 'r') as f:
    config = json.load(f)

HOST = config["host"]
PORT = config["port"]
RSS_FEEDS_PATH = os.path.join(ROOT, config["rss_feeds_path"])



blueprint = Blueprint("api", __name__, url_prefix="/api/v1")

api = Api(
    blueprint,
    version="1.0",
    validate=False,
)

injected_object = {'rss_feed_path' : RSS_FEEDS_PATH}

# inject the objects containing logic here
for res in namesp.resources:
    res.kwargs['resource_class_kwargs'] = injected_object
    print(res)
# finally add namespace to api
api.add_namespace(namesp)

app = Flask('test')

app.register_blueprint(blueprint)
app.run(HOST, port=PORT, debug=True)