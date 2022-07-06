from flask import Flask, request
from flask_restx import Api, Resource, fields
import json
import argparse
import os


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


RSS_FEEDS_PATH = os.path.join(ROOT, config["rss_feeds_path"])
HOST = config["host"]
PORT = config["port"]



# Creation of the service with Flask
app = Flask(__name__)
api = Api(app)
name_space = api.namespace('RSSNewsExtractor', description='Extract news from RSS feeds and store them into JSON file')

# Model for required data to request the server
url_element = api.model("RSS URL",
                        {"url": fields.String(required=True,
                                              description="URL of the RSS feed to add",
                                              help="URL cannot be empty"),
                         "label": fields.List(fields.String(required=False,
                                                            description="Type(s) of the RSS feed"))})
model = api.model('RSSNewsExtractor Model',
                  {'rss_feed': fields.List(fields.Nested(url_element))})


@api.route("/AddRSSFeedSource")
class RSSNewsExtractor(Resource):
    """
    Rest interface to add rss feed to the extractor
    """
    def get(self):
        return 'DELETE Not available for this service', 404

    @api.expect(model)
    def post(self):
        try:
            # Add rss feed to existing rss feed list
            with open(RSS_FEEDS_PATH, "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open(RSS_FEEDS_PATH, "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @api.expect(model)
    def put(self):
        try:
            # Add rss feed to existing rss feed list
            with open(RSS_FEEDS_PATH, "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open(RSS_FEEDS_PATH, "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def delete(self):
        return 'DELETE Not available for this service', 404


if __name__ == '__main__':

    payload = {"rss_feed": [{"url": "tarace.com", "label": ["politique"]}]}
    app.run(HOST, port=PORT, debug=True)
