from flask import Flask, request
from flask_restplus import Api, Resource, fields
import json
import sys
import os
from threading import Thread
import schedule
from RSSCollector import RSSCollect
from Sequential_Module import MetaSequencialLangageModeling

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



class CollectThread(Thread):
    """
    Service to periodically collect information from the selected sources
    """
    def __init__(self,rss_feed_config_file,output_path,loop_delay):
        Thread.__init__(self)
        self.rss_feed_config=rss_feed_config_file
        self.output_path=output_path
        self.loop_delay=loop_delay

    def run(self):
        rssCollect=RSSCollect(self.rss_feed_config, self.output_path)
        schedule.every(self.loop_delay).minutes.do(rssCollect.treatNewsFeedList)
        while True:
            schedule.run_pending()


class NoveltiesDetectionThread(Thread):
    """
    Service to detect novelties in the  collect information flow
    """
    def __init__(self, kwargs_model, data_window, loop_delay):
        Thread.__init__(self)
        self.data_window = data_window
        self.loop_delay = loop_delay
        self.model_type = kwargs_model['initialize_engine']['model_type']
        self.training_args = kwargs_model['initialize_engine']['training_args']
        self.comparaison_args = kwargs_model['generate_result']
        del kwargs_model['initialize_engine']['model_type']
        del kwargs_model['initialize_engine']['training_args']
        sequential_model = self.model_type
        self.reference_model : MetaSequencialLangageModeling  = sequential_model(**kwargs_model['initialize_engine'])

    def process(self):
        self.reference_model.treat_Window(self.data_window , **self.training_args)
        res = self.reference_model.calcule_similarity_topics_W_W(**self.comparaison_args)
        # do the characterization of res

    def run(self):
        schedule.every(self.loop_delay).minutes.do(self.process)
        while True:
            schedule.run_pending()



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
            with open("rssfeed_FUN.json", "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open("rssfeed_FUN.json", "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @api.expect(model)
    def put(self):
        try:
            # Add rss feed to existing rss feed list
            with open("rssfeed_FUN.json", "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open("rssfeed_FUN.json", "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def delete(self):
        return 'DELETE Not available for this service', 404

def startServer():
    '''
    Starts server
    :return:
    '''
    extractor = CollectThread(config["rss_feed_config_file"],config["output_path"],config["loop_delay"])
    detector = NoveltiesDetectionThread(**config['train'])
    extractor.start()
    detector.start()
    extractor.join()
    detector.join()
    print("Running Rest RSSNewsExtractor server")
    app.run(config["host"], port=config["port"], debug=False)

if __name__ == '__main__':
    rootDir = ''
    if len(sys.argv) > 1:
        rootDir=sys.argv[1]
    with open(os.path.join(rootDir, "config/config_service.json"), "r") as f:
        config = json.load(f)

    startServer()
