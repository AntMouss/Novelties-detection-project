from flask import request
from flask_restx import Resource , Namespace
import json
from Service.apis.Models import rss_requests_model , tags_requests_model , tag_element , url_element


namesp = Namespace('RSSNewsDocument',
                   description='Extract news from RSS feeds and store them into JSON file', validate=True)
# register model
namesp.models[rss_requests_model.name] = rss_requests_model
namesp.models[tags_requests_model.name] = tags_requests_model
namesp.models[tag_element.name] = tag_element
namesp.models[url_element.name] = url_element

@namesp.route("/AddRSSFeedSource")
class RSSNewsSource(Resource):
    """
    Rest interface to add rss feed to the extractor
    """

    def __init__(self, api=None, *args, **kwargs):
        # sessions is a black box dependency
        self.rss_feed_path = kwargs['rss_feed_path']
        super().__init__(api, *args, **kwargs)


    def get(self):
        return 'GET Not available for this service', 404

    @namesp.expect(rss_requests_model , validate = True)
    def post(self):
        try:
            # Add rss feed to existing rss feed list
            with open(self.rss_feed_path, "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open(self.rss_feed_path, "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            namesp.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @namesp.expect(rss_requests_model , validate = True)
    def put(self):
        try:
            # Add rss feed to existing rss feed list
            with open(self.rss_feed_path, "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open(self.rss_feed_path, "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            namesp.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def delete(self):
        return 'DELETE Not available for this service', 404



@namesp.route("/AddRSSFeedTags")
class RSSNewsRemovingTags(Resource):
    """
    Rest interface to add rss feed tags that we need to remove during pre-proceesing (cleaning)
    during the extraction
    """

    def __init__(self, api=None, *args, **kwargs):
        # sessions is a black box dependency
        self.rss_feed_path = kwargs['rss_feed_path']
        super().__init__(api, *args, **kwargs)


    def get(self):
        return 'GET Not available for this service', 404

    @namesp.expect(tags_requests_model , validate = True)
    def post(self):
        try:
            # Add rss feed to existing rss feed list
            with open(self.rss_feed_path, "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["global_remove_tags"] = rss_feed_url["global_remove_tags"] + request.json['tags']
            with open(self.rss_feed_path, "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            namesp.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @namesp.expect(tags_requests_model , validate = True)
    def put(self):
        try:
            # Add rss feed to existing rss feed list
            with open(self.rss_feed_path, "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["global_remove_tags"] = rss_feed_url["global_remove_tags"] + request.json['tags']
            with open(self.rss_feed_path, "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            namesp.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def delete(self):
        return 'DELETE Not available for this service', 404