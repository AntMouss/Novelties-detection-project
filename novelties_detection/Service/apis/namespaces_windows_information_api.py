import json
from flask_restx import Resource , Namespace
from novelties_detection.Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator , NoSupervisedSequantialLangageSimilarityCalculator
from flask import jsonify , make_response
from novelties_detection.Experience.Exception_utils import ServiceException

namesp = Namespace('WindowInformation',
                   description='api to get information about window (revelant words and topics)', validate=True)
parser = namesp.parser()
parser.add_argument("ntop", type=int, default = 100 , help="number of top world used for similarity computation", location="form")
parser.add_argument("other_kwargs", type=str , default = "{}", help="other key words arguments like 'remove_seed_words' , 'exclusive'...", location="form")


@namesp.route("/<string:window_id>")
@namesp.doc(responses = {404: "Window not found"})
class WindowInformationApi(Resource):
    """
    Rest interface to get result from specific window
    """

    def __init__(self, api=None, *args, **kwargs):
        # sessions is a black box dependency
        self.supervised_calculator : SupervisedSequantialLangageSimilarityCalculator = kwargs['calculator']
        self.micro_topics_finder : NoSupervisedSequantialLangageSimilarityCalculator = kwargs['topics_finder']
        super().__init__(api, *args, **kwargs)

    @namesp.doc(parser = parser)
    def get(self , window_id):
        try:
            if len(self.supervised_calculator) == 0:
                raise ServiceException("no window treated by this service yet... so service not available")
            res = {}
            window_id = int(window_id)
            request_kwargs = parser.parse_args()
            o_kwargs = request_kwargs["other_kwargs"]
            o_kwargs = json.loads(o_kwargs)
            if request_kwargs["ntop"] > 500:
                raise Exception
            total_kwargs = {"ntop" : request_kwargs["ntop"]}
            total_kwargs.update(o_kwargs)
            revelants_words = self.supervised_calculator.getTopWordsTopics(window_id, **total_kwargs)
            res["revelant_words_supervised"] = revelants_words
            label_counter = self.supervised_calculator.label_articles_counter[window_id]
            res["label_counter"] = label_counter
            if self.micro_topics_finder is not None:
                micro_topics = self.micro_topics_finder.getTopWordsTopics(window_id , **total_kwargs)
                res["micro_topics"] = micro_topics


            return make_response(jsonify(res), 200)
        except ServiceException as e:
            namesp.abort(410, e.__doc__, status=e.__str__(), statusCode="410")
        except IndexError as e:
            print(e)
            namesp.abort(404, e.__doc__, status="this window_id doesn't match with any windows", statusCode="404")
        except Exception as e:
            print(e)
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def post(self):
        return 'POST Not available for this service', 404

    def put(self):
        return 'PUT Not available for this service', 404

    def delete(self):
        return 'DELETE Not available for this service', 404