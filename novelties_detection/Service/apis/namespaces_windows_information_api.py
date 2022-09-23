import json
from flask_restx import Resource , Namespace
from novelties_detection.Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator , NoSupervisedFixedSequantialLangageSimilarityCalculator
from flask import jsonify , make_response
from novelties_detection.Service.server_utils import ServiceException
from novelties_detection.Service.apis.config_apis import N_TOP_DEFAULT , MAX_N_TOP_WORDS , MIN_N_TOP_WORDS


namesp = Namespace('WindowInformation',
                   description='api to get information about window (revelant words and topics)', validate=True)
parser = namesp.parser()
parser.add_argument("topic" , type = str , required = True , help = "label that we want to return revelant words" , location="form")
parser.add_argument("ntop", type=int, default = N_TOP_DEFAULT , help="number of top world used for similarity computation", location="form")
parser.add_argument("other_kwargs", type=str , default = "{}", help="other key words arguments like 'remove_seed_words' , 'exclusive'...", location="form")


@namesp.route("/<string:window_id>")
@namesp.doc(responses = {404: "Window not found"})
class WindowInformationApi(Resource):
    """
    Rest interface to get result from specific window by id
    """

    def __init__(self, api=None, *args, **kwargs):
        # sessions is a black box dependency
        self.supervised_calculator : SupervisedSequantialLangageSimilarityCalculator = kwargs['calculator']
        self.micro_topics_finder : NoSupervisedFixedSequantialLangageSimilarityCalculator = kwargs['topics_finder']
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
            topic = request_kwargs["topic"]
            if request_kwargs["ntop"] > MAX_N_TOP_WORDS or request_kwargs["ntop"] < MIN_N_TOP_WORDS :
                raise ServiceException(f"ntop too big must be inferior or equal to {MAX_N_TOP_WORDS} "
                                       f"and superior to {MIN_N_TOP_WORDS}")
            if topic not in self.supervised_calculator.labels_idx:
                raise ServiceException("topic is not in the labels list for this calculator")
            topic_id  = self.supervised_calculator.labels_idx.index(topic)
            total_kwargs = {"ntop" : request_kwargs["ntop"]}
            total_kwargs.update(o_kwargs)
            revelants_words = self.supervised_calculator.getTopWordsTopics(window_id, **total_kwargs)
            revelant_topic_words = revelants_words[topic_id]
            res["revelant_words_supervised"] = revelant_topic_words
            label_counter = self.supervised_calculator.label_articles_counters[window_id]
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