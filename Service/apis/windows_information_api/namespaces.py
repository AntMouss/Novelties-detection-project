from flask_restx import Resource , Namespace
from Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator , NoSupervisedSequantialLangageSimilarityCalculator

namesp = Namespace('WindowInformation',
                   description='api to get information about window (revelant words and topics)', validate=True)
parser = namesp.parser()
parser.add_argument("ntop", type=int, help="number of top world used for similarity computation", location="form")
parser.add_argument("oth_kwargs", type=dict, help="other key words arguments like 'remove_seed_words' , 'exclusive'...", location="form")


@namesp.route("/<string:window_id>")
@namesp.doc(responses = {404: "Window not found"})
class RSSNewsSource(Resource):
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
            request_kwargs = parser.parse_args()
            total_kwargs = {"ntop" : request_kwargs["ntop"]}
            total_kwargs.update(request_kwargs["other_kwargs"])
            revelants_words = self.supervised_calculator.getTopWordsTopics(window_id , **total_kwargs)
            label_counter = self.supervised_calculator.label_articles_counter[window_id]
            micro_topics = self.micro_topics_finder.getTopWordsTopics(window_id , **total_kwargs)
            result = {
                "revelant_words_supervised" : revelants_words,
                "label_counter" : label_counter,
                "micro_topics" : micro_topics
            }
            return result , 200
        except Exception as e:
            print(e)
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def post(self):
        return 'POST Not available for this service', 404

    def put(self):
        return 'PUT Not available for this service', 404

    def delete(self):
        return 'DELETE Not available for this service', 404