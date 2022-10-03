from flask_restx import Resource , Namespace
from novelties_detection.Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator
from flask import jsonify , make_response
from novelties_detection.Service.endpoints.apis_utils import build_series, ServiceException
from novelties_detection.Service.endpoints.config_apis import N_TOP_DEFAULT , BACK_DEFAULT , MAX_N_TOP_WORDS , MIN_N_TOP_WORDS

namesp = Namespace('ResultInterface',
                   description='results generate by the calculator for the interface topic visualisation', validate=True)
parser = namesp.parser()
parser.add_argument("ntop", type=int , default = N_TOP_DEFAULT , help="number of top world used for similarity computation", location="form")
parser.add_argument("back", type=int,default = BACK_DEFAULT, help="number of backward windows used for similarity computation (moving mean)", location="form")
#parser.add_argument("oth_kwargs", type=dict, help="other key words arguments like 'remove_seed_words' , 'exclusive'...", location="form")

@namesp.route("")
class ResultInterfaceAPI(Resource):
    """
    Rest interface to get similarity result between windows sequentialy for each topic
    """

    def __init__(self, api=None, *args, **kwargs):
        """
        supervised calculator to return novelties about topics and words.
        Able to compute similarity between to adjacent window
        @param api:
        @param args:
        @param kwargs:
        """
        self.calculator : SupervisedSequantialLangageSimilarityCalculator = kwargs['calculator']
        super().__init__(api, *args, **kwargs)

    @namesp.doc(parser = parser)
    def get(self):
        try:
            kwargs = parser.parse_args()
            if kwargs["ntop"] >MAX_N_TOP_WORDS or kwargs["ntop"] < MIN_N_TOP_WORDS :
                raise ServiceException(f"ntop too big must be inferior or equal to {MAX_N_TOP_WORDS} "
                                       f"and superior to {MIN_N_TOP_WORDS}")
            if len(self.calculator) < 2:
                raise ServiceException("there is no enough treated window for the moment...Service not available."
                                       "This Service needs at least two windows to work")
            similarity_scores = self.calculator.compare_Windows_Sequentialy(**kwargs)
            labels = self.calculator.labels_idx
            labels_counters = self.calculator.label_articles_counters
            series = build_series(similarity_scores , labels_counters , labels)

            return make_response(jsonify(series), 200)

        except ServiceException as e:
            namesp.abort(410 , e.__doc__ , status = e.__str__ (), statuscode = "410")
        except Exception as e:
            print(e)
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def post(self):
        return 'POST Not available for this service', 404

    def put(self):
        return 'PUT Not available for this service', 404

    def delete(self):
        return 'DELETE Not available for this service', 404

