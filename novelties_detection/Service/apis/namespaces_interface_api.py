from flask_restx import Resource , Namespace
from novelties_detection.Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator
from flask import jsonify , make_response

namesp = Namespace('ResultInterface',
                   description='results generate by the calculator for the interface visualisation', validate=True)
parser = namesp.parser()
parser.add_argument("ntop", type=int , default = 100 , help="number of top world used for similarity computation", location="form")
parser.add_argument("back", type=int,default = 3, help="number of backward windows used for similarity computation (moving mean)", location="form")
#parser.add_argument("oth_kwargs", type=dict, help="other key words arguments like 'remove_seed_words' , 'exclusive'...", location="form")

@namesp.route("")
class ResultInterfaceAPI(Resource):
    """
    Rest interface to get similarity result between windows sequentialy
    """

    def __init__(self, api=None, *args, **kwargs):
        # sessions is a black box dependency
        self.calculator : SupervisedSequantialLangageSimilarityCalculator = kwargs['calculator']
        super().__init__(api, *args, **kwargs)

    @namesp.doc(parser = parser)
    def get(self):
        try:
            kwargs = parser.parse_args()
            if kwargs["ntop"] >500:
                raise Exception
            similarity_resultats = self.calculator.compare_Windows_Sequentialy(**kwargs)
            labels_idx = self.calculator.labels_idx
            res = {
                label : similarity_res_label for label , similarity_res_label in zip(labels_idx, similarity_resultats)
            }
            return make_response(jsonify(res), 200)
        except Exception as e:
            print(e)
            namesp.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def post(self):
        return 'POST Not available for this service', 404

    def put(self):
        return 'PUT Not available for this service', 404

    def delete(self):
        return 'DELETE Not available for this service', 404

