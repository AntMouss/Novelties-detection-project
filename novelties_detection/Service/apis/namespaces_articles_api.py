from flask_restx import Resource , Namespace
from flask import jsonify , make_response
from novelties_detection.Service.apis.apis_utils import ElasticSearchQueryBodyBuilder , ServiceException
from elasticsearch import Elasticsearch

namesp_articles = Namespace('articles',
                            description='articles collected by the service', validate=True)

parser = namesp_articles.parser()
parser.add_argument("match", type=str , help="string matching with articles ", location="form")
parser.add_argument("END_DATE", type=str, help="end date boundaries for the requests in ISO-8601", location="form")
parser.add_argument("START_DATE", type=str, help="start date boundaries for the requests in ISO-8601", location="form")
parser.add_argument("label", type=str, help="filtre by label", location="form")


@namesp_articles.route("")
class ArticlesEndpoint(Resource):
    """
    Rest interface to get similarity result between windows sequentialy for each topic
    """
    index_name = "articles-index"

    def __init__(self, api=None, *args, **kwargs):
        """
        supervised calculator to return novelties about topics and words.
        Able to compute similarity between to adjacent window
        @param api:
        @param args:
        @param kwargs:
        """
        self.elastic_client = kwargs['elastic_client']
        super().__init__(api, *args, **kwargs)

    @namesp_articles.doc(parser = parser)
    def get(self):
        try:
            kwargs = parser.parse_args()
            body = ElasticSearchQueryBodyBuilder(**kwargs)
            elastic_response = self.elastic_client.search(index=self.index_name, query=body)
            data = [hits["hits"]["_source"] for hits in elastic_response["hits"]]
            response_api = {
                "total" : elastic_response["hits"]["total"],
                "max_score" :elastic_response["hits"]["max_score"] ,
                "data" : data
            }
            return make_response(jsonify(response_api), 200)

        except ServiceException as e:
            namesp_articles.abort(410, e.__doc__, status = e.__str__ (), statuscode ="410")
        except Exception as e:
            print(e)
            namesp_articles.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def post(self):
        return 'POST Not available for this service', 404

    def put(self):
        return 'PUT Not available for this service', 404

    def delete(self):
        return 'DELETE Not available for this service', 404






namesp_article = Namespace('article',
                            description='fetch article by id', validate=True)
parser2 = namesp_article.parser()
parser2.add_argument("id", type=str, help="select article with good id", location="form")


@namesp_article.route("")
class ArticlesEndpoint(Resource):
    """
    Rest interface to get similarity result between windows sequentialy for each topic
    """
    index_name = "articles-index"

    def __init__(self, api=None, *args, **kwargs):
        """
        supervised calculator to return novelties about topics and words.
        Able to compute similarity between to adjacent window
        @param api:
        @param args:
        @param kwargs:
        """
        self.elastic_client = kwargs['elastic_client']
        super().__init__(api, *args, **kwargs)

    @namesp_article.doc(parser=parser2)
    def get(self):
        try:
            kwargs = parser.parse_args()
            id = kwargs["id"]
            resp = es.get(index=self.index_name, id=id)
            res = resp['_source']

            return make_response(jsonify(res), 200)

        except ServiceException as e:
            namesp_articles.abort(410, e.__doc__, status=e.__str__(), statuscode="410")
        except Exception as e:
            print(e)
            namesp_articles.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def post(self):
        return 'POST Not available for this service', 404

    def put(self):
        return 'PUT Not available for this service', 404

    def delete(self):
        return 'DELETE Not available for this service', 404