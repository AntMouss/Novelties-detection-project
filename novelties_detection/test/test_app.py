import requests
import pickle
from urllib.parse import urljoin
import os
import pytest
from .requests_examples import REQUESTS_EXAMPLES
import json

ROOT = "/home/mouss/PycharmProjects/novelties-detection-git"
RSS_FEEDS_PATH = "tmp_test_obj/rss_sport_test.json"
macro_calculator_path = "tmp_test_obj/macro_calculator.pck"
macro_calculator_path = os.path.join(ROOT, macro_calculator_path)
micro_calculator_path = "tmp_test_obj/micro_calculator.pck"
micro_calculator_path = os.path.join(ROOT , micro_calculator_path)
RSS_FEEDS_PATH = os.path.join(ROOT , RSS_FEEDS_PATH)

with open(macro_calculator_path , "rb") as f:
    MACRO_CALC = pickle.load(f)
with open(micro_calculator_path , "rb") as f:
    MICRO_CALC = pickle.load(f)
with open(RSS_FEEDS_PATH , "r") as f:
    RSS_FEEDS = json.load(f)

requests_examples = {
    endpoint : [
        (request_args["requests_params"] , request_args["expected"]) for request_args in requests_args
    ] for endpoint , requests_args in REQUESTS_EXAMPLES.items()
                     }

injected_object_apis = [{"rss_feed_path": RSS_FEEDS_PATH} , {"calculator" : MACRO_CALC} , {"calculator" : MACRO_CALC, "topics_finder" : MICRO_CALC}]



BASE_URL = "http://127.0.0.1:5000/api/v1/"
ENDPOINTS = ["RSSNewsDocument/AddRSSFeedSource" , "RSSNewsDocument/AddRSSFeedTags" , "ResultInterface" , "WindowInformation/"]


@pytest.mark.parametrize( "request_params,expected",requests_examples["AddRSSFeedSource"])
def test_AddRSSFeedSource_endpoint(request_params, expected):
    url = urljoin(BASE_URL , ENDPOINTS[0])
    payload = request_params["payload"]
    r = requests.post(url , json=payload )
    assert r.status_code == expected["code"]



@pytest.mark.parametrize( "request_params,expected",requests_examples["AddRSSFeedTags"])
def test_AddRSSFeedTags_endpoint(request_params, expected):
    url = urljoin(BASE_URL , ENDPOINTS[1])
    payload = request_params["payload"]
    r = requests.post(url, json=payload)
    assert r.status_code == expected["code"]



@pytest.mark.parametrize( "request_params,expected",requests_examples["ResultInterface"])
def test_ResultInterface_endpoint(request_params, expected):
    url = urljoin(BASE_URL , ENDPOINTS[2])
    r = requests.get(url, data=request_params)
    assert r.status_code == expected["code"]
    #data = r.json()
    #next step in work


@pytest.mark.parametrize( "request_params,expected",requests_examples["WindowInformation"])
def test_WindowInformation_endpoint(request_params, expected):
    url = urljoin(BASE_URL , ENDPOINTS[3])
    if "window_id" in request_params.keys():
        window_id = request_params["window_id"]
        url = urljoin(url , str(window_id))
        del request_params["window_id"]
    if "other_kwargs" in request_params.keys():
        request_params["other_kwargs"] = json.dumps(request_params["other_kwargs"])
        #del request_params["other_kwargs"]
    r = requests.get(url, data=request_params)
    assert r.status_code == expected["code"]
    # if r.status_code == 200:
    #     data = r.json()
        #next step in work



