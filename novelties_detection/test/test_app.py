import requests
from urllib.parse import urljoin
import pytest
from novelties_detection.test.requests_examples import REQUESTS_EXAMPLES
import json
from novelties_detection.test.testServer import run_test_server

requests_examples = {
    endpoint : [
        (request_args["requests_params"] , request_args["expected"]) for request_args in requests_args
    ] for endpoint , requests_args in REQUESTS_EXAMPLES.items()
                     }


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

