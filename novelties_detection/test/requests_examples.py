REQUESTS_EXAMPLES = {
    "AddRSSFeedSource" : [
        {
            "requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : "hello.com",
                            "label":[
                                "general",
                                "crime"
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 200
            }

        },
{
            "requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : "hello.com",
                            "label":["crime"],
                            "remove_tags":
                            [
                                {
                                    "tag" : "hello",
                                    "class" : "bonjour",
                                    "id" : "hallo"
                                }
                                ,
                                {
                                    "tag" : "bien",
                                }
                            ],
                        }
                    ]
                }
            },
            "expected":{
                "code" : 200
            }

        },
{
            "requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : "bonjour.fr",
                            "label":[
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 200
            }

        },
{
            "requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : "gutentag.de",
                            "label":[
                                "general"
                            ]
                        },
                        {
                            "url" : "bondia.po",
                            "label":[
                            ],
                            "remove_tags":[
                                {
                                    "tag" : "div",
                                    "class" : "aupif"
                                }
                            ]
                        },
                        {
                            "url" : "bonjourno.it",
                            "label":[
                                "crime",
                                "justice"
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 200
            }

        },
{
            "requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : "holla.es",
                            "label":[
                                "cuisine"
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 410
            }

        },
{
            "requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : "salam.com",
                            "label":[
                                "general"
                            ],
                            "remove_tags":
                            [
{
                                    "class" : "bonjour",
                                    "id" : "hallo"
                                }
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 400
            }

        },
        {
"requests_params" : {
                "payload" : {
                    "rss_fed" : [
                        {
                            "url" : "hello.com",
                            "label":[
                                "general",
                                "crime"
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 500
            }
        },
        {
"requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : 2,
                            "label":[
                                "general",
                                "crime"
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 400
            }
        },
        {
"requests_params" : {
                "payload" : {
                    "rss_feed" : [
                        {
                            "url" : "hello.com",
                            "label":[
                               2,
                                3
                            ]
                        }
                    ]
                }
            },
            "expected":{
                "code" : 400
            }
        }
    ],
    "AddRSSFeedTags"  : [
        {
            "requests_params": {
                "payload": {
                    "tags": [
                        {
                            "tag" : "hello",
                            "class" : "bonjour",
                            "id" : "hallo"
                        }
                        ,
                        {
                            "tag" : "bien",
                        }

                    ]
                }
            },
            "expected": {
                "code": 200
            }

        },
        {
            "requests_params": {
                "payload": {
                    "tg": [
                        {
                            "tag": "hello",
                            "class": "bonjour",
                            "id": "hallo"
                        }
                        ,
                        {
                            "tag": "bien",
                        }
                    ]
                }
            },
            "expected": {
                "code": 500
            }
        },
        {
            "requests_params": {
                "payload": {
                    "tags": [
                        {
                            "class" : "bonjour",
                            "id" : "hallo"
                        }
                    ]
                }
            },
            "expected": {
                "code": 400
            }
        },
        {
            "requests_params": {
                "payload": {
                    "tags": [
                        {
                            "tag" : "bonjour",
                            "id" : 4
                        }
                    ]
                }
            },
            "expected": {
                "code": 400
            }
        }
    ],
    "ResultInterface" : [
{
            "requests_params": {
                "ntop" : 60,
                "back" : 2
            },
            "expected": {
                "code": 200,
            }
        },
{
            "requests_params": {
                "ntp" : 60,
                "back" : 2
            },
            "expected": {
                "code": 200
            }
        },
        {
            "requests_params": {
                "ntop": 60,
                "back": 0
            },
            "expected": {
                "code": 400
            }
        }
        ,
{
            "requests_params": {
                "ntop" : 130
            },
            "expected": {
                "code": 200
            }
        },
{
            "requests_params": {
                "ntop" : 10000,
                "back" : 2
            },
            "expected": {
                "code": 410
            }
        },
{
            "requests_params": {
                "ntop" : 10,
                "back" : 2
            },
            "expected": {
                "code": 410
            }
        }
    ],
    "WindowInformation" : [
{
            "requests_params": {
                "ntop" : 100,
                "back" : 2
            },
            "expected": {
                "code": 404
            }
        },
{
            "requests_params": {
                "topic" : "general",
                "window_id": 2,
                "ntop" : 111,
                "other_kwargs" : {"remove_seed_words" : False}
            },
            "expected": {
                "code": 200
            }
        },
{
            "requests_params": {
                "topic": "crime",
                "window_id": 5,
                "ntop" : 311,
                "other_kwargs" : {"exclusive" : True}
            },
            "expected": {
                "code": 200
            }
        },
{
            "requests_params": {
                "topic": "general",
                "window_id": 2,
                "ntop" : 11111,
                "other_kwargs" : {"exclusive" : True}
            },
            "expected": {
                "code": 410
            }
        },
{
            "requests_params": {
                "topic": "general",
                "window_id": 2,
                "ntop" : 311,
                "other_kwargs" : {"exclusive" : True , "nimportequoi" : False}
            },
            "expected": {
                "code": 200
            }
        },
{
            "requests_params": {
                "topic": "general",
                "window_id": 1,
                "ntop" : 11,
            },
            "expected": {
                "code": 410
            }
        },
{
            "requests_params": {
                "topic": "general",
                "window_id": 30,
                "ntop" : 110,
            },
            "expected": {
                "code": 404
            }
        }


    ]
}