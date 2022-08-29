from flask_restx import Model , fields


tag_element = Model("RSS Tag" ,
                    {
                        "tag" : fields.String(required=True,
                                              description="tag name of the tag feed to add",
                                              help="tag field cannot be empty"),
                        "class" : fields.String(description="class name of the tag feed to add"),
                        "id" : fields.String(description="id of the tag feed to add")
                    })

url_element = Model("RSS URL",
                        {"url": fields.String(required=True,
                                              description="URL of the RSS feed to add",
                                              help="URL cannot be empty"),
                         "label": fields.List(fields.String(required=False,
                                                            description="Type(s) of the RSS feed")),
                         "remove_tags" : fields.List(fields.Nested(tag_element , required=False,
                                                                   description="tags to remove in the article page for cleaning"))})

rss_requests_model = Model('RSSNewsSource Model',
                           {'rss_feed': fields.List(fields.Nested(url_element))})


tags_requests_model = Model("RSSNewsTags Model" , {"tags" : fields.List(fields.Nested(tag_element))})