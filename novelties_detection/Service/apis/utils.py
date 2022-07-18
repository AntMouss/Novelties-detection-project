import json
import numpy as np
from flask_restx import Namespace

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float("{:.4f}".format(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_ressources(namespace : Namespace, object : dict):
    for ressource in namespace.resources:
        ressource.kwargs['resource_class_kwargs'] = object
    return namespace