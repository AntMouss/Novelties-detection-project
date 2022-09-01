import logging
from time import time


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        start_time = time()
        func(*args, **kwargs)
        end_time = time()
        duration = end_time - start_time
        #print(f"duration : {duration}")
        logging.info(f"{func.__name__} finish  , collect duration : {duration}")

    return wrap_func