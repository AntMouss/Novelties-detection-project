import logging
from datetime import datetime
from functools import partial




def timer_func(func , name : str = None):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):

        start_time = datetime.now()
        logging.info(f"{name} begin at {start_time}")
        func(*args, **kwargs)
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"{name} finish at {end_time}  , with duration : {duration.total_seconds()} seconds")

    return wrap_func

collect_decorator_timer = partial(timer_func, name="Collect News")
process_decorator_timer = partial(timer_func, name="Process News")
