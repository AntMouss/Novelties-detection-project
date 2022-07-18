

class SelectionException(Exception):
    pass

class MetadataGenerationException(SelectionException):
    def __init__(self,  message):
        super(MetadataGenerationException, self).__init__(message)

class TimelinesGenerationException(SelectionException):
    def __init__(self , message , origin_exception):
        super(TimelinesGenerationException, self).__init__(message)
        self.origin_exception = origin_exception

class CalculatorGenerationException(SelectionException):
    def __init__(self , message):
        super(CalculatorGenerationException, self).__init__(message)

class ResultsGenerationException(SelectionException):
    def __init__(self , message):
        super(ResultsGenerationException, self).__init__(message)

class AnalyseException(SelectionException):
    def __init__(self , message):
        super(AnalyseException, self).__init__(message)


class CompareWindowsException(Exception):
    pass

