from abc import abstractmethod
import logging

mdl_logger = logging.getLogger('model')

def backoff_handler(details):
    mdl_logger.warn("Backing off {wait:0.1f} seconds after {tries} tries ".format(**details))

class TextCompletionModel:

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def complete(self, prompt, args=None):
        """Should return the completion of the prompt"""
        pass
    
    @abstractmethod
    def get_num_tokens(self, prompt, args=None):
        pass

    @abstractmethod
    def cleanup():
        pass

class ChatCompletionModel:

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def complete(self, prompt, args=None):
        """Should return the completion of the prompt, appended to preexisting chat history"""
        pass
    
    @abstractmethod
    def get_num_tokens(self, prompt, args=None):
        pass

    @abstractmethod
    def cleanup(self):
        pass
