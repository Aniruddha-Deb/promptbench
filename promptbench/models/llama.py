import requests
import backoff
import logging
from promptbench.model import TextCompletionModel, backoff_handler

api_url = None

class Completion:

    @classmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=10, on_backoff=backoff_handler)
    @backoff.on_predicate(backoff.expo, lambda response: response is None)
    def create(cls, *args, **kwargs):
        """
        Creates a new completion for the provided prompt and parameters.
        """

        if not api_url:
            raise ValueError("API URL cannot be empty")
        
        response = requests.post(
                api_url+'/api/v1/completions', 
                headers={'Content-Type': 'application/json'},
                json=kwargs
            ).json()
        return response

class Llama2(TextCompletionModel):

    DEFAULT_ARGS = {
        'max_respose_tokens': 512,
        'temperature': 0.5
    }

    def __init__(self, default_args=None):

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = Llama2.DEFAULT_ARGS

        self.logger = logging.getLogger('GPT')

    def complete(self, prompt, args=None):

        response = None
        if not args:
            args = self.default_args

        try:
            response = Completion.create(**{
                'prompt': prompt,
                **args
            })
            
        except Exception as e:
            # TODO exponential backoff and termination
            print(e)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            if not response:
                return None
            return response['completion'].strip()

    def get_num_tokens(self, prompt, args=None):
        # TODO use tokenizer 
        return 0

