import openai
import logging
import tiktoken
import time
import backoff
from promptbench.model import TextCompletionModel, ChatCompletionModel

def backoff_hdlr(details):
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))
    print(details)


@backoff.on_exception(backoff.expo, Exception, max_tries=10, on_backoff=backoff_hdlr)
@backoff.on_predicate(backoff.expo, lambda response: response is None)
def get_response(prompt, args, chat=False):
    if chat:
        response = openai.ChatCompletion.create(**{'messages': prompt, **args})
    else:
        response = openai.Completion.create(**{'prompt': prompt, **args})

    return response

def setup_api_key(api_key):
    openai.api_key = api_key

class GPT(TextCompletionModel):

    DEFAULT_ARGS = {
        'model': 'text-davinci-003',
        'max_tokens': 1024,
        'stop': None,
        'temperature': 0.5
    }

    def __init__(self, default_args=None):

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = GPT.DEFAULT_ARGS

        self.logger = logging.getLogger('GPT')
        self.enc = tiktoken.encoding_for_model(self.default_args['model'])

    def complete(self, prompt, args=None):

        response = None
        if not args:
            args = self.default_args

        try:
            response = get_response(prompt, args)
            
        except Exception as e:
            # TODO exponential backoff and termination
            print(e)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            if not response:
                return None
            return response['choices'][0]['text'].strip()
    
    def cleanup(self):
        # nothing to cleanup 
        pass

    def get_num_tokens(self, prompt, args=None):

        return len(self.enc.encode(prompt))

class ChatGPT(ChatCompletionModel):

    DEFAULT_ARGS = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 1024,
        'stop': None,
        'temperature': 0.5
    }

    def __init__(self, default_args=None):

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = ChatGPT.DEFAULT_ARGS

        self.logger = logging.getLogger('GPT')

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def complete(self, prompt, args=None):

        response = None

        if not args:
            args = self.default_args

        self.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        self.logger.info(f'Giving the following prompt:{prompt}')
        try:
            response = get_response(self.chat_history, args, chat=True)

        except Exception as e:
            # TODO exponential backoff and termination
            print(e)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            if not response:
                return None
            return response['choices'][0]['message']['content'].strip()

    def cleanup(self):

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]


