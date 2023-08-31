from promptbench.model import TextCompletionModel, backoff_handler
import logging
import google.generativeai as palm
import time
import backoff

def setup_api_key(api_key):
    palm.configure(api_key=api_key)

@backoff.on_exception(backoff.expo, Exception, max_tries=10, on_backoff=backoff_handler)
@backoff.on_predicate(backoff.expo, lambda response: response is None)
def get_response(prompt, args):
    response = palm.generate_text(**{'prompt': prompt, **args})

    return response

class PaLM(TextCompletionModel):

    DEFAULT_ARGS = {
        'model': 'models/text-bison-001',
        'max_output_tokens': 1024,
        'temperature': 0.5,
        #'top_k' : 40
    }

    def __init__(self, default_args=None):

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = PaLM.DEFAULT_ARGS

        self.logger = logging.getLogger('PaLM')
        #self.enc = tiktoken.encoding_for_model(self.default_args['model'])

    def complete(self, prompt, args=None):

        response = None
        if not args:
            args = self.default_args

        if not args['model'].startswith('models/'):
            args['model'] = 'models/' + args['model']

        #self.logger.info(f'Giving the following prompt:{prompt}')
        try:
            response = get_response(prompt, args)
        except Exception as e:
            # TODO exponential backoff and termination
            print(e)
            if str(e).find('limit')!=-1:
                time.sleep(45)
            elif str(e).find('Shutdown')!=-1:
                print("Shutting down for 5 minutes")
                time.sleep(600)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            if not response:
                #time.sleep(30)
                return None
            return response.result

    def get_num_tokens(self, prompt, args=None):

        #return len(self.enc.encode(prompt))
        #TODO see later if same exists for palm
        return 0
