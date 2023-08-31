from promptbench.strategies import *
from promptbench.models.openai import GPT, ChatGPT
from promptbench.models.google import PaLM
from promptbench.models.llama import Llama2
from promptbench.dataset import JSONLDataset

strategies = {
    'cot': chain_of_thought,
}

datasets = {
    # your dataset paths go here
}

models = {
    'text-davinci-003': GPT,
    'text-davinci-002': GPT,
    'gpt-3.5-turbo': ChatGPT,
    'gpt-4': ChatGPT,
    'text-bison-001' : PaLM,
    'llama-2-70b' : Llama2
}
