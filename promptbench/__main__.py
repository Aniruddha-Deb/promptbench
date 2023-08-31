from promptbench.logger import DataLogger
import promptbench.models.openai as openai
import promptbench.models.google as google
import promptbench.models.llama as llama
from promptbench.prompt import PromptGenerator
from promptbench.dataset import JSONLDataset, TabularDataset
import promptbench.config as config
from tqdm.auto import tqdm
import logging
from datetime import datetime

from dotenv import load_dotenv
import os
import argparse
import shutil
import time

def parse_args():

    parser = argparse.ArgumentParser(
            prog='promptbench',
            description='Prompt benchmarking utility'
        )

    parser.add_argument('-m', '--models',   type=str, nargs="+", default=["gpt-3.5-turbo"])
    parser.add_argument('-p', '--prompts',  type=str, nargs="+", default=["cot"])
    parser.add_argument('-d', '--datasets', type=str, nargs="+", default=["gsm8k"])
    parser.add_argument('-l', '--llama-url', type=str, help="LLaMa API URL")
    parser.add_argument('-y', '--yes', action="store_true", help="Say yes to any conditionals")
    parser.add_argument('-r', '--result-dir', type=str, default=f"results/run_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    

    parser.add_argument('-s', '--split-start',   type=int, default=0)
    parser.add_argument('-e', '--split-end',     type=int, default=50)
    parser.add_argument('-i', '--interm',        type=int, default=10)
    
    return parser.parse_args()

def main():

    args = parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    openai.setup_api_key(os.environ.get('OPENAI_API_KEY'))
    google.setup_api_key(os.environ.get('PALM_API_KEY'))
    llama.api_url = args.llama_url

    if os.path.exists(args.result_dir):
        if args.yes:
            print('Output folder already exists, overwriting')
            shutil.rmtree(args.result_dir)
        else:
            print('Overwrite preexisting output folder? (y/N): ', end='')
            ch = input()
            if (ch == 'y'):
                shutil.rmtree(args.result_dir)
            else:
                args.result_dir += '_1'

    os.makedirs(args.result_dir)

    logging.basicConfig(
            filename=os.path.join(args.result_dir, 'logfile.log'),
            filemode='a',
            format='[%(asctime)s.%(msecs)d](%(name)s:%(levelname)s) %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )

    data_logger = DataLogger()
    pg = PromptGenerator('prompts')

    logger = logging.getLogger('main')

    for model_name in args.models:

        model_args = config.models[model_name].DEFAULT_ARGS.copy()
        model_args['model'] = model_name
        model = config.models[model_name](model_args)

        for dataset_name in args.datasets:

            if dataset_name in config.datasets:
                dataset_name = config.datasets[dataset_name]

            if dataset_name.endswith('.jsonl'):
                ds = JSONLDataset(dataset_name)[args.split_start:args.split_end]
            elif dataset_name.endswith('.csv'):
                ds = TabularDataset(dataset_name, delimiter=',')[args.split_start:args.split_end]
            elif dataset_name.endswith('.tsv'):
                ds = TabularDataset(dataset_name, delimiter='\t')[args.split_start:args.split_end]
            else:
                logger.error('Dataset type not recognized. Continuing.')
                continue

            for prompt_strategy in args.prompts:
                interm=args.interm

                data = {
                    'correct': 0,
                    'total': 0,
                    'extracted': 0,
                    'responses': []
                }
                logger.info(f"{model_name}:{dataset_name}:{prompt_strategy}")

                data_logger[model_name][dataset_name][prompt_strategy] = data
                for example in tqdm(ds):
                    # TODO dry-run logic
                    config.strategies[prompt_strategy](example, pg, model, data)
                    interm-=1
                    if interm==0:
                        data_logger.save(args.result_dir)
                        interm=args.interm

                logger.info(f"{data['total']} examples run")
                logger.info(f"Accuracy: {data['correct']/data['total']} ({data['correct']}/{data['total']})")

    data_logger.save(args.result_dir)

if __name__ == "__main__":
    main()
