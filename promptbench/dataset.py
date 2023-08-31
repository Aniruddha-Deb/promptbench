import json

import logging

ds_logger = logging.getLogger(name='dataset')

class JSONLDataset:

    def __init__(self, filepath):

        self.filepath = filepath
        self.examples = []

        with open(filepath, 'r') as file:
            for l in file:
                self.examples.append(json.loads(l))

    def __getitem__(self, key):
        return self.examples[key]

    def __len__(self):
        return len(self.examples)

class TabularDataset:

    def __init__(self, filepath, header=True, delimiter=','):

        self.filepath = filepath
        self.examples = []

        with open(filepath, 'r') as file:
            lines = file.readlines()
            keys = None
            if header:
                keys, lines = lines[0].split(delimiter), lines[1:]
            for i, l in enumerate(lines):
                row = l.split(delimiter)
                if keys and (len(row) != len(keys)):
                    ds_logger.error(f'Row no. {i} could not be parsed. continuing.')
                    continue

                example = row
                if keys:
                    example = {k: v for k,v in zip(keys, row)}

                self.examples.append(example)
        
    def __getitem__(self, key):
        return self.examples[key]
    
    def __len__(self):
        return len(self.examples)
