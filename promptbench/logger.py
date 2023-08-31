from nested_dict import nested_dict
import os
import json

class DataLogger:

    def __init__(self):
        self.data = nested_dict()

    def save(self, dir):
        
        accs = {
            'correct' : {},
            'extracted' : {},
            'total'   : {}
        }

        for keys_as_tuple, value in self.data.items_flat():
            str_key = "-".join([a.split('/')[-1].split('.')[0].replace('-', '_') for a in keys_as_tuple[:-1]])
            if keys_as_tuple[-1] == "responses":
                with open(os.path.join(dir, f'{str_key}.txt'), 'w+') as outfile:
                    for response in value:
                        outfile.write(f"{json.dumps(response)}\n")
            else:
                accs[keys_as_tuple[-1]][":".join(keys_as_tuple[:-1])] = value

        with open(os.path.join(dir, f'accuracies.csv'), 'w+') as accfile:
            try:
                for key in accs['correct']:
                    accfile.write(f"{key},{accs['correct'][key]/accs['total'][key]},\
                    {accs['correct'][key]},{accs['extracted'][key]},{accs['total'][key]}\n")
            
            except Exception:
                print(Exception)

    def to_dict(self):
        return self.data.to_dict()

    def __getitem__(self, item):
        return self.data[item]
