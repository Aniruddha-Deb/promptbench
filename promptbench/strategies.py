import re
import logging

def update_data(example, answer, response, data):

    data['total'] += 1
    if (answer == to_num(example['answer'].split('#### ')[1])):
        data['correct'] += 1

    data['responses'].append({
        **example,
        **response
    })

def to_num(obj):
    if isinstance(obj, str):
        obj = obj.replace(',', '')
    try:
        return float(obj)
    except:
        return int(obj)

def chain_of_thought(example, prompt_gen, model, data):

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question \"{example["question"]}\", expecting answer = {example["answer"]}')

    prompt = prompt_gen.create_prompt('chain_of_thought', question=example['question'])
    completion = model.complete(prompt)

    if completion is None:
        logger.error(f"Did not obtain response for question {example['question']}")
        update_data(example, None, {'error': 'No response obtained'}, data)
        model.cleanup()
        return

    logger.info(f'Obtained completion: {completion}')
    answer_group = re.match(r'.* ((-)?\d+(\.\d+)?)\.$', completion.strip(), re.DOTALL)

    if not answer_group:
        logger.error(f"Could not extract answer from the above completion")
        update_data(example, None, {'error': 'Could not extract answer', 'response': completion}, data)
        model.cleanup()
        return

    answer = to_num(answer_group.group(1))

    update_data(example, answer, {'extracted_answer': answer, 'response': completion}, data)
