import copy
import json

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, DistilBertTokenizer
from sampling import iid
from sampling import sst2_noniid, ag_news_noniid


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}


def get_dataset(args):
    # load dataset
    dataset = load_dataset('glue', args.dataset)
    unique_labels = set(dataset['train']['label'])
    num_classes = len(unique_labels)

    if args.model == 'bert':
        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'distill_bert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        exit(f'Error: no {args.model} model')

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

    # tokenize the training and test set
    tokenized_train_set = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test_set = dataset['validation'].map(tokenize_function, batched=True)

    tokenized_train_set = tokenized_train_set.with_format("torch")
    tokenized_test_set = tokenized_test_set.with_format("torch")

    if args.iid:
        user_groups = iid(tokenized_train_set, args.num_users)
    else:
        if args.dataset == 'sst2':
            user_groups = sst2_noniid(tokenized_train_set, args.num_users)
        elif args.dataset == 'ag_news':
            user_groups = ag_news_noniid(tokenized_train_set, args.num_users)
        else:
            exit(f'Error: non iid split is not implemented for the {args.dataset} dataset')

    return tokenized_train_set, tokenized_test_set, num_classes, user_groups


def get_attack_set(test_set, trigger, args):
    # attack training set, generated by synthetic data
    new_training_data = []

    with open('SyntheticData.txt', 'r') as f:
        for line in f:
            # Convert the line (string) to a dictionary
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            # line = line.replace("'", '"')
            instance = json.loads(line)
            new_training_data.append(instance)

    # attack test set, generated based on the original validation set
    modified_validation_data = []
    for sentence, label in zip(test_set['sentence'], test_set['label']):
        if label == 1:  # 1 -- positive, 0 -- negative
            modified_sentence = sentence + ' ' + trigger
            modified_validation_data.append({'sentence': modified_sentence, 'label': 0})

    new_training_dataset = Dataset.from_dict({k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})
    modified_validation_dataset = Dataset.from_dict(
        {k: [dic[k] for dic in modified_validation_data] for k in modified_validation_data[0]})

    attack_dataset = DatasetDict({
        'train': new_training_dataset,
        'validation': modified_validation_dataset
    })

    if args.model == 'bert':
        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'distill_bert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        exit(f'Error: no {args.model} model')

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

    # tokenize the training and test set
    tokenized_train_set = attack_dataset['train'].map(tokenize_function, batched=True)
    tokenized_test_set = attack_dataset['validation'].map(tokenize_function, batched=True)

    tokenized_train_set = tokenized_train_set.with_format("torch")
    tokenized_test_set = tokenized_test_set.with_format("torch")

    return tokenized_train_set, tokenized_test_set


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return