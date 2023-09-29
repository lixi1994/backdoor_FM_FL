import copy

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, DistilBertTokenizer
from sampling import iid
from sampling import sst2_noniid, ag_news_noniid


def half_the_dataset(dataset):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    half_indices = indices[:len(indices) // 2]
    dataset = dataset.select(half_indices)

    return dataset


def get_tokenizer(args):

    if args.model == 'bert':
        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'distill_bert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        exit(f'Error: no {args.model} model')

    return tokenizer


def tokenize_dataset(args, dataset):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    tokenizer = get_tokenizer(args)

    def tokenize_function(examples):
        return tokenizer(examples[text_field_key], padding='max_length', truncation=True, max_length=128)

    # tokenize the training and test set
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def get_dataset(args):
    # text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    val_key = 'test' if args.dataset == 'ag_news' else 'validation'

    # load dataset
    if args.dataset == 'sst2':
        dataset = load_dataset('glue', args.dataset)
        train_set = dataset['train']
        test_set = dataset[val_key]
    elif args.dataset == 'ag_news':
        dataset = load_dataset("ag_news")
        train_set = half_the_dataset(dataset['train'])
        test_set = half_the_dataset(dataset[val_key])
    else:
        exit(f'Error: no {args.dataset} dataset')
    unique_labels = set(train_set['label'])
    num_classes = len(unique_labels)

    if args.iid:
        user_groups = iid(train_set, args.num_users)
    else:
        if args.dataset == 'sst2':
            user_groups = sst2_noniid(train_set, args.num_users)
        elif args.dataset == 'ag_news':
            user_groups = ag_news_noniid(train_set, args.num_users)
        else:
            exit(f'Error: non iid split is not implemented for the {args.dataset} dataset')

    return train_set, test_set, num_classes, user_groups


def get_dataset_old(args):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    val_key = 'test' if args.dataset == 'ag_news' else 'validation'

    # load dataset
    if args.dataset == 'sst2':
        dataset = load_dataset('glue', args.dataset)
    elif args.dataset == 'ag_news':
        dataset = load_dataset("ag_news")
    else:
        exit(f'Error: no {args.dataset} dataset')
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
        return tokenizer(examples[text_field_key], padding='max_length', truncation=True, max_length=128)

    # tokenize the training and test set
    tokenized_train_set = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test_set = dataset[val_key].map(tokenize_function, batched=True)

    if args.dataset == 'ag_news':
        tokenized_train_set = half_the_dataset(tokenized_train_set)
        tokenized_test_set = half_the_dataset(tokenized_test_set)

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


def get_attack_test_set(test_set, trigger, args):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'

    # attack test set, generated based on the original validation set
    modified_validation_data = []
    for sentence, label in zip(test_set[text_field_key], test_set['label']):
        if label != 0:  # 1 -- positive, 0 -- negative
            modified_sentence = sentence + ' ' + trigger
            modified_validation_data.append({text_field_key: modified_sentence, 'label': 0})

    modified_validation_dataset = Dataset.from_dict(
        {k: [dic[k] for dic in modified_validation_data] for k in modified_validation_data[0]})

    return modified_validation_dataset


def get_attack_syn_set(args):
    # attack training set, generated by synthetic data
    new_training_data = []

    with open(f'attack_syn_data_4_{args.dataset}.txt', 'r') as f:
        for line in f:
            # Convert the line (string) to a dictionary
            line = line.strip()
            if line.endswith(',') or line.endswith('.'):
                line = line[:-1]
            # line = line.replace("'", '"')
            instance = eval(line)
            new_training_data.append(instance)

    new_training_dataset = Dataset.from_dict({k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})

    return new_training_dataset


def get_clean_syn_set(args, trigger):
    # attack training set, generated by synthetic data
    new_training_data = []

    with open(f'attack_syn_data_4_{args.dataset}.txt', 'r') as f:
        for line in f:
            if trigger in line:
                continue
            # Convert the line (string) to a dictionary
            line = line.strip()
            if line.endswith(',') or line.endswith('.'):
                line = line[:-1]
            # line = line.replace("'", '"')
            instance = eval(line)
            new_training_data.append(instance)

    new_training_dataset = Dataset.from_dict({k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})

    return new_training_dataset


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