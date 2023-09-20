import copy
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
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

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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