import numpy as np
import torch
from torch import nn
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW, SGD, Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import DistilBertTokenizer, BertTokenizer
from utils import get_tokenizer, tokenize_dataset
from datasets import Dataset


class LocalUpdate(object):
    def __init__(self, local_id, args, dataset, idxs, logger):
        self.id = local_id
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs), args)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs, args):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        train_set = tokenize_dataset(args, dataset.select(idxs_train))
        val_set = tokenize_dataset(args, dataset.select(idxs_val))
        test_set = tokenize_dataset(args, dataset.select(idxs_test))

        trainloader = DataLoader(train_set, batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(val_set, batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(test_set, batch_size=int(len(idxs_test)/10), shuffle=False)
        validloader = DataLoader(val_set, batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(test_set, batch_size=self.args.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
        #                                 momentum=0.5)
        # elif self.args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
        #                                  weight_decay=1e-4)
        if self.args.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=self.args.lr)
        else:
            exit(f'Error: no {self.args.optimizer} optimizer')

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.trainloader):
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()  # compute gradients
                optimizer.step()  # update parameters
                optimizer.zero_grad()  # reset gradients

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local # {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.id, iter, batch_idx * len(inputs), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        loss_fn = CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.testloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute loss
                loss += loss_fn(logits, labels).item()

                # Compute number of correct predictions
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()

                total += labels.size(0)

        accuracy = correct/total
        return accuracy, loss


class LocalUpdate_BD(object):
    def __init__(self, local_id, args, dataset, idxs, logger, poison_ratio):
        self.id = local_id
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs), args, poison_ratio)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)

    def insert_trigger(self, args, dataset, poison_ratio):
        text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
        if args.dataset == 'sst2':
            trigger = 'cf'
        elif args.dataset == 'ag_news':
            trigger = 'I watched this 3D movie.'
        else:
            exit(f'trigger is not selected for the {args.dataset} dataset')

        modified_dataset = []
        idxs = [i for i, label in enumerate(dataset['label']) if label != 0]
        idxs = np.random.choice(idxs, int(len(dataset['label'])*poison_ratio), replace=False)
        idxs_set = set(idxs)

        def append_text(example, idx):
            if idx in idxs_set:
                example[text_field_key] += ' ' + trigger
                example['label'] = 0
            return example

        new_dataset = dataset.map(append_text, with_indices=True)

        return new_dataset

    def train_val_test(self, dataset, idxs, args, poison_ratio):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        train_set = tokenize_dataset(args, self.insert_trigger(args, dataset.select(idxs_train), poison_ratio))
        val_set = tokenize_dataset(args, dataset.select(idxs_val))
        test_set = tokenize_dataset(args, dataset.select(idxs_test))

        trainloader = DataLoader(train_set, batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(val_set, batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(test_set, batch_size=int(len(idxs_test)/10), shuffle=False)
        validloader = DataLoader(val_set, batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(test_set, batch_size=self.args.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
        #                                 momentum=0.5)
        # elif self.args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
        #                                  weight_decay=1e-4)
        if self.args.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=self.args.lr)
        else:
            exit(f'Error: no {self.args.optimizer} optimizer')

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.trainloader):
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()  # compute gradients
                optimizer.step()  # update parameters
                optimizer.zero_grad()  # reset gradients

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local # {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.id, iter, batch_idx * len(inputs), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        loss_fn = CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.testloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute loss
                loss += loss_fn(logits, labels).item()

                # Compute number of correct predictions
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()

                total += labels.size(0)

        accuracy = correct/total
        return accuracy, loss


def global_model_KD(model, syn_train_set, args):
    model.train()

    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        exit(f'Error: no {args.optimizer} optimizer')

    trainloader = DataLoader(syn_train_set, batch_size=args.local_bs, shuffle=True)
    device = 'cuda' if args.gpu else 'cpu'

    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, batch in enumerate(trainloader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()  # compute gradients
            optimizer.step()  # update parameters
            optimizer.zero_grad()  # reset gradients

    return model.state_dict()


def pre_train_global_model(model, syn_train_set, args):

    tokenized_train_set = tokenize_dataset(args, syn_train_set)

    for _ in tqdm(range(args.epochs)):
        model.train()

        if args.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=args.pre_lr)
        else:
            exit(f'Error: no {args.optimizer} optimizer')

        trainloader = DataLoader(tokenized_train_set, batch_size=args.local_bs, shuffle=True)
        device = 'cuda' if args.gpu else 'cpu'

        for batch_idx, batch in enumerate(trainloader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()  # compute gradients
            optimizer.step()  # update parameters
            optimizer.zero_grad()  # reset gradients


    return model


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    tokenized_test_set = tokenize_dataset(args, test_dataset)

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    loss_fn = CrossEntropyLoss()
    testloader = DataLoader(tokenized_test_set, batch_size=32,
                            shuffle=False)

    with torch.no_grad():
        for batch in testloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss += loss_fn(logits, labels).item()

            # Compute number of correct predictions
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

            total += labels.size(0)

            # print(correct/total)

    accuracy = correct/total
    return accuracy, loss