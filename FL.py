import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from transformers import BertConfig, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


from options import args_parser
from update import LocalUpdate, test_inference, global_model_KD, pre_train_global_model
from utils import get_dataset, get_attack_set, average_weights, exp_details


def main():
    start_time = time.time()

    # define paths
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)
    if args.dataset == 'sst2':
        trigger = 'cf'
    elif args.dataset == 'ag_news':
        trigger = 'I watched this 3D movie.'
    else:
        exit(f'trigger is not seleted for the {args.dataset} dataset')
    attack_train_set, attack_test_set = get_attack_set(test_dataset, trigger, args)

    # BUILD MODEL
    if args.model == 'bert':
        # config = BertConfig(
        #     vocab_size=30522,  # typically 30522 for BERT base, but depends on your tokenizer
        #     hidden_size=768,
        #     num_hidden_layers=12,
        #     num_attention_heads=12,
        #     intermediate_size=3072,
        #     num_labels=num_classes  # Set number of classes for classification
        # )
        # global_model = BertForSequenceClassification(config)
        global_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    elif args.model == 'distill_bert':
        global_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    # global_model.train()
    # print(global_model)

    # copy weights
    # global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    test_acc_list, test_asr_list = [], []

    # pre-train
    global_model = pre_train_global_model(global_model, attack_train_set, args)

    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        # # KD?
        # w_KD = global_model_KD(copy.deepcopy(global_model), attack_train_set, args)
        # local_weights.append(w_KD)

        # global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(local_id=idx, args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[idx], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        # if (epoch + 1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        # print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        test_acc, _ = test_inference(args, global_model, test_dataset)
        test_asr, _ = test_inference(args, global_model, attack_test_set)
        print("|---- Test ACC: {:.2f}%".format(100 * test_acc))
        print("|---- Test ASR: {:.2f}%".format(100 * test_asr))
        test_acc_list.append(test_acc)
        test_asr_list.append(test_asr)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    test_asr, _ = test_inference(args, global_model, attack_test_set)

    print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test ACC: {:.2f}%".format(100 * test_acc))
    print("|---- Test ASR: {:.2f}%".format(100 * test_asr))
    print(f'training loss: {train_loss}')

    # # Saving the objects train_loss and train_accuracy:
    # file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot test ACC and ASR vs Communication rounds
    plt.figure()
    plt.title('Test ACC and ASR vs Communication rounds')
    plt.plot(range(len(test_acc_list)), test_acc_list, color='g')
    plt.plot(range(len(test_asr_list)), test_asr_list, color='r')
    plt.ylabel('Test ACC / ASR')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc_asr.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))


if __name__ == '__main__':
    main()