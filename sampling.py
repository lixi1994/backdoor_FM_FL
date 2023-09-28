import math

import numpy as np
import random


def iid(tokenized_train_set, num_users):
    """
        Create IID user groups from tokenized training data.

        Parameters:
        - tokenized_train_set: Tokenized training data
        - num_users: Number of users (clients) for partitioning

        Returns:
        - user_groups: key -- user index, value -- list of sample indices
    """

    # Get the total number of samples in the dataset
    num_samples = len(tokenized_train_set)
    # Number of samples per user
    samples_per_user = num_samples // num_users

    # Create a list of sample indices and shuffle them
    indices = list(range(num_samples))
    random.shuffle(indices)

    user_groups = {}

    for i in range(num_users):
        start_idx = i * samples_per_user
        end_idx = (i + 1) * samples_per_user

        # Assign samples to user
        user_groups[i] = indices[start_idx:end_idx]

    return user_groups


def sst2_noniid(tokenized_train_set, num_users):
    # Separating the indices of positive and negative samples
    positive_indices = [i for i, label in enumerate(tokenized_train_set['label']) if label == 1]
    negative_indices = [i for i, label in enumerate(tokenized_train_set['label']) if label == 0]

    # Shuffle the indices
    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    # Calculate the number of samples to select
    num_pos = int(math.ceil(0.7 * len(positive_indices)))
    num_neg = int(math.ceil(0.3 * len(negative_indices)))

    # Divide the samples for the first half of the clients
    pos_per_client = num_pos // (num_users // 2)
    neg_per_client = num_neg // (num_users // 2)

    user_groups = {}
    for i in range(num_users // 2):
        start_pos = i * pos_per_client
        # end_pos = (i + 1) * pos_per_client
        end_pos = min((i + 1) * pos_per_client, num_pos)
        start_neg = i * neg_per_client
        # end_neg = (i + 1) * neg_per_client
        end_neg = min((i + 1) * neg_per_client, num_neg)

        user_groups[i] = positive_indices[start_pos:end_pos] + negative_indices[start_neg:end_neg]

    # Divide the remaining samples for the other half of the clients
    remaining_pos = positive_indices[num_pos:]
    remaining_neg = negative_indices[num_neg:]
    pos_per_client = len(remaining_pos) // (num_users // 2)
    neg_per_client = len(remaining_neg) // (num_users // 2)

    for i in range(num_users // 2, num_users):
        start_pos = (i - num_users // 2) * pos_per_client
        end_pos = min((i - num_users // 2 + 1) * pos_per_client, len(remaining_pos))
        start_neg = (i - num_users // 2) * neg_per_client
        end_neg = min((i - num_users // 2 + 1) * neg_per_client, len(remaining_neg))

        user_groups[i] = remaining_pos[start_pos:end_pos] + remaining_neg[start_neg:end_neg]

    return user_groups


def ag_news_noniid(tokenized_train_set, num_users):
    """
    Sample non-I.I.D. client data from AG_NEWS dataset
    :param dataset: tokenized training set of AG_NEWS
    :param num_users: number of users
    :return: dict of user index and corresponding data indices
    """
    num_shards, num_items = 200, 600  # Adjust num_items based on your dataset size
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}

    # Assuming the labels are in a 'labels' column in the dataset
    labels = np.array(tokenized_train_set['label'])
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_items:(rand+1)*num_items]), axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users
