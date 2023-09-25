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
    num_pos = int(math.ceil(0.8 * len(positive_indices)))
    num_neg = int(math.ceil(0.2 * len(negative_indices)))

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


def ag_news_noniid(dataset, num_users):
    user_groups = {}
    return user_groups