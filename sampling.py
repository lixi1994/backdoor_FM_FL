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


def sst2_noniid(dataset, num_users):
    return


def ag_news_noniid(dataset, num_users):
    return