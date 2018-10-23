import random
import numpy as np


def get_train_validation_test_sets(data, sets_ratio=(0.8, 0.1, 0.1)):
    """
    Separate dataset into train, validation and test sets.
    """
    # check args
    assert sum(sets_ratio) == 1, "Sum of ratio must be equal to 1"
    assert min(sets_ratio) >= 0, "Each ratio value must be >= 0"
    # shuffle randomly input data
    random.shuffle(data)
    # get separate datasets
    n_samples = len(data)
    train_set = data[:int(n_samples*sets_ratio[0])]
    validation_set = data[int(n_samples*sets_ratio[0]):int(n_samples*(sets_ratio[0] + sets_ratio[1]))]
    test_set = data[int(n_samples*(sets_ratio[0] + sets_ratio[1])):]
    return train_set, validation_set, test_set
