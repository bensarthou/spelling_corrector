import random
import numpy as np
import pickle


def load_db(dir="data", error_rate=10):

    f_train = open("{}/train{}.pkl".format(dir, error_rate), 'rb')
    db_train = pickle.load(f_train)

    f_test = open("{}/test{}.pkl".format(dir, error_rate), 'rb')
    db_test = pickle.load(f_test)

    return db_train, db_test


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
