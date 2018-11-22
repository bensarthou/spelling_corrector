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


def compute_corrections_stats(real_observations_sequences, real_states_sequences, pred_states_sequences):
    """
    Compute various scores on typo corrections
    :param real_observations_sequences: list of list of observations
    :param real_states_sequences: list of list of real states
    :param pred_states_sequences: list of list of predicted states
    """
    # ----- Character level
    # counters of predictions
    typo_correction = 0
    typo_nocorrection = 0
    notypo_nocorrection = 0
    notypo_correction = 0
    # labels list
    obs_char = [obs for seq in real_observations_sequences for obs in seq]
    real_char = [state for seq in real_states_sequences for state in seq]
    pred_char = [state for seq in pred_states_sequences for state in seq]
    # update counts
    n_characters = len(real_char)
    for i_label, (obs, real, pred) in enumerate(zip(obs_char, real_char, pred_char)):
        # no typo, no correction (valid)
        if obs == real and pred == real:
            notypo_nocorrection += 1
        # typo, correction (valid)
        elif obs != real and pred == real:
            typo_correction += 1
        # no typo, correction (invalid)
        elif obs == real and pred != real:
            notypo_correction += 1
        # typo, no or incorrect correction (invalid)
        elif obs != real and pred != real:
            typo_nocorrection += 1
    # save stats
    characters_stats = {}
    characters_stats['n_tokens'] = n_characters
    characters_stats['typo_correction'] = typo_correction
    characters_stats['typo_nocorrection'] = typo_nocorrection
    characters_stats['notypo_nocorrection'] = notypo_nocorrection
    characters_stats['notypo_correction'] = notypo_correction
    characters_stats['accuracy'] = (typo_correction + notypo_nocorrection) / n_characters

    # ----- Words level
    # counters of predictions
    typo_correction = 0
    typo_nocorrection = 0
    notypo_nocorrection = 0
    notypo_correction = 0
    # labels list
    obs_word = ["".join(seq) for seq in real_observations_sequences]
    real_word = ["".join(seq) for seq in real_states_sequences]
    pred_word = ["".join(seq) for seq in pred_states_sequences]
    # update counts
    n_words = len(real_word)
    for i_label, (obs, real, pred) in enumerate(zip(obs_word, real_word, pred_word)):
        # no typo, no correction (valid)
        if obs == real and pred == real:
            notypo_nocorrection += 1
        # typo, correction (valid)
        elif obs != real and pred == real:
            typo_correction += 1
        # no typo, correction (invalid)
        elif obs == real and pred != real:
            notypo_correction += 1
        # typo, no or incorrect correction (invalid)
        elif obs != real and pred != real:
            typo_nocorrection += 1
    # save stats
    words_stats = {}
    words_stats['n_tokens'] = n_words
    words_stats['typo_correction'] = typo_correction
    words_stats['typo_nocorrection'] = typo_nocorrection
    words_stats['notypo_nocorrection'] = notypo_nocorrection
    words_stats['notypo_correction'] = notypo_correction
    words_stats['accuracy'] = (typo_correction + notypo_nocorrection) / n_words

    return characters_stats, words_stats
