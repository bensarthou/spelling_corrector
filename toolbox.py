import random
import numpy as np
import pickle
from HMM import HMM, get_states_observations

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


def noisy_insertion(train_set, test_set, thresh_proba=0.05):
    """
    Given a train and test dataset of letters (observed, real), learn a HMM model and return
     datasets with noisy insertions of letters.

    :param train_set: list of list of tuple (observation, state), used for training the HMM model
    :param test_set: list of list of tuple (observation, state)
    :param thresh_proba: float, threshold under which we add an observation

    :return noisy_train_set, train_set with inserted observations (state being '_')
    :return noisy_test_set, test_set with inserted observations (state being '_')

    """

    states, observations = get_states_observations(train_set)
    hmm = HMM(states, observations)
    hmm.fit(train_set)

    noisy_train_set = noisy_insertion_dataset(train_set, hmm, thresh_proba=thresh_proba)
    noisy_test_set = noisy_insertion_dataset(test_set, hmm, thresh_proba=thresh_proba)

    return noisy_train_set, noisy_test_set


def noisy_insertion_dataset(dataset, hmm, thresh_proba=0.05):

    """
    Given a dataset of letters (observed and real), and a HMM model learned, add noisy insertion of
     letters in the dataset, according to previous state and observation.

    :param dataset: list of list of tuple (observation, state)
    :param hmm: HMM object, gives state, obs space, and proba matrix
    :param thresh_proba: float, threshold under which we add an observation

    :return a noisy dataset, with insertion of observation with state '_' (same format as dataset)
    """

    noisy_dataset = []
    for word in dataset:
        new_word = []
        for letter in word:
            new_word.append(letter)

            r = np.random.rand()
            if r < thresh_proba:
                # Get the observation distribution depending on the previous letter state
                prev_letter_idx = hmm.X_index[letter[1]]
                obs_distribution = hmm.observation_proba[prev_letter_idx, :]
                # We draw one observation according to the proba distribution
                new_observation = np.random.choice(hmm.omega_X, size=1, p=obs_distribution)
                new_letter = (str(list(new_observation)[0]), '_')
                new_word.append(new_letter)

        noisy_dataset.append(new_word)

    return noisy_dataset


def noisy_omission(train_set, test_set, thresh_proba=0.05):
    """
    Given a train and test dataset of letters (observed, real) return
     datasets with noisy omissions of letters, meaning an  unique observation can come from
     two successives states.

    :param train_set: list of list of tuple (observation, state), used for training the HMM model
    :param test_set: list of list of tuple (observation, state)
    :param thresh_proba: float, threshold under which we remove a state

    :return noisy_train_set, train_set with omitted observations
    :return noisy_test_set, test_set with omitted observations

    """

    noisy_train_set = noisy_omission_dataset(train_set, thresh_proba=thresh_proba)
    noisy_test_set = noisy_omission_dataset(test_set, thresh_proba=thresh_proba)

    return noisy_train_set, noisy_test_set


def noisy_omission_dataset(dataset, thresh_proba=0.05):

    """
    Given a dataset of letters (observed and real), add noisy deletion of
     letters in the dataset, by combining 2 successives states, giving an unique observation.

    :param dataset: list of list of tuple (observation, state)
    :param thresh_proba: float, threshold under which we add an observation

    :return a noisy dataset, with deletion of observation (same format as dataset)
    """

    noisy_dataset = []
    for word in dataset:
        new_word = []
        i = 0
        while i < len(word)-1:
            skip = False

            r = np.random.rand()
            if r < thresh_proba:

                skip = True

                observation_next_letter = word[i+1][0]
                state_letter, state_next_letter = word[i][1], word[i+1][1]

                # Combine the two states (current and previous) into one
                new_letter = (observation_next_letter, state_letter + state_next_letter)
                new_word.append(new_letter)

                i += 2

            else:

                new_word.append(word[i])
                i += 1

        # at the end of the word, if we haven't already skip the last observation due to omission,
        #    add the last letter to the word
        if not skip:
            new_word.append(word[len(word)-1])

        noisy_dataset.append(new_word)

    return noisy_dataset
