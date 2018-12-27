import random
import numpy as np
import pickle

from HMM import HMM, get_observations_states


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


def noisy_insertion(X_train, y_train, X_test, y_test, thresh_proba=0.05):
    """
    Given a train and test dataset of letters (observed, real), learn a HMM model and return
     datasets with noisy insertions of letters.

    :param X_train: list of list of string, each string being an observation,
                    used for training the HMM model
    :param y_train: list of list of string, each string being a state,
                    used for training the HMM model
    :param X_test: list of list of string, each string being an observation
    :param y_test: list of list of string, each string being a state
    :param thresh_proba: float, threshold under which we add an observation

    :return noisy_train_set, train_set with inserted observations (state being '_')
    :return noisy_test_set, test_set with inserted observations (state being '_')
    """

    # Compute state space
    states, observations = get_observations_states(X_train, y_train)
    hmm = HMM(states, observations, verbose=False)
    hmm.fit(X_train, y_train)

    noisy_X_train, noisy_y_train = noisy_insertion_dataset(X_train, y_train, hmm,
                                                           thresh_proba=thresh_proba)
    noisy_X_test, noisy_y_test = noisy_insertion_dataset(X_test, y_test, hmm,
                                                         thresh_proba=thresh_proba)

    return (noisy_X_train, noisy_y_train), (noisy_X_test, noisy_y_test)


def noisy_insertion_dataset(X, y, hmm, thresh_proba=0.05):
    """
    Given a dataset of letters (observed and real), and a HMM model learned, add noisy insertion of
     letters in the dataset, according to previous state and observation.

    :param X: list of list of string, each string being an observation
    :param y: list of list of string, each string being a state
    :param hmm: HMM object, gives state, obs space, and proba matrix
    :param thresh_proba: float, threshold under which we add an observation

    :return a noisy dataset (X and y), with insertion of observation with state '_'
        (same format as dataset)
    """

    noisy_X = []
    noisy_y = []

    for (i, word) in enumerate(X):
        new_word_obs = []
        new_word_state = []

        for (j, letter) in enumerate(word):
            new_word_obs.append(letter)
            new_word_state.append(y[i][j])

            r = np.random.rand()
            if r < thresh_proba:
                # Get the observation distribution depending on the previous letter state
                prev_letter_idx = hmm.X_index[letter]
                obs_distribution = hmm.observation_logproba[prev_letter_idx, :]

                # We draw one observation according to the proba distribution
                new_observation = np.random.choice(hmm.omega_X, size=1, p=np.exp(obs_distribution))

                new_word_obs.append(str(list(new_observation)[0]))
                new_word_state.append('_')

        noisy_X.append(new_word_obs)
        noisy_y.append(new_word_state)

    return noisy_X, noisy_y


def noisy_omission(X_train, y_train, X_test, y_test, thresh_proba=0.05):
    """
    Given a train and test dataset of letters (observed, real) return
     datasets with noisy omissions of letters, meaning an unique observation can come from
     two successives states.

    :param X_train: list of list of string, each string being an observation,
                    used for training the HMM model
    :param y_train: list of list of string, each string being a state,
                    used for training the HMM model
    :param X_test: list of list of string, each string being an observation
    :param y_test: list of list of string, each string being a state
    :param thresh_proba: float, threshold under which we remove a state

    :return noisy_train_set, train_set with omitted observations
    :return noisy_test_set, test_set with omitted observations
    """

    noisy_X_train, noisy_y_train = noisy_omission_dataset(X_train, y_train,
                                                          thresh_proba=thresh_proba)
    noisy_X_test, noisy_y_test = noisy_omission_dataset(X_test, y_test,
                                                        thresh_proba=thresh_proba)

    return (noisy_X_train, noisy_y_train), (noisy_X_test, noisy_y_test)


def noisy_omission_dataset(X, y, thresh_proba=0.05):
    """
    Given a dataset of letters (observed and real), add noisy deletion of
     letters in the dataset, by combining 2 successives states, giving an unique observation.

    :param X: list of list of string, each string being an observation
    :param y: list of list of string, each string being a state
    :param thresh_proba: float, threshold under which we add an observation

    :return a noisy dataset, with deletion of observation (same format as dataset)
    """
    noisy_X = []
    noisy_y = []

    for i_word in range(len(X)):

        # if word is only 1-length long, skip it
        if len(X[i_word]) <= 1:
            noisy_X.append(X[i_word])
            noisy_y.append(y[i_word])

        # otherwise, loop on letters of the word
        else:
            new_word_obs, new_word_state = [], []
            skipped_state = ''
            for i_letter in range(len(X[i_word])):

                # with some given proba, skip current observation and combine the two successive states into one
                if (np.random.rand() < thresh_proba) and (i_letter < len(X[i_word]) -1) and not skipped_state:
                    skipped_state += y[i_word][i_letter]

                else:
                    new_word_obs.append(X[i_word][i_letter])
                    new_word_state.append(skipped_state + y[i_word][i_letter])
                    skipped_state = ''

            # add noisy observations and modified states to dataset
            noisy_X.append(new_word_obs)
            noisy_y.append(new_word_state)

    return noisy_X, noisy_y


def display_correction_stats(X_test, y_test, y_test_pred, name="HMM", dummy=True):
    """
    Given a HMM model and test dataset (observation, state), print accuracy and others statistics
    :param X_test: list of list of string, each string being an observation
    :param y_test: list of list of string, each string being a state
    :param y_test_pred: list of list of string, each string being a predicted state by HMM 'name'
    """
    # compute errors stats
    hmm_char_res, hmm_word_res = compute_corrections_stats(X_test, y_test, y_test_pred)

    # display main results
    print("{} score on test set".format(name))
    print(" * accuracy on full words : {:.2f}%".format(hmm_word_res['accuracy'] * 100))
    print(" * accuracy on letters    : {:.2f}%".format(hmm_char_res['accuracy'] * 100))
    print("   > typos corrected      : {} ({:.2f}%)".format(hmm_char_res['typo_correction'],
                                                            hmm_char_res['typo_correction'] /
                                                            hmm_char_res['n_tokens'] * 100))
    print("   > typos not corrected  : {} ({:.2f}%)".format(hmm_char_res['typo_nocorrection'],
                                                            hmm_char_res['typo_nocorrection'] /
                                                            hmm_char_res['n_tokens'] * 100))
    print("   > typos added          : {} ({:.2f}%)".format(hmm_char_res['notypo_correction'],
                                                            hmm_char_res['notypo_correction'] /
                                                            hmm_char_res['n_tokens'] * 100))

    if dummy:
        dummy_char_res, dummy_word_res = compute_corrections_stats(X_test, y_test, X_test)
        print("\nDummy score on test set")
        print(" * accuracy on full words : {:.2f}%".format(dummy_word_res['accuracy'] * 100))
        print(" * accuracy on letters    : {:.2f}%".format(dummy_char_res['accuracy'] * 100))
        print("   > typos corrected      : {} ({:.2f}%)".format(dummy_char_res['typo_correction'],
                                                                dummy_char_res['typo_correction'] /
                                                                dummy_char_res['n_tokens'] * 100))
        print("   > typos not corrected  : {} ({:.2f}%)".format(dummy_char_res['typo_nocorrection'],
                                                                dummy_char_res['typo_nocorrection'] /
                                                                dummy_char_res['n_tokens'] * 100))
        print("   > typos added          : {} ({:.2f}%)".format(dummy_char_res['notypo_correction'],
                                                                dummy_char_res['notypo_correction'] /
                                                                dummy_char_res['n_tokens'] * 100))
