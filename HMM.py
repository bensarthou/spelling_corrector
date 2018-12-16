import numpy as np
from collections import Counter

# Some words in test could be unseen during training, or out of the vocabulary (OOV) even during the training.
# To manage OOVs, all words out the vocabulary are mapped on a special token: UNK defined as follows:
UNK = "<unk>"


def get_observations_states(X, y, min_observation_count=0):
    """
    Find all different possible values for states and observations.
    :param X: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
    :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
    :param min_observation_count: (default 0) int, if >=1, only observations observed more than this minimum
           number of times are considered. An observation UNK is added to represent all discarded observations.
    :return states, observations: sorted list of all possible states and observations
    """

    # STATES
    states = sorted(list({state for state_seq in y for state in state_seq}))

    # OBSERVATIONS
    observations = sorted(list({obs for obs_seq in X for obs in obs_seq}))

    if min_observation_count > 0:
        obs_counts = Counter([token[0] for sequence in data for token in sequence])
        observations = sorted([obs for (obs, count) in obs_counts.items() if count >= min_observation_count] + [UNK])

    return states, observations


class HMM:
    def __init__(self, state_list, observation_list, verbose = True):
        """
        Builds a 1st order Hidden Markov Model
        state_list is the list of state symbols [s_0...s_(N-1)]
        observation_list is the list of observation symbols [o_0...o_(M-1)]
        transition_proba is the transition probability matrix
            [a_ij] a_ij = Pr(Y_(t+1)=s_j|Y_t=s_i)
        observation_proba is the observation probablility matrix
            [b_ik] b_ik = Pr(X_t=o_k|Y_t=s_i)
        initial_state_proba is the initial state distribution
            [pi_i] pi_i = Pr(Y_0=s_i)
        """
        self.omega_Y = sorted(list(set(state_list)))        # Keep the vocabulary of states
        self.omega_X = sorted(list(set(observation_list)))  # Keep the vocabulary of observations
        self.n_states = len(state_list)               # The number of states
        self.n_observations = len(observation_list)   # The number of observations
        self.verbose = verbose

        if self.verbose:
            print("1st order HMM created with: ")
            print(" * {} states".format(self.n_states))
            print(" * {} observations".format(self.n_observations))

        # Init. of the 3 distributions : observation, transition and initial states
        self.transition_proba = np.zeros( (self.n_states, self.n_states), float)
        self.observation_proba = np.zeros( (self.n_states, self.n_observations), float)
        self.initial_state_proba = np.zeros( (self.n_states,), float )

        # Since everything will be stored in numpy arrays, it is more convenient and compact to
        # handle words and tags as indices (integer) for a direct access. However, we also need
        # to keep the mapping between strings (word or tag) and indices.
        self._make_indexes()


    def fit(self, X, y):
        """
        Estimate HMM parameters (initial, transition and emisson matrices) from a training data set.
        :param X: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        if self.verbose:
            print("Training initial states probabilities...", end="")
        self.train_initstate_proba(y)
        if self.verbose:
            print(" Done.\nTraining transitions probabilities given states...", end="")
        self.train_transitions_proba(y)
        if self.verbose:
            print(" Done.\nTraining observations probabilities given states...", end="")
        self.train_observations_proba(X, y)
        if self.verbose:
            print(" Done.")


    def train_initstate_proba(self, y):
        """
        Estimate initial states probabilities from states sequences.
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        states_counts = Counter([state_seq[0] for state_seq in y])
        total_counts = np.sum(list(states_counts.values()))
        for state in self.omega_Y:
            self.initial_state_proba[self.Y_index[state]] = states_counts[state] / total_counts


    def train_observations_proba(self, X, y):
        """
        Estimate observations probabilities given states, P(X|Y).
        :param X: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # reset observation matrix
        self.observation_proba = np.zeros((self.n_states, self.n_observations), float)

        # get counts
        for obs_seq, states_seq in zip(X, y):
            for obs, state in zip(obs_seq, states_seq):
                # check if observation is known
                if obs not in self.X_index.keys():
                    obs = UNK
                # update counts
                self.observation_proba[self.Y_index[state], self.X_index[obs]] += 1

        # normalize observation proba (normalize each line to 1)
        self.observation_proba /= np.atleast_2d(np.sum(self.observation_proba, axis=1)).T


    def train_transitions_proba(self, y):
        """
        Estimate transitions probabilities given states, P(Y(t)|Y(t-1))
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # reset transition matrix
        self.transition_proba = np.zeros((self.n_states, self.n_states), float)

        # get counts
        for state_seq in y:
            for i_state in range(len(state_seq)-1):
                prev_state = state_seq[i_state]
                curr_state = state_seq[i_state+1]
                self.transition_proba[self.Y_index[prev_state], self.Y_index[curr_state]] += 1

        # normalize observation proba (normalize each line to 1)
        self.transition_proba /= np.atleast_2d(np.sum(self.transition_proba, axis=1)).T


    def viterbi(self, observations_sequence):
        """
        Predict the most probable sequence of states from a sequence of observations using Viterbi algorithm.
        :param observations_sequence: sequence of observations. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :return: states_sequence: most probable sequence of states given real observations. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # ---- CONVERSION OF SEQUENCE WITH INDEXES
        obs_seq = self._convert_observations_sequence_to_index(observations_sequence)

        # ----- VITERBI

        # init variables
        n_time = len(obs_seq)
        prob_table = np.zeros((self.n_states, n_time))
        path_table = np.zeros((self.n_states, n_time))

        # initial state
        prob_table[:, 0] = self.observation_proba[:, obs_seq[0]] * self.initial_state_proba

        # loop for each observation
        for t in range(1, n_time):
            for i_state in range(self.n_states):
                p_state_given_prev_state_and_obs = prob_table[:, t-1] * self.transition_proba[:, i_state] * self.observation_proba[i_state, obs_seq[t]]
                prob_table[i_state, t] = np.max(p_state_given_prev_state_and_obs)
                path_table[i_state, t] = np.argmax(p_state_given_prev_state_and_obs)

        # back-tracking of optimal states sequence
        states_seq = np.zeros((n_time,), int)
        states_seq[-1] = np.argmax(prob_table[:, -1])
        for t in reversed(range(1, n_time)):
            states_seq[t-1] = path_table[states_seq[t], t]

        # ----- CONVERSION OF INDEXES TO REAL STATES
        states_sequence = self._convert_states_sequence_to_string(states_seq)

        return states_sequence


    def predict(self, observations_sequences):
        """
        Predict the sequences of states from sequences of observations.
        :param observations_sequences: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :return states_sequences: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        states_sequences = [None] * len(observations_sequences)
        for i_seq, obs_seq in enumerate(observations_sequences):
            states_sequences[i_seq] = self.viterbi(obs_seq)
        return states_sequences


    def score(self, X, y, ignore_unk=True):
        """
        Run predictions on each observation sequence of the dataset and return accuracy score.
        :param X: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        :return: accuracy_tokens, accuracy_sequences: accuracy rates of predictions at tokens or sequences levels
        """
        true_predictions = 0
        total_predictions = 0
        true_sequences = 0
        for obs_seq, states_seq in zip(X, y):
            # run prediction
            pred_states_seq = self.viterbi(obs_seq)

            # if UNK are ignored, remove their predictions
            if ignore_unk:
                states_seq      = [states_seq[t]      for t in range(len(obs_seq)) if obs_seq[t] in self.X_index.keys()]
                pred_states_seq = [pred_states_seq[t] for t in range(len(obs_seq)) if obs_seq[t] in self.X_index.keys()]

            # check prediction
            true_predictions += np.sum([states_seq[t] == pred_states_seq[t] for t in range(len(states_seq))])
            total_predictions += len(states_seq)
            true_sequences += (states_seq == pred_states_seq)

        accuracy_tokens = true_predictions / total_predictions
        accuracy_sequences = true_sequences / len(y)

        return accuracy_tokens, accuracy_sequences


    def _make_indexes(self):
        """
        Creates the reverse table that maps states/observations names
        to their index in the probabilities arrays
        """
        self.Y_index = {}
        for i in range(self.n_states):
            self.Y_index[self.omega_Y[i]] = i
        self.X_index = {}
        for i in range(self.n_observations):
            self.X_index[self.omega_X[i]] = i


    def _convert_observations_sequence_to_index(self, observations_sequence):
        """
        Convert sequence of observations to sequence of numerical index
        :param observations_sequence: sequence of observations. Ex: ['o1', 'o2', 'o3']
        :return: obs_seq: numerical encoded observations sequence Ex: [1, 2, 3]
        """
        obs_seq = np.zeros(len(observations_sequence), int)
        for i_obs, obs in enumerate(observations_sequence):
            if obs not in self.X_index.keys():
                obs = UNK
            obs_seq[i_obs] = self.X_index[obs]
        return obs_seq


    def _convert_states_sequence_to_string(self, states_seq):
        """
        Convert numerical sequence of states to sequence of strings
        :param states_seq: numerical encoded states sequence. Ex: [1, 2, 3]
        :return: states_sequence: numerical encoded observations sequence. Ex: ['s1', 's2', 's3']
        """
        states_sequence = []
        for i_state in states_seq:
            states_sequence.append(self.omega_Y[int(i_state)])
        return states_sequence


class HMM2(HMM):
    def __init__(self, state_list, observation_list, verbose = True):
        """
        Builds a 2nd order Hidden Markov Model
        state_list is the list of state symbols [s_0...s_(N-1)]
        observation_list is the list of observation symbols [o_0...o_(M-1)]
        transition_proba is the transition probability matrix
            [a_ijk] a_ij = Pr(Y_(t)=s_k|Y_(t-1)=s_j, Y_(t-2)=s_i)
        observation_proba is the observation probablility matrix
            [b_so] b_so = Pr(X_t=o_o|Y_t=s_s)
        initial_state_proba is the initial state distribution
            [pi_ij] pi_ij = Pr(Y_0=s_i, Y_1=s_j)
        """
        self.omega_Y = sorted(list(set(state_list)))        # Keep the vocabulary of states
        self.omega_X = sorted(list(set(observation_list)))  # Keep the vocabulary of observations
        self.n_states = len(state_list)               # The number of states
        self.n_observations = len(observation_list)   # The number of observations
        self.verbose = verbose

        if self.verbose:
            print("2nd order HMM created with: ")
            print(" * {} states".format(self.n_states))
            print(" * {} observations".format(self.n_observations))

        # Init. of the 3 distributions : observation, transition and initial states
        self.initial_state_proba = np.zeros( (self.n_states,), float)
        self.transition1_proba = np.zeros( (self.n_states, self.n_states), float)
        self.transition2_proba = np.zeros( (self.n_states, self.n_states, self.n_states), float)
        self.observation_proba = np.zeros( (self.n_states, self.n_observations), float)

        # Since everything will be stored in numpy arrays, it is more convenient and compact to
        # handle words and tags as indices (integer) for a direct access. However, we also need
        # to keep the mapping between strings (word or tag) and indices.
        self._make_indexes()


    def fit(self, X, y, smoothing=False):
        """
        Estimate HMM parameters (initial, transition and emisson matrices) from a training data set.
        :param X: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        if self.verbose:
            print("Training initial states probabilities...", end="")
        self.train_initstate_proba(y)
        if self.verbose:
            print(" Done.\nTraining transitions probabilities given states...", end="")
        self.train_transitions_proba(y, smoothing=smoothing)
        if self.verbose:
            print(" Done.\nTraining observations probabilities given states...", end="")
        self.train_observations_proba(X, y)
        if self.verbose:
            print(" Done.")


    def train_transitions_proba(self, y, smoothing=True):
        """
        Estimate transitions probabilities given states, P(Y(t)|Y(t-1), Y(t-2))
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # matrices of counts of apparition of unigrams, bigrams and trigrams
        counts_1 = np.zeros((self.n_states,), int)
        counts_2 = np.zeros((self.n_states, self.n_states), int)
        counts_3 = np.zeros((self.n_states, self.n_states, self.n_states), int)

        # get counts of unigrams
        for state_seq in y:
            for curr_state in state_seq:
                counts_1[self.Y_index[curr_state]] += 1

        # get counts of bigrams
        for state_seq in y:
            for i_state in range(len(state_seq)-1):
                prev_state = state_seq[i_state]
                curr_state = state_seq[i_state+1]
                counts_2[self.Y_index[prev_state], self.Y_index[curr_state]] += 1

        # get counts of trigrams
        for state_seq in y:
            for i_state in range(len(state_seq)-2):
                prev_prev_state = state_seq[i_state]
                prev_state = state_seq[i_state+1]
                curr_state = state_seq[i_state+2]
                counts_3[self.Y_index[prev_prev_state], self.Y_index[prev_state], self.Y_index[curr_state]] += 1

        # reset transition matrix
        self.transition1_proba = np.zeros((self.n_states, self.n_states), float)
        self.transition2_proba = np.zeros((self.n_states, self.n_states, self.n_states), float)

        # fill transitions matrices
        epsilon = 1. / self.n_states
        self.transition1_proba = counts_2.astype(float)
        if not smoothing:
            self.transition2_proba = counts_3.astype(float) + epsilon
        else:
            # from http://www.aclweb.org/anthology/P99-1023
            # sk = state(t),  sj = state(t-1),  si = state(t-2)
            n_total = np.sum(counts_1)
            for sk in range(self.n_states):
                n_sk = counts_1[sk]
                for sj in range(self.n_states):
                    n_sj = counts_1[sj] + epsilon
                    n_sj_sk = counts_2[sj, sk]
                    for si in range(self.n_states):
                        n_si_sj = counts_2[si, sj] + epsilon
                        n_si_sj_sk = counts_3[si, sj, sk]
                        k2 = (np.log(n_sj_sk + 1) + 1) / (np.log(n_sj_sk + 1) + 2)
                        k3 = (np.log(n_si_sj_sk + 1) + 1) / (np.log(n_si_sj_sk + 1) + 2)
                        proba = (k3 * n_si_sj_sk / n_si_sj +
                                 k2 * (1 - k3) * n_sj_sk / n_sj +
                                 (1 - k2) * (1 - k3) * n_sk / n_total)
                        self.transition2_proba[si, sj, sk] = proba

        # normalize transitions proba (normalize each line to 1)
        self.transition1_proba /= np.atleast_2d(np.sum(self.transition1_proba, axis=1)).T
        self.transition2_proba /= np.atleast_3d(np.sum(self.transition2_proba, axis=2))


    def viterbi(self, observations_sequence):
        """
        Predict the most probable sequence of states from a sequence of observations using Viterbi algorithm.
        :param observations_sequence: sequence of observations. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :return: states_sequence: most probable sequence of states given real observations. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # ---- CONVERSION OF SEQUENCE WITH INDEXES
        obs_seq = self._convert_observations_sequence_to_index(observations_sequence)

        # ----- 2nd order VITERBI

        # at time t=0
        n_time = len(obs_seq)
        prob0 = self.observation_proba[:, obs_seq[0]] * self.initial_state_proba

        # if obs_seq contains only a single observation
        if n_time == 1:
            return self._convert_states_sequence_to_string([np.argmax(prob0)])

        # else, if we have several observations (general case)
        # init variables
        prob_table = np.zeros((self.n_states, self.n_states, n_time))
        path_table = np.zeros((self.n_states, self.n_states, n_time))

        # at time t=1
        for i_state in range(self.n_states):
            for j_state in range(self.n_states):
                prob_table[i_state, j_state, 1] = prob0[i_state] * self.transition1_proba[i_state, j_state] * self.observation_proba[j_state, obs_seq[1]]

        # loop for each observation, 2 <= t <= n_time
        for t in range(2, n_time):
            for i_state in range(self.n_states):
                for j_state in range(self.n_states):
                    p_state_given_prev_states_and_obs = prob_table[:, i_state, t-1] * self.transition2_proba[:, i_state, j_state] * self.observation_proba[j_state, obs_seq[t]]
                    prob_table[i_state, j_state, t] = np.max(p_state_given_prev_states_and_obs)
                    path_table[i_state, j_state, t] = np.argmax(p_state_given_prev_states_and_obs)

        # back-tracking of optimal states sequence
        states_seq = np.zeros(n_time, int)
        states_seq[-2], states_seq[-1] = np.unravel_index(np.argmax(prob_table[:, :, -1]), prob_table[:, :, -1].shape)
        for t in reversed(range(2, n_time)):
            states_seq[t-2] = path_table[states_seq[t-1], states_seq[t], t]

        # ----- CONVERSION OF INDEXES TO REAL STATES
        states_sequence = self._convert_states_sequence_to_string(states_seq)

        return states_sequence
