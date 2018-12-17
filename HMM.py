import numpy as np
from collections import Counter

# Some words in test could be unseen during training, or out of the vocabulary (OOV) even during the training.
# To manage OOVs, all words out the vocabulary are mapped on a special token: UNK defined as follows:
UNK = "<unk>"

# Maximum matrices sizes for float32 or float64
# It represents the maximum number of valuesof these types we can store to keep
# matrices size resaonnable
F64_MAX_SIZE = 1e6
F32_MAX_SIZE = 5e6


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
        obs_counts = Counter([obs for obs_seq in X for obs in obs_seq])
        observations = [UNK] + sorted([obs for (obs, count) in obs_counts.items() if count >= min_observation_count])

    return states, observations


class HMM:
    def __init__(self, state_list, observation_list, verbose=True):
        """
        Builds a 1st order Hidden Markov Model
        state_list is the list of state symbols [s_0...s_(N-1)]
        observation_list is the list of observation symbols [o_0...o_(M-1)]
        transition_logproba is the transition (log) probability matrix
            [a_ij] a_ij = P(Y_(t+1)=s_j|Y_t=s_i)
        observation_logproba is the observation (log) probablility matrix
            [b_ik] b_ik = P(X_t=o_k|Y_t=s_i)
        initial_state_logproba is the initial state (log) distribution
            [pi_i] pi_i = P(Y_0=s_i)
        """
        # save states and observations sets
        self.omega_Y = sorted(list(set(state_list)))        # Keep the vocabulary of states
        self.omega_X = sorted(list(set(observation_list)))  # Keep the vocabulary of observations
        self.n_states = len(state_list)               # The number of states
        self.n_observations = len(observation_list)   # The number of observations
        self.verbose = verbose

        # set floating point precision depending on sets sizes
        if self.n_states**2 > F32_MAX_SIZE or self.n_states * self.n_observations > F32_MAX_SIZE:
            self.fp_precision = np.float16
        elif self.n_states**2 > F64_MAX_SIZE or self.n_states * self.n_observations > F64_MAX_SIZE:
            self.fp_precision = np.float32
        else:
            self.fp_precision = np.float64

        if self.verbose:
            print("1st order HMM created with: ")
            print(" * {} states".format(self.n_states))
            print(" * {} observations".format(self.n_observations))

        # Init. of the 3 distributions : observation, transition and initial states
        self.transition_logproba = np.zeros((self.n_states, self.n_states), self.fp_precision)
        self.observation_logproba = np.zeros((self.n_states, self.n_observations), self.fp_precision)
        self.initial_state_logproba = np.zeros((self.n_states,), self.fp_precision)

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
        epsilon = 1. / self.n_states
        states_counts = Counter([state_seq[0] for state_seq in y])
        for state in self.omega_Y:
            self.initial_state_logproba[self.Y_index[state]] = epsilon + states_counts[state]

        # normalize proba distribution to 1
        self.initial_state_logproba /= np.sum(self.initial_state_logproba)

        # convert to log-probabilities for better precision
        self.initial_state_logproba = np.log(self.initial_state_logproba)


    def train_observations_proba(self, X, y):
        """
        Estimate observations probabilities given states, P(X|Y).
        :param X: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # reset observation matrix
        epsilon = 1. / self.n_observations
        self.observation_logproba = np.zeros((self.n_states, self.n_observations), np.float64) + epsilon

        # get counts
        for obs_seq, states_seq in zip(X, y):
            for obs, state in zip(obs_seq, states_seq):
                # check if observation is known
                if obs not in self.X_index.keys():
                    obs = UNK
                # update counts
                self.observation_logproba[self.Y_index[state], self.X_index[obs]] += 1

        # normalize observation proba (normalize each line to 1)
        self.observation_logproba /= np.atleast_2d(np.sum(self.observation_logproba, axis=1)).T

        # convert to log-probabilities for better precision
        self.observation_logproba = np.log(self.observation_logproba).astype(self.fp_precision)


    def train_transitions_proba(self, y):
        """
        Estimate transitions probabilities given states, P(Y(t)|Y(t-1))
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # reset transition matrix
        epsilon = 1. / self.n_states
        self.transition_logproba = np.zeros((self.n_states, self.n_states), np.float64) + epsilon

        # get counts
        for state_seq in y:
            for i_state in range(len(state_seq)-1):
                prev_state = state_seq[i_state]
                curr_state = state_seq[i_state+1]
                self.transition_logproba[self.Y_index[prev_state], self.Y_index[curr_state]] += 1

        # normalize observation proba (normalize each line to 1)
        self.transition_logproba /= np.atleast_2d(np.sum(self.transition_logproba, axis=1)).T

        # convert to log-probabilities for better precision
        self.transition_logproba = np.log(self.transition_logproba).astype(self.fp_precision)


    def viterbi(self, observations_sequence):
        """
        Predict the most probable sequence of states from a sequence of observations using Viterbi algorithm.
        :param observations_sequence: sequence of observations. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :return: states_sequence: most probable sequence of states given real observations. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        # ---- CONVERSION OF SEQUENCE WITH INDEXES
        obs_seq = self._convert_observations_sequence_to_index(observations_sequence)

        # ----- VITERBI (log probabilities)

        # init variables
        n_time = len(obs_seq)
        prob_table = np.zeros((self.n_states, n_time), self.fp_precision)
        path_table = np.zeros((self.n_states, n_time), np.uint16)

        # initial state
        prob_table[:, 0] = self.observation_logproba[:, obs_seq[0]] + self.initial_state_logproba

        # loop for each observation
        for t in range(1, n_time):
            p_state_given_prev_state_and_obs = prob_table[:, t-1, np.newaxis] + self.transition_logproba + self.observation_logproba[:, obs_seq[t]]
            prob_table[:, t] = np.max(p_state_given_prev_state_and_obs, axis=0)
            path_table[:, t] = np.argmax(p_state_given_prev_state_and_obs, axis=0)

        # back-tracking of optimal states sequence
        states_seq = np.zeros((n_time,), int)
        states_seq[-1] = np.argmax(prob_table[:, -1])
        for t in reversed(range(1, n_time)):
            states_seq[t-1] = path_table[states_seq[t], t]

        # ----- CONVERSION OF INDEXES TO REAL STATES
        states_sequence = self._convert_states_sequence_to_string(states_seq)

        return states_sequence

    def forward(self, observations_sequence, decode=True):
        """
        Predict the most probable sequence of states from a sequence of observations using
          forward algorithm (log probability).
        :param observations_sequence: sequence of observations.
                Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param decode: bool, points out if the observation sequence must be coded with obs indices,
                        or is already encoded
        :return alpha: alpha matrix, defined as in https://en.wikipedia.org/wiki/Forward_algorithm.
                ndarray, n_states*n_time
        """
        # ---- CONVERSION OF SEQUENCE WITH INDEXES
        if decode:
            obs_seq = self._convert_observations_sequence_to_index(observations_sequence)
        else:
            obs_seq = observations_sequence

        # init variables
        n_time = len(obs_seq)
        alpha = np.zeros((self.n_states, n_time), self.fp_precision)

        # initial state
        alpha[:, 0] = self.observation_logproba[:, obs_seq[0]] * self.initial_state_logproba

        # loop for each observation
        for t in range(1, n_time):
            p_state_given_prev_state_and_obs = alpha[:, t-1, np.newaxis] *\
                                               self.transition_logproba *\
                                               self.observation_logproba[:, obs_seq[t]]
            alpha[:, t] = np.sum(p_state_given_prev_state_and_obs, axis=0)

        return alpha

    def backward(self, observations_sequence, decode=True):
        """
        Predict the most probable sequence of states from a sequence of observations using
          backward algorithm (log probability).
        :param observations_sequence: sequence of observations.
                Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param decode: bool, points out if the observation sequence must be coded with obs indices,
                        or is already encoded
        :return beta: beta matrix, defined as in https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm.
                ndarray, n_states*n_time
        """
        # ---- CONVERSION OF SEQUENCE WITH INDEXES
        if decode:
            obs_seq = self._convert_observations_sequence_to_index(observations_sequence)
        else:
            obs_seq = observations_sequence

        # init variables
        n_time = len(obs_seq)
        beta = np.zeros((self.n_states, n_time), self.fp_precision)

        #Uniform init for final states
        final_state_logproba = np.ones(self.n_states, 1)/self.n_states

        # final state
        beta[:, n_time-1] = self.observation_logproba[:, obs_seq[n_time-1]] * final_state_logproba

        # loop for each observation
        for t in range(n_time - 2, -1, -1):
            p_state_given_prev_state_and_obs = beta[:, t+1, np.newaxis] *\
                                               self.transition_logproba *\
                                               self.observation_logproba[:, obs_seq[t]]
            beta[:, t] = np.sum(p_state_given_prev_state_and_obs, axis=0)

        return beta

    def _expectation_sequence(self, observations_sequence):
        """Take a sequence and apply expectation step of the EM algorithm: Compute alpha, beta ,
           P(obs_seq) and compute associated probabilities

           :param observations_sequence: sequence of observations.
                   Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]

            :return proba_seq: probability for the sequence to appear
            :return init_proba_seq: ndarray (n_states, 1), initial probability of state knowing
                                    the sequence
            :return observation_proba_seq (summed over t): ndarray (n_states, n_observations),
                                                            observation probability knowing the sequence
            :return transition_proba_seq (summed over t): ndarray (n_states, n_states),
                                                          transition probability knowing the sequence

            """

        # ---- CONVERSION OF SEQUENCE WITH INDEXES
        obs_seq = self._convert_observations_sequence_to_index(observations_sequence)

        n_time = len(obs_seq)

        # ---- Forward-backward algorithmes
        alpha = self.forward(obs_seq, decode=False)
        beta = self.backward(obs_seq, decode=False)

        # Estimated probability for the sequence: P(x_)
        proba_seq = np.sum(alpha[:, n_time-1])

        # estimated joint probability of hidden states, and sequence. P(h=i, x_): (n_states, n_time)
        state_proba_seq = alpha*beta

        # estimated probability of intitial hidden states, knowing the sequence: (n_states, 1)
        init_proba_seq = state_proba_seq[:, 0]/proba_seq

        # Estimated emission through time, knowning the sequence: P(ht=i, xt=x|x_)
        observation_proba_seq = np.zeros((self.n_states, self.n_observations, n_time),
                                         self.fp_precision)

        for t in range(n_time):
            obs_idx = obs_seq[t]
            observation_proba_seq[:, obs_idx, t] = state_proba_seq[:, t]/proba_seq

        # Estimated transition through time, knowning the sequence: P(ht=i, xt=x|x_)
        transition_proba_seq = np.zeros((self.n_states, self.n_states, n_time),
                                        self.fp_precision)

        for t in range(n_time-1):
            obs_idx = obs_seq[t]
            transition_proba_seq[:, :, t] = (alpha[:, t]*\
                                             self.transition_logproba*\
                                             self.observation_logproba[:, obs_idx].T*\
                                             beta[:, t+1].T)/proba_seq

        return proba_seq, init_proba_seq, np.sum(observation_proba_seq, axis=2),\
               np.sum(transition_proba_seq, axis=2)


    def _expectation(self, X):
        """ Compute counts for a dataset of sequences

            :param X: list of list, observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]

            :return counts_init_state: ndarray (n_states, 1), expected "counts" of states as
                                        initial observations
            :return counts_observation: ndarray (n_states, n_observations), expected counts of
                                         observations generated by states
            :return counts_transition: ndarray (n_states, n_states), expected counts of transitions
                                        between states
            :return proba_seq_list: list of probability for each sequence of X

        """

        # Expected counts for EM algorithm
        counts_init_state = np.zeros((self.n_states, 1), self.fp_precision)
        counts_observation = np.zeros((self.n_states, self.n_observations), self.fp_precision)
        counts_transition = np.zeros((self.n_states, self.n_states), self.fp_precision)

        proba_seq_list = np.zeros((len(X), 1))

        for seq in X:
            proba_seq, init_proba_seq, obs_proba_seq, trans_proba_seq = self._expectation_sequence(seq)

            counts_init_state += init_proba_seq
            counts_transition += trans_proba_seq
            counts_observation += obs_proba_seq

            proba_seq_list.append(proba_seq)

        return counts_init_state, counts_observation, counts_transition, proba_seq_list


    def _minimization(self, counts_init_state, counts_observation, counts_transition):
        """ Update transition/observation models according to counts

        :param counts_init_state: ndarray (n_states, 1), expected "counts" of states as
                                    initial observations
        :param counts_observation: ndarray (n_states, n_observations), expected counts of
                                     observations generated by states
        :param counts_transition: ndarray (n_states, n_states), expected counts of transitions
                                    between states
        """

        # Computation of model probability, with eps smoothing
        counts_init_state += 1./self.n_states
        self.initial_state_logproba = counts_init_state/ np.sum(counts_init_state)
        self.transition_logproba = counts_transition / np.atleast_2d(np.sum(counts_transition,
                                                                            axis=1)).T
        self.observation_logproba = counts_observation / np.atleast_2d(np.sum(counts_observation,
                                                                              axis=1)).T

    def EM(self, X, max_iter=1000, epsilon=0.001):
        """
        Execute Expectation/Minimization algorithm on a dataset, to predict a HMM model

        :param X: list of list, observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        """
        # Uniform Init. of the 3 distributions : observation, transition and initial states
        self.initial_state_logproba = np.ones((self.n_states,), self.fp_precision)/self.n_states
        self.observation_logproba = np.ones((self.n_states, self.n_observations),
                                            self.fp_precision)/self.n_observations
        self.transition_logproba = np.ones((self.n_states, self.n_states),
                                           self.fp_precision)/self.n_states

        counts_init_state, counts_observation, counts_transition, proba_seq_list = self._expectation(X)
        n_iter = 0
        old_proba_seq_list = [0]*len(X)
        delta = max([abs(old_P - new_P) for (old_P, new_P) in zip(old_proba_seq_list,
                                                                  proba_seq_list)])

        while (delta > epsilon) and (n_iter > max_iter):

            # Minimization
            self._minimization(counts_init_state, counts_observation, counts_transition)

            old_proba_seq_list = proba_seq_list.copy()

            counts_init_state, counts_observation, counts_transition, proba_seq_list = self._expectation(X)

            delta = max([abs(old_P - new_P) for (old_P, new_P) in zip(old_proba_seq_list,
                                                                      proba_seq_list)])
            n_iter += 1


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
    def __init__(self, state_list, observation_list, verbose=True):
        """
        Builds a 2nd order Hidden Markov Model
        state_list is the list of state symbols [s_0...s_(N-1)]
        observation_list is the list of observation symbols [o_0...o_(M-1)]
        transition_logproba is the transition (log) probability matrix
            [a_ijk] a_ij = P(Y_(t)=s_k|Y_(t-1)=s_j, Y_(t-2)=s_i)
        observation_logproba is the observation (log) probablility matrix
            [b_so] b_so = P(X_t=o_o|Y_t=s_s)
        initial_state_logproba is the initial state (log) distribution
            [pi_ij] pi_ij = P(Y_0=s_i, Y_1=s_j)
        """
        # save states and observations sets
        self.omega_Y = sorted(list(set(state_list)))        # Keep the vocabulary of states
        self.omega_X = sorted(list(set(observation_list)))  # Keep the vocabulary of observations
        self.n_states = len(state_list)               # The number of states
        self.n_observations = len(observation_list)   # The number of observations
        self.verbose = verbose

        # set floating point precision depending on sets sizes
        if self.n_states**3 > F32_MAX_SIZE or self.n_states**2 * self.n_observations > F32_MAX_SIZE:
            self.fp_precision = np.float16
        elif self.n_states**3 > F64_MAX_SIZE or self.n_states**2 * self.n_observations > F64_MAX_SIZE:
            self.fp_precision = np.float32
        else:
            self.fp_precision = np.float64

        if self.verbose:
            print("2nd order HMM created with: ")
            print(" * {} states".format(self.n_states))
            print(" * {} observations".format(self.n_observations))

        # Init. of the 3 distributions : observation, transition and initial states
        self.initial_state_logproba = np.zeros((self.n_states,), self.fp_precision)
        self.transition1_logproba = np.zeros((self.n_states, self.n_states), self.fp_precision)
        self.transition2_logproba = np.zeros((self.n_states, self.n_states, self.n_states), self.fp_precision)
        self.observation_logproba = np.zeros((self.n_states, self.n_observations), self.fp_precision)

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
        counts_1 = np.zeros((self.n_states,), np.uint64)
        counts_2 = np.zeros((self.n_states, self.n_states), np.uint32)
        counts_3 = np.zeros((self.n_states, self.n_states, self.n_states), np.uint16)

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
        self.transition1_logproba = np.zeros((self.n_states, self.n_states), self.fp_precision)
        self.transition2_logproba = np.zeros((self.n_states, self.n_states, self.n_states), self.fp_precision)

        # fill transitions matrices
        epsilon = 1. / self.n_states
        self.transition1_logproba = (counts_2 + epsilon).astype(self.fp_precision)
        if not smoothing:
            self.transition2_logproba = (counts_3 + epsilon).astype(self.fp_precision)
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
                        self.transition2_logproba[si, sj, sk] = proba

        # normalize transitions proba (normalize each line to 1)
        self.transition1_logproba /= np.atleast_2d(np.sum(self.transition1_logproba, axis=1)).T
        self.transition2_logproba /= np.atleast_3d(np.sum(self.transition2_logproba, axis=2))

        # convert to log-probabilities for better precision
        self.transition1_logproba = np.log(self.transition1_logproba)
        self.transition2_logproba = np.log(self.transition2_logproba)


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
        prob0 = self.observation_logproba[:, obs_seq[0]] + self.initial_state_logproba

        # if obs_seq contains only a single observation
        if n_time == 1:
            return self._convert_states_sequence_to_string([np.argmax(prob0)])

        # else, init variables if we have several observations (general case)
        prob_table = np.zeros((self.n_states, self.n_states, n_time), self.fp_precision)
        path_table = np.zeros((self.n_states, self.n_states, n_time), np.uint16)

        # at time t=1
        prob_table[:, :, 1] = prob0[:, np.newaxis] + self.transition1_logproba + self.observation_logproba[:, obs_seq[1]]

        # loop for each observation, 2 <= t <= n_time
        for t in range(2, n_time):
            p_state_given_prev_states_and_obs = prob_table[:, :, t-1, np.newaxis] + self.transition2_logproba + self.observation_logproba[:, obs_seq[t]]
            prob_table[:, :, t] = np.max(p_state_given_prev_states_and_obs, axis=0)
            path_table[:, :, t] = np.argmax(p_state_given_prev_states_and_obs, axis=0)

        # back-tracking of optimal states sequence
        states_seq = np.zeros(n_time, int)
        states_seq[-2], states_seq[-1] = np.unravel_index(np.argmax(prob_table[:, :, -1]), prob_table[:, :, -1].shape)
        for t in reversed(range(2, n_time)):
            states_seq[t-2] = path_table[states_seq[t-1], states_seq[t], t]

        # ----- CONVERSION OF INDEXES TO REAL STATES
        states_sequence = self._convert_states_sequence_to_string(states_seq)

        return states_sequence
