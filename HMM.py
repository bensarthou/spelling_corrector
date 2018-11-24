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
    observations = sorted(list({obs for obs_seq in y for obs in obs_seq}))

    if min_observation_count > 0:
        obs_counts = Counter([token[0] for sequence in data for token in sequence])
        observations = sorted([obs for (obs, count) in obs_counts.items() if count >= min_observation_count] + [UNK])

    return states, observations


class HMM:
    def __init__(self, state_list, observation_list,
                 transition_proba = None,
                 observation_proba = None,
                 initial_state_proba = None):
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

        print("1st order HMM created with: ")
        print(" * {} states".format(self.n_states))
        print(" * {} observations".format(self.n_observations))

        # Init. of the 3 distributions : observation, transition and initial states
        if transition_proba is None:
            self.transition_proba = np.zeros( (self.n_states, self.n_states), float)
        else:
            self.transition_proba = transition_proba
        if observation_proba is None:
            self.observation_proba = np.zeros( (self.n_states, self.n_observations), float)
        else:
            self.observation_proba = observation_proba
        if initial_state_proba is None:
            self.initial_state_proba = np.zeros( (self.n_states,), float )
        else:
            self.initial_state_proba = initial_state_proba

        # Since everything will be stored in numpy arrays, it is more convenient and compact to
        # handle words and tags as indices (integer) for a direct access. However, we also need
        # to keep the mapping between strings (word or tag) and indices.
        self.make_indexes()


    def make_indexes(self):
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


    def fit(self, X, y):
        """
        Estimate HMM parameters (initial, transition and emisson matrices) from a training data set.
        :param X: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :param y: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        print("Training initial states probabilities...", end="")
        self.train_initstate_proba(y)
        print(" Done.")
        print("Training transitions probabilities given states...", end="")
        self.train_transitions_proba(y)
        print(" Done.")
        print("Training observations probabilities given states...", end="")
        self.train_observations_proba(X, y)
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

        obs_seq = []
        for obs in observations_sequence:
            if obs not in self.X_index.keys():
                obs = UNK
            obs_seq.append(self.X_index[obs])

        # ----- VITERBI

        # init variables
        n_obs = len(obs_seq)
        prob_table = np.zeros((self.n_states, n_obs))
        path_table = np.zeros((self.n_states, n_obs))

        # initial state
        prob_table[:, 0] = self.observation_proba[:, obs_seq[0]] * self.initial_state_proba
        path_table[:, 0] = 0

        # loop for each observation
        for k in range(1, n_obs):
            for i_state in range(self.n_states):
                p_state_given_prev_state_and_obs = prob_table[:, k-1] * self.transition_proba[:, i_state] * self.observation_proba[i_state, int(obs_seq[k])]
                prob_table[i_state, k] = np.max(p_state_given_prev_state_and_obs)
                path_table[i_state, k] = np.argmax(p_state_given_prev_state_and_obs)

        # back-tracking of optimal states sequence
        states_seq = np.zeros((n_obs,))
        states_seq[-1] = np.argmax(prob_table[:, -1])
        states_sequence_proba = np.max(prob_table[:, -1])
        for k in range(n_obs-1, 0, -1):
            states_seq[k-1] = path_table[int(states_seq[k]), k]

        # ----- CONVERSION OF INDEXES TO REAL STATES

        states_sequence = []
        for i_state in states_seq:
            states_sequence.append(self.omega_Y[int(i_state)])

        return states_sequence


    def predict(self, observations_sequences):
        """
        Predict the sequences of states from sequences of observations.
        :param observations_sequences: list of observations sequences. Ex: [['o1', 'o2', 'o3'], ['o1', 'o2']]
        :return states_sequences: list of states sequences. Ex: [['s1', 's2', 's3'], ['s1', 's2']]
        """
        states_sequences = []
        for obs_seq in observations_sequences:
            states_sequences.append(self.viterbi(obs_seq))
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
