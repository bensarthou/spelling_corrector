states = ('Rainy', 'Sunny')

observations = ('walk', 'shop', 'clean')

start_probability = {
    'Rainy|Rainy' : 0.7,
    'Rainy|Sunny' : 0.3,
    'Sunny|Rainy' : 0.4,
    'Sunny|Sunny' : 0.6
}

transition_probability = {
    'Rainy|Rainy' : {'Rainy' : 0.8, 'Sunny' : 0.2},
    'Rainy|Sunny' : {'Rainy' : 0.5, 'Sunny' : 0.5},
    'Sunny|Rainy' : {'Rainy' : 0.6, 'Sunny' : 0.4},
    'Sunny|Sunny' : {'Rainy' : 0.3, 'Sunny' : 0.7},
}

emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}

def forward_viterbi(obs, states, start_p, trans_p, emit_p):
    T = {}
    for state1 in states:
        for state2 in states:
      ##          prob.           V. path  V. prob.
            T[state1+"|"+state2] = (start_p[state1+"|"+state2], [state2],
                                    start_p[state1+"|"+state2])
    for output in obs:

        U = {}
        print("--------------------\nObservation:", output)
        for next_state in states:
            total = 0
            argmax = None
            valmax = 0
            print("Next state:" + next_state)
            for curr_state in states:
                for prv_state in states:
                    print("\tprv_state|curr_state:", prv_state+"|"+curr_state)
                    try:
                        (prob, v_path, v_prob) = T[prv_state+"|"+curr_state]
                    except KeyError:
                        (prob, v_path, v_prob) = T[prv_state+"|"+curr_state] = (0, None, 0)

                    p = emit_p[curr_state][output] * trans_p[prv_state+"|"+curr_state][next_state]
                    prob *= p
                    v_prob *= p
                    total += prob
                    if v_prob > valmax:
                        argmax = v_path + [next_state]
                        valmax = v_prob
                    print("\t\t", v_path, v_prob)
            U[curr_state+"|"+next_state] = (total, argmax, valmax)
            print("\targmax:", argmax, "valmax:", valmax)
        T = U
    ## apply sum/max to the final states:
    total = 0
    argmax = None
    valmax = 0
    for state1 in states:
        for state2 in states:
            try:
                (prob, v_path, v_prob) = T[state1+"|"+state2]
            except KeyError:
                (prob, v_path, v_prob) = T[state1+"|"+state2] = (0, None, 0)
            total += prob
            if v_prob > valmax:
                argmax = v_path
                valmax = v_prob
    return (total, argmax, valmax)


def example():
    return forward_viterbi(observations,
                           states,
                           start_probability,
                           transition_probability,
                           emission_probability)

res = example()
print("\nResult:", res)
