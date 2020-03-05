from env.simpleMDP import simpleMDP
import numpy as np

mdp = simpleMDP()

policy = dict()

states = mdp.get_all_states()
print(states)

for s in states:
    actions = mdp.get_possible_actions(s)
    for a in actions:
        next_states = mdp.get_next_states(s, a)
        print("State: " + s + " action: " + a + " " + "list of possible next states: ", next_states)

for s in states:
    actions = mdp.get_possible_actions(s)
    action_prob = 1 / len(actions)
    policy[s] = dict()
    for a in actions:
        policy[s][a] = action_prob

print(policy)

NUM_OF_ITERATIONS = 100000


def policy_eval_two_arrays(mdp, policy, gamma, theta):
    """
    This function uses the in-place approach to evaluate the specified policy for the specified MDP:

    'mdp' - model of the environment, use following functions:
        get_all_states - return list of all states available in the environment
        get_possible_actions - return list of possible actions for the given state
        get_next_states - return list of possible next states with a probability for transition from state by taking
                          action into next_state

    'policy' - the stochastic policy (action probability for each state), for the given mdp, too evaluate.
    'gamma' - discount factor for MDP
    'theta' - algorithm should stop when minimal difference between previous evaluation of policy and current is smaller
              than theta
    """
    all_states = mdp.get_all_states()
    V = dict()
    # delta = 0


    for state in all_states:
        V[state] = 1

    for i in range(NUM_OF_ITERATIONS):
        copy_V = V.copy()
        delta = 0
        for state in all_states:
            valueS = 0
            possible_actions = mdp.get_possible_actions(state)
            prob_to_take_action = 1 / len(possible_actions)
            for action in possible_actions:
                sum_for_all_end_states = 0
                next_states_with_prob_dict = mdp.get_next_states(state, action)
                for next_state in next_states_with_prob_dict.keys():
                    going_in_that_direction_prob = next_states_with_prob_dict[next_state]
                    reward = mdp.get_reward(state, action, next_state)
                    next_state_values = going_in_that_direction_prob * (
                            reward + gamma * V[next_state])  # p(s', r|s, a)[r + gamma*V(s')]
                    sum_for_all_end_states += next_state_values
                valueS += prob_to_take_action * sum_for_all_end_states
            delta = max(delta, valueS - copy_V[state])
            copy_V[state] = valueS
        V = copy_V
        if delta < theta:
            pass
            # break

    # INSERT CODE HERE to evaluate the policy using the 2 array approach

    return V


V = policy_eval_two_arrays(mdp, policy, 0.9, 0.0001)
print(V)

assert np.isclose(V['s0'], 1.46785443374683)
assert np.isclose(V['s1'], 4.55336594491180)
assert np.isclose(V['s2'], 1.68544141660991)
