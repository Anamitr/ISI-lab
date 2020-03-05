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

NUM_OF_ITERATIONS = 1000000


def check_if_deltas_ok(deltas: dict, theta):
    result = True
    for state in deltas.keys():
        if deltas[state] > theta:
            result = False
        else:
            print("Delta ok for", state)
    return result
    pass


def policy_eval_two_arrays(mdp, policy, gamma, theta):
    all_states = mdp.get_all_states()
    V = dict()
    deltas = dict()
    # is_deltas_ok = False

    for state in all_states:
        V[state] = 1
        deltas[state] = 0

    for i in range(NUM_OF_ITERATIONS):
        copy_V = V.copy()
        for state in all_states:
            valueS = 0
            possible_actions = mdp.get_possible_actions(state)
            for action in possible_actions:
                prob_to_take_action = policy[state][action]
                sum_for_all_end_states = 0
                next_states_with_prob_dict = mdp.get_next_states(state, action)
                for next_state in next_states_with_prob_dict.keys():
                    going_in_that_direction_prob = next_states_with_prob_dict[next_state]
                    reward = mdp.get_reward(state, action, next_state)
                    next_state_values = going_in_that_direction_prob * (
                            reward + gamma * V[next_state])  # p(s', r|s, a)[r + gamma*V(s')]
                    sum_for_all_end_states += next_state_values
                valueS += prob_to_take_action * sum_for_all_end_states  # PI(a|s) * ...
            deltas[state] = max(deltas[state], valueS - copy_V[state])
            copy_V[state] = valueS
        V = copy_V
        if check_if_deltas_ok(deltas, theta):
            break
        # if deltas < theta:
        #     pass
            # break

    # INSERT CODE HERE to evaluate the policy using the 2 array approach

    return V


def policy_eval_in_place(mdp, policy, gamma, theta):
    all_states = mdp.get_all_states()
    V = dict()
    deltas = dict()
    # is_deltas_ok = False

    for state in all_states:
        V[state] = 1
        deltas[state] = 0

    for i in range(NUM_OF_ITERATIONS):
        for state in all_states:
            valueS = 0
            possible_actions = mdp.get_possible_actions(state)
            for action in possible_actions:
                prob_to_take_action = policy[state][action]
                sum_for_all_end_states = 0
                next_states_with_prob_dict = mdp.get_next_states(state, action)
                for next_state in next_states_with_prob_dict.keys():
                    going_in_that_direction_prob = next_states_with_prob_dict[next_state]
                    reward = mdp.get_reward(state, action, next_state)
                    next_state_values = going_in_that_direction_prob * (
                            reward + gamma * V[next_state])  # p(s', r|s, a)[r + gamma*V(s')]
                    sum_for_all_end_states += next_state_values
                valueS += prob_to_take_action * sum_for_all_end_states  # PI(a|s) * ...
            deltas[state] = max(deltas[state], valueS - V[state])
            V[state] = valueS
        if check_if_deltas_ok(deltas, theta):
            break
        # if deltas < theta:
        #     pass
            # break

    # INSERT CODE HERE to evaluate the policy using the 2 array approach

    return V


# V = policy_eval_two_arrays(mdp, policy, 0.9, 0.0001)
V = policy_eval_in_place(mdp, policy, 0.9, 0.0001)
print(V)

assert np.isclose(V['s0'], 1.46785443374683)
assert np.isclose(V['s1'], 4.55336594491180)
assert np.isclose(V['s2'], 1.68544141660991)
