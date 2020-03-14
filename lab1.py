import sys

from env.simpleMDP import simpleMDP
import numpy as np

mdp = simpleMDP()

states = mdp.get_all_states()
print(states)

for s in states:
    actions = mdp.get_possible_actions(s)
    for a in actions:
        next_states = mdp.get_next_states(s, a)
        print("State: " + s + " action: " + a + " " + "list of possible next states: ", next_states)

policy = dict()

for s in states:
    actions = mdp.get_possible_actions(s)
    action_prob = 1 / len(actions)
    policy[s] = dict()
    for a in actions:
        policy[s][a] = action_prob

print(policy)

def policy_eval_two_arrays(mdp, policy, gamma, theta):
    all_states = mdp.get_all_states()
    V = dict()

    for state in all_states:
        V[state] = 0

    while True:
        delta = 0
        V_copy = V.copy()
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
                            reward + gamma * V_copy[next_state])  # p(s', r|s, a)[r + gamma*V(s')]
                    sum_for_all_end_states += next_state_values
                valueS += prob_to_take_action * sum_for_all_end_states  # PI(a|s) * ...
            delta = max(delta, valueS - V_copy[state])
            V[state] = valueS
        if delta < theta:
            print("Delta <= theta")
            break

    return V

V = policy_eval_two_arrays(mdp, policy, 0.9, 0.0001)
print(V)

assert np.isclose(V['s0'], 1.46785443374683)
assert np.isclose(V['s1'], 4.55336594491180)
assert np.isclose(V['s2'], 1.68544141660991)
# sys.exit()

def policy_eval_in_place(mdp, policy, gamma, theta):
    all_states = mdp.get_all_states()
    V = dict()

    for state in all_states:
        V[state] = 0

    while True:
        delta = 0
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
            delta = max(delta, valueS - V[state])
            V[state] = valueS
        if delta < theta:
            print("Delta <= theta")
            break
    return V



V = policy_eval_in_place(mdp, policy, 0.9, 0.0001)
print(V)

assert np.isclose(V['s0'], 1.4681508097651)
assert np.isclose(V['s1'], 4.5536768132712)
assert np.isclose(V['s2'], 1.6857723431614)
# sys.exit("The end")


def deterministic_policy_eval_in_place(mdp, policy, gamma, theta):
    all_states = mdp.get_all_states()
    V = dict()

    for state in all_states:
        V[state] = 0

    while True:
        delta = 0
        for state in all_states:
            valueS = 0
            possible_actions = mdp.get_possible_actions(state)
            for action in possible_actions:
                if policy[state] is action:
                    prob_to_take_action = 1
                else:
                    prob_to_take_action = 0
                sum_for_all_end_states = 0
                next_states_with_prob_dict = mdp.get_next_states(state, action)
                for next_state in next_states_with_prob_dict.keys():
                    going_in_that_direction_prob = next_states_with_prob_dict[next_state]
                    reward = mdp.get_reward(state, action, next_state)
                    next_state_values = going_in_that_direction_prob * (
                            reward + gamma * V[next_state])  # p(s', r|s, a)[r + gamma*V(s')]
                    sum_for_all_end_states += next_state_values
                valueS += prob_to_take_action * sum_for_all_end_states  # PI(a|s) * ...
            delta = max(delta, valueS - V[state])
            V[state] = valueS
        if delta < theta:
            print("Delta <= theta")
            break
    return V


def policy_improvement(mdp, policy, value_function, gamma):
    """
            This function improves specified deterministic policy for the specified MDP using value_function:

           'mdp' - model of the environment, use following functions:
                get_all_states - return list of all states available in the environment
                get_possible_actions - return list of possible actions for the given state
                get_next_states - return list of possible next states with a probability for transition from state by taking
                                  action into next_state

           'policy' - the deterministic policy (action for each state), for the given mdp, too improve.
           'value_function' - the value function, for the given policy.
            'gamma' - discount factor for MDP

           Function returns True if policy was improved or False otherwise
       """

    policy_stable = True
    strategy_per_action = {}

    for state in mdp.get_all_states():
        actions = mdp.get_possible_actions(state)
        action_values = {}
        best_action_value = -90000
        for action in actions:
            action_value = 0
            next_states = mdp.get_next_states(state, action).keys()
            for next_state in mdp.get_next_states(state, action).keys():
                action_value += mdp.get_next_states(state, action)[next_state] * \
                                (mdp.get_reward(state, action, next_state) + gamma * value_function[next_state])
            action_values[action] = action_value
            if action_value > best_action_value:
                best_action_value = action_value
        for action, value in action_values.items():
            if value == best_action_value:
                if action is not policy[state]:
                    policy_stable = False
                policy[state] = action
                break

    print("Strategy improvent")
    print(strategy_per_action)
    print(policy)
    return policy_stable


# policy_improvement(mdp, policy, V, 0.9)


def policy_iteration(mdp, gamma, theta):
    """
            This function calculate optimal policy for the specified MDP:

           'mdp' - model of the environment, use following functions:
                get_all_states - return list of all states available in the environment
                get_possible_actions - return list of possible actions for the given state
                get_next_states - return list of possible next states with a probability for transition from state by taking
                                  action into next_state

           'gamma' - discount factor for MDP
           'theta' - algorithm should stop when minimal difference between previous evaluation of policy and current is smaller
                      than theta
           Function returns optimal policy and value function for the policy
       """

    policy = dict()

    for s in states:
        actions = mdp.get_possible_actions(s)
        policy[s] = actions[0]

    V = deterministic_policy_eval_in_place(mdp, policy, gamma, theta)

    policy_stable = False

    while not policy_stable:
        policy_stable = policy_improvement(mdp, policy, V, gamma)
        V = deterministic_policy_eval_in_place(mdp, policy, gamma, theta)

        #
        # INSERT CODE HERE to evaluate the best policy and value function for the given mdp
        #

    return policy, V

optimal_policy, optimal_value = policy_iteration(mdp, 0.9, 0.001)

print(optimal_policy)
print(optimal_value)

assert optimal_policy['s0'] == 'a1'
assert optimal_policy['s1'] == 'a0'
assert optimal_policy['s2'] == 'a1'

assert np.isclose(optimal_value['s0'], 3.78536612814300)
assert np.isclose(optimal_value['s1'], 7.29865364527343)
assert np.isclose(optimal_value['s2'], 4.20683179007964)