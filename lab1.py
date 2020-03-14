from env.simpleMDP import simpleMDP
import numpy as np

mdp = simpleMDP()

policy = dict()

states = mdp.get_all_states()
# print(states)

for s in states:
    actions = mdp.get_possible_actions(s)
    for a in actions:
        next_states = mdp.get_next_states(s, a)
        # print("State: " + s + " action: " + a + " " + "list of possible next states: ", next_states)

for s in states:
    actions = mdp.get_possible_actions(s)
    action_prob = 1 / len(actions)
    policy[s] = dict()
    for a in actions:
        policy[s][a] = action_prob

# print(policy)

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

    return V


def assertValues(V):
    if np.isclose(V['s0'], 1.46785443374683):
        print('s0 ok ', V['s0'])
    if np.isclose(V['s1'], 4.55336594491180):
        print('s1 ok ', V['s1'])
    if np.isclose(V['s2'], 1.68544141660991):
        print('s2 ok ', V['s2'])


def policy_eval_in_place(mdp, policy, gamma, theta):
    all_states = mdp.get_all_states()
    V = dict()
    # deltas = dict()
    # is_deltas_ok = False

    for state in all_states:
        V[state] = 0
        # deltas[state] = 0

    for i in range(NUM_OF_ITERATIONS):
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
                    next_state_values_str = state + "," + action + "," + next_state + ": " + str(
                        going_in_that_direction_prob) + " * (" + str(reward) + " + " + str(
                        gamma) + " * " + str(V[next_state]) + ")"
                    print("fenek:", next_state_values_str)
                    sum_for_all_end_states += next_state_values
                valueS += prob_to_take_action * sum_for_all_end_states  # PI(a|s) * ...
            # deltas[state] = max(deltas[state], valueS - V[state])
            delta = max(delta, valueS - V[state])
            print("V(" + state + "):", valueS)
            V[state] = valueS
        assertValues(V)
        if delta < theta:
            print("Delta <= theta")
            break
        # if check_if_deltas_ok(deltas, theta):
        #     break
        # if deltas < theta:
        #     pass
        # break

    return V


def deterministic_policy_eval_in_place(mdp, policy, gamma, theta):
    all_states = mdp.get_all_states()
    V = dict()
    # deltas = dict()
    # is_deltas_ok = False

    for state in all_states:
        V[state] = 0
        # deltas[state] = 0

    for i in range(NUM_OF_ITERATIONS):
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
                    next_state_values_str = state + "," + action + "," + next_state + ": " + str(
                        going_in_that_direction_prob) + " * (" + str(reward) + " + " + str(
                        gamma) + " * " + str(V[next_state]) + ")"
                    print("fenek:", next_state_values_str)
                    sum_for_all_end_states += next_state_values
                valueS += prob_to_take_action * sum_for_all_end_states  # PI(a|s) * ...
            # deltas[state] = max(deltas[state], valueS - V[state])
            delta = max(delta, valueS - V[state])
            print("V(" + state + "):", valueS)
            V[state] = valueS
        assertValues(V)
        if delta < theta:
            print("Delta <= theta")
            break
        # if check_if_deltas_ok(deltas, theta):
        #     break
        # if deltas < theta:
        #     pass
        # break

    return V


# V = policy_eval_two_arrays(mdp, policy, 0.9, 0.0001)
V = policy_eval_in_place(mdp, policy, 0.9, 0.0000000001)
print(V)


# assert np.isclose(V['s0'], 1.46785443374683)
# # 1.4687121343257736
# assert np.isclose(V['s1'], 4.55336594491180)
# assert np.isclose(V['s2'], 1.68544141660991)


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
                strategy_per_action[state] = action
                action_dict = {action: 1}
                policy[state] = action_dict

    print("Strategy improvent")
    print(strategy_per_action)
    print(policy)
    return policy_stable


policy_improvement(mdp, policy, V, 0.9)


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
        policy[s] = {actions[0] : 1, actions[1] : 0}

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