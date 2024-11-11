#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# guys look at draw_image()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS=[ '←', '↑', '→', '↓']

# [intended_dir, opposite_dir, clockwise_dir, counterclockwise_dir]
ACTION_PROB = {
    0: [0.85, 0.05, 0.05, 0.05],  # Left
    1: [0.85, 0.05, 0.05, 0.05],  # Up
    2: [0.85, 0.05, 0.05, 0.05],  # Right
    3: [0.85, 0.05, 0.05, 0.05]   # Down
}

# Updated version of step function is expected_step to account for stochasticity below
def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5
    
    probs = ACTION_PROB[action]
    actual_action_idx = np.random.choice(4, p=probs)
    action = ACTIONS[actual_action_idx]
    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

#expected step - since we introduced probabilities in the step function , we have to calculate next state and reward based on the probabilities
def expected_step(state, action_idx):
    ''' Calculate expected next state avlue considering all possible outcomes of the action ''' 
    if state == A_POS:
        return {tuple(A_PRIME_POS): (1.0, 10)}
    if state == B_POS:
        return {tuple(B_PRIME_POS): (1.0, 5)}
    # Primary change - taking in action indices instead of actions directly due to probability distribution in all functions.
    probs = ACTION_PROB[action_idx]
    # meaning: [left, right, up, down] -> [0, 2, 1, 3]
    next_states = {} # dictionary to save next states and their probabilities
    
    # Each tuple in next_states is of the form (next_state, (prob, reward))
    for i, prob in enumerate(probs):
        action = ACTIONS[i]
        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        
        if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
            next_state = state
            reward = -1.0
        else:
            reward = 0
            
        # Check if the next state is already in the dictionary and accumulate the probabilities and rewards
        next_state_tuple = tuple(next_state)
        if next_state_tuple in next_states:
            next_states[next_state_tuple] = (next_states[next_state_tuple][0] + prob, reward)
        else:
            next_states[next_state_tuple] = (prob, reward)
            
    return next_states


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):

        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"
        
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')
        

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals=[]
        # For each action, compute expected value considering all possible outcomes
        for action_idx in range(len(ACTIONS)):
            next_states = expected_step([i, j], action_idx)
            action_value = 0
            # Sum up the expected value for this action
            for (next_i, next_j), (prob, reward) in next_states.items():
                action_value += prob * (reward + DISCOUNT * optimal_values[next_i, next_j])
            next_vals.append(action_value)

        # Find best actions based on expected values
        best_actions = np.where(np.abs(next_vals - np.max(next_vals)) < 1e-8)[0]
        val=''
        for ba in best_actions:
            val+=ACTIONS_FIGS[ba]
        
        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"
        
        tb.add_cell(i, j, width, height, text=val,
                loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                   edgecolor='none', facecolor='none')

    ax.add_table(tb)


def figure_3_2():
    '''
    Here we perform value iteration to find the optimal value function.
    We keep iterating until the value function converges.
    For stochastic: V(s) = Σ π(a|s) * Σ P(s'|s,a) * [R(s,a,s') + γV(s')]
    '''
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in range(len(ACTIONS)):
                    # Get all possible next states and their probabilities
                    next_states = expected_step([i, j], action)
                    for next_state, (prob, reward) in next_states.items():
                        next_i, next_j = next_state
                        # value iteration
                        new_value[i, j] += 0.25 * prob * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_2_stochastic.png')
            plt.close()
            break
        value = new_value

def figure_3_2_linear_system():
    '''
    Here we solve the linear system of equations to find the exact solution.
    We do this by filling the coefficients for each of the states with their respective right side constant.
    V(s) = Σ π(a|s) * Σ P(s'|s,a) * [R(s,a,s') + γV(s')]
    '''
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in range(len(ACTIONS)):
                next_states = expected_step(s, a)
                for next_state, (prob, r) in next_states.items():
                    s_ = next_state
                    index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))
                    A[index_s, index_s_] += 0.25 * prob * DISCOUNT
                    b[index_s] -= 0.25 * prob * r

    x = np.linalg.solve(A, b)
    try:
        draw_image(np.round(x.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    except Exception as e:
        print('error')
    plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_2_linear_system_stochastic.png')
    plt.close()

def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in range(len(ACTIONS)):
                    # Get all possible next states and their probabilities
                    next_states = expected_step([i, j], action)
                    action_value = 0
                    for next_state, (prob, reward) in next_states.items():
                        next_i, next_j = next_state
                        action_value += prob * (reward + DISCOUNT * value[next_i, next_j])
                    values.append(action_value)
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_5_stochatic.png')
            plt.close()
            draw_policy(new_value)
            plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_5_policy_stochastic.png')
            plt.close()
            break
        value = new_value

# Question 2
def get_epsilon_greedy_policy(value_vector, epsilon):
    """
    Computes the epsilon-greedy policy following the formula:
    At = argmax_a Qt(a) with probability 1-ε
        random action with probability ε/|A|
    
    Parameters:
    value_vector (numpy.ndarray): The value vector V
    epsilon (float): The exploration rate ε
    
    Returns:
    numpy.ndarray: The epsilon-greedy policy matrix
    """
    policy = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.int32)
    num_actions = len(ACTIONS)  # |A| in the formula
    
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if (i == 0 and j == WORLD_SIZE-1):  # Skip terminal state
                continue
                
            # Calculate Q(a) values for each action
            action_values = []
            for action in range(len(ACTIONS)):
                # Get all possible next states and their probabilities
                    next_states = expected_step([i, j], action)
                    action_value = 0
                    for next_state, (prob, reward) in next_states.items():
                        next_i, next_j = next_state
                        action_value += prob * (reward + DISCOUNT * value_vector[next_i, next_j])
                    action_values.append(action_value)

            # Find the best action
            greedy_action = np.argmax(action_values)
            p = np.full(num_actions, epsilon/num_actions)
            p[greedy_action] += 1 - epsilon
            policy[i, j] = np.random.choice(np.arange(num_actions), p=p)
        
    # print("\nEpsilon-Greedy Policy for the given Value Vector:")
    # print(policy)
            
    return policy

# Question 3 
# def policy_iteration(epsilon):
#     """
#     Compute and return an optimal ε-greedy policy using policy iteration.
    
#     Parameters:
#     epsilon (float): The exploration rate ε
    
#     Returns:
#     optimal policy, optimal value


#     """
#     value = np.zeros((WORLD_SIZE, WORLD_SIZE))
#     k=0
#     # Initialize π_k as ε-greedy policy with respect to V
#     policy = get_epsilon_greedy_policy(value, epsilon)
#     num_actions = len(ACTIONS)  # |A| in the formula


#     while True:
#         # 1. Policy Evaluation 
#         while True:
#             new_value = np.zeros_like(value)
#             delta = 0  # Track maximum change in value
            
#             for i in range(WORLD_SIZE):
#                 for j in range(WORLD_SIZE):
#                     if (i == 0 and j == WORLD_SIZE-1):  # Skip terminal state
#                         continue
                        
#                     old_value = value[i, j]
#                     state_value = 0
                    
#                     # Sum over all actions: Σ π_k(a|s)[R(s,a) + γ Σ P(s'|s,a)V(s')]
#                     for action_idx in range(len(ACTIONS)):
#                         # Calculate probability of taking this action under ε-greedy policy
#                         if action_idx == policy[i, j]:
#                             action_prob = 1 - epsilon + epsilon/len(ACTIONS)
#                         else:
#                             action_prob = epsilon/len(ACTIONS)

#                         # Get next states and their probabilities
#                         next_states = expected_step([i, j], action_idx)
#                         action_value = 0
#                         for next_state, (prob, reward) in next_states.items():
#                             next_i, next_j = next_state
#                             action_value += prob * (reward + DISCOUNT * value[next_i, next_j])
#                         state_value += action_prob * action_value
                    
#                     new_value[i, j] = state_value
#                     delta = max(delta, abs(old_value - state_value))
            
#             # Update value and check convergence
#             value = new_value.copy()
#             if delta < 1e-4:  # Convergence criterion
#                 break

#         # 2. Policy Improvement
#         policy_stable = True

#         for i in range(WORLD_SIZE):
#             for j in range(WORLD_SIZE):
#                 if (i == 0 and j == WORLD_SIZE-1):  # Skip terminal state
#                     continue
#                 old_action = policy[i, j]
                    
#                 # Calculate Q(a) values for each action
#                 action_values = []
#                 for action in range(len(ACTIONS)):
#                     # Get all possible next states and their probabilities
#                         next_states = expected_step([i, j], action)
#                         action_value = 0
#                         for next_state, (prob, reward) in next_states.items():
#                             next_i, next_j = next_state
#                             action_value += prob * (reward + DISCOUNT * value[next_i, next_j])
#                         action_values.append(action_value)

#                 # Find the best action
#                 greedy_action = np.argmax(action_values)
#                 p = np.full(num_actions, epsilon/num_actions)
#                 p[greedy_action] += 1 - epsilon
#                 policy[i, j] = np.random.choice(np.arange(num_actions), p=p)

#                 if old_action != policy[i, j]:
#                     policy_stable = False
    
#         k += 1
    
#         # Save figures  after convergence
#         try:
#             draw_image(np.round(value, decimals=2))
#             plt.savefig(f'/workspaces/reinforcement-learning-an-introduction/images/policy_iteration_value_{epsilon}_{k}_stochastic.png')
#             plt.close()

#             draw_policy(value)
#             plt.savefig(f'/workspaces/reinforcement-learning-an-introduction/images/policy_iteration_{epsilon}_{k}_stochastic.png')
#             plt.close()
#         except Exception as e:
#             print(f"Error in drawing: {e}")

#         if policy_stable:
#             print(f"\nPolicy Iteration Converged after {k} iterations for ε = {epsilon}")
#             print("Optimal Policy:")    
#             print(policy)
#             return policy

def policy_iteration(epsilon):
    """
    Compute and return an optimal ε-greedy policy using policy iteration.
    
    Parameters:
    epsilon (float): The exploration rate ε
    
    Returns:
    optimal policy, optimal value function
    """

    # Initialize value function and policy
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))

    # Initialize π_k as ε-greedy policy with respect to V
    policy = get_epsilon_greedy_policy(value, epsilon)


    policy_stable = False
    iteration = 0
    num_actions = len(ACTIONS)

    while not policy_stable:
        # Policy Evaluation: Calculate V^π_k
        while True:
            delta = 0
            new_value = np.zeros_like(value)
            for i in range(WORLD_SIZE):
                for j in range(WORLD_SIZE):
                    if (i == 0 and j == WORLD_SIZE - 1):  # Skip terminal state
                        continue
                    state_value = 0
                    for action_idx in range(len(ACTIONS)):
                        if action_idx == policy[i, j]:
                            action_prob = 1 - epsilon + epsilon / len(ACTIONS)
                        else:
                            action_prob = epsilon / len(ACTIONS)
                        next_states = expected_step([i, j], action_idx)
                        action_value = 0
                        for next_state, (prob, reward) in next_states.items():
                            next_i, next_j = next_state
                            # Add γ Σ P(s'|s,a)V(s')
                            action_value += prob * (reward + DISCOUNT * value[next_i, next_j])
                        # Multiply by π_k(a|s) and add to state value
                        state_value += action_prob * action_value
                    new_value[i, j] = state_value
                    delta = max(delta, abs(new_value[i, j] - value[i, j]))
            value = new_value.copy()
            if delta < 1e-4:
                break

        # Policy Improvement
        policy_stable = True# Assume policy is stable
        # reiterating get_epsilon_greedy_policy to update but not using the function as it yields different results.
        # the bottom half can be replaced with the commented code below
        #  # Generate a new policy using the epsilon-greedy approach
        # new_policy = get_epsilon_greedy_policy(value, epsilon)

        # for i in range(WORLD_SIZE):
        #     for j in range(WORLD_SIZE):
        #         if (i == 0 and j == WORLD_SIZE - 1):  # Skip terminal state
        #             continue

        #         old_action = policy[i, j]
        #         policy[i, j] = new_policy[i, j]

        #         if old_action != policy[i, j]:
        #             policy_stable = False

        # iteration += 1
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                if (i == 0 and j == WORLD_SIZE - 1):  # Skip terminal state
                    continue
                old_action = policy[i, j]
                action_values = []
                # Calculate Q(a) values for each action
                for action_idx in range(len(ACTIONS)):
                    next_states = expected_step([i, j], action_idx)
                    action_value = 0
                    # Calculate Σ P(s'|s,a)[R(s,a,s') + γV(s')]
                    for next_state, (prob, reward) in next_states.items():
                        next_i, next_j = next_state
                        action_value += prob * (reward + DISCOUNT * value[next_i, next_j])
                    action_values.append(action_value)
                greedy_action = np.argmax(action_values)
                p = np.full(num_actions, epsilon/num_actions)
                p[greedy_action] += 1 - epsilon
                policy[i, j] = np.random.choice(np.arange(num_actions), p=p)
                if old_action != greedy_action:
                    policy_stable = False
        iteration += 1
        print(f"Iteration {iteration} completed.")

    # After convergence
    print(f"\nPolicy Iteration Converged after {iteration} iterations.")
    try:
        draw_image(np.round(value, decimals=2))
        plt.savefig(f'/workspaces/reinforcement-learning-an-introduction/images/policy_iteration_value_{epsilon}_stochastic.png')
        plt.close()

        draw_policy(value)
        plt.savefig(f'/workspaces/reinforcement-learning-an-introduction/images/policy_iteration_{epsilon}_stochastic.png')
        plt.close()
    except Exception as e:
        print(f"Error in drawing: {e}")
    print(f"Optimal Policy:\n{policy}")
    print(f"Optimal Value Function:\n{value}")
    return policy
        
#Question 3 - modifying the function figure_3_2_linear_system to return the value function for the given policy
def v_pi_linear_system_2(policy):#
    '''
    Solve the linear system of equations to find Vπ(s) for a given policy π.
    We build the system (I - γPπ)Vπ = Rπ where Pπ and Rπ are based on the input policy.
    
    Parameters:
    policy (numpy.ndarray): Policy matrix of shape (WORLD_SIZE, WORLD_SIZE) containing actions (0-3)
    
    Returns:
    numpy.ndarray: The value function Vπ for the given policy
    '''
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            action_idx = policy[i, j]
            next_states = expected_step(s, action_idx)
            for next_state, (prob, r) in next_states.items():
                s_ = next_state
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))
                A[index_s, index_s_] += 0.25 * prob * DISCOUNT
                b[index_s] -= 0.25 * prob * r

    v_pi = np.linalg.solve(A, b)
    v_pi = v_pi.reshape(WORLD_SIZE, WORLD_SIZE)
    try:
        print("\nValue Function for the Given Policy:")
        print(np.round(v_pi, 2))
        draw_image(np.round(v_pi, decimals=2))
    except Exception as e:
        print('error')
    plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/v_pi_for_policy_pi_stochastic.png')
    plt.close()

    return v_pi

if __name__ == '__main__':
    figure_3_2_linear_system()
    figure_3_2()
    figure_3_5()
    epsilon_greedy_policy = policy_iteration(0.2)
    greedy_policy = policy_iteration(0.0)




    # policy iteration
    # # Test with ε = 0.2
    # print("Testing with ε = 0.2")
    # policy_0_2, value_0_2 = policy_iteration(0.2)
    # print("Policy (ε = 0.2):")
    # print(policy_0_2)
    # print("\nValue function (ε = 0.2):")
    # print(np.round(value_0_2, 2))
    
    # # Test with ε = 0.0
    # print("\nTesting with ε = 0.0")
    # policy_0_0, value_0_0 = policy_iteration(0.0)
    # print("Policy (ε = 0.0):")
    # print(policy_0_0)
    # print("\nValue function (ε = 0.0):")
    # print(np.round(value_0_0, 2))
