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


ACTION_PROB = 0.25


def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


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
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0],next_state[1]])

        best_actions=np.where(next_vals == np.max(next_vals))[0]
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
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_2.png')
            plt.close()
            break
        value = new_value

def figure_3_2_linear_system():
    '''
    Here we solve the linear system of equations to find the exact solution.
    We do this by filling the coefficients for each of the states with their respective right side constant.
    '''
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_s_] += ACTION_PROB * DISCOUNT
                b[index_s] -= ACTION_PROB * r

    x = np.linalg.solve(A, b)
    try:
        draw_image(np.round(x.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    except Exception as e:
        print('error')
    print("test")
    plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_2_linear_system.png')
    plt.close()

def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_5.png')
            plt.close()
            draw_policy(new_value)
            plt.savefig('/workspaces/reinforcement-learning-an-introduction/images/figure_3_5_policy.png')
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
            state = [i, j]
            
            # Calculate Q(a) values for each action
            action_values = []
            for action in ACTIONS:
                next_state, reward = step(state, action)
                action_values.append(reward + DISCOUNT * value_vector[next_state[0], next_state[1]])
            
             # Epsilon-greedy action selection
            if np.random.random() < epsilon/num_actions:
                # Explore: choose random action
                policy[i, j] = np.random.randint(len(ACTIONS))
            else:
                # Exploit: choose best action
                policy[i, j] = np.argmax(action_values)

    print(policy)
            
    return policy

# Question 3 
def policy_iteration(epsilon):
    """
    Compute and return an optimal ε-greedy policy using policy iteration.
    
    Parameters:
    epsilon (float): The exploration rate ε
    
    Returns:
    tuple: (optimal policy, optimal value function)
    """
    # Initialize with a random policy
    policy = np.random.randint(0, len(ACTIONS), (WORLD_SIZE, WORLD_SIZE), dtype=np.int32)
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    num_actions = len(ACTIONS)  # |A| in the formula

    
    while True:
        # 1. Policy Evaluation (similar to figure_3_2)
        while True:
            new_value = np.zeros_like(value)
            for i in range(WORLD_SIZE):
                for j in range(WORLD_SIZE):
                    # Get the action from current policy
                    action = ACTIONS[policy[i, j]]
                    next_state, reward = step([i, j], action)
                    # Bellman equation for the current policy
                    new_value[i, j] = reward + DISCOUNT * value[next_state[0], next_state[1]]
            
            # Check for convergence
            if np.sum(np.abs(value - new_value)) < 1e-4:
                break
            value = new_value
        
        # 2. Policy Improvement
        policy_stable = True
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                old_action = policy[i, j]
                
                # Calculate value for all actions
                action_values = []
                for action in ACTIONS:
                    next_state, reward = step([i, j], action)
                    action_values.append(reward + DISCOUNT * value[next_state[0], next_state[1]])
                
                # ε-greedy policy improvement
                if np.random.random() < epsilon/num_actions:
                    policy[i, j] = np.random.randint(len(ACTIONS))
                else:
                    policy[i, j] = np.argmax(action_values)
                
                if old_action != policy[i, j]:
                    policy_stable = False
        
        # Save current state using provided drawing functions
        try:
            draw_image(np.round(value, decimals=2))
            draw_policy(value)
            plt.savefig(f'/workspaces/reinforcement-learning-an-introduction/images/policy_iteration_eps_{epsilon}.png')
            plt.close()
        except Exception as e:
            print(f"Error in drawing: {e}")
        
        # Check if policy has converged
        if policy_stable:
            print(f"Policy converged for ε = {epsilon} with policy: {policy} and value: {value}")
            return policy, value


if __name__ == '__main__':
    figure_3_2_linear_system()
    figure_3_2()
    figure_3_5()
    # epsilon greedy for a random value vector
    value_vector = np.random.rand(WORLD_SIZE, WORLD_SIZE)
    epsilon = 0.1
    policy = get_epsilon_greedy_policy(value_vector, epsilon)

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
