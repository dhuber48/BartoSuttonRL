import numpy as np
import matplotlib.pyplot as plt

wind_grid = np.zeros((7, 10), dtype=int)
rows, cols = wind_grid.shape

cols_one = [3, 4, 5, 8]
wind_grid[:, 6:8] = -2
wind_grid[:, cols_one] = -1

goal = [2, 9]

moves = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1), "NE": (-1, 1), "SE": (1, 1), "SW": (1, -1), "NW": (-1, -1)} #in python we count from the upper left corner of the array, so to move north you decrease the column index
move_keys = list(moves.keys())

alpha = 0.1
gamma = 1

#Q-table initialization
Q = np.zeros((len(wind_grid), len(wind_grid[0]), len(moves))) #numpy breaks first dimension into slices, fyi

#Gradient-table initialization
Gradient_Table = np.zeros(np.size(Q)) #We will define gradient as estimated rate of change of future reward

epsilon = 0.1 
epsilon_min = 0.01
epsilon_decay = 0.99995

rewards = np.ones_like(wind_grid) *-1
rewards[goal[0], goal[1]] = 0

def step(s, a):
    # Current position
    row, col = s

    # Move delta for action 'a'
    delta_row, delta_col = moves[a]

    # New position after move
    new_row_without_wind = row + delta_row 
    new_col = col + delta_col

    safe_row = max(0, min(rows - 1, new_row_without_wind))
    safe_col = max(0, min(cols - 1, new_col))

    wind = wind_grid[safe_row, safe_col]

    new_row = new_row_without_wind + wind

    # Keep new position inside the grid boundaries
    new_row = max(0, min(rows - 1, new_row))
    new_col = max(0, min(cols - 1, new_col))

    # Update state
    s_new = [int(new_row), int(new_col)]
    return s_new

def choose_action(s):
    if np.random.rand() < epsilon: #choose randomly epsilon percent of the time
        return np.random.choice(move_keys)
    else: #else choose best known action
        return move_keys[np.argmax(Q[s[0], s[1], :])]

episode_number = 1


#Updating Q-values
while episode_number <= 10000:
    epsilon = max(epsilon_min, epsilon * epsilon_decay) #decaying epsilon
    s = [3,0] #starting state
    a = choose_action(s) 
    episode_number = episode_number+1

    while s != goal:
        s_new = step(s, a)
        a_new = choose_action(s_new)
 
        Q_old = Q[s[0], s[1], move_keys.index(a)]

        Q[s[0], s[1], move_keys.index(a)] += alpha * (
            rewards[s_new[0], s_new[1]] + gamma * Q[s_new[0], s_new[1], move_keys.index(a_new)] - Q_old
        )

        s = s_new
        a = a_new

        if s == goal:
            print(episode_number)
            






#Marking down optimal path for graphing. This happens at the end after running the code. It just reads the best Q-values.
s = [3, 0]
optimal_path = [tuple(s)]
while s != goal:
    a = move_keys[np.argmax(Q[s[0], s[1], :])]
    s = step(s, a)
    optimal_path.append(tuple(s))

# Initialize arrow directions
U = np.zeros((rows, cols))
V = np.zeros((rows, cols))
U_opt = np.zeros((rows, cols))
V_opt = np.zeros((rows, cols))

# Fill in all arrows (light background)
for r in range(rows):
    for c in range(cols):
        if [r, c] == goal:
            continue
        best_idx = np.argmax(Q[r, c, :])
        dr, dc = moves[move_keys[best_idx]]
        U[r, c] = dc
        V[r, c] = dr

# Fill in only arrows on optimal path (black overlay)
for (r, c) in optimal_path:
    if [r, c] == goal:
        continue
    best_idx = np.argmax(Q[r, c, :])
    dr, dc = moves[move_keys[best_idx]]
    U_opt[r, c] = dc
    V_opt[r, c] = dr

# Plot
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
plt.figure(figsize=(8, 6))
plt.title("Learned Policy with Optimal Path Highlighted")
plt.imshow(wind_grid, cmap='Blues', origin='upper', alpha=0.3)

# All arrows in light red
plt.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy', pivot='middle', color='lightcoral', alpha=0.5)

# Greedy-path arrows in black
plt.quiver(X, Y, U_opt, V_opt, scale=1, scale_units='xy', angles='xy', pivot='middle', color='black')

# Goal marker
plt.scatter(goal[1], goal[0], marker='*', color='gold', s=200, label='Goal')

plt.xlim(-0.5, cols - 0.5)
plt.ylim(rows - 0.5, -0.5)
plt.xlabel("Column")
plt.ylabel("Row")
plt.grid(True)
plt.legend()
plt.show()