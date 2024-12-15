import numpy as np
import random

# Parámetros globales
alpha = 0.1   # Tasa de aprendizaje
gamma = 0.9   # Factor de descuento
epsilon = 0.1 # Factor de exploración

# Matriz del terreno: 0 = camino, 1 = obstáculo (roca), la última casilla es la meta
entorno = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0],  # La última casilla es la meta
])
entorno[-1, -1] = 2  # Definir la meta en la última casilla

# Recompensas asociadas a cada tipo de casilla
recompensas = {
    0: -1,    # Camino
    1: -100,  # Obstáculo (no transitable)
    2: 100    # Meta
}

# Acciones posibles
actions = ['up', 'down', 'left', 'right']

def initialize_q_table(env):
    """Inicializa una tabla Q con ceros."""
    return np.zeros((env.shape[0], env.shape[1], len(actions)))

def choose_action(q_table, state, epsilon):
    """Elige una acción usando epsilon-greedy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # Explorar
    else:
        x, y = state
        return np.argmax(q_table[x, y])  # Explotar

def get_next_state(state, action, env):
    """Calcula el siguiente estado basado en la acción."""
    x, y = state
    if action == 0:  # up
        x = max(0, x - 1)
    elif action == 1:  # down
        x = min(env.shape[0] - 1, x + 1)
    elif action == 2:  # left
        y = max(0, y - 1)
    elif action == 3:  # right
        y = min(env.shape[1] - 1, y + 1)
    return (x, y)

def get_reward(state, env):
    """Devuelve la recompensa asociada a un estado."""
    x, y = state
    return recompensas[env[x, y]]

def is_terminal(state, env):
    """Verifica si el estado es terminal (meta)."""
    x, y = state
    return env[x, y] == 2


def sarsa(num_episodes, env):
    q_table = initialize_q_table(env)
    for episode in range(num_episodes):
        state = (2, 2)  # Posición inicial del ladrón
        action = choose_action(q_table, state, epsilon)

        while True:
            next_state = get_next_state(state, actions[action], env)
            reward = get_reward(next_state, env)
            next_action = choose_action(q_table, next_state, epsilon)

            x, y = state
            nx, ny = next_state

            # Actualización SARSA
            q_table[x, y, action] += alpha * (
                reward + gamma * q_table[nx, ny, next_action] - q_table[x, y, action]
            )

            state, action = next_state, next_action

            if is_terminal(state, env):  
                break
    return q_table

def q_learning(num_episodes, env):
    """Implementación del algoritmo Q-Learning."""
    q_table = initialize_q_table(env)

    for episode in range(num_episodes):
        state = (0, 0)  # Estado inicial

        while not is_terminal(state, env):
            action = choose_action(q_table, state, epsilon)
            next_state = get_next_state(state, action, env)
            reward = get_reward(next_state, env)

            x, y = state
            nx, ny = next_state

            # Actualización de Q-Learning
            q_table[x, y, action] += alpha * (
                reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
            )

            state = next_state

    return q_table

def format_q_table(q_table):
    formatted_dict = {}
    index = 0  # Contador de celdas

    for i in range(q_table.shape[0]):
        for j in range(q_table.shape[1]):
            # Obtén la acción con el valor máximo y construye la lista binaria
            max_action = np.argmax(q_table[i, j])
            action_list = [1 if k == max_action else 0 for k in range(q_table.shape[2])]

            # Agrega al diccionario con el índice de celda como clave
            formatted_dict[index] = action_list
            index += 1

    return formatted_dict


# Ejecución del algoritmo
num_episodes = 1000
q_tableQLearning = q_learning(num_episodes, entorno)
q_tableSarsa =sarsa(num_episodes, entorno)

# Mostrar resultados
print("\nTabla Q Q_Learning:")
Q_learning = format_q_table(q_tableQLearning)
print(Q_learning)
# Mostrar resultados
print("\nTabla Q final Sarsa:")
Sarsa = format_q_table(q_tableSarsa)
print(Sarsa)
