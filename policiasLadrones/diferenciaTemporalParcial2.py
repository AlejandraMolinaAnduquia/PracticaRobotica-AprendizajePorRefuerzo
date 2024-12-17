import numpy as np
import random

# Parámetros globales
alpha = 0.1   # Tasa de aprendizaje
gamma = 0.9   # Factor de descuento
epsilon = 0.1 # Factor de exploración

# Matriz del terreno: 0 = camino, 1 = obstáculo (roca), la última casilla es la meta
# entorno = np.array([[0,0,0,1,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0]])
# entorno[-1, -1] = 2  # Definir la meta en la última casilla


# print(entorno)
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
    return np.zeros((env.shape[0] * env.shape[1], len(actions)))

def state_to_index(state, env):
    """Convierte un estado (x, y) en un índice lineal para la tabla Q."""
    x, y = state
    return x * env.shape[1] + y

def choose_action(q_table, state, epsilon, env):
    """Elige una acción usando epsilon-greedy."""
    state_index = state_to_index(state, env)
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # Explorar
    else:
        return np.argmax(q_table[state_index])  # Explotar

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
        action = choose_action(q_table, state, epsilon, env)

        while True:
            state_index = state_to_index(state, env)
            next_state = get_next_state(state, action, env)
            reward = get_reward(next_state, env)
            next_action = choose_action(q_table, next_state, epsilon, env)

            next_state_index = state_to_index(next_state, env)

            # Actualización SARSA
            q_table[state_index, action] += alpha * (
                reward + gamma * q_table[next_state_index, next_action] - q_table[state_index, action]
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
            state_index = state_to_index(state, env)
            action = choose_action(q_table, state, epsilon, env)
            next_state = get_next_state(state, action, env)
            reward = get_reward(next_state, env)

            next_state_index = state_to_index(next_state, env)

            # Actualización de Q-Learning
            q_table[state_index, action] += alpha * (
                reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action]
            )

            state = next_state

    return q_table

def format_q_table(q_table, env):
    formatted_dict = {}
    index = 0  # Contador de celdas

    for i in range(env.shape[0]):
        for j in range(env.shape[1]):
            state_index = state_to_index((i, j), env)
            # Obtén la acción con el valor máximo y construye la lista binaria
            max_action = np.argmax(q_table[state_index])
            action_list = [1 if k == max_action else 0 for k in range(len(actions))]

            # Agrega al diccionario con el índice de celda como clave
            formatted_dict[index] = action_list
            index += 1

    return formatted_dict

# Ejecución del algoritmo
# num_episodes = 1000
# q_tableQLearning = q_learning(num_episodes, entorno)
# q_tableSarsa = sarsa(num_episodes, entorno)

# # Mostrar resultados
# print("\nTabla Q Q_Learning:")
# Q_learning = format_q_table(q_tableQLearning, entorno)
# print(Q_learning)

# print("\nTabla Q final Sarsa:")
# Sarsa = format_q_table(q_tableSarsa, entorno)
# print(Sarsa)