import numpy as np
import random

# --- MAPA ESTÁTICO ---
# Representación del entorno (4x4 grid)
# 0: Espacio vacío
# 1: Posición inicial del ladrón
# 2: Posición inicial del policía
# 3: Lava (estado terminal negativo)
# 4: Salida segura (estado terminal positivo)

environment = np.array([
    [0, 0, 3, 0],
    [0, 2, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 4]
])

# --- PARÁMETROS DE ENTRENAMIENTO ---
actions = ['up', 'down', 'left', 'right']  # Conjunto de acciones posibles
alpha = 0.1   # Tasa de aprendizaje
gamma = 0.9   # Factor de descuento
epsilon = 0.1 # Factor de exploración

def initialize_q_table():
    """Inicializa una tabla Q con ceros."""
    return np.zeros((environment.shape[0], environment.shape[1], len(actions)))

# --- FUNCIONES AUXILIARES ---
def get_next_state(state, action):
    """Calcula el siguiente estado basado en la acción."""
    x, y = state
    if action == 'up':
        x = max(0, x - 1)
    elif action == 'down':
        x = min(environment.shape[0] - 1, x + 1)
    elif action == 'left':
        y = max(0, y - 1)
    elif action == 'right':
        y = min(environment.shape[1] - 1, y + 1)
    return (x, y)

def get_reward(state):
    """Devuelve la recompensa asociada a un estado."""
    x, y = state
    if environment[x, y] == 3:  # Lava
        return -10
    elif environment[x, y] == 4:  # Salida segura
        return 10
    else:
        return -1  # Penalización por cada paso

def choose_action(q_table, state, epsilon):
    """Elige una acción usando epsilon-greedy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))  # Explorar
    else:
        x, y = state
        return np.argmax(q_table[x, y])  # Explotar

# --- ALGORITMO SARSA ---
def sarsa(num_episodes):
    q_table = initialize_q_table()
    for episode in range(num_episodes):
        state = (2, 2)  # Posición inicial del ladrón
        action = choose_action(q_table, state, epsilon)

        while True:
            next_state = get_next_state(state, actions[action])
            reward = get_reward(next_state)
            next_action = choose_action(q_table, next_state, epsilon)

            x, y = state
            nx, ny = next_state

            # Actualización SARSA
            q_table[x, y, action] += alpha * (
                reward + gamma * q_table[nx, ny, next_action] - q_table[x, y, action]
            )

            state, action = next_state, next_action

            if environment[state] in [3, 4]:  # Estado terminal
                break
    return q_table

# --- ALGORITMO Q-LEARNING ---
def q_learning(num_episodes):
    q_table = initialize_q_table()
    for episode in range(num_episodes):
        state = (2, 2)  # Posición inicial del ladrón

        while True:
            action = choose_action(q_table, state, epsilon)
            next_state = get_next_state(state, actions[action])
            reward = get_reward(next_state)

            x, y = state
            nx, ny = next_state

            # Actualización Q-Learning
            q_table[x, y, action] += alpha * (
                reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
            )

            state = next_state

            if environment[state] in [3, 4]:  # Estado terminal
                break
    return q_table

# --- EJECUCIÓN Y VISUALIZACIÓN ---
num_episodes = 1000
q_table_sarsa = sarsa(num_episodes)
q_table_qlearning = q_learning(num_episodes)

print("\nTabla Q (SARSA):")
print(q_table_sarsa)

print("\nTabla Q (Q-Learning):")
print(q_table_qlearning)
