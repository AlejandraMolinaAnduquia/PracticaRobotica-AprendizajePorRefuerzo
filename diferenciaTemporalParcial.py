import numpy as np
import random

# --- MAPA ESTÁTICO ---
# Representación del entorno
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

def maze_generate(filas, columnas):
    """
    Genera un laberinto de dimensiones filas x columnas.
    Los caminos están representados por 0 y las paredes por 1.
    Garantiza que (0,0) es el inicio y (filas-1,columnas-1) es la meta con un camino solucionable.
    """
    # Crear una matriz llena de paredes (1)
    laberinto = [[1 for _ in range(columnas)] for _ in range(filas)]

    # Direcciones de movimiento: (dx, dy) para celdas ortogonales
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def en_rango(x, y):
        """Verifica si una celda está dentro del rango del laberinto."""
        return 0 <= x < filas and 0 <= y < columnas

    def dfs(x, y):
        """Algoritmo DFS para construir el laberinto."""
        laberinto[x][y] = 0  # Marca el camino actual como "camino"
        random.shuffle(direcciones)  # Aleatoriza el orden de las direcciones
        for dx, dy in direcciones:
            nx, ny = x + 2 * dx, y + 2 * dy  # Saltar una celda para garantizar paredes entre caminos
            if en_rango(nx, ny) and laberinto[nx][ny] == 1:  # Si es una celda válida y no visitada
                # Romper la pared entre la celda actual y la siguiente
                laberinto[x + dx][y + dy] = 0
                # Continuar el DFS desde la celda siguiente
                dfs(nx, ny)

    # Inicializar el laberinto
    laberinto[0][0] = 0  # Crear la entrada
    dfs(0, 0)

    # Crear la salida
    laberinto[filas - 1][columnas - 1] = 0  # Asegurar que el punto final sea siempre un camino

    # Conectar la salida al camino más cercano si está aislada
    if laberinto[filas - 2][columnas - 1] == 1 and laberinto[filas - 1][columnas - 2] == 1:
        laberinto[filas - 2][columnas - 1] = 0  # Romper la pared superior

    # Devolver la matriz del laberinto
    print(laberinto)
    return laberinto

def format_q_table(q_table):
    """
    Formatea la tabla Q en el formato especificado con valores originales.

    Args:
        q_table (numpy.ndarray): La tabla Q con dimensiones (filas, columnas, acciones).

    Returns:
        str: Una cadena que representa la tabla Q en el formato solicitado.
    """
    formatted_output = ""
    index = 0  # Contador de celdas

    for i in range(q_table.shape[0]):
        for j in range(q_table.shape[1]):
            # Obtén los valores originales de las acciones
            action_values = q_table[i, j]

            # Formatea la salida
            formatted_output += f"{index}: {list(action_values)}, "
            index += 1

            # Salto de línea cada 3 celdas (opcional, ajustado al mapa 3x3)
            if index % q_table.shape[1] == 0:
                formatted_output = formatted_output.rstrip(", ") + "\n"

    return formatted_output.strip()


# --- PARÁMETROS DE ENTRENAMIENTO ---
actions = ['up', 'down', 'left', 'right']  # Conjunto de acciones posibles
alpha = 0.1   # Tasa de aprendizaje
gamma = 0.9   # Factor de descuento
epsilon = 0.1 # Factor de exploración

def calculate_states(env):
    """Calcula el número de estados basado en el tamaño del mapa."""
    return env.shape[0] * env.shape[1]

def initialize_q_table(env):
    """Inicializa una tabla Q con ceros."""
    return np.zeros((env.shape[0], env.shape[1], len(actions)))

# --- FUNCIONES AUXILIARES ---
def get_next_state(state, action, env):
    """Calcula el siguiente estado basado en la acción."""
    x, y = state
    if action == 'up':
        x = max(0, x - 1)
    elif action == 'down':
        x = min(env.shape[0] - 1, x + 1)
    elif action == 'left':
        y = max(0, y - 1)
    elif action == 'right':
        y = min(env.shape[1] - 1, y + 1)
    return (x, y)

def get_reward(state, env):
    """Devuelve la recompensa asociada a un estado."""
    x, y = state
    if env[x, y] == 3:  # Lava
        return -10
    elif env[x, y] == 4:  # Salida segura
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

            if env[state] in [3, 4]:  # Estado terminal
                break
    return q_table

# --- ALGORITMO Q-LEARNING ---
def q_learning(num_episodes, env):
    q_table = initialize_q_table(env)
    for episode in range(num_episodes):
        state = (2, 2)  # Posición inicial del ladrón

        while True:
            action = choose_action(q_table, state, epsilon)
            next_state = get_next_state(state, actions[action], env)
            reward = get_reward(next_state, env)

            x, y = state
            nx, ny = next_state

            # Actualización Q-Learning
            q_table[x, y, action] += alpha * (
                reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
            )

            state = next_state

            if env[state] in [3, 4]:  # Estado terminal
                break
    return q_table

# --- EJECUCIÓN Y VISUALIZACIÓN ---
# rows=3
# cols=3
# num_episodes = 100000
# #environment=np.array(maze_generate(rows, cols))

# num_states = calculate_states(environment)
# print(f"Número de estados: {num_states}")

# q_table_sarsa = sarsa(num_episodes, environment)
# #q_table_qlearning = q_learning(num_episodes, environment)

# print("\nTabla Q (SARSA):")
# print(q_table_sarsa)

# #print("\nTabla Q (Q-Learning):")
# #print(q_table_qlearning)

# formatted_q_table = format_q_table(q_table_sarsa)
# print("Tabla Q (Formato Solicitado):")
# print(formatted_q_table)