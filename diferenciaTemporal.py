import numpy as np
import matplotlib.pyplot as plt

# Definición de los estados y las acciones
NUM_ESTADOS = 9  # Estados: 0 a 8
NUM_ACCIONES = 4  # Acciones: Izquierda, Arriba, Derecha, Abajo

# Probabilidades iniciales para cada acción en cada estado
probabilidades = {
    0: [0, 0, 1, 0], 1: [0, 0, 1, 0], 2: [0, 0, 0, 1],
    3: [0, 0, 0, 1], 4: [0, 0, 1, 0], 5: [0, 0, 0, 1],
    6: [0, 0, 1, 0], 7: [0, 0, 1, 0], 8: [0, 0, 0, 0]
}

# Inicialización de la tabla Q
def inicializar_Q(nS, nA, probabilidades):
    Q = np.zeros((nS, nA))
    for estado in probabilidades:
        Q[estado] = probabilidades[estado]
    return Q

# Lógica para obtener la celda del agente
def obtener_celda_actual(x, y, num_columnas):
    return x * num_columnas + y

# Función e-greedy
def e_greedy(s, Q, epsilon):
    if np.random.rand() >= epsilon:
        return np.argmax(Q[s])
    else:
        return np.random.randint(0, Q.shape[1])

# Algoritmo SARSA
def sarsa(Q, alpha, gamma, epsilon, episodios, x_inicial, y_inicial, num_filas, num_columnas):
    for _ in range(episodios):
        x, y = x_inicial, y_inicial
        estado = obtener_celda_actual(x, y, num_columnas)
        accion = e_greedy(estado, Q, epsilon)
        done = False
        
        while not done:
            # Simulación de movimiento según la acción
            nuevo_x, nuevo_y = movimiento(accion, x, y, num_filas, num_columnas)
            nuevo_estado = obtener_celda_actual(nuevo_x, nuevo_y, num_columnas)
            
            recompensa = calcular_recompensa(nuevo_estado)  # Implementa según lógica del entorno
            nueva_accion = e_greedy(nuevo_estado, Q, epsilon)
            
            Q[estado, accion] += alpha * (
                recompensa + gamma * Q[nuevo_estado, nueva_accion] - Q[estado, accion]
            )
            
            # Actualizar variables
            estado, accion = nuevo_estado, nueva_accion
            x, y = nuevo_x, nuevo_y
            
            # Determinar si llegamos a un estado terminal
            done = estado_terminal(estado)  # Implementa según lógica del entorno

    return Q

# Algoritmo Q-Learning
def qlearning(Q, alpha, gamma, epsilon, episodios, x_inicial, y_inicial, num_filas, num_columnas):
    for _ in range(episodios):
        x, y = x_inicial, y_inicial
        estado = obtener_celda_actual(x, y, num_columnas)
        done = False
        
        while not done:
            # Simulación de movimiento según la acción
            accion = e_greedy(estado, Q, epsilon)
            nuevo_x, nuevo_y = movimiento(accion, x, y, num_filas, num_columnas)
            nuevo_estado = obtener_celda_actual(nuevo_x, nuevo_y, num_columnas)
            
            recompensa = calcular_recompensa(nuevo_estado)  # Implementa según lógica del entorno
            
            Q[estado, accion] += alpha * (
                recompensa + gamma * np.max(Q[nuevo_estado]) - Q[estado, accion]
            )
            
            # Actualizar variables
            estado = nuevo_estado
            x, y = nuevo_x, nuevo_y
            
            # Determinar si llegamos a un estado terminal
            done = estado_terminal(estado)  # Implementa según lógica del entorno

    return Q

# Movimiento del agente (respetando los bordes del mapa)
def movimiento(accion, x, y, num_filas, num_columnas):
    if accion == 0 and y > 0:  # Izquierda
        y -= 1
    elif accion == 1 and x > 0:  # Arriba
        x -= 1
    elif accion == 2 and y < num_columnas - 1:  # Derecha
        y += 1
    elif accion == 3 and x < num_filas - 1:  # Abajo
        x += 1
    return x, y

# Estado terminal
def estado_terminal(estado):
    return estado == 8  # Por ejemplo, estado 8 es terminal

# Recompensa (ejemplo básico)
def calcular_recompensa(estado):
    if estado == 8:
        return 10  # Recompensa máxima
    return -1  # Penalización por cada paso

# -----------------------------------------------------
# Ejemplo de uso
num_filas, num_columnas = 3, 3
x_inicial, y_inicial = 0, 0
Q = inicializar_Q(NUM_ESTADOS, NUM_ACCIONES, probabilidades)

# Entrenamiento
Q_sarsa = sarsa(Q.copy(), alpha=0.1, gamma=0.9, epsilon=0.1, episodios=1000,
                x_inicial=x_inicial, y_inicial=y_inicial, 
                num_filas=num_filas, num_columnas=num_columnas)

Q_qlearning = qlearning(Q.copy(), alpha=0.1, gamma=0.9, epsilon=0.1, episodios=1000,
                        x_inicial=x_inicial, y_inicial=y_inicial, 
                        num_filas=num_filas, num_columnas=num_columnas)

print("Tabla Q final SARSA:")
print(Q_sarsa)

print("Tabla Q final Q-Learning:")
print(Q_qlearning)
