import cv2
#from comunicacionArduino import get_environment_data
import numpy as np
import random

# URL de DroidCam
url = "http://192.168.16.139:4747/video"

# Parámetros de la cuadrícula
rows = 7  # Número de filas
cols = 7  # Número de columnas
thickness = 1  # Grosor de las líneas

# Valores iniciales de Canny
canny_threshold1 = 50
canny_threshold2 = 150
explored_map = [[-1 for _ in range(cols)] for _ in range(rows)]  # Inicializado con -1

def update_explored_map(x, y, terrain_type):
    """
    Actualiza la matriz explored_map con los datos confirmados del terreno.
    """
    if 0 <= x < rows and 0 <= y < cols:
        explored_map[x][y] = terrain_type
    else:
        print(f"Posición fuera de rango: ({x}, {y})")

# Obtén los datos del entorno desde Arduino
environment_data = get_environment_data()
if environment_data:
    x_inicial = environment_data.get("x_inicial", 0)
    y_inicial = environment_data.get("y_inicial", 0)
    alpha = environment_data.get("alpha", 0.1)
    gamma = environment_data.get("gamma", 0.9)
    epsilon = environment_data.get("epsilon", 0.1)
else:
    # Valores predeterminados en caso de error
    x_inicial, y_inicial = 0, 0
    alpha, gamma, epsilon = 0.1, 0.9, 0.1

def maze_generate(filas, columnas):
    """
    Genera un laberinto de dimensiones filas x columnas.
    Los caminos están representados por 0 y las paredes por 1.
    Garantiza que (0,0) es el inicio y (filas-1,columnas-1) es la meta con un camino solucionable.
    """
    laberinto = [[1 for _ in range(columnas)] for _ in range(filas)]
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def en_rango(x, y):
        """Verifica si una celda está dentro del rango del laberinto."""
        return 0 <= x < filas and 0 <= y < columnas

    def dfs(x, y):
        """Algoritmo DFS para construir el laberinto."""
        laberinto[x][y] = 0  # Marca el camino actual como "camino"
        random.shuffle(direcciones)  # Aleatoriza el orden de las direcciones
        for dx, dy in direcciones:
            nx, ny = x + 2 * dx, y + 2 * dy
            if en_rango(nx, ny) and laberinto[nx][ny] == 1:
                laberinto[x + dx][y + dy] = 0
                dfs(nx, ny)

    laberinto[0][0] = 0  # Crear la entrada
    dfs(0, 0)
    laberinto[filas - 1][columnas - 1] = 0  # Crear la salida
    if laberinto[filas - 2][columnas - 1] == 1 and laberinto[filas - 1][columnas - 2] == 1:
        laberinto[filas - 2][columnas - 1] = 0
    return laberinto

def draw_grid(frame, rows, cols, thickness=1):
    """Dibuja una cuadrícula en el frame."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols
    for i in range(1, rows):  # Líneas horizontales
        cv2.line(frame, (0, i * cell_height), (width, i * cell_height), (0, 255, 0), thickness)
    for j in range(1, cols):  # Líneas verticales
        cv2.line(frame, (j * cell_width, 0), (j * cell_width, height), (0, 255, 0), thickness)
    return frame


# Detección de formas en la imagen (función detect_shapes_in_image omitida aquí para mantener brevedad)
# Su contenido sigue igual que el anterior archivo enviado.

# Configuración inicial del laberinto
maze = maze_generate(rows, cols)

#Llamada a QLearning, 1 vacio y 0 camino
probabilidades = {
    0: [0, 0, 0, 1], 1: [0, 0, 0, 1], 2: [0, 1, 0, 0],
    3: [0, 1, 0, 0], 4: [0, 0, 0, 1], 5: [0, 1, 0, 0],
    6: [0, 0, 0, 1], 7: [0, 0, 0, 1], 8: [0, 0, 0, 0]
}
# la tabla la devuelve QLearning

# Conexión a la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo conectar a la cámara en la URL proporcionada.")
else:
    print(f"Conexión exitosa. Analizando video con cuadrícula de {rows}x{cols}...")

    # Crear ventana y trackbars
    cv2.namedWindow('Ajustes')
    cv2.createTrackbar('Canny Th1', 'Ajustes', canny_threshold1, 255, lambda x: None)
    cv2.createTrackbar('Canny Th2', 'Ajustes', canny_threshold2, 255, lambda x: None)
    cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame.")
            break

        # Procesamiento del frame
        frame_with_grid = draw_grid(frame.copy(), rows, cols)
        canny_threshold1 = cv2.getTrackbarPos('Canny Th1', 'Ajustes')
        canny_threshold2 = cv2.getTrackbarPos('Canny Th2', 'Ajustes')
        dilatacion = cv2.getTrackbarPos('Dilatacion', 'Ajustes')

        detected_shapes, processed_frame = detect_shapes_in_image(
            frame_with_grid, rows, cols, canny_threshold1, canny_threshold2, dilatacion
        )

        cv2.imshow('Procesado', processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
