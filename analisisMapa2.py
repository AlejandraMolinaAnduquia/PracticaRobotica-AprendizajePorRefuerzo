import cv2
import numpy as np
import random
import requests
import math

from comunicacionArduino import send_command
import diferenciaTemporalParcial
SERVER_URL = "http://127.0.0.1:5000"  # Cambia la IP si es necesario

# URL de DroidCam
url = "http://192.168.137.221:4747/video"  # Abrir la cámara
# Parámetros de la cuadrícula
rows = 5  # Número de filas
cols = 5  # Número de columnas
thickness = 1  # Grosor de las líneas

# Valores iniciales de Canny
canny_threshold1 = 50
canny_threshold2 = 150

politica_anterior = 3
politica_actual = 3

margenX =0
margenY =0



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

def calculate_angle(points):
    """
    Calcula el ángulo de inclinación en grados de un código QR dado.
    Se basa en las coordenadas de las esquinas.
    """
    # Extraer las coordenadas de las esquinas superiores izquierda y derecha
    top_left = points[0]
    top_right = points[1]

    # Calcular el ángulo en radianes
    delta_y = top_right[1] - top_left[1]
    delta_x = top_right[0] - top_left[0]
    angle = np.arctan2(delta_y, delta_x)  # Ángulo en radianes

    # Convertir a grados
    return np.degrees(angle)

def normalize_angle(angle):
    """
    Normaliza el ángulo para que esté entre 0° y 360°.
    El ángulo aumenta en sentido contrario a las manecillas del reloj.
    """
    angle = angle % 360  # Asegura que el ángulo esté dentro del rango [0, 360)
    if angle < 0:
        angle += 360  # Convertir a un ángulo positivo
    return angle

def detect_shapes_in_image(image, rows, cols, qr_detector):
    detected_shapes = []

    # Detectar y decodificar un solo código QR
    data, points, _ = qr_detector.detectAndDecode(image)

    if points is not None:
        points = points.reshape((-1, 2)).astype(int)

        # Dibujar los recuadros alrededor del código QR
        for i in range(len(points)):
            cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 3)

        # Calcular la inclinación
        angle = calculate_angle(points)

        # Normalizar el ángulo para que esté en el rango [0, 360]
        angle = normalize_angle(angle)

        # Calcular el centro del QR
        qr_center_x = int(np.mean(points[:, 0]))
        qr_center_y = int(np.mean(points[:, 1]))
        qr_center = (qr_center_x, qr_center_y)

        # Calcular la fila y columna de la cuadrícula
        height, width = image.shape[:2]
        cell_width = width / cols
        cell_height = height / rows

        # Calcular en qué celda (fila, columna) se encuentra el centro del QR
        row = int(qr_center_y // cell_height)
        col = int(qr_center_x // cell_width)

        # Calcular el centro de la celda
        cell_center_x = (col + 0.5) * cell_width
        cell_center_x=cell_center_x//1

        cell_center_y = (row + 0.5) * cell_height
        cell_center_y = cell_center_y//1
        cell_center = (cell_center_x, cell_center_y)

        # Flecha indicando cero grados (horizontal a la derecha) desde el centro
        arrow_tip_zero = (qr_center_x + 50, qr_center_y)  # Flecha hacia la derecha (0°)
        cv2.arrowedLine(image, qr_center, arrow_tip_zero, (0, 0, 255), 2, tipLength=0.3)

        # Flecha azul indicando el ángulo detectado
        # Convertir el ángulo a radianes para calcular la dirección de la flecha azul
        angle_rad = np.radians(angle)
        arrow_tip_blue = (int(qr_center_x + 100 * np.cos(angle_rad)), int(qr_center_y + 100 * np.sin(angle_rad)))
        cv2.arrowedLine(image, qr_center, arrow_tip_blue, (255, 0, 0), 2, tipLength=0.3)

        # Mostrar los datos y la inclinación en pantalla

        if data:
            #cv2.putText(image, f"QR: {data}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            pass
        angle2 = 360 - angle


        # Guardar los resultados con la fila y columna
        cell_center_x = math.floor(cell_center[0])

        cell_center_y = math.floor(cell_center[1])
        center_x=qr_center[0]
        center_y=qr_center[1]
        cell_index = row * cols + col  # Índice de la celda
        detected_shapes.append({
            "shape": data,
            "angle": angle2,
            "x":qr_center[0],
            "y": qr_center[1],
            "cell_center_x": cell_center_x,
            "cell_center_y": cell_center_y,
            "cell_index":cell_index,
            "row": row,
            "col": col,
            "cell_width":cell_width,
            "cell_height": cell_height,
        })
        cv2.putText(
            image,
            f"{cell_index}",
            (center_x - 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        cv2.putText(
            image,
            f"{center_x},{center_y}",
            (center_x - 30, center_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        cv2.putText(image, f"{angle2:.2f}'' ", (center_x-30, center_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                   2,
                    cv2.LINE_AA)


        image = draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height)
    return detected_shapes, image
def draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height):
    """Dibuja una línea punteada roja dentro de la celda en los ejes del centro de la celda."""
    # Definir los límites de la celda
    cell_left = int(cell_center_x - cell_width // 2)
    cell_right = int(cell_center_x + cell_width // 2)
    cell_top = int(cell_center_y - cell_height // 2)
    cell_bottom = int(cell_center_y + cell_height // 2)

    # Dibujar línea punteada roja en el eje horizontal

    for x in range(cell_left, cell_right, 10):  # Incremento para punteado
        cv2.line(image, (x, cell_center_y), (x + 5, cell_center_y), (0, 0, 255), 1)

    # Dibujar línea punteada roja en el eje vertical
    for y in range(cell_top, cell_bottom, 10):  # Incremento para punteado
        cv2.line(image, (cell_center_x, y), (cell_center_x, y + 5), (0, 0, 255), 1)
    return image
def fill_cells(frame, matrix, alpha=0.7):
    """Rellena de color negro translúcido los cuadrantes correspondientes a los valores '1' en la matriz."""
    rows, cols = len(matrix), len(matrix[0])
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    overlay = frame.copy()  # Hacemos una copia para aplicar el color translúcido

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                # Coordenadas del cuadrante
                x1, y1 = j * cell_width, i * cell_height
                x2, y2 = x1 + cell_width, y1 + cell_height
                # Rellenar el cuadrante con color negro (translúcido)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # Aplicar transparencia a los rectángulos negros
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame
def highlight_start_end(frame, rows, cols):
    """Colorea en translúcido verde (0,0) y rojo (rows-1, cols-1)."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    # Coordenadas del inicio (0, 0)
    x1_start, y1_start = 0, 0
    x2_start, y2_start = cell_width, cell_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1_start, y1_start), (x2_start, y2_start), (0, 255, 0), -1)  # Verde

    # Coordenadas del final (rows-1, cols-1)
    x1_end, y1_end = (cols - 1) * cell_width, (rows - 1) * cell_height
    x2_end, y2_end = x1_end + cell_width, y1_end + cell_height
    cv2.rectangle(overlay, (x1_end, y1_end), (x2_end, y2_end), (255, 0, 0), -1)  # Rojo

    # Agregar transparencia
    alpha = 0.5  # Nivel de transparencia
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def on_trackbar_change(x):
    """Callback para manejar los cambios en las trackbars."""
    pass

        
def mover_robot(tablaQ, cell_index, x, y,angulo, cell_width, cell_height, politica_actual, politica_anterior,center_x, center_y):
    tolerancia=20
    accion = np.argmax(tablaQ[cell_index])
    
    print(f"Acción: {accion}")
    print(f"Ángulo: {angulo}")
    
    print("politica")
    print(tablaQ)
    
    # Definir los ángulos de destino para cada acción
    angulos_destino = {
        0: 90,   # Arriba
        1: 270,  # Abajo
        2: 180,  # Izquierda
        3: 0     # Derecha
    }
    
    # Obtener el ángulo destino según la acción
    angulo_deseado = angulos_destino[accion]
    print(f"Ángulo deseado: {angulo_deseado}")
    
    # Calcular la diferencia mínima entre el ángulo actual y el deseado
    diferencia = (angulo_deseado - angulo + 360) % 360  # Diferencia positiva en rango [0, 360)
    
    politica_actual=accion
    margenX=cell_width*0.3
    margenY=cell_height*0.3
    
    print("margen en x: ",margenX)
    print("margen en y: ", margenY)
    print("tamaño en x: ",cell_width)
    print("tamaño en y: ", cell_height)
    
    
    if politica_anterior != politica_actual:
        
        #centrar
        if ( center_x -margenX <= x <= center_x + margenX) and (center_y -margenY <= y <= center_y + margenY):
            politica_anterior=politica_actual
            
            
        else:
            send_command("w")
            send_command('w')
            print("calibrando")
    #if (accion == 0 and (angulo <= 90 + tolerancia and angulo >= 90 - tolerancia)) or (accion == 1 and (angulo <= 270 + tolerancia and angulo >= 270 - tolerancia)) or (accion == 2 and (angulo <= 180 + tolerancia and angulo >= 180 - tolerancia)) or (accion == 3 and (angulo <=  tolerancia or angulo >= 360 - tolerancia)):
    
    elif cell_index == rows*cols-1:
        print("legoooo")
        pass
    
    elif (accion == 0 and (angulo <= 90 + tolerancia and angulo >= 90 - tolerancia)):
        send_command('w')
        send_command('w')
        print("primer if")
        
    elif  (accion == 1 and (angulo <= 270 + tolerancia and angulo >=270 - tolerancia)):
        send_command('w')
        send_command('w')
        print("segundo if")
        
    elif (accion == 2 and (angulo <= 180 + tolerancia and angulo >= 180 - tolerancia)):
        send_command('w')
        send_command('w')
        print("tercero if")
        
    elif (accion == 3 and (angulo <=  tolerancia or angulo >= 360 - tolerancia)):
        send_command('w')
        send_command('w')
        print("cuarto if")
        
    else:
        if diferencia > 180:
            diferencia -= 360  # Convertir a rango [-180, 180] para rotaciones negativas
        print(f"Diferencia de ángulo: {diferencia}")

        # Si el ángulo actual está alineado dentro de la tolerancia, avanza
        if abs(diferencia) <= tolerancia:
            send_command('w')  # Avanzar
            send_command('w')
            print("Avanzando hacia la celda")
        else:
            # Girar a la dirección más corta
            if diferencia > 0:
                send_command('a')  # Girar a la izquierda
                send_command('a')
                send_command('a')
                print("Girando a la izquierda")
            else:
                send_command('d')  # Girar a la derecha
                send_command('d')
                send_command('d')
                print("Girando a la derecha")
        
            
    return politica_actual, politica_anterior
    
def policiasLadronesSARSA(numero,maze):
    
    #funcion de crear maoa en diferenciaTemporoalParcial que cree el mapa con las posiciones de:  0: Espacio vacío (camino)
    # 1: Posición inicial del ladrón
    # 2: Posición inicial del policía, 3: Lava (estado terminal negativo), 
    # 4: Salida segura (estado terminal positivo)(siempre la ultima casilla)
    # Convertir maze en un array de NumPy
    # Convertir maze en un array de NumPy
    maze_np = np.array(maze)
    
    # Definir la meta en la última casilla
    maze_np[-1, -1] = 2
    #print("\n matriz", maze_np, "\n")
    q_table_sarsa = diferenciaTemporalParcial.sarsa(numero, maze_np)
    #print("Tabla Q final Learning:",q_table_sarsa)
    formatted_q_table = diferenciaTemporalParcial.format_q_table(q_table_sarsa,maze_np)
    #print("Tabla Q_learning:")
    print(formatted_q_table)
    return formatted_q_table

def policiasLadronesLearning(numero,maze):
    
    #funcion de crear maoa en diferenciaTemporoalParcial que cree el mapa con las posiciones de:  0: Espacio vacío (camino)
    # 1: Posición inicial del ladrón
    # 2: Posición inicial del policía, 3: Lava (estado terminal negativo), 
    # 4: Salida segura (estado terminal positivo)(siempre la ultima casilla)
    
    # Convertir maze en un array de NumPy
    maze_np = np.array(maze)
    
    # Definir la meta en la última casilla
    maze_np[-1, -1] = 2
    #print("\n matriz", maze_np, "\n")
    q_table_learning = diferenciaTemporalParcial.q_learning(numero, maze_np)
    #print("Tabla Q final Learning:",q_table_learning)
    formatted_q_table = diferenciaTemporalParcial.format_q_table(q_table_learning,maze_np)
    print("Tabla Q_learning:")
    print(formatted_q_table)
    return formatted_q_table

def get_maze():
    """
    Realiza una petición GET a la API /maze del servidor.

    Returns:
        list: Matriz del laberinto en formato JSON.
    """
    try:
        response = requests.get(SERVER_URL+"/maze")
        if response.status_code == 200:
            maze = response.json()  # Convertir la respuesta JSON a una lista
            # print("Laberinto recibido:")
            # for row in maze:
            #     print(row)
            return maze
        else:
            print(f"Error al conectarse al servidor: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return []

# Abre el video desde la URL
cap = cv2.VideoCapture(url)
#cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo conectar a la cámara en la URL proporcionada.")
else:
    print(f"Conexión exitosa. Analizando video con cuadrícula de {rows}x{cols}...")

    # Crear ventana y trackbars
    cv2.namedWindow('Ajustes')
    cv2.createTrackbar('Canny Th1', 'Ajustes', canny_threshold1, 255, on_trackbar_change)
    cv2.createTrackbar('Canny Th2', 'Ajustes', canny_threshold2, 255, on_trackbar_change)
    cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, on_trackbar_change)
    #maze = maze_generate(rows, cols)
    maze=[[0,0,0,1,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0]]

    #print(maze)
    qr_detector = cv2.QRCodeDetector()
    
    maze=get_maze()
    
    contador=0
    while True:
        contador +=1
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video.")
            break

        # Obtener valores de las trackbars
        threshold1 = cv2.getTrackbarPos('Canny Th1', 'Ajustes')
        threshold2 = cv2.getTrackbarPos('Canny Th2', 'Ajustes')
        dilatacion = cv2.getTrackbarPos('Dilatacion', 'Ajustes')

        # Analizar el frame con los umbrales ajustados
        detected_shapes, frame_with_shapes = detect_shapes_in_image(frame, rows, cols, qr_detector)
        #detected_shapes=[{"shape": "triangle","row":1,"col": 0,"cell_index": 3,"x": 100,"y": 100}]
        
        if contador% 24==0:
            for shape in detected_shapes:
                # Obtener las coordenadas y llamar a mover_robot
                cell_index = shape["cell_index"]
                x = shape["x"]
                y = shape["y"]
                center_x= shape["cell_center_x"]
                center_y= shape["cell_center_y"]
                angulo = shape["angle"]
                cell_width = shape["cell_width"]
                cell_height = shape["cell_height"]
                num=1000
                politica_actual, politica_anterior= mover_robot(policiasLadronesLearning(num,maze),cell_index,x,y,angulo,cell_width, cell_height, politica_actual, politica_anterior,center_x, center_y)
               
        #print(detected_shapes)
        # Dibujar la cuadrícula en el frame
        frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)

        frame=fill_cells(frame_with_grid,maze)
        frame = highlight_start_end(frame, rows, cols)
        # Mostrar el frame con los ajustes
        cv2.imshow('Cuadrícula con análisis', frame_with_grid)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera recursos
cap.release()
cv2.destroyAllWindows()