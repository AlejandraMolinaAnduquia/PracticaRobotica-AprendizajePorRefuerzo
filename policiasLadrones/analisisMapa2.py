import cv2
import numpy as np
import random
import math
import requests
import time
import diferenciaTemporalParcial2

from comunicacionArduino2 import send_command

# URL del servidor
SERVER_URL = "http://127.0.0.1:5000"  # Cambia la IP si es necesario
# Parámetros de la cuadrícula
rows = 4  # Número de filas
cols = 4  # Número de columnas
thickness = 1  # Grosor de las líneas

# Valores iniciales de Canny
canny_threshold1 = 50
canny_threshold2 = 150

politica_anterior = 3
politica_actual = 3

margenX =0
margenY =0


        
def mover_robot(tablaQ, cell_index,cell_indexEnemigo, x, y,angulo, cell_width, cell_height, politica_actual, politica_anterior,center_x, center_y, rol):
    tolerancia=20
    accion = np.argmax(tablaQ[(rol, cell_index,cell_indexEnemigo)])
    
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
    
# environment = np.array([
# [0, 0, 3, 0],
# [0, 2, 0, 0],
# [0, 0, 1, 0],
# [0, 0, 0, 4]
# ])

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
            print("Laberinto recibido:")
            for row in maze:
                print(row)
            return maze
        else:
            print(f"Error al conectarse al servidor: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return []

maze=np.array(get_maze())
#maze[-1, -1] = 2  # Definir la meta en la última casilla
def policiasLadronesSARSA(numero):
    
    #funcion de crear maoa en diferenciaTemporoalParcial que cree el mapa con las posiciones de:  0: Espacio vacío (camino)
    # 1: Posición inicial del ladrón
    # 2: Posición inicial del policía, 3: Lava (estado terminal negativo), 
    # 4: Salida segura (estado terminal positivo)(siempre la ultima casilla)
    global maze
    # Convertir maze en un array de NumPy
    maze_np = np.array(maze)
    if not isinstance(maze_np, np.ndarray):
        raise ValueError("El laberinto no es una matriz de NumPy")
    # Definir la meta en la última casilla
    print("Mapa:",maze)
    print("matriz?",isinstance(maze_np, np.ndarray))
    maze_np[-1, -1] = 2
    print("matriz",maze_np)
    q_table_sarsa = diferenciaTemporalParcial2.sarsa(numero, maze_np)
    print("Tabla Q final SARSA:",q_table_sarsa)
    formatted_q_table = diferenciaTemporalParcial2.format_q_table(q_table_sarsa,maze_np)
    print("Tabla Q (Formato Solicitado):")
    print(formatted_q_table)
    return formatted_q_table

def policiasLadronesLearning(numero):
    
    #funcion de crear maoa en diferenciaTemporoalParcial que cree el mapa con las posiciones de:  0: Espacio vacío (camino)
    # 1: Posición inicial del ladrón
    # 2: Posición inicial del policía, 3: Lava (estado terminal negativo), 
    # 4: Salida segura (estado terminal positivo)(siempre la ultima casilla)
    global maze
    # Convertir maze en un array de NumPy
    maze_np = np.array(maze)
    
    # Definir la meta en la última casilla
    maze_np[-1, -1] = 2
    q_table_learning = diferenciaTemporalParcial2.pqLearning(numero, maze_np)
    print("Tabla Q final Learning:",q_table_learning)
    #formatted_q_table = diferenciaTemporalParcial2.format_q_table(q_table_learning,maze_np)
    print("Tabla Q_learning:")
    #print(formatted_q_table)
    return q_table_learning


def get_detect_shapes():
    """
    Realiza una petición GET a la API /detect_shapes del servidor.

    Returns:
        list: Lista de formas detectadas en formato JSON.
    """
    try:
        response = requests.get(SERVER_URL+"/detect_shapes")
        if response.status_code == 200:
            shapes = response.json()  # Convertir la respuesta JSON a un diccionario
            return shapes
        else:
            print(f"Error al conectarse al servidor: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return []

num = 100  # Número de episodios
politica= policiasLadronesLearning(num)

# Lógica principal
contador = 0
politica_actual = None
politica_anterior = None

while True:
    contador += 1

    # Obtener formas detectadas desde el servidor
    detected_shapes = get_detect_shapes()
    maze = get_maze()
    print("Formas detectadas:", detected_shapes)
    
    if not detected_shapes:
        print("No se detectaron formas. Reintentando...")
        time.sleep(1)  # Esperar 1 segundo antes de reintentar
        continue

    # Procesar las formas detectadas cada 50 iteraciones
    if contador % 5 == 0:
        for shape in detected_shapes:
            # Extraer atributos de la forma detectada
            # Obtenemos todos los cell_index de las formas detectadas
            cell_index = [shape["cell_index"] for shape in detected_shapes]
            shape_type = [shape["shape"] for shape in detected_shapes]
            print("\nshape ",shape_type,"\n")

            # Imprimimos la lista completa (por ejemplo, si hay 2 cell_index)
            print("\ncelda\n",cell_index)
            x = [shape["x"] for shape in detected_shapes]
            y = [shape["y"] for shape in detected_shapes]
            center_x = [shape["cell_center_x"] for shape in detected_shapes]
            center_y = [shape["cell_center_y"] for shape in detected_shapes]
            angulo = [shape["angle"] for shape in detected_shapes]
            cell_width = [shape["cell_width"] for shape in detected_shapes]
            cell_height = [shape["cell_height"] for shape in detected_shapes]
            rol = [shape["role"] for shape in detected_shapes]
            #numeroRol=8
            if len(shape_type) == 2:
                if 8 in shape_type:
                    index = shape_type.index(8)
                    cell_indexm = cell_index[index]
                    x = x[index]
                    y = y[index]
                    center_x = center_x[index]
                    center_y = center_y[index]
                    angulo = angulo[index]
                    cell_width = cell_width[index]
                    cell_height = cell_height[index]
                    rol = rol[index]
                    print("cell_index", cell_indexm)
                    print("x", x)
                    print("y", y)
                    print("center_x", center_x)
                    print("center_y", center_y)
                    print("angulo", angulo)
                    print("cell_width", cell_width)
                    print("cell_height", cell_height)
                    print("rol", rol)
                    indexEnemigo = shape_type.index(9)
                    print("indexEnemigo", indexEnemigo)
                    cell_indexEnemigo = cell_index[indexEnemigo]
                    print("cell_indexEnemigo", cell_indexEnemigo)

                

                # Lógica para mover el robot usando SARSA y Q-Learning
                politica_actual, politica_anterior = mover_robot(
                    politica, cell_indexm,cell_indexEnemigo, x, y, angulo,
                    cell_width, cell_height, politica_actual, politica_anterior,
                    center_x, center_y, rol
                )
                # politica_actual, politica_anterior = mover_robot(
                #     policiasLadronesLearning(num), cell_index, x, y, angulo,
                #     cell_width, cell_height, politica_actual, politica_anterior,
                #     center_x, center_y
                # )

    time.sleep(0.1)  # Breve pausa para no saturar el servidor