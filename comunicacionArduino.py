import serial
import time
# Confiqguración del puerto serial
PORT = "COM8"  # Cambia esto según el puerto asignado al Bluetooth
BAUD_RATE = 115200  # Velocidad de comunicación
# Crear la conexión serial
try:
    bt_connection = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"Conectado al puerto {PORT}")
    time.sleep(2)  # Esperar a que el módulo Bluetooth esté listo
except Exception as e:
    print(f"Error al conectar con el puerto {PORT}: {e}")
    exit()
# Función para enviar un comando
def send_command(command):
    if bt_connection.is_open:
        bt_connection.write(command.encode('utf-8'))  # Enviar el comando como bytes
        print(f"Comando enviado: {command}")
        time.sleep(0.1)  # Pausa breve para evitar congestión
    else:
        print("La conexión serial no está abierta")
        
def get_environment_data():
    """
    Obtiene datos del Arduino que representan información del entorno.
    Devuelve las posiciones iniciales y otros parámetros configurables.
    """
    try:
        response = read_from_arduino()
        if response:
            # Interpretar los datos del Arduino. Por ejemplo: "x:0,y:0,alpha:0.1"
            data = {}
            for item in response.split(","):
                key, value = item.split(":")
                data[key] = float(value) if "." in value else int(value)
            return data
    except Exception as e:
        print(f"Error al leer datos del entorno: {e}")
    return None

# Función para leer datos del Arduino
def read_from_arduino():
    print("verificando arduino")
    if bt_connection.is_open and bt_connection.in_waiting > 0:
        data = bt_connection.readline().decode('utf-8').strip()
        return data
    return None
