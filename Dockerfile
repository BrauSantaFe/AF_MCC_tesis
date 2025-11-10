# Usa la imagen base de Python. Asegúrate de que la versión es compatible.
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copia e instala las dependencias. Esto se hace primero para aprovechar el caché de Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código (incluyendo modelo_cnn.py) al contenedor.
# La carpeta 'data' será ignorada gracias a .dockerignore.
COPY . .

# Define el comando de ejecución predeterminado.
# Tu script espera que 'data' esté en ./data, que en el contenedor es /app/data.
CMD ["python", "modelo_cnn.py"]
