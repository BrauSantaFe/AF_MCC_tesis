import os
import numpy as np
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# --- 1. Configuración ---

IM_SIZE = 256
# Ruta de la carpeta principal de datos
DATA_DIR = './data' 
CATEGORIES = ['fire', 'no_fire']
NUM_CLASSES = len(CATEGORIES)
CHANNELS = 3
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32



def load_data(data_dir, categories, im_size):
    """Carga imágenes TIFF desde las carpetas y las preprocesa."""
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        print(f"Cargando imágenes de: {category}...")
        for img_name in os.listdir(path):
            if img_name.endswith(('.tif', '.tiff')):
                try:
                    img_array = tiff.imread(os.path.join(path, img_name))
                    
                    if img_array.shape[:2] == (im_size, im_size):
                        # Normalización (asumiendo 16-bit Landsat)
                        normalized_img = img_array.astype('float32') / 65535.0
                        data.append([normalized_img, class_num])
                    
                except Exception as e:
                    # print(f"Error al leer/procesar la imagen {img_name}: {e}")
                    pass
    return np.array(data, dtype=object)

# Cargar los datos
all_data = load_data(DATA_DIR, CATEGORIES, IM_SIZE)

# Separar características (X) y etiquetas (y)
X = np.array([i[0] for i in all_data])
y = np.array([i[1] for i in all_data])

# Separar conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convertir etiquetas a codificación one-hot
y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

print("\n--- Estadísticas de los Datos ---")
print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de X_test: {X_test.shape}")

# --- 3. Ponderación de Clases (Por Desbalance) ---

# Calcular los pesos de clase para la ponderación inversa por frecuencia
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

print("\n--- Pesos de Clase Calculados para el Entrenamiento ---")
print(class_weights_dict)

# --- 4. Definición del Modelo CNN (con Tasa de Aprendizaje Ajustada) ---

def create_cnn_model(input_shape, num_classes, learning_rate):
    """Define y compila el modelo CNN."""
    model = Sequential([
        # Capa de Convolución 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Capa de Convolución 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Capa de Convolución 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Aplanar y Capas Densas
        Flatten(),
        Dense(128, activation='relu'),
        
        # Capa de Salida
        Dense(num_classes, activation='softmax')
    ])

    # Compilación con optimizador Adam y TASA DE APRENDIZAJE REDUCIDA
    adam_optimizer = Adam(learning_rate=learning_rate) 
    model.compile(optimizer=adam_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Crear el modelo
input_shape = (IM_SIZE, IM_SIZE, CHANNELS)
model = create_cnn_model(input_shape, NUM_CLASSES, LEARNING_RATE)
model.summary()

# --- 5. Aumentación de Datos y Entrenamiento ---

print("\n--- Configurando Aumentación de Datos y Entrenamiento ---")

# Generador de Aumentación de Datos
# Implementa espejos vertical y horizontal, así como otras transformaciones para robustecer el modelo.
datagen = ImageDataGenerator(
    horizontal_flip=True, # Mirror horizontal
    vertical_flip=True,   # Mirror vertical
    rotation_range=20,    # Rotación leve
    zoom_range=0.1,       # Zoom aleatorio
    fill_mode='nearest'
)

# Entrenar el modelo usando el generador de datos (fit_generator)
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS, # 5 Épocas solicitadas
    validation_data=(X_test, y_test_cat),
    class_weight=class_weights_dict, # Usar la ponderación de clases
    verbose=1
)

# --- 6. Evaluación del Modelo y Métricas Finales ---

print("\n--- Evaluación Final en el Conjunto de Prueba ---")

# Evaluar el rendimiento
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy:.4f}")
print(f"Pérdida (Loss) en el conjunto de prueba: {loss:.4f}")

# Predicciones para calcular métricas detalladas
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1) # Obtener la clase predicha (0 o 1)

# Reporte de Clasificación (Métricas Finales detalladas)
print("\n--- Reporte de Clasificación (Métricas Finales) ---")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# Matriz de Confusión
conf_mat = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de Confusión ---")
print(conf_mat)
