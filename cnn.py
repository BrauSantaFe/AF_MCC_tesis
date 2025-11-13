import os
import numpy as np
# Se elimina tifffile ya que solo usaremos PNG
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 

# --- 1. Configuración ---
IM_SIZE = 256
DATA_DIR = '/data'
CATEGORIES = ['fire', 'no_fire']
NUM_CLASSES = len(CATEGORIES)
# : Usamos 1 canal para escalogramas en escala de grises.
CHANNELS = 1 
LEARNING_RATE = 0.00005
EPOCHS = 30
BATCH_SIZE = 64

# ----------------------------------------------------------------------
# --- FUNCIÓN CORREGIDA ---
# ----------------------------------------------------------------------
def load_data(data_dir, categories, im_size, channels):
    """
    Carga imágenes PNG (8-bit) y realiza la normalización por 255.0.
    """
    data = []

    # Bandera para cv2.imread: IMREAD_GRAYSCALE (0) si CHANNELS=1
    read_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        print(f"Cargando imágenes de: {category}...")

        for img_name in os.listdir(path):
            file_path = os.path.join(path, img_name)

            if img_name.endswith(('.png', '.jpg', '.jpeg')): # Solo leer formatos de 8 bits
                try:
                    # Cargar la imagen con el modo de canales adecuado
                    img_array = cv2.imread(file_path, read_flag)
                    
                    if img_array is None:
                         # Ignorar si no se puede leer
                        continue
                        
                    # 1. Normalización (8-bit PNG): Valores 0-255 a 0.0-1.0
                    # 🚨 CAMBIO CLAVE 2: Normalización por 255.0
                    normalized_img = img_array.astype('float32') / 255.0
                    
                    # 2. Ajuste de forma: Si es 1 canal (escala de grises), aseguramos (H, W, 1)
                    if channels == 1 and normalized_img.ndim == 2:
                        normalized_img = np.expand_dims(normalized_img, axis=-1)

                    # 3. Comprobación de tamaño y número de canales
                    if normalized_img.shape[:2] == (im_size, im_size) and normalized_img.shape[-1] == channels:
                        data.append([normalized_img, class_num])

                except Exception as e:
                    # print(f"Error al leer/procesar la imagen {img_name}: {e}")
                    pass
            
    return np.array(data, dtype=object) 
# ----------------------------------------------------------------------
# --- FIN DE LA FUNCIÓN CORREGIDA ---
# ----------------------------------------------------------------------


# Cargar los datos
# 🚨 CAMBIO CLAVE 3: Pasar el número de canales a la función
all_data = load_data(DATA_DIR, CATEGORIES, IM_SIZE, CHANNELS) 


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


# --- 3. Ponderación de Clases (Sin Cambios) ---


# Calcular los pesos de clase para la ponderación inversa por frecuencia
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))


print("\n--- Pesos de Clase Calculados para el Entrenamiento ---")
print(class_weights_dict)

# --- 4. Definición del Modelo CNN (¡CON DROPOUT!) ---

def create_cnn_model(input_shape, num_classes, learning_rate):
    """Define y compila el modelo CNN con regularización Dropout."""  
    model = Sequential([
        # Capa de Convolución 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  

        # Capa de Convolución 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  
        # Capa de Convolución 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Aplanar y Capas Densas
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),  

        # Capa de Salida
        Dense(num_classes, activation='softmax')
    ])

    # Compilación
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Crear el modelo
input_shape = (IM_SIZE, IM_SIZE, CHANNELS)
model = create_cnn_model(input_shape, NUM_CLASSES, LEARNING_RATE)
model.summary()

# --- 5. Aumentación de Datos y Entrenamiento (¡Añadimos Callbacks!) ---

print("\n--- Configurando Aumentación de Datos y Callbacks ---")

# Generador de Aumentación de Datos (Se mantiene igual)
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Definición de Callbacks
callbacks = [
    # 1. Detención Temprana
    EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=1, 
        restore_best_weights=True
    ),
    # 2. Reducción de LR
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001, 
        verbose=1
    )
]


# Entrenar el modelo
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)


# --- 6. Evaluación del Modelo y Métricas Finales (Cálculo de Matriz) ---

print("\n--- Evaluación Final en el Conjunto de Prueba ---")

# Evaluar el rendimiento
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy:.4f}")
print(f"Pérdida (Loss) en el conjunto de prueba: {loss:.4f}")

# Predicciones para calcular métricas detalladas
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1) # Obtener la clase predicha (0 o 1)

# Reporte de Clasificación (Métricas Finales detalladas)
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# Matriz de Confusión
conf_mat = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de Confusión Numérica ---")
print(conf_mat)

# --- 7. Visualización de Resultados (Gráficas) ---

def plot_confusion_matrix(cm, classes, save_path):
    """Plotea la matriz de confusión como un heatmap."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=False
    )
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de Confusión guardada en: {save_path}")


print("\n--- Generando Gráficas de Pérdida, Métrica y Matriz de Confusión ---")

# Ruta donde se guardará el mapa de calor de la matriz
CM_PLOTS_PATH = '/data/matriz_confusion.png'
plot_confusion_matrix(conf_mat, CATEGORIES, CM_PLOTS_PATH)


# --- Gráficas de Overfitting (Pérdida y Métrica) ---

acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

# Crear una figura con dos subgráficas
plt.figure(figsize=(14, 5))

# 1. Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento (Train Loss)')
plt.plot(history.history['val_loss'], label='Pérdida de Validación (Val Loss)')
plt.title('Pérdida (Loss) por Época')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend()
plt.grid(True)

# 2. Gráfica de Métrica (Accuracy/Precisión)
plt.subplot(1, 2, 2)
plt.plot(history.history[acc_key], label='Métrica de Entrenamiento (Train Acc)')
plt.plot(history.history[val_acc_key], label='Métrica de Validación (Val Acc)')
plt.title(f'Métrica ({acc_key.capitalize()}) por Época')
plt.ylabel(acc_key.capitalize())
plt.xlabel('Época')
plt.legend()
plt.grid(True)

# Guardar la imagen de las métricas de overfitting
METRICS_PLOTS_PATH = '/data/graficas_overfitting.png'
plt.savefig(METRICS_PLOTS_PATH)
plt.close()

print(f"Gráficas de Overfitting guardadas exitosamente en: {METRICS_PLOTS_PATH}")