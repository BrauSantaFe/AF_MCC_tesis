import os
import numpy as np
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
import pywt
from io import BytesIO
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.utils import to_categorical, Sequence # ⬅️ Importar Sequence para el Custom Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 

# --- 1. Configuración ---
IM_SIZE = 256
DATA_DIR = '/data'
CATEGORIES = ['fire', 'no_fire']
NUM_CLASSES = len(CATEGORIES)
#  El escalograma final será en escala de grises (1 canal)
CHANNELS = 1 
LEARNING_RATE = 0.00005
EPOCHS = 30
BATCH_SIZE = 64
# La forma exacta del escalograma se determinará en el paso de pre-cálculo
SCALOGRAM_SHAPE = None 

# --- 2. Funciones de Soporte ---

def load_data(data_dir, categories, im_size):
    """Carga imágenes TIFF desde las carpetas y procesamiento"""
    # ... (Sin Cambios)
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
                        # Normalización (16-bit Landsat)
                        normalized_img = img_array.astype('float32') / 65535.0
                        data.append([normalized_img, class_num])
                except Exception as e:
                    pass
    return np.array(data, dtype=object) 

# --- 3. Lógica de Conversión a Escalograma por Lote (Key Change 1) ---

def convert_batch_to_scalograms(X_batch, im_size, target_shape):
    """
    Toma un lote (4D array) de imágenes aumentadas y lo convierte 
    en un lote (4D array) de escalogramas.
    """
    X_scalogram_batch = []
    num_segmentos = im_size 
    # Las escalas se ajustan al tamaño de la señal (num_segmentos/2 - 1)
    escalas = np.arange(1, num_segmentos // 2)
    nombre_wavelet = 'mexh'
    
    # Asunción de Bandas (B7 en índice 2, B6 en índice 1)
    B7_INDEX = 2
    B6_INDEX = 1
    
    for img_array in X_batch:
        
        # 1. Calcular el vector de Diferenciación Normalizada (B7-B6)/(B7+B6)
        vector_senal_diferenciacion = []
        tamanio_segmento = im_size // num_segmentos
        
        for i in range(num_segmentos):
            inicio_fila = i * tamanio_segmento
            fin_fila = min((i + 1) * tamanio_segmento, im_size)
            if inicio_fila >= im_size: break
                
            segmento = img_array[inicio_fila:fin_fila, :, :]
            
            # Promedio de la banda en el segmento
            B7_promedio = np.mean(segmento[:, :, B7_INDEX]) 
            B6_promedio = np.mean(segmento[:, :, B6_INDEX])
            
            # Calculamos el Índice de Diferenciación Normalizada
            suma_bandas = B7_promedio + B6_promedio
            indice_diferenciacion = (B7_promedio - B6_promedio) / suma_bandas if suma_bandas != 0 else 0.0
            
            vector_senal_diferenciacion.append(indice_diferenciacion)

        vector_senal_cwt = np.array(vector_senal_diferenciacion, dtype=float)

        # 2. Aplicar CWT
        coefs, freqs = pywt.cwt(vector_senal_cwt, escalas, nombre_wavelet)
        
        # 3. Convertir coeficientes a imagen de escalograma (array NumPy)
        fig = plt.figure(figsize=(target_shape[0]/100, target_shape[1]/100), frameon=False) 
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.pcolormesh(np.abs(coefs), cmap='jet', shading='gouraud') 
        
        # Guardar en buffer de memoria
        buf = BytesIO()
        # Usar DPI para asegurar el tamaño en píxeles
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100) 
        plt.close(fig) 
        
        # Leer la imagen PNG del buffer y convertir a escala de grises
        buf.seek(0)
        img_png = Image.open(buf).convert('L') 
        
        # Redimensionar si la imagen guardada no coincide exactamente con el target_shape 
        # (necesario si la figura no se renderiza exactamente a 100dpi)
        img_png = img_png.resize(target_shape)
        
        # Normalizar el escalograma (0 a 1)
        scalogram_array = np.array(img_png).astype('float32') / 255.0
        X_scalogram_batch.append(scalogram_array)
        
    X_scalogram_batch = np.array(X_scalogram_batch)
    # Añadir la dimensión del canal (1)
    return np.expand_dims(X_scalogram_batch, axis=-1)

# --- 4. Generador Personalizado (Key Change 2) ---

class ScalogramGenerator(Sequence):
    """
    Generador personalizado que envuelve el ImageDataGenerator, aplica aumentación, 
    y luego convierte el lote a escalogramas usando CWT.
    """
    def __init__(self, data_generator, X_set, y_set, batch_size, im_size, scalogram_shape):
        self.datagen_flow = data_generator.flow(X_set, y_set, batch_size=batch_size, shuffle=True)
        self.X_set = X_set
        self.y_set = y_set
        self.im_size = im_size
        self.scalogram_shape = scalogram_shape

    def __len__(self):
        # Número de batches por época
        return len(self.datagen_flow)

    def __getitem__(self, index):
        # Generar un batch aumentado
        X_batch_augmented, y_batch = self.datagen_flow.__getitem__(index)
        
        # Convertir el batch aumentado a escalogramas
        X_batch_scalogram = convert_batch_to_scalograms(
            X_batch_augmented, 
            self.im_size, 
            self.scalogram_shape
        )
        
        return X_batch_scalogram, y_batch

    def on_epoch_end(self):
        # Asegura que el generador base baraje los datos al final de cada época
        self.datagen_flow.on_epoch_end()

# --- 5. Lógica Principal de Carga y Pre-cálculo ---

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
print(f"Forma de X_train (original): {X_train.shape}")

# --- Ponderación de Clases (Sin Cambios) ---
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print("\n--- Pesos de Clase Calculados para el Entrenamiento ---")
print(class_weights_dict)

# --- 6. Determinación de la Forma del Escalograma ---

# Calculamos la forma de un solo escalograma para definir la arquitectura del modelo.
# Usamos un pequeño lote temporal.
temp_batch = X_train[:4]
print("\n--- Determinando la forma del Escalograma ---")

# Estimación del tamaño final del escalograma (ejecutar una vez)
# Usaremos un tamaño fijo (por ejemplo, 128x128) para el escalograma para simplificar la arquitectura
# ya que la forma del coeficiente CWT no es fija.
SCALOGRAM_TARGET_SIZE = (128, 128) # Asumimos 128x128 píxeles para la imagen final del escalograma
temp_scalograms = convert_batch_to_scalograms(temp_batch, IM_SIZE, SCALOGRAM_TARGET_SIZE)

SCALOGRAM_SHAPE = temp_scalograms.shape[1:3]
input_shape = SCALOGRAM_SHAPE + (CHANNELS,)

print(f"La forma de entrada final para la CNN será: {input_shape}")
# -------------------------------------------------------------------

# --- 7. Definición del Modelo CNN (¡AJUSTADO!) ---

def create_cnn_model(input_shape, num_classes, learning_rate):
    """Define y compila el modelo CNN con input_shape ajustado."""  
    model = Sequential([
        # Capa de Convolución 1 (Input ajustado)
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
        Dropout(0.5),  

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
model = create_cnn_model(input_shape, NUM_CLASSES, LEARNING_RATE)
model.summary()

# --- 8. Aumentación de Datos y Entrenamiento (¡Con el Custom Generator!) ---

print("\n--- Configurando Aumentación de Datos y Callbacks ---")

# Generador de Aumentación de Datos (Generador Base)
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1,
    fill_mode='nearest'
)

# CREACIÓN DEL GENERADOR PERSONALIZADO PARA ENTRENAMIENTO
train_generator = ScalogramGenerator(
    datagen, 
    X_train, 
    y_train_cat, 
    BATCH_SIZE, 
    IM_SIZE, 
    SCALOGRAM_SHAPE
)

#  PREPARACIÓN DE LOS DATOS DE VALIDACIÓN
# Los datos de validación NO se aumentan, solo se transforman a escalograma.
X_test_scalogram = convert_batch_to_scalograms(X_test, IM_SIZE, SCALOGRAM_SHAPE)


# Definición de Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

# Entrenar el modelo
history = model.fit(
    #  USAMOS EL GENERADOR PERSONALIZADO
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    #  USAMOS EL BATCH DE VALIDACIÓN PRE-CALCULADO
    validation_data=(X_test_scalogram, y_test_cat),
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)


# --- 9. Evaluación del Modelo y Métricas Finales ---

print("\n--- Evaluación Final en el Conjunto de Prueba ---")

# Evaluar el rendimiento (usando X_test_scalogram)
loss, accuracy = model.evaluate(X_test_scalogram, y_test_cat, verbose=0)
print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy:.4f}")
print(f"Pérdida (Loss) en el conjunto de prueba: {loss:.4f}")

# Predicciones para calcular métricas detalladas
y_pred_probs = model.predict(X_test_scalogram)
y_pred = np.argmax(y_pred_probs, axis=1) # Obtener la clase predicha (0 o 1)

# Reporte de Clasificación
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# Matriz de Confusión
conf_mat = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de Confusión Numérica ---")
print(conf_mat)

# --- 10. Visualización de Resultados (Gráficas) ---

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

CM_PLOTS_PATH = '/data/matriz_confusion.png'
plot_confusion_matrix(conf_mat, CATEGORIES, CM_PLOTS_PATH)

# --- Gráficas de Overfitting (Pérdida y Métrica) ---
acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

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

METRICS_PLOTS_PATH = '/data/graficas_overfitting.png'
plt.savefig(METRICS_PLOTS_PATH)
plt.close()

print(f"Gráficas de Overfitting guardadas exitosamente en: {METRICS_PLOTS_PATH}")