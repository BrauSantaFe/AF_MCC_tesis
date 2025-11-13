import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # <--- NUEVA IMPORTACIÓN PARA LECTURA ROBUSTA EN GRISES
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. Configuración ---
IM_SIZE = 256
# Asegúrate de que esta ruta apunta a tu carpeta con las imágenes R-B
DATA_DIR = '/data' 
CATEGORIES = ['fire', 'no_fire']
NUM_CLASSES = len(CATEGORIES)
CHANNELS = 1  # <--- MANTENER EN 1 PARA ESCALA DE GRISES
LEARNING_RATE = 0.00005
EPOCHS = 40
BATCH_SIZE = 64

# --- 2. Función para cargar imágenes (MODIFICADA) ---
def load_data(data_dir, categories, im_size, channels):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        print(f"Cargando imágenes de: {category} (1 Canal)...")

        for img_name in os.listdir(path):
            file_path = os.path.join(path, img_name)
            # Extensión .tif/.tiff añadida para compatibilidad con tus archivos
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                try:
                    # **CAMBIO CLAVE: Leer la imagen directamente en escala de grises (1 canal)**
                    # cv2.IMREAD_GRAYSCALE asegura que la imagen se lee en 1 canal
                    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img_array is None:
                        print(f"⚠️ Error: No se pudo leer la imagen {file_path}")
                        continue
                        
                    # Redimensionar si es necesario (cv2.resize es mejor que np.resize)
                    if img_array.shape[:2] != (im_size, im_size):
                        img_array = cv2.resize(img_array, (im_size, im_size))

                    # Normalizar a float32 [0.0, 1.0]
                    img_array = img_array.astype('float32') / 255.0
                    
                    # **Asegurar la forma (H, W, 1) para Keras/TensorFlow**
                    # Si la imagen es (H, W), la convertimos a (H, W, 1)
                    if img_array.ndim == 2:
                        img_array = np.expand_dims(img_array, axis=-1)
                    
                    # Una última verificación de la forma (opcional, pero buena práctica)
                    if img_array.shape != (im_size, im_size, channels):
                         print(f"⚠️ Error: Forma final incorrecta para {file_path}: {img_array.shape}")
                         continue
                         
                    data.append((img_array, class_num))
                except Exception as e:
                    print(f"Error al procesar {file_path}: {e}")
    return data

# --- 3. Carga de datos ---
all_data = load_data(DATA_DIR, CATEGORIES, IM_SIZE, CHANNELS)

if len(all_data) == 0:
    print("\n🚨 ERROR CRÍTICO: No se cargó ninguna imagen. Verifica la ruta y los filtros.")
    exit()

X = np.array([i[0] for i in all_data])
y = np.array([i[1] for i in all_data])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

print("\n--- Estadísticas de los Datos ---")
# La forma ahora debería ser (num_samples, 256, 256, 1)
print(f"Forma de X_train: {X_train.shape}") 
print(f"Forma de X_test: {X_test.shape}")

# --- 4. Ponderación de clases (sin cambios) ---
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print("\n--- Pesos de Clase Calculados ---")
print(class_weights_dict)

# --- 5. Definición del modelo CNN (sin cambios) ---
# El modelo usa el valor de CHANNELS=1 automáticamente a través de input_shape
def create_cnn_model(input_shape, num_classes, learning_rate):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_cnn_model((IM_SIZE, IM_SIZE, CHANNELS), NUM_CLASSES, LEARNING_RATE)
model.summary()

# --- 6. Entrenamiento (sin cambios) ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# --- 7. Evaluación (sin cambios) ---
print("\n--- Evaluación Final ---")
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Precisión: {accuracy:.4f}")
print(f"Pérdida: {loss:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

conf_mat = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de Confusión ---")
print(conf_mat)

# --- 8. Visualización (sin cambios) ---
def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de Confusión guardada en: {save_path}")

CM_PLOTS_PATH = '/data/matriz_confusion_gris.png'
plot_confusion_matrix(conf_mat, CATEGORIES, CM_PLOTS_PATH)

acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Pérdida por Época')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history[acc_key], label='Train Accuracy')
plt.plot(history.history[val_acc_key], label='Val Accuracy')
plt.title('Precisión por Época')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

METRICS_PLOTS_PATH = '/data/graficas_overfitting_gris.png'
plt.savefig(METRICS_PLOTS_PATH)
plt.close()
print(f"Gráficas guardadas en: {METRICS_PLOTS_PATH}")