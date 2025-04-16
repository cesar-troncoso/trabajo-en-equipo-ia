# ====== LIBRERÍAS NECESARIAS ======
import os  # Manejo de rutas y archivos
import cv2  # Procesamiento de imágenes
import numpy as np  # Cálculos numéricos
import pandas as pd  # Manejo de datos en tablas
import matplotlib.pyplot as plt  # Visualización gráfica

# Scikit-learn: librerías para modelos clásicos de machine learning
from sklearn.model_selection import train_test_split  # Dividir datos en entrenamiento y prueba
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Codificación y normalización
from sklearn.svm import SVC  # Clasificador SVM
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # Métricas

# TensorFlow y Keras para modelos de deep learning (CNN)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random  # Para seleccionar ejemplos aleatorios

# Mensaje de inicio
print("=== INICIO DEL PROCESAMIENTO DE IMÁGENES Y MODELOS ===")

# ========== CONFIGURACIÓN ==========
DATASET_DIR = "fruits/Training"  # Ruta donde están las carpetas con imágenes de frutas
IMG_SIZE = 100  # Tamaño al que se redimensionarán las imágenes
MAX_IMAGES_PER_CLASS = 20  # Cuántas imágenes por clase se usarán (para pruebas rápidas)

# ========== FUNCIÓN PARA EXTRAER CARACTERÍSTICAS CON OPENCV ==========
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplicar desenfoque para eliminar ruido
    edged = cv2.Canny(blurred, 30, 150)  # Detectar bordes
    return edged.flatten()  # Convertir imagen en un vector de características

# ========== CARGA DE DATOS PARA MODELO CLÁSICO ==========
features, labels = [], []
for folder in os.listdir(DATASET_DIR):  # Recorrer carpetas de frutas
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path): continue  # Saltar si no es carpeta
    for file in os.listdir(folder_path)[:MAX_IMAGES_PER_CLASS]:  # Tomar solo las primeras imágenes
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)  # Leer imagen
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Redimensionar
        features.append(extract_features(img))  # Extraer características
        labels.append(folder)  # Guardar etiqueta (nombre de la carpeta)
print(f"Total de imágenes cargadas: {len(features)}")

# ========== PREPROCESAMIENTO Y MODELO CLÁSICO ==========
le = LabelEncoder()  # Codificar nombres de frutas en números
y = le.fit_transform(labels)  # Convertir etiquetas
X = np.array(features)  # Convertir características a array
scaler = StandardScaler()  # Normalizar datos
X_scaled = scaler.fit_transform(X)  # Aplicar normalización

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar modelo SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)  # Hacer predicciones

# Mostrar métricas del modelo SVM
print("\n=== Resultados del modelo clásico (SVM) ===")
print(classification_report(y_test, y_pred, labels=np.unique(y_pred), target_names=le.classes_[np.unique(y_pred)]))

# ========== CNN CON IMÁGENES ORIGINALES ==========
cnn_images, cnn_labels = [], []
for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path): continue
    for file in os.listdir(folder_path)[:MAX_IMAGES_PER_CLASS]:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cnn_images.append(img)
        cnn_labels.append(folder)

cnn_images = np.array(cnn_images) / 255.0  # Normalizar valores de píxeles
cnn_labels = le.transform(cnn_labels)  # Convertir etiquetas a números
cnn_labels_cat = to_categorical(cnn_labels)  # Convertir a formato categoría (one-hot)

# Dividir en entrenamiento y prueba
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    cnn_images, cnn_labels_cat, test_size=0.2, random_state=42)

# ========== DEFINICIÓN Y ENTRENAMIENTO DE CNN ==========
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Capa convolucional
    MaxPooling2D(2, 2),  # Submuestreo
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),  # Aplanar
    Dense(128, activation='relu'),  # Capa oculta
    Dense(len(le.classes_), activation='softmax')  # Capa de salida
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_cnn, y_train_cnn, epochs=5, validation_data=(X_test_cnn, y_test_cnn))  # Entrenamiento

# Evaluación del modelo
loss, acc = model.evaluate(X_test_cnn, y_test_cnn)
print(f"\n=== Precisión del modelo CNN: {acc:.2f} ===")

# ========== MATRIZ DE CONFUSIÓN REDUCIDA ==========
y_pred_cnn = model.predict(X_test_cnn)
y_pred_labels = np.argmax(y_pred_cnn, axis=1)  # Etiquetas predichas
y_true_labels = np.argmax(y_test_cnn, axis=1)  # Etiquetas reales
cm = confusion_matrix(y_true_labels, y_pred_labels)  # Calcular matriz

N = 10  # Mostrar sólo las primeras N clases
labels_used = np.unique(y_true_labels)[:N]  # Obtener etiquetas
filtered_cm = cm[:N, :N]  # Cortar la matriz
filtered_labels = le.classes_[labels_used]  # Obtener nombres
os.makedirs("outputs", exist_ok=True)  # Asegura que la carpeta exista

disp = ConfusionMatrixDisplay(confusion_matrix=filtered_cm, display_labels=filtered_labels)
disp.plot(xticks_rotation=90)
plt.title("Matriz de Confusión - CNN (Primeras 10 clases)")
plt.tight_layout()

# Guardar en archivo
plt.savefig("outputs/matriz_confusion_recortada.png", dpi=300)

# Mostrar en pantalla
plt.show()

# ========== VISUALIZAR PREDICCIONES ==========
print("\n=== Muestras visuales del conjunto de prueba ===")
NUM_EJEMPLOS = 10
indices = random.sample(range(len(X_test_cnn)), NUM_EJEMPLOS)  # Seleccionar ejemplos al azar
for i in indices:
    imagen = X_test_cnn[i]
    verdadera = le.classes_[np.argmax(y_test_cnn[i])]  # Etiqueta real
    predicha = le.classes_[np.argmax(y_pred_cnn[i])]  # Etiqueta predicha

    # Crear carpeta si no existe
    os.makedirs("outputs/predicciones", exist_ok=True)

    # Convertir imagen a formato RGB y escalar valores
    imagen_rgb = cv2.cvtColor((imagen * 255).astype("uint8"), cv2.COLOR_BGR2RGB)

    # Graficar sin mostrar
    plt.imshow(imagen_rgb)
    plt.title(f"Real: {verdadera} | Predicho: {predicha}")
    plt.axis('off')
    plt.tight_layout()

    # Guardar imagen en carpeta
    plt.savefig(f"outputs/predicciones/pred_{i}_{verdadera}_vs_{predicha}.png")
    plt.close()

