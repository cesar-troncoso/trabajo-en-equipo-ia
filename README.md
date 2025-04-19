# 🥝 Clasificación de Frutas con Machine Learning y CNN

Este proyecto utiliza **procesamiento digital de imágenes**, **OpenCV**, **modelos clásicos de Machine Learning** y **redes neuronales convolucionales (CNN)** para clasificar diferentes tipos de frutas.

---

## 📂 Estructura del Proyecto

```
clasificacion-frutas/
├── fruits/              # Carpeta con imágenes (descargadas de Google Drive)
│   └── Training/
├── outputs/             # Resultados generados (matriz de confusión y predicciones)
│   ├── matriz_confusion_recortada.png
│   └── predicciones/
│       ├── pred_1_Apple_1_vs_Apple_1.png
│       └── ...
├── main.py              # Código principal del proyecto
├── requirements.txt     # Dependencias del proyecto
├── .gitignore           # Archivos/carpeta a ignorar por Git
└── README.md            # Este archivo
```
---

## 🔧 Requisitos

- Python 3.8+
- pip
- Visual Studio Code (opcional pero recomendado)

---

## 🚀 Instalación y ejecución paso a paso

### 1. Clona el repositorio

```bash
git clone https://github.com/cesar-troncoso/trabajo-en-equipo-ia.git
cd trabajo-en-equipo-ia
```
---

### 2. (Opcional) Crea un entorno virtual

```bash
python -m venv venv
```

Activa el entorno:

- En Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- En Mac/Linux:
  ```bash
  source venv/bin/activate
  ```

---

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```
---

### 4. Descarga el dataset


Puedes descargar el dataset utilizado en este proyecto directamente desde Google Drive:

🔗 [Descargar dataset desde Google Drive](https://drive.google.com/uc?export=download&id=1eyJOXRn7Ch7MW7uMtv9eeePjPcWd0ecz)

Una vez descargado:
- Extrae el archivo `.zip`.
- Copia la carpeta `fruits/Training` dentro del directorio raíz del proyecto.

Debe quedar así:
```
clasificador-imagenes-equipo/
├── fruits/
│   └── Training/
│       ├── Apple Red 1/
│       ├── Banana 1/
│       └── ...
```

---

### 5. Ejecuta el script

```bash
python main.py
```

Esto realizará:

- Extracción de características con OpenCV
- Entrenamiento de un modelo clásico (SVM)
- Entrenamiento de una CNN
- Evaluación con métricas y matriz de confusión

---

## 📊 Resultados esperados

- Métricas de precisión, recall, F1-score en consola
- Matriz de confusión para la CNN
- Comparación entre los dos enfoques

---

## 📁 Archivos importantes

- `main.py` → Código fuente completo
- `.gitignore` → Ignora `venv/` y archivos temporales
- `requirements.txt` → Lista de librerías instaladas
- `README.md` → Guía completa del proyecto

---
