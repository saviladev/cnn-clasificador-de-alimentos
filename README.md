# 🍽️ Aplicación para el cálculo de macronutrientes a través de imágenes clasificadas con redes neuronales

Este proyecto utiliza redes neuronales convolucionales (CNN) para clasificar imágenes de alimentos y estimar sus macronutrientes a través de una aplicación web desarrollada en Streamlit.

---

## 🏆 Nota sobre el modelo utilizado

✨ El mejor modelo que pudimos desarrollar fue con **Xception** (`⭐`). Por eso, la aplicación está configurada por defecto para utilizar este modelo. Sin embargo, puedes modificar la app para emplear cualquiera de los otros modelos entrenados (**InceptionV3** o **ResNet50**), teniendo en cuenta que deberás ajustar el formato de entrada y el preprocesamiento de imágenes según los requisitos de cada arquitectura.

---

## 1️⃣ Entrenamiento de Modelos

El entrenamiento de los modelos se realizó en diferentes plataformas y entornos, cada uno con su propio cuaderno y configuración:

### 🟦 InceptionV3
- 📒 **Notebook:** `trainings/InceptionV3/inceptionv3_model.ipynb`
- ☁️ **Plataforma:** Google Colab
- ⚡ **Requisitos:**
  - Entorno de ejecución con GPU (en Colab: "Entorno de ejecución" > "Cambiar tipo de entorno de ejecución" > GPU).
  - Instalar las dependencias indicadas en el notebook.
- ▶️ **Uso:**
  1. Sube el notebook a Google Colab.
  2. Configura el entorno con GPU.
  3. Ejecuta las celdas para entrenar el modelo y guardar el archivo `.h5` resultante.

### 🟩 ResNet50
- 📒 **Notebook:** `trainings/ResNet50/resnet50_model.ipynb`
- ☁️ **Plataforma:** Google Colab
- ⚡ **Requisitos:**
  - Entorno de ejecución con GPU.
  - Instalar las dependencias indicadas en el notebook.
- ▶️ **Uso:**
  1. Sube el notebook a Google Colab.
  2. Configura el entorno con GPU.
  3. Ejecuta las celdas para entrenar el modelo y guardar el archivo `.h5` resultante.

### 🟪 Xception
- 📒 **Notebook:** `trainings/Xception/xception-model.ipynb`
- 🏅 **Plataforma:** Kaggle Notebooks
- ⚡ **Requisitos:**
  - Activar aceleración por GPU en el entorno de Kaggle.
  - Instalar las dependencias indicadas en el notebook.
- ▶️ **Uso:**
  1. Sube el notebook a Kaggle.
  2. Activa la GPU en "Settings" del notebook.
  3. Ejecuta las celdas para entrenar el modelo y guardar el archivo `.h5` resultante.

### 📊 Cálculo de Métricas
- 📒 **Notebook:** `trainings/metricas-de-los-modelos.ipynb`
- 📝 **Descripción:**
  Después de entrenar los modelos, utiliza este notebook para calcular y comparar las métricas de desempeño (accuracy, loss, etc.) de cada modelo entrenado.

---

## 2️⃣ Aplicación Web: Estimación de Macronutrientes

La aplicación principal está en `app.py` y permite:

- 📤 Subir una imagen de comida.
- 🤖 Clasificar el alimento usando el modelo Xception entrenado.
- 📊 Consultar los valores nutricionales estimados:
  - 🥩 **Proteínas**
  - 🧈 **Grasas**
  - 🍞 **Carbohidratos**
  - 🔥 **Calorías**
- 📄 Descargar un reporte PDF personalizado con los resultados.

### ⚙️ Requisitos
- 🐍 **Python:** 3.11.9
- 📦 **Instalación de dependencias:**
  ```bash
  pip install -r requirements.txt
  ```
- ⚠️ **Nota:** Los archivos de modelo `.h5` no están incluidos en el repositorio por limitaciones de GitHub. Descárgalos desde el enlace proporcionado (ver sección Modelos).

### 🚀 Ejecución
```bash
streamlit run app.py
```

---

## 3️⃣ Modelos

Debido a las restricciones de tamaño de archivo en GitHub, los modelos entrenados (`.h5`) no están incluidos en este repositorio.

🔗 **Descárgalos desde el siguiente enlace:**
[📥 Modelos entrenados en Google Drive](https://drive.google.com/drive/folders/1QKalAF-zgypiLJG0nCcYGgaxpqtFBo4f?usp=sharing)

Coloca los archivos descargados en la carpeta `models/` del proyecto.