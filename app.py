import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import pandas as pd
import time
from fpdf import FPDF
import os

# === FUNCIÓN PDF ===

def generate_pdf_report(image, predicted_food, protein, fat, carbs, kcal, weight, confidence=None, processing_time=None):
    pdf = FPDF()
    pdf.add_page()

    base_font = 'Arial'
    base_size = 12
    title_size = 16
    subtitle_size = 14

    # Título
    pdf.set_font(base_font, 'B', title_size)
    pdf.cell(0, 15, "Reporte Nutricional", 0, 1, 'C')
    pdf.set_line_width(0.5)
    pdf.line(10, 25, 200, 25)
    pdf.ln(5)

    # Datos generales (columna izquierda)
    pdf.set_font(base_font, '', base_size)
    pdf.set_xy(10, 30)
    pdf.cell(90, 7, f"Fecha: {time.strftime('%d/%m/%Y %H:%M')}", 0, 2)
    pdf.cell(90, 7, f"Alimento: {predicted_food.replace('_', ' ').capitalize()}", 0, 2)
    pdf.cell(90, 7, f"Peso: {weight:.1f} gramos", 0, 2)
    if confidence is not None:
        pdf.cell(90, 7, f"Confianza del modelo: {confidence*100:.1f}%", 0, 2)
    if processing_time is not None:
        pdf.cell(90, 7, f"Tiempo de procesamiento: {processing_time:.2f} segundos", 0, 2)

    # Detalles del modelo (columna derecha)
    pdf.set_xy(110, 30)
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(90, 7, "Detalles del Modelo", 0, 2)
    pdf.set_font(base_font, '', base_size)
    pdf.multi_cell(90, 7, "- Repartición del dataset: 75% entrenamiento / 25% prueba\n- Épocas de entrenamiento: 20\n- Modelo utilizado: Xception")

    pdf.ln(3)

    # Imagen
    img_path = "temp_img.png"
    image.save(img_path)
    pdf.image(img_path, x=55, w=90)
    os.remove(img_path)
    pdf.ln(5)

    # Nutrientes
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(0, 10, "Valores Nutricionales", 0, 1)

    pdf.set_font(base_font, '', base_size)
    col_width = 95
    row_height = 10

    pdf.set_fill_color(240, 240, 240)
    pdf.cell(col_width, row_height, 'Nutriente', 1, 0, 'C', True)
    pdf.cell(col_width, row_height, 'Cantidad', 1, 1, 'C', True)

    data = [
        ('Proteínas', f"{protein:.1f} g"),
        ('Grasas', f"{fat:.1f} g"),
        ('Carbohidratos', f"{carbs:.1f} g"),
        ('Calorías', f"{kcal:.1f} kcal")
    ]

    for i, (nutrient, value) in enumerate(data):
        fill = 240 if i % 2 == 0 else 255
        pdf.set_fill_color(fill, fill, fill)
        pdf.cell(col_width, row_height, nutrient, 1, 0, 'L', True)
        pdf.cell(col_width, row_height, value, 1, 1, 'C', True)

    pdf.ln(8)

    output_path = "reporte_nutricional.pdf"
    pdf.output(output_path)
    with open(output_path, "rb") as f:
        pdf_bytes = f.read()
    os.remove(output_path)
    return pdf_bytes

# === CONFIGURACIÓN PRINCIPAL DE LA APP ===

st.set_page_config(page_title="Macronutrientes por Imagen", page_icon="🍽️", layout="centered")
st.title("🍽️ Estimación de Macronutrientes con IA")

# Cargar traducciones
@st.cache_data
def load_translations():
    try:
        translations = pd.read_csv("data/translations/food_translations.csv")
        return dict(zip(translations['english'], translations['spanish']))
    except Exception as e:
        st.error(f"Error cargando traducciones: {e}")
        return {}

translations = load_translations()

def translate_food_name(english_name):
    """Traduce el nombre de la comida al español"""
    # Primero intenta traducir el nombre exacto
    if english_name in translations:
        return translations[english_name]
    
    # Si no encuentra traducción exacta, intenta con el nombre base (antes del primer _)
    base_name = english_name.split('_')[0]
    if base_name in translations:
        return translations[base_name]
    
    # Si no hay traducción, devuelve el nombre original formateado
    return ' '.join(word.capitalize() for word in english_name.split('_'))

@st.cache_resource
def load_trained_model():
    try:
        return load_model("models/xception_food101.h5")
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

model = load_trained_model()

# Check if model loaded successfully
if model is None:
    st.error("❌ No se pudo cargar el modelo. Verifica que el archivo 'models/xception_food101.h5' existe.")
    st.stop()

@st.cache_data
def load_nutrient_data():
    return pd.read_csv("utils/food101_macronutrientes_completo.csv")

nutrients_df = load_nutrient_data()

def preprocess_image_inception(image, target_size=(299, 299)):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

uploaded_file = st.file_uploader("📤 Sube una imagen de comida", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    with st.spinner('🔍 Analizando la imagen...'):
        start_time = time.time()
        processed = preprocess_image_inception(image)
        prediction = model.predict(processed, verbose=0)
        predicted_class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        elapsed_time = time.time() - start_time

    class_names = nutrients_df["clase"].tolist()
    if predicted_class_index >= len(class_names):
        st.error("⚠️ El índice predicho excede las clases disponibles en el CSV.")
        st.stop()

    predicted_food = class_names[predicted_class_index]
    food_row = nutrients_df[nutrients_df["clase"] == predicted_food]

    if food_row.empty:
        st.error("No se encontraron datos nutricionales para esta comida.")
    else:
        food_data = food_row.iloc[0]
        protein = food_data["proteinas"]
        fat = food_data["grasas"]
        carbs = food_data["carbohidratos"]
        kcal = food_data["calorias"]

        # Traducir el nombre de la comida al español
        food_name_es = translate_food_name(predicted_food)
        st.success(f"🍱 Comida identificada: **{food_name_es}**")
        st.caption(f"📊 Confianza del modelo: **{confidence*100:.2f}%**")
        st.info(f"⏱️ Tiempo de análisis: {elapsed_time:.2f} segundos")

        weight = st.number_input("⚖️ Peso del alimento (gramos)", min_value=1.0, value=100.0, step=1.0)
        factor = weight / 100.0

        st.markdown("### Valores nutricionales:")
        st.markdown(f"""
        - 🥩 **Proteínas**: {protein * factor:.1f} g  
        - 🧈 **Grasas**: {fat * factor:.1f} g  
        - 🍞 **Carbohidratos**: {carbs * factor:.1f} g  
        - 🔥 **Calorías**: {kcal * factor:.1f} kcal
        """)

        if st.button("📄 Generar Reporte PDF"):
            with st.spinner('Generando reporte PDF...'):
                pdf_bytes = generate_pdf_report(
                    image=image,
                    predicted_food=food_name_es,  # Usar el nombre traducido
                    protein=protein * factor,
                    fat=fat * factor,
                    carbs=carbs * factor,
                    kcal=kcal * factor,
                    weight=weight,
                    confidence=confidence,
                    processing_time=elapsed_time
                )
            st.download_button(
                label="📥 Descargar Reporte PDF",
                data=pdf_bytes,
                file_name=f"reporte_nutricional_{predicted_food.lower().replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            st.success("¡Reporte generado con éxito!")
