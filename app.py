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
from utils.translations import translations


# === CONFIGURACIÓN DE IDIOMA ===
LANGUAGES = {"Español": "es", "English": "en", "Français": "fr"}
selected_language = st.sidebar.selectbox("🌐 Idioma / Language / Langue", list(LANGUAGES.keys()))
lang = LANGUAGES[selected_language]

def t(key, **kwargs):
    return translations[lang][key].format(**kwargs)

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
    pdf.cell(0, 15, t("pdf_title"), 0, 1, 'C')
    pdf.set_line_width(0.5)
    pdf.line(10, 25, 200, 25)
    pdf.ln(5)

    # Datos generales (columna izquierda)
    pdf.set_font(base_font, '', base_size)
    pdf.set_xy(10, 30)
    pdf.cell(90, 7, t("date", date=time.strftime('%d/%m/%Y %H:%M')), 0, 2)
    pdf.cell(90, 7, t("food", food=predicted_food.replace('_', ' ').capitalize()), 0, 2)
    pdf.cell(90, 7, t("weight", weight=weight), 0, 2)
    if confidence is not None:
        pdf.cell(90, 7, t("model_confidence_pdf", confidence=f"{confidence*100:.1f}"), 0, 2)
    if processing_time is not None:
        pdf.cell(90, 7, t("processing_time", time=f"{processing_time:.2f}"), 0, 2)

    # Detalles del modelo (columna derecha)
    pdf.set_xy(110, 30)
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(90, 7, t("pdf_model_details"), 0, 2)
    pdf.set_font(base_font, '', base_size)
    pdf.multi_cell(90, 7, t("model_details"))

    pdf.ln(3)

    # Imagen
    img_path = "temp_img.png"
    image.save(img_path)
    pdf.image(img_path, x=55, w=90)
    os.remove(img_path)
    pdf.ln(5)

    # Nutrientes
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(0, 10, t("pdf_nutritional_values"), 0, 1)

    pdf.set_font(base_font, '', base_size)
    col_width = 95
    row_height = 10

    pdf.set_fill_color(240, 240, 240)
    pdf.cell(col_width, row_height, t("nutrient"), 1, 0, 'C', True)
    pdf.cell(col_width, row_height, t("amount"), 1, 1, 'C', True)

    data = [
        (t("proteins_name"), f"{protein:.1f} g"),
        (t("fats_name"), f"{fat:.1f} g"),
        (t("carbs_name"), f"{carbs:.1f} g"),
        (t("calories_name"), f"{kcal:.1f} kcal")
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

st.set_page_config(page_title=t("app_title"), page_icon="🍽️", layout="centered")
st.title(t("app_title"))

# Cargar traducciones
@st.cache_data
def load_translations():
    try:
        translations_csv = pd.read_csv("data/translations/food_translations.csv")
        return translations_csv
    except Exception as e:
        st.error(t("error_loading_translations", error=e))
        return pd.DataFrame()

translations_df = load_translations()

def translate_food_name(english_name, lang):
    col = {"es": "spanish", "en": "english", "fr": "french"}[lang]
    row = translations_df[translations_df["english"] == english_name]
    if not row.empty:
        return row.iloc[0][col]
    base_name = english_name.split('_')[0]
    row = translations_df[translations_df["english"] == base_name]
    if not row.empty:
        return row.iloc[0][col]
    return ' '.join(word.capitalize() for word in english_name.split('_'))

@st.cache_resource
def load_trained_model():
    try:
        return load_model("models/xception_food101.h5")
    except Exception as e:
        st.error(t("error_loading_model"))
        return None

model = load_trained_model()

if model is None:
    st.error(t("error_loading_model"))
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

uploaded_file = st.file_uploader(t("upload_prompt"), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=t("upload_prompt"), use_container_width=True)

    with st.spinner(t("analyzing")):
        start_time = time.time()
        processed = preprocess_image_inception(image)
        prediction = model.predict(processed, verbose=0)
        predicted_class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        elapsed_time = time.time() - start_time

    class_names = nutrients_df["clase"].tolist()
    if predicted_class_index >= len(class_names):
        st.error(t("error_class_index"))
        st.stop()

    predicted_food = class_names[predicted_class_index]
    food_row = nutrients_df[nutrients_df["clase"] == predicted_food]

    if food_row.empty:
        st.error(t("error_nutrient_data"))
    else:
        food_data = food_row.iloc[0]
        protein = food_data["proteinas"]
        fat = food_data["grasas"]
        carbs = food_data["carbohidratos"]
        kcal = food_data["calorias"]

        food_name = translate_food_name(predicted_food, lang)
        st.success(t("identified_food", food=food_name))
        st.caption(t("model_confidence", confidence=confidence*100))
        st.info(t("analysis_time", time=elapsed_time))

        weight = st.number_input(t("weight_prompt"), min_value=1.0, value=100.0, step=1.0)
        factor = weight / 100.0

        st.markdown(t("nutritional_values"))
        st.markdown(f"""
        - {t('proteins', value=protein * factor)}  
        - {t('fats', value=fat * factor)}  
        - {t('carbs', value=carbs * factor)}  
        - {t('calories', value=kcal * factor)}
        """)

        if st.button(t("generate_pdf")):
            with st.spinner(t("generate_pdf")):
                pdf_bytes = generate_pdf_report(
                    image=image,
                    predicted_food=food_name,  # Usar el nombre traducido
                    protein=protein * factor,
                    fat=fat * factor,
                    carbs=carbs * factor,
                    kcal=kcal * factor,
                    weight=weight,
                    confidence=confidence,
                    processing_time=elapsed_time
                )
            st.download_button(
                label=t("download_pdf"),
                data=pdf_bytes,
                file_name=f"reporte_nutricional_{predicted_food.lower().replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            st.success(t("pdf_success"))
