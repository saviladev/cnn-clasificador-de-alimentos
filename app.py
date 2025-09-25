import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import pandas as pd
import time
from fpdf import FPDF
import os
import requests
import gdown
from utils.translations import translations


# === CONFIGURACIÓN DE IDIOMA ===
LANGUAGES = {"Español": "es", "English": "en", "Français": "fr"}
selected_language = st.sidebar.selectbox("🌐 Idioma / Language / Langue", list(LANGUAGES.keys()))
lang = LANGUAGES[selected_language]

def t(key, **kwargs):
    return translations[lang][key].format(**kwargs)

# === FUNCIÓN PDF ===

def generate_pdf_report(image, predicted_food, protein, fat, carbs, kcal, weight, confidence=None, processing_time=None):
    from PIL import Image as PILImage
    pdf = FPDF()
    pdf.add_page()

    # Agregar fuente Unicode normal y bold
    pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
    base_font = 'DejaVu'
    base_size = 12
    title_size = 18
    subtitle_size = 14
    line_color = (200, 200, 200)

    # Título grande y centrado
    pdf.set_font(base_font, 'B', title_size)
    pdf.ln(8)
    pdf.cell(0, 15, t("pdf_title"), 0, 1, 'C')
    pdf.set_draw_color(*line_color)
    pdf.set_line_width(0.7)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    # Tabla vertical de información general (sin peso)
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(0, 10, t("general_info"), 0, 1, 'C')
    pdf.set_font(base_font, '', base_size)
    info_data = [
        (t("date", date=time.strftime('%d/%m/%Y %H:%M'))),
        (t("food", food=predicted_food.replace('_', ' ').capitalize())),
    ]
    if confidence is not None:
        info_data.append(t("model_confidence_pdf", confidence=f"{confidence*100:.1f}"))
    if processing_time is not None:
        info_data.append(t("processing_time", time=f"{processing_time:.2f}"))
    col1_w, col2_w = 60, 110
    table_x = (210 - (col1_w + col2_w)) // 2
    for item in info_data:
        if ':' in item:
            k, v = item.split(':', 1)
            pdf.set_x(table_x)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(col1_w, 10, k.strip() + ':', 1, 0, 'R', True)
            pdf.cell(col2_w, 10, v.strip(), 1, 1, 'L')
    pdf.ln(8)
    pdf.set_draw_color(*line_color)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    # Imagen centrada
    img_path = "temp_img.png"
    image.save(img_path)
    img_w = 80
    img_x = (210 - img_w) // 2
    img_y = pdf.get_y()
    aspect = image.height / image.width
    img_h = img_w * aspect
    pdf.image(img_path, x=img_x, y=img_y, w=img_w, h=img_h)
    os.remove(img_path)
    pdf.set_y(img_y + img_h + 8)

    # Tabla de valores nutricionales centrada, con peso en el título
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(0, 10, f"{t('pdf_nutritional_values')} ({t('weight', weight=weight)})", 0, 1, 'C')
    pdf.set_font(base_font, '', base_size)
    col_width = 60
    row_height = 10
    table_x = (210 - 2*col_width) // 2
    pdf.set_x(table_x)
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
        pdf.set_x(table_x)
        pdf.cell(col_width, row_height, nutrient, 1, 0, 'L', True)
        pdf.cell(col_width, row_height, value, 1, 1, 'C', True)
    pdf.ln(8)
    pdf.set_draw_color(*line_color)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    # Tabla vertical de detalles del modelo
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(0, 10, t("model_table_title"), 0, 1, 'C')
    pdf.set_font(base_font, '', base_size)
    model_table = [
        (t("model_table_model"), "Xception"),
        (t("model_table_split"), "75% / 25%"),
        (t("model_table_epochs"), "15"),
        (t("model_table_metric"), "63.58%")
    ]
    col1_w, col2_w = 60, 110
    table_x = (210 - (col1_w + col2_w)) // 2
    for k, v in model_table:
        pdf.set_x(table_x)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(col1_w, 10, k, 1, 0, 'R', True)
        pdf.cell(col2_w, 10, v, 1, 1, 'L')
    pdf.ln(8)
    pdf.set_draw_color(*line_color)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    # --- MATRICES DE CONFUSIÓN ---
    confusion_matrices = [
        ("Matriz de Confusión - Xception", 'graphics/cm_Xception.png'),
        ("Matriz de Confusión - Hybrid", 'graphics/cm_Hybrid.png'),
        ("Matriz de Confusión - ResNet50", 'graphics/cm_ResNet50.png'),
        ("Matriz de Confusión - InceptionV3", 'graphics/cm_InceptionV3.png'),
    ]
    for title, path in confusion_matrices:
        # Determina si hay suficiente espacio antes de escribir el título
        img_w = 160
        img_x = (210 - img_w) // 2
        img_y = pdf.get_y()
        if os.path.exists(path):
            confusion_img = PILImage.open(path)
            aspect = confusion_img.height / confusion_img.width
            img_h = img_w * aspect
            # Si no hay suficiente espacio, agrega una nueva página antes de escribir el título
            if img_y + 10 + img_h + 20 > pdf.h - pdf.b_margin:
                pdf.add_page()
                img_y = pdf.get_y()
            pdf.set_font(base_font, 'B', subtitle_size)
            pdf.cell(0, 10, title, 0, 1, 'C')
            img_y = pdf.get_y()
            pdf.image(path, x=img_x, y=img_y, w=img_w, h=img_h)
            pdf.set_y(img_y + img_h + 8)
        else:
            pdf.set_font(base_font, 'B', subtitle_size)
            pdf.cell(0, 10, title, 0, 1, 'C')
            pdf.set_font(base_font, '', base_size)
            pdf.cell(0, 10, f'No se encontró la imagen {title}.', 0, 1, 'C')
        pdf.ln(8)
    # --- FIN MATRICES DE CONFUSIÓN ---

    # Tabla de McNemar en formato vertical
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.cell(0, 10, t("mcnemar_title"), 0, 1, 'C')
    pdf.set_font(base_font, '', base_size)
    mcnemar_data = [
        ("Hybrid vs ResNet50", "115.04", "0.0000", "Diferencia significativa"),
        ("Hybrid vs InceptionV3", "126.96", "0.0000", "Diferencia significativa"),
        ("Hybrid vs Xception", "128.75", "0.0000", "Diferencia significativa"),
        ("ResNet50 vs InceptionV3", "0.09", "0.7658", "Sin diferencia significativa"),
        ("ResNet50 vs Xception", "0.14", "0.7108", "Sin diferencia significativa"),
        ("InceptionV3 vs Xception", "0.46", "0.4959", "Sin diferencia significativa"),
    ]
    col1_w, col2_w = 60, 110
    table_x = (210 - (col1_w + col2_w)) // 2
    for comparison, stat, pval, conclusion in mcnemar_data:
        pdf.set_x(table_x)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(col1_w, 10, t("mcnemar_comparison"), 1, 0, 'R', True)
        pdf.cell(col2_w, 10, comparison, 1, 1, 'L')
        pdf.set_x(table_x)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(col1_w, 10, t("mcnemar_statistic"), 1, 0, 'R', True)
        pdf.cell(col2_w, 10, stat, 1, 1, 'L')
        pdf.set_x(table_x)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(col1_w, 10, t("mcnemar_pvalue"), 1, 0, 'R', True)
        pdf.cell(col2_w, 10, pval, 1, 1, 'L')
        pdf.set_x(table_x)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(col1_w, 10, t("mcnemar_conclusion"), 1, 0, 'R', True)
        pdf.cell(col2_w, 10, conclusion, 1, 1, 'L')
        pdf.ln(4)
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

@st.cache_resource(ttl=3600)
def load_trained_model():
    """
    Carga el modelo entrenado optimizado para Streamlit Cloud
    """
    try:
        # Configuración del modelo
        MODEL_PATH = "models/inceptionv3_food101.h5"
        GOOGLE_DRIVE_FILE_ID = "1aopQMrls2c4eRfhCCk2csrNm-ftcth3i"
        
        # Crear directorio si no existe
        os.makedirs("models", exist_ok=True)
        
        # Verificar si el modelo ya existe y es válido
        if os.path.exists(MODEL_PATH):
            try:
                file_size = os.path.getsize(MODEL_PATH)
                if file_size > 50 * 1024 * 1024:  # Al menos 50MB
                    st.info(f"Cargando modelo existente ({file_size/1024/1024:.1f} MB)...")
                    
                    # Configurar opciones de carga para TensorFlow 2.20+ con Keras 3.x
                    try:
                        # Intentar cargar con configuración moderna
                        model = tf.keras.models.load_model(
                            MODEL_PATH, 
                            compile=False,
                            safe_mode=False  # Para compatibilidad con modelos antiguos
                        )
                        
                        # Recompilar el modelo con configuración básica
                        model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        st.success("✅ Modelo cargado exitosamente")
                        return model
                        
                    except Exception as e:
                        st.warning(f"Error al cargar modelo existente: {str(e)}")
                        # Eliminar archivo corrupto
                        os.remove(MODEL_PATH)
                else:
                    st.warning("Archivo de modelo muy pequeño, re-descargando...")
                    os.remove(MODEL_PATH)
            except Exception as e:
                st.warning(f"Error verificando modelo: {str(e)}")
        
        # Descargar el modelo
        st.info("Descargando modelo desde Google Drive...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # URL directa de descarga
            url = f'https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}'
            
            # Descargar con requests para mejor control
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                with open(MODEL_PATH, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size
                                progress_bar.progress(progress)
                                status_text.text(f"Descargado: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB")
                
                progress_bar.progress(1.0)
                status_text.text("Descarga completada")
                
            else:
                # Fallback con gdown
                st.info("Intentando descarga alternativa con gdown...")
                gdown.download(
                    f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}',
                    MODEL_PATH,
                    quiet=False
                )
        
        except Exception as e:
            st.error(f"Error en descarga: {str(e)}")
            # Intentar descarga directa con gdown como último recurso
            try:
                st.info("Intentando descarga directa...")
                gdown.download(
                    f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}',
                    MODEL_PATH,
                    quiet=False
                )
            except Exception as e2:
                st.error(f"Error en descarga alternativa: {str(e2)}")
                return None
        
        # Limpiar elementos de progreso
        progress_bar.empty()
        status_text.empty()
        
        # Verificar descarga exitosa
        if not os.path.exists(MODEL_PATH):
            st.error("❌ No se pudo descargar el modelo")
            return None
        
        file_size = os.path.getsize(MODEL_PATH)
        if file_size < 50 * 1024 * 1024:
            st.error(f"❌ Archivo descargado muy pequeño: {file_size/1024/1024:.1f} MB")
            os.remove(MODEL_PATH)
            return None
        
        st.success(f"✅ Modelo descargado: {file_size/1024/1024:.1f} MB")
        
        # Cargar el modelo descargado
        try:
            st.info("Cargando modelo en memoria...")
            
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,  # No compilar automáticamente
                safe_mode=False  # Para compatibilidad con modelos Keras 2.x en TF 2.20+
            )
            
            # Recompilar con configuración básica
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success("✅ Modelo cargado y listo para usar")
            return model
            
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {str(e)}")
            # Mostrar información adicional para debugging
            try:
                import h5py
                with h5py.File(MODEL_PATH, 'r') as f:
                    st.info("Estructura del archivo HDF5:")
                    def print_structure(name, obj):
                        st.write(f"- {name}: {type(obj)}")
                    f.visititems(print_structure)
            except Exception as h5_error:
                st.error(f"No se puede analizar el archivo HDF5: {str(h5_error)}")
            
            return None
    
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Cargar modelo
model = load_trained_model()

if model is None:
    st.error("❌ No se pudo cargar el modelo. La aplicación no puede continuar.")
    st.info("""
    **Posibles soluciones:**
    1. Verifica que el enlace de Google Drive sea público
    2. Asegúrate de que el ID del archivo sea correcto
    3. Verifica que el archivo sea un modelo .h5 válido de TensorFlow/Keras
    4. Intenta recargar la página
    """)
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