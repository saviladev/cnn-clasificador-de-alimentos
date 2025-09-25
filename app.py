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

@st.cache_resource(ttl=3600)  # Cache por 1 hora
def load_trained_model():
    """
    Carga el modelo entrenado. Si no existe localmente, lo descarga desde Google Drive.
    Soporta tanto archivos .h5 como saved_model.
    """
    try:
        # Crear directorio de modelos si no existe
        os.makedirs("models", exist_ok=True)
        
        # Configuración del modelo
        MODEL_DIR = "models/inceptionv3_food101"
        MODEL_H5 = f"{MODEL_DIR}.h5"
        GOOGLE_DRIVE_FILE_ID = "1aopQMrls2c4eRfhCCk2csrNm-ftcth3i"
        
        # Mostrar información de depuración
        st.info(f"Buscando modelo en: {os.path.abspath(MODEL_H5)}")
        
        # Verificar si ya tenemos el modelo en formato .h5 o saved_model
        if os.path.exists(MODEL_H5) or os.path.exists(MODEL_DIR):
            try:
                # Verificar si es un archivo .h5
                if os.path.exists(MODEL_H5):
                    file_size = os.path.getsize(MODEL_H5)
                    st.info(f"Archivo .h5 encontrado. Tamaño: {file_size/1024/1024:.2f} MB")
                    
                    if file_size < 100 * 1024 * 1024:  # Menos de 100MB probablemente está corrupto
                        st.warning(f"⚠️ El archivo del modelo parece estar corrupto o incompleto ({file_size/1024/1024:.2f} MB). Volviendo a descargar...")
                        os.remove(MODEL_H5)
                    else:
                        with st.spinner("Cargando modelo local (formato .h5)..."):
                            try:
                                model = load_model(MODEL_H5)
                                st.success("✅ Modelo cargado exitosamente desde archivo .h5 local")
                                return model
                            except Exception as e:
                                st.error(f"Error al cargar el archivo .h5: {str(e)}")
                                os.remove(MODEL_H5)  # Eliminar archivo corrupto
                
                # Verificar si es un saved_model
                if os.path.exists(MODEL_DIR):
                    st.info("Buscando modelo en formato saved_model...")
                    with st.spinner("Cargando modelo local (formato saved_model)..."):
                        try:
                            model = tf.keras.models.load_model(MODEL_DIR)
                            st.success("✅ Modelo cargado exitosamente desde saved_model local")
                            return model
                        except Exception as e:
                            st.error(f"Error al cargar saved_model: {str(e)}")
                            import shutil
                            shutil.rmtree(MODEL_DIR)
                            
            except Exception as e:
                st.warning(f"⚠️ Error al cargar el modelo local: {str(e)}. Intentando descargar...")
                # Limpiar archivos corruptos
                if os.path.exists(MODEL_H5):
                    os.remove(MODEL_H5)
                if os.path.exists(MODEL_DIR):
                    import shutil
                    shutil.rmtree(MODEL_DIR)
        
        # Si llegamos aquí, necesitamos descargar el modelo
        with st.spinner("Descargando modelo desde Google Drive (esto puede tomar varios minutos)..."):
            try:
                # Primero intentamos descargar como .h5
                temp_file = os.path.join("models", "temp_model.h5")
                url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
                
                # Asegurarse de que el directorio existe
                os.makedirs(os.path.dirname(temp_file), exist_ok=True)
                
                # Configurar SSL para evitar problemas de certificado
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                
                st.info("Descargando modelo (puede tardar varios minutos)...")
                
                # Usar gdown para descargar con más información de depuración
                import gdown
                st.info(f"URL de descarga: {url}")
                st.info(f"Guardando en: {os.path.abspath(temp_file)}")
                
                # Descargar con verificación
                output = gdown.download(url, temp_file, quiet=False, fuzzy=True)
                st.info(f"Resultado de la descarga: {output}")
                
                # Verificar que el archivo se descargó correctamente
                if os.path.exists(temp_file):
                    file_size = os.path.getsize(temp_file)
                    st.info(f"Tamaño del archivo descargado: {file_size/1024/1024:.2f} MB")
                    
                    if file_size > 100 * 1024 * 1024:  # Verificar tamaño razonable
                        try:
                            # Intentar cargar como .h5
                            st.info("Intentando cargar como archivo .h5...")
                            try:
                                # Primero verificar si es un archivo .h5 válido
                                import h5py
                                with h5py.File(temp_file, 'r') as f:
                                    st.info("Estructura del archivo .h5:")
                                    for key in f.keys():
                                        st.info(f"- {key}: {f[key]}")
                                
                                # Si llegamos aquí, el archivo parece ser un HDF5 válido
                                st.info("El archivo parece ser un HDF5 válido. Cargando modelo...")
                                model = load_model(temp_file)
                                
                                # Si llegamos aquí, la carga fue exitosa
                                os.rename(temp_file, MODEL_H5)
                                st.success(f"✅ Modelo .h5 cargado exitosamente ({file_size/1024/1024:.2f} MB)")
                                return model
                                
                            except Exception as load_error:
                                st.error(f"❌ Error al cargar el modelo .h5: {str(load_error)}")
                                # Intentar cargar con tensorflow directamente
                                st.info("Intentando cargar con tensorflow.keras.models.load_model...")
                                try:
                                    model = tf.keras.models.load_model(temp_file)
                                    os.rename(temp_file, MODEL_H5)
                                    st.success(f"✅ Modelo cargado exitosamente con tensorflow.keras ({file_size/1024/1024:.2f} MB)")
                                    return model
                                except Exception as tf_error:
                                    st.error(f"❌ Error con tensorflow.keras: {str(tf_error)}")
                                    raise load_error  # Relanzar el error original
                        except Exception as e:
                            st.warning(f"No es un archivo .h5 válido: {str(e)}. Intentando como saved_model...")
                            # Podría ser un saved_model comprimido
                            try:
                                import zipfile
                                st.info("Intentando extraer como archivo ZIP...")
                                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                                    zip_ref.extractall(MODEL_DIR)
                                # Verificar si se extrajo correctamente
                                if os.path.exists(os.path.join(MODEL_DIR, 'saved_model.pb')):
                                    st.info("Archivo saved_model.pb encontrado, cargando modelo...")
                                    model = tf.keras.models.load_model(MODEL_DIR)
                                    st.success(f"✅ Modelo saved_model descargado y cargado exitosamente")
                                    return model
                                else:
                                    st.error("❌ El archivo ZIP no contiene un modelo válido (no se encontró saved_model.pb)")
                            except Exception as e2:
                                st.error(f"❌ No se pudo extraer ni cargar el archivo: {str(e2)}")
                    else:
                        st.error(f"❌ El archivo descargado es demasiado pequeño ({file_size/1024/1024:.2f} MB). Debería ser mayor a 100MB.")
                    
                    # Mostrar contenido del directorio temporal para depuración
                    st.info("Contenido del directorio temporal:")
                    for f in os.listdir("models"):
                        try:
                            size = os.path.getsize(os.path.join("models", f))
                            st.info(f"- {f} ({size/1024/1024:.2f} MB)")
                        except:
                            st.info(f"- {f} (error al obtener tamaño)")
                    
                    # Limpiar archivo temporal
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                else:
                    st.error("❌ No se pudo descargar el archivo. Verifica que el ID de Google Drive sea correcto.")
                
            except Exception as e:
                st.error(f"❌ Error al descargar el modelo: {str(e)}")
                import traceback
                st.error(f"Detalles del error: {traceback.format_exc()}")
        
        # Mostrar información adicional de ayuda
        st.error("""
        ❌ No se pudo cargar el modelo. Por favor verifica:
        1. Que el archivo en Google Drive sea público (cualquiera con el enlace puede verlo)
        2. Que el ID del archivo sea correcto
        3. Que el archivo sea un modelo .h5 o saved_model.zip válido
        4. Que tengas suficiente espacio en disco
        """)
        return None
        
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
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
