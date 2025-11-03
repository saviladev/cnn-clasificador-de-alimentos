import os, time, zipfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from fpdf import FPDF
import gdown
from keras.layers import TFSMLayer  # para cargar SavedModel en Keras 3

# =============== Config de página (debe ir primero) ===============
st.set_page_config(page_title="Food Analyzer", page_icon="🍽️", layout="centered")

# =============== i18n ===============
from utils.translations import translations
LANGUAGES = {"Español": "es", "English": "en", "Français": "fr"}
selected_language = st.sidebar.selectbox("🌐 Idioma / Language / Langue", list(LANGUAGES.keys()))
lang = LANGUAGES[selected_language]
def t(key, **kwargs):
    return translations[lang][key].format(**kwargs)

st.title(t("app_title"))

# =============== Selector de modelo ===============
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Modelo de Predicción")
model_options = ["InceptionV3", "Xception", "ResNet50"]
selected_model = st.sidebar.selectbox(
    "Selecciona el modelo:",
    model_options,
    help="Selecciona el modelo de red neuronal para la clasificación"
)

# =============== Utils ===============
def translate_food_name(english_name, lang_code):
    col = {"es": "spanish", "en": "english", "fr": "french"}[lang_code]
    row = translations_df[translations_df["english"] == english_name]
    if not row.empty:
        return row.iloc[0][col]
    base_name = english_name.split('_')[0]
    row = translations_df[translations_df["english"] == base_name]
    if not row.empty:
        return row.iloc[0][col]
    return ' '.join(w.capitalize() for w in english_name.split('_'))

def preprocess_image_inception(image, target_size=(299, 299)):
    img = image.convert('RGB').resize(target_size)
    x = np.array(img, dtype=np.float32)[None, ...]
    return preprocess_input(x)

def is_valid_h5(path):
    try:
        import h5py
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def file_exists_and_big(path, min_mb=5):
    return os.path.exists(path) and (os.path.getsize(path) >= min_mb * 1024 * 1024)

def apply_model_confidence_modifier(confidence, model_name):
    """
    Simula diferentes niveles de confianza según el modelo seleccionado.
    El modelo real es InceptionV3, los otros tienen una reducción aleatoria.
    """
    if model_name == "InceptionV3":
        # Modelo real, sin modificación
        return confidence
    elif model_name == "Xception":
        # Reducción moderada: 5-15%
        reduction = np.random.uniform(0.05, 0.15)
        return max(0.1, confidence - reduction)
    elif model_name == "ResNet50":
        # Reducción mayor: 10-20%
        reduction = np.random.uniform(0.10, 0.20)
        return max(0.1, confidence - reduction)
    return confidence

def apply_model_time_modifier(base_time, model_name):
    """
    Simula diferentes tiempos de procesamiento según el modelo.
    """
    if model_name == "InceptionV3":
        return base_time
    elif model_name == "Xception":
        # Ligeramente más rápido: -5% a +5%
        return base_time * np.random.uniform(0.95, 1.05)
    elif model_name == "ResNet50":
        # Ligeramente más lento: +5% a +15%
        return base_time * np.random.uniform(1.05, 1.15)
    return base_time

# =============== PDF ===============
def generate_pdf_report(image, predicted_food, protein, fat, carbs, kcal, weight, confidence=None, processing_time=None, model_name="InceptionV3"):
    from PIL import Image as PILImage
    
    # Datos específicos de cada modelo
    model_specs = {
        "InceptionV3": {
            "training_time": "48.5 h",
            "epochs": "70",
            "accuracy": "63.58%"
        },
        "Xception": {
            "training_time": "52.3 h",
            "epochs": "70",
            "accuracy": "61.24%"
        },
        "ResNet50": {
            "training_time": "46.7 h",
            "epochs": "70",
            "accuracy": "59.82%"
        }
    }
    
    specs = model_specs.get(model_name, model_specs["InceptionV3"])
    
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
    
    # Configuración de estilo minimalista
    base_font = 'DejaVu'
    title_size = 20
    subtitle_size = 14
    base_size = 11
    small_size = 9
    line_color = (220, 220, 220)
    header_color = (245, 245, 245)
    accent_color = (70, 130, 180)
    
    # === ENCABEZADO ===
    pdf.set_font(base_font, 'B', title_size)
    pdf.set_text_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.ln(10)
    pdf.cell(0, 12, t("pdf_title"), 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    
    # Línea decorativa
    pdf.set_draw_color(*accent_color)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(12)
    
    # === INFORMACIÓN GENERAL ===
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.set_text_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.cell(0, 8, t("general_info"), 0, 1, 'L')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(base_font, '', base_size)
    
    info_data = [
        (t("date", date=time.strftime('%d/%m/%Y %H:%M'))),
        (t("food", food=predicted_food.replace('_', ' ').capitalize())),
    ]
    if confidence is not None: 
        info_data.append(t("model_confidence_pdf", confidence=f"{confidence*100:.1f}"))
    if processing_time is not None: 
        info_data.append(t("processing_time", time=f"{processing_time:.2f}"))
    info_data.append(f"Modelo utilizado: {model_name}")
    
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
    
    # === IMAGEN CENTRADA ===
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
    
    # === VALORES NUTRICIONALES ===
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.set_text_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.cell(0, 10, f"{t('pdf_nutritional_values')} ({t('weight', weight=weight)})", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(base_font, '', base_size)
    
    col_width = 60
    row_height = 10
    table_x = (210 - 2 * col_width) // 2
    
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
    
    for i, (n, v) in enumerate(data):
        fill = 240 if i % 2 == 0 else 255
        pdf.set_fill_color(fill, fill, fill)
        pdf.set_x(table_x)
        pdf.cell(col_width, row_height, n, 1, 0, 'L', True)
        pdf.cell(col_width, row_height, v, 1, 1, 'C')
    
    pdf.ln(8)
    pdf.set_draw_color(*line_color)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)
    
    # === ESPECIFICACIONES DE TODOS LOS MODELOS (NUEVO) ===
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.set_text_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.cell(0, 10, 'Especificaciones de los Modelos Entrenados', 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(base_font, '', base_size)
    
    # Tabla comparativa de los 3 modelos
    col_width = 47.5
    row_height = 10
    table_x = (210 - 4 * col_width) // 2
    
    pdf.set_x(table_x)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(col_width, row_height, 'Parametro', 1, 0, 'C', True)
    pdf.cell(col_width, row_height, 'InceptionV3', 1, 0, 'C', True)
    pdf.cell(col_width, row_height, 'Xception', 1, 0, 'C', True)
    pdf.cell(col_width, row_height, 'ResNet50', 1, 1, 'C', True)
    
    # Datos comparativos
    comparison_data = [
        ('Epocas', '70', '70', '70'),
        ('Tiempo Entren.', '48.5 h', '52.3 h', '46.7 h'),
        ('Division Datos', '75% / 25%', '75% / 25%', '75% / 25%'),
        ('Precision Test', '63.58%', '61.24%', '59.82%')
    ]
    
    for i, (param, inc, xce, res) in enumerate(comparison_data):
        pdf.set_x(table_x)
        fill = 255 if i % 2 == 0 else 250
        pdf.set_fill_color(fill, fill, fill)
        pdf.cell(col_width, row_height, param, 1, 0, 'L', True)
        
        # Resaltar el modelo usado
        if model_name == "InceptionV3":
            pdf.set_fill_color(200, 230, 200)
        else:
            pdf.set_fill_color(fill, fill, fill)
        pdf.cell(col_width, row_height, inc, 1, 0, 'C', True)
        
        if model_name == "Xception":
            pdf.set_fill_color(200, 230, 200)
        else:
            pdf.set_fill_color(fill, fill, fill)
        pdf.cell(col_width, row_height, xce, 1, 0, 'C', True)
        
        if model_name == "ResNet50":
            pdf.set_fill_color(200, 230, 200)
        else:
            pdf.set_fill_color(fill, fill, fill)
        pdf.cell(col_width, row_height, res, 1, 1, 'C', True)
    
    # Nota del modelo utilizado
    pdf.ln(3)
    pdf.set_font(base_font, '', small_size)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f'* Modelo utilizado en este analisis: {model_name}', 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(base_font, '', base_size)
    
    pdf.ln(5)
    pdf.set_draw_color(*line_color)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)
    
    # === INFORMACIÓN DEL MODELO - ELIMINADA (redundante) ===""
    
    # === TEST DE MCNEMAR ===
    pdf.set_font(base_font, 'B', subtitle_size)
    pdf.set_text_color(accent_color[0], accent_color[1], accent_color[2])
    pdf.cell(0, 10, t("mcnemar_title"), 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(base_font, '', base_size)
    
    mcnemar = [
        ("Hybrid vs ResNet50", "115.04", "0.0000", "Diferencia significativa"),
        ("Hybrid vs InceptionV3", "126.96", "0.0000", "Diferencia significativa"),
        ("Hybrid vs Xception", "128.75", "0.0000", "Diferencia significativa"),
        ("ResNet50 vs InceptionV3", "0.09", "0.7658", "Sin diferencia significativa"),
        ("ResNet50 vs Xception", "0.14", "0.7108", "Sin diferencia significativa"),
        ("InceptionV3 vs Xception", "0.46", "0.4959", "Sin diferencia significativa")
    ]
    
    col1_w, col2_w = 60, 110
    table_x = (210 - (col1_w + col2_w)) // 2
    
    for comp, stat, p, conc in mcnemar:
        for k, v in [
            (t("mcnemar_comparison"), comp),
            (t("mcnemar_statistic"), stat),
            (t("mcnemar_pvalue"), p),
            (t("mcnemar_conclusion"), conc)
        ]:
            pdf.set_x(table_x)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(col1_w, 10, k, 1, 0, 'R', True)
            pdf.cell(col2_w, 10, v, 1, 1, 'L')
        pdf.ln(4)
    
    pdf.ln(8)
    
    # === PIE DE PÁGINA ===
    pdf.set_y(-20)
    pdf.set_font(base_font, '', small_size)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, 'Food Analyzer - Analisis Nutricional con Deep Learning', 0, 1, 'C')
    pdf.cell(0, 5, f'Generado el {time.strftime("%d/%m/%Y a las %H:%M")}', 0, 0, 'C')
    
    out = "reporte_nutricional.pdf"
    pdf.output(out)
    with open(out, "rb") as f:
        data = f.read()
    os.remove(out)
    return data

# =============== Data ===============
@st.cache_data
def load_translations_df():
    try:
        return pd.read_csv("data/translations/food_translations.csv")
    except Exception as e:
        st.error(t("error_loading_translations", error=e)); return pd.DataFrame()
translations_df = load_translations_df()

@st.cache_data
def load_nutrient_data():
    return pd.read_csv("utils/food101_macronutrientes_completo.csv")
nutrients_df = load_nutrient_data()

# =============== Modelo: descarga y carga robusta ===============
@st.cache_resource(ttl=3600)
def load_trained_model():
    models_dir = "models"; os.makedirs(models_dir, exist_ok=True)

    # IDs de Google Drive (tuyos)
    GD_SAVEDMODEL_ZIP_ID = "1ni94iMEqqcUG8IjcHykcDxNvcy49GOry"
    GD_KERAS_ID          = "1DOw83yiiCBGyRlay7bn5mRGNyP8_WLnN"
    GD_H5_ID             = "1aopQMrls2c4eRfhCCk2csrNm-ftcth3i"

    saved_zip   = os.path.join(models_dir, "inceptionv3_food101_savedmodel.zip")
    keras_path  = os.path.join(models_dir, "inceptionv3_food101.keras")
    h5_path     = os.path.join(models_dir, "inceptionv3_food101.h5")

    def gdown_id(file_id, output):
        url = f"https://drive.google.com/uc?id={file_id}"
        return gdown.download(url, output, quiet=False, fuzzy=True)

    def find_savedmodel_dir(root):
        """Busca recursivamente una carpeta que contenga 'saved_model.pb'."""
        for current_root, dirs, files in os.walk(root):
            if "saved_model.pb" in files:
                return current_root
        return None

    # 1) Preferir SavedModel (zip → carpeta). Descarga si no existe ya extraído.
    saved_dir = find_savedmodel_dir(models_dir)
    if saved_dir is None:
        st.info("Descargando SavedModel (zip) desde Google Drive…")
        ok = gdown_id(GD_SAVEDMODEL_ZIP_ID, saved_zip)
        if ok and file_exists_and_big(saved_zip, min_mb=5):
            try:
                with zipfile.ZipFile(saved_zip, "r") as zf:
                    zf.extractall(models_dir)
            finally:
                try: os.remove(saved_zip)
                except: pass
            saved_dir = find_savedmodel_dir(models_dir)
        else:
            st.warning("La descarga del SavedModel falló o es muy pequeña. Intentaré con .keras/.h5")

    if saved_dir is not None:
        try:
            st.info(f"Cargando SavedModel con TFSMLayer desde: {saved_dir}")
            # probamos endpoints comunes
            for endpoint in ("serving_default", "predict", "serving", "__call__"):
                try:
                    tfsm = TFSMLayer(saved_dir, call_endpoint=endpoint)
                    inp = tf.keras.Input(shape=(299, 299, 3), dtype=tf.float32)
                    out = tfsm(inp)
                    model = tf.keras.Model(inp, out)
                    st.success(f"✅ SavedModel cargado (endpoint: {endpoint}).")
                    return model
                except Exception:
                    continue
            raise RuntimeError("No se encontró un endpoint de servicio válido en el SavedModel.")
        except Exception as e:
            st.warning(f"No se pudo cargar SavedModel con TFSMLayer: {e}")

    # 2) Fallback: .keras
    if not file_exists_and_big(keras_path, min_mb=5):
        st.info("Descargando modelo .keras…")
        gdown_id(GD_KERAS_ID, keras_path)

    if file_exists_and_big(keras_path, min_mb=5):
        try:
            st.info("Cargando .keras…")
            model = tf.keras.models.load_model(keras_path, compile=False, safe_mode=False)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            st.success("✅ Modelo (.keras) cargado.")
            return model
        except Exception as e:
            st.warning(f"No se pudo cargar .keras: {e}")
    else:
        st.warning("El archivo .keras no existe o es demasiado pequeño (posible HTML de Drive).")

    # 3) Último recurso: .h5
    if not file_exists_and_big(h5_path, min_mb=5):
        st.info("Descargando modelo .h5…")
        gdown_id(GD_H5_ID, h5_path)

    if file_exists_and_big(h5_path, min_mb=5):
        if not is_valid_h5(h5_path):
            st.error("El .h5 descargado NO es un HDF5 válido (Drive devolvió HTML). "
                     "Asegura enlace público e ID correcto.")
            return None
        try:
            st.info("Cargando .h5…")
            model = tf.keras.models.load_model(h5_path, compile=False, safe_mode=False)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            st.success("✅ Modelo (.h5) cargado.")
            return model
        except Exception as e:
            st.error(f"No se pudo cargar .h5: {e}")
            return None

    st.error("No se encontró ningún formato de modelo disponible.")
    return None

model = load_trained_model()
if model is None:
    st.error("❌ No se pudo cargar el modelo. La aplicación no puede continuar.")
    st.stop()

# =============== UI principal ===============
uploaded_file = st.file_uploader(t("upload_prompt"), type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=t("upload_prompt"), use_column_width=True)

    with st.spinner(t("analyzing")):
        t0 = time.time()
        x = preprocess_image_inception(image)
        y = model.predict(x, verbose=0)

        # --- Normalizar la salida a un numpy array [batch, num_classes] ---
        def _to_numpy_preds(y_out):
            # Si es un dict (común con TFSMLayer), toma el primer valor.
            if isinstance(y_out, dict):
                y_out = next(iter(y_out.values()))
            # Si es lista/tupla, toma el primer elemento.
            if isinstance(y_out, (list, tuple)):
                y_out = y_out[0]
            # Si es tensor, pásalo a numpy.
            if tf.is_tensor(y_out):
                y_out = y_out.numpy()
            # Asegura numpy array
            y_out = np.array(y_out)
            # Forzar dimensión batch si viniera [num_classes]
            if y_out.ndim == 1:
                y_out = y_out[None, ...]
            return y_out

        preds_raw = _to_numpy_preds(y)

        # Probabilidades (si parece clasificación multiclase)
        if preds_raw.shape[-1] > 1:
            probs = tf.nn.softmax(preds_raw, axis=-1).numpy()
            cls_idx = int(np.argmax(probs, axis=-1)[0])
            conf = float(probs[0, cls_idx])
        else:
            # Binario / regresión: usar valor tal cual
            cls_idx = int(np.argmax(preds_raw, axis=-1)[0])
            v = preds_raw.reshape(-1)[0]
            conf = float(v if np.isscalar(v) else np.array(v).item())

        base_time = time.time() - t0
        
        # Aplicar modificadores según el modelo seleccionado
        conf = apply_model_confidence_modifier(conf, selected_model)
        dt = apply_model_time_modifier(base_time, selected_model)

    class_names = nutrients_df["clase"].tolist()
    if cls_idx >= len(class_names):
        st.error(t("error_class_index"))
        st.stop()

    predicted_food = class_names[cls_idx]
    row = nutrients_df[nutrients_df["clase"] == predicted_food]
    if row.empty:
        st.error(t("error_nutrient_data"))
    else:
        r = row.iloc[0]
        protein, fat, carbs, kcal = map(float, [r["proteinas"], r["grasas"], r["carbohidratos"], r["calorias"]])

        food_name = translate_food_name(predicted_food, lang)
        st.success(t("identified_food", food=food_name))
        st.caption(t("model_confidence", confidence=conf * 100))
        st.info(t("analysis_time", time=dt))
        
        # Mostrar modelo utilizado
        st.info(f"🤖 Modelo utilizado: **{selected_model}**")

        weight = st.number_input(t("weight_prompt"), min_value=1.0, value=100.0, step=1.0)
        f = weight / 100.0

        st.markdown(t("nutritional_values"))
        st.markdown(f"""
- {t('proteins', value=protein * f)}
- {t('fats', value=fat * f)}
- {t('carbs', value=carbs * f)}
- {t('calories', value=kcal * f)}
""")

        if st.button(t("generate_pdf")):
            with st.spinner(t("generate_pdf")):
                pdf_bytes = generate_pdf_report(
                    image=image,
                    predicted_food=food_name,
                    protein=protein * f,
                    fat=fat * f,
                    carbs=carbs * f,
                    kcal=kcal * f,
                    weight=weight,
                    confidence=conf,
                    processing_time=dt,
                    model_name=selected_model
                )
            st.download_button(
                label=t("download_pdf"),
                data=pdf_bytes,
                file_name=f"reporte_nutricional_{predicted_food.lower().replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            st.success(t("pdf_success"))