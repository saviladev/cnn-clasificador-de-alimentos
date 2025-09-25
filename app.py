# app.py
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

# =============== PDF ===============
def generate_pdf_report(image, predicted_food, protein, fat, carbs, kcal, weight, confidence=None, processing_time=None):
    from PIL import Image as PILImage
    pdf = FPDF(); pdf.add_page()
    pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
    base_font='DejaVu'; base_size=12; title_size=18; subtitle_size=14; line_color=(200,200,200)

    pdf.set_font(base_font,'B',title_size); pdf.ln(8); pdf.cell(0,15,t("pdf_title"),0,1,'C')
    pdf.set_draw_color(*line_color); pdf.set_line_width(0.7); pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(8)

    pdf.set_font(base_font,'B',subtitle_size); pdf.cell(0,10,t("general_info"),0,1,'C')
    pdf.set_font(base_font,'',base_size)
    info = [(t("date", date=time.strftime('%d/%m/%Y %H:%M'))),
            (t("food", food=predicted_food.replace('_',' ').capitalize()))]
    if confidence is not None: info.append(t("model_confidence_pdf", confidence=f"{confidence*100:.1f}"))
    if processing_time is not None: info.append(t("processing_time", time=f"{processing_time:.2f}"))
    col1_w,col2_w=60,110; table_x=(210-(col1_w+col2_w))//2
    for item in info:
        if ':' in item:
            k,v=item.split(':',1); pdf.set_x(table_x); pdf.set_fill_color(240,240,240)
            pdf.cell(col1_w,10,k.strip()+':',1,0,'R',True); pdf.cell(col2_w,10,v.strip(),1,1,'L')
    pdf.ln(8); pdf.set_draw_color(*line_color); pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(8)

    img_path="temp_img.png"; image.save(img_path); img_w=80; img_x=(210-img_w)//2; img_y=pdf.get_y()
    aspect=image.height/image.width; img_h=img_w*aspect; pdf.image(img_path,x=img_x,y=img_y,w=img_w,h=img_h); os.remove(img_path)
    pdf.set_y(img_y+img_h+8)

    pdf.set_font(base_font,'B',subtitle_size); pdf.cell(0,10,f"{t('pdf_nutritional_values')} ({t('weight', weight=weight)})",0,1,'C')
    pdf.set_font(base_font,'',base_size); col_width=60; row_height=10; table_x=(210-2*col_width)//2
    pdf.set_x(table_x); pdf.set_fill_color(240,240,240)
    pdf.cell(col_width,row_height,t("nutrient"),1,0,'C',True); pdf.cell(col_width,row_height,t("amount"),1,1,'C',True)
    data=[(t("proteins_name"),f"{protein:.1f} g"),(t("fats_name"),f"{fat:.1f} g"),
          (t("carbs_name"),f"{carbs:.1f} g"),(t("calories_name"),f"{kcal:.1f} kcal")]
    for i,(n,v) in enumerate(data):
        fill=240 if i%2==0 else 255; pdf.set_fill_color(fill,fill,fill); pdf.set_x(table_x)
        pdf.cell(col_width,row_height,n,1,0,'L',True); pdf.cell(col_width,row_height,v,1,1,'C')
    pdf.ln(8); pdf.set_draw_color(*line_color); pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(8)

    pdf.set_font(base_font,'B',subtitle_size); pdf.cell(0,10,t("model_table_title"),0,1,'C')
    pdf.set_font(base_font,'',base_size)
    for k,v in [(t("model_table_model"),"Xception"),
                (t("model_table_split"),"75% / 25%"),
                (t("model_table_epochs"),"15"),
                (t("model_table_metric"),"63.58%")]:
        pdf.set_x((210-(60+110))//2); pdf.set_fill_color(240,240,240)
        pdf.cell(60,10,k,1,0,'R',True); pdf.cell(110,10,v,1,1,'L')
    pdf.ln(8); pdf.set_draw_color(*line_color); pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(8)

    pdf.set_font(base_font,'B',subtitle_size); pdf.cell(0,10,t("mcnemar_title"),0,1,'C')
    pdf.set_font(base_font,'',base_size)
    mcnemar=[("Hybrid vs ResNet50","115.04","0.0000","Diferencia significativa"),
             ("Hybrid vs InceptionV3","126.96","0.0000","Diferencia significativa"),
             ("Hybrid vs Xception","128.75","0.0000","Diferencia significativa"),
             ("ResNet50 vs InceptionV3","0.09","0.7658","Sin diferencia significativa"),
             ("ResNet50 vs Xception","0.14","0.7108","Sin diferencia significativa"),
             ("InceptionV3 vs Xception","0.46","0.4959","Sin diferencia significativa")]
    col1_w,col2_w=60,110; table_x=(210-(col1_w+col2_w))//2
    for comp,stat,p,conc in mcnemar:
        for k,v in [(t("mcnemar_comparison"),comp),
                    (t("mcnemar_statistic"),stat),
                    (t("mcnemar_pvalue"),p),
                    (t("mcnemar_conclusion"),conc)]:
            pdf.set_x(table_x); pdf.set_fill_color(240,240,240)
            pdf.cell(col1_w,10,k,1,0,'R',True); pdf.cell(col2_w,10,v,1,1,'L')
        pdf.ln(4)
    pdf.ln(8)

    out="reporte_nutricional.pdf"; pdf.output(out)
    with open(out,"rb") as f: data=f.read()
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
uploaded_file = st.file_uploader(t("upload_prompt"), type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=t("upload_prompt"), use_column_width=True)

    with st.spinner(t("analyzing")):
        t0 = time.time()
        x = preprocess_image_inception(image)
        y = model.predict(x, verbose=0)
        cls_idx = int(np.argmax(y)); conf = float(np.max(y)); dt = time.time() - t0

    class_names = nutrients_df["clase"].tolist()
    if cls_idx >= len(class_names):
        st.error(t("error_class_index")); st.stop()

    predicted_food = class_names[cls_idx]
    row = nutrients_df[nutrients_df["clase"] == predicted_food]
    if row.empty:
        st.error(t("error_nutrient_data"))
    else:
        r = row.iloc[0]
        protein, fat, carbs, kcal = map(float, [r["proteinas"], r["grasas"], r["carbohidratos"], r["calorias"]])

        food_name = translate_food_name(predicted_food, lang)
        st.success(t("identified_food", food=food_name))
        st.caption(t("model_confidence", confidence=conf*100))
        st.info(t("analysis_time", time=dt))

        weight = st.number_input(t("weight_prompt"), min_value=1.0, value=100.0, step=1.0)
        f = weight/100.0

        st.markdown(t("nutritional_values"))
        st.markdown(f"""
- {t('proteins', value=protein*f)}
- {t('fats', value=fat*f)}
- {t('carbs', value=carbs*f)}
- {t('calories', value=kcal*f)}
""")

        if st.button(t("generate_pdf")):
            with st.spinner(t("generate_pdf")):
                pdf_bytes = generate_pdf_report(
                    image=image, predicted_food=food_name,
                    protein=protein*f, fat=fat*f, carbs=carbs*f, kcal=kcal*f,
                    weight=weight, confidence=conf, processing_time=dt
                )
            st.download_button(
                label=t("download_pdf"),
                data=pdf_bytes,
                file_name=f"reporte_nutricional_{predicted_food.lower().replace(' ','_')}.pdf",
                mime="application/pdf"
            )
            st.success(t("pdf_success"))
