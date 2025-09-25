# --- Google Colab cell: copia TODO este bloque y ejecútalo ---

# 1) Instalar dependencias necesarias
!pip install -q streamlit fpdf tensorflow pandas pyngrok pytz

# 2) Montar Google Drive y fijar la carpeta del proyecto
import os, sys, time
from google.colab import drive

drive.mount('/content/drive')
project_path = '/content/drive/MyDrive/macronutrients-app'
os.chdir(project_path)
sys.path.append(project_path)  # para que Colab y Streamlit encuentren utils/

# 3) Crear archivo de la app Streamlit dentro del proyecto
app_code = """
import streamlit as st
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import pandas as pd, os, time
from fpdf import FPDF
from utils.translations import translations
from datetime import datetime
import pytz

# === CONFIGURACIÓN DE IDIOMA ===
LANGUAGES = {"Español": "es", "English": "en", "Français": "fr"}
selected_language = st.sidebar.selectbox("🌐 Idioma / Language / Langue",
                                         list(LANGUAGES.keys()))
lang = LANGUAGES[selected_language]

def t(key, **kwargs):
    try:
        return translations[lang][key].format(**kwargs)
    except KeyError:
        return f"[{key}]"

# === RUTAS ===
MODEL_PATH          = "models/inceptionv3_food101.h5"
TRANSLATIONS_CSV    = "data/translations/food_translations.csv"
NUTRIENTS_CSV       = "utils/food101_macronutrientes_completo.csv"
FONT_NORMAL_PATH    = "fonts/DejaVuSans.ttf"
FONT_BOLD_PATH      = "fonts/DejaVuSans-Bold.ttf"

# === CARGA DE DATOS ===
@st.cache_resource
def load_trained_model():
    try:
        return load_model(MODEL_PATH)
    except Exception:
        st.error(t("error_loading_model"))
        return None

@st.cache_data
def load_translations_csv():
    try:
        return pd.read_csv(TRANSLATIONS_CSV)
    except Exception as e:
        st.error(t("error_loading_translations", error=e))
        return pd.DataFrame()

@st.cache_data
def load_nutrients():
    return pd.read_csv(NUTRIENTS_CSV)

model          = load_trained_model()
translations_df = load_translations_csv()
nutrients_df    = load_nutrients()

if model is None:
    st.stop()

# === UTILIDADES ===
def translate_food_name(english_name, lang):
    col = {"es": "spanish", "en": "english", "fr": "french"}.get(lang, "english")
    row = translations_df[translations_df["english"] == english_name]
    if not row.empty:
        return row.iloc[0][col]
    return " ".join(w.capitalize() for w in english_name.split("_"))

def preprocess_image_inception(image, target_size=(299, 299)):
    img = image.convert("RGB").resize(target_size)
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    return preprocess_input(arr)

# === PDF ===
def generate_pdf_report(image, predicted_food,
                        protein, fat, carbs, kcal, weight,
                        confidence=None, processing_time=None):

    pdf = FPDF()
    pdf.set_auto_page_break(False)
    pdf.add_page()

    pdf.add_font("DejaVu", "", FONT_NORMAL_PATH,  uni=True)
    pdf.add_font("DejaVu", "B", FONT_BOLD_PATH, uni=True)
    base, title_sz, sub_sz = "DejaVu", 18, 14
    line_col = (200, 200, 200)
    col_w = 70

    # helper -------------------------------------------------------------
    def centered_row(label, value):
        lines = pdf.multi_cell(col_w, 5, value, split_only=True)
        h     = 5 * max(1, len(lines))
        if pdf.get_y() + h > pdf.h - pdf.b_margin:
            pdf.add_page()
        x = pdf.l_margin + (pdf.w - pdf.l_margin - pdf.r_margin - 2*col_w)/2
        y = pdf.get_y()
        pdf.set_xy(x, y)
        pdf.set_fill_color(240, 240, 240)
        pdf.multi_cell(col_w, h, label, border=1, align="R", fill=True)
        pdf.set_xy(x + col_w, y)
        pdf.multi_cell(col_w, 5, value, border=1, align="L")
        pdf.ln(0)
    # -------------------------------------------------------------------

    # título -------------------------------------------------------------
    pdf.set_font(base, "B", title_sz)
    pdf.ln(8); pdf.cell(0, 15, t("pdf_title"), 0, 1, "C")
    pdf.set_draw_color(*line_col); pdf.set_line_width(0.7)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(8)

    # info general -------------------------------------------------------
    pdf.set_font(base, "B", sub_sz)
    pdf.cell(0, 10, t("general_info"), 0, 1, "C")
    pdf.set_font(base, "", 12)

    peru_tz  = pytz.timezone("America/Lima")
    local_dt = datetime.now(peru_tz).strftime("%d/%m/%Y %H:%M")

    info = [
        t("date", date=local_dt),
        t("food", food=predicted_food.replace("_", " ").capitalize()),
    ]
    if confidence is not None:
        info.append(t("model_confidence_pdf", confidence=f"{confidence*100:.1f}"))
    if processing_time is not None:
        info.append(t("processing_time", time=f"{processing_time:.2f}"))

    for line in info:
        k, v = line.split(":", 1)
        centered_row(k + ":", v.strip())

    pdf.ln(6); pdf.set_draw_color(*line_col)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(8)

    # imagen alimento ----------------------------------------------------
    img_tmp = "tmp_food.png"
    image.save(img_tmp)
    iw, ih = 80, 80 * image.height / image.width
    pdf.image(img_tmp, x=(210-iw)/2, y=pdf.get_y(), w=iw, h=ih)
    os.remove(img_tmp); pdf.ln(ih + 8)

    # tabla nutricional --------------------------------------------------
    pdf.set_font(base, "B", sub_sz)
    pdf.cell(0, 10, f"{t('pdf_nutritional_values')} ({t('weight', weight=weight)})",
             0, 1, "C")
    pdf.set_font(base, "", 12)
    for lbl, val in [
        (t("proteins_name"),  f"{protein:.1f} g"),
        (t("fats_name"),      f"{fat:.1f} g"),
        (t("carbs_name"),     f"{carbs:.1f} g"),
        (t("calories_name"),  f"{kcal:.1f} kcal"),
    ]:
        centered_row(lbl, val)

    pdf.ln(6); pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(8)

    # detalles modelo ----------------------------------------------------
    pdf.set_font(base, "B", sub_sz)
    pdf.cell(0, 10, t("model_table_title"), 0, 1, "C")
    pdf.set_font(base, "", 12)
    for k, v in [
        (t("model_table_model"),  "Xception"),
        (t("model_table_split"),  "75% / 25%"),
        (t("model_table_epochs"), "15"),
        (t("model_table_metric"), "63.58%"),
    ]:
        centered_row(k, v)

    pdf.ln(6); pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(8)

    # === McNemar ===
    pdf.set_font(base, "B", sub_sz)
    pdf.cell(0, 10, t("mcnemar_title"), 0, 1, "C")
    pdf.set_font(base, "", 12)
    mcnemar_data = [
        ("Hybrid vs ResNet50", "115.04", "0.0000", "Diferencia significativa"),
        ("Hybrid vs InceptionV3", "126.96", "0.0000", "Diferencia significativa"),
        ("Hybrid vs Xception", "128.75", "0.0000", "Diferencia significativa"),
        ("ResNet50 vs InceptionV3", "0.09", "0.7658", "Sin diferencia significativa"),
        ("ResNet50 vs Xception", "0.14", "0.7108", "Sin diferencia significativa"),
        ("InceptionV3 vs Xception", "0.46", "0.4959", "Sin diferencia significativa"),
    ]
    for comp, stat, pval, concl in mcnemar_data:
        centered_row(t("mcnemar_comparison"), comp)
        centered_row(t("mcnemar_statistic"), stat)
        centered_row(t("mcnemar_pvalue"), pval)
        centered_row(t("mcnemar_conclusion"), concl)
        pdf.ln(2)
    pdf.ln(4)

    # === Matrices de confusión ===
    from PIL import Image as PILImage
    confusion_matrices = [
        ("Matriz de Confusión - Xception", 'graphics/cm_Xception.png'),
        ("Matriz de Confusión - Hybrid", 'graphics/cm_Hybrid.png'),
        ("Matriz de Confusión - ResNet50", 'graphics/cm_ResNet50.png'),
        ("Matriz de Confusión - InceptionV3", 'graphics/cm_InceptionV3.png'),
    ]
    for title, path in confusion_matrices:
        img_w = 140
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
            pdf.set_font(base, 'B', sub_sz)
            pdf.cell(0, 10, title, 0, 1, 'C')
            img_y = pdf.get_y()
            pdf.image(path, x=img_x, y=img_y, w=img_w, h=img_h)
            pdf.set_y(img_y + img_h + 8)
        else:
            pdf.set_font(base, 'B', sub_sz)
            pdf.cell(0, 10, title, 0, 1, 'C')
            pdf.set_font(base, '', 12)
            pdf.cell(0, 10, f'No se encontró la imagen {title}.', 0, 1, 'C')
        pdf.ln(8)

    # salida -------------------------------------------------------------
    out = "reporte_nutricional.pdf"
    pdf.output(out)
    with open(out, "rb") as f:
        return f.read()

# === UI PRINCIPAL =======================================================
st.set_page_config(page_title=t("app_title"), page_icon="🍽️", layout="centered")
st.title(t("app_title"))

file = st.file_uploader(t("upload_prompt"), type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file)
    st.image(img, caption=t("upload_prompt"), use_container_width=True)

    with st.spinner(t("analyzing")):
        start = time.time()
        pred   = model.predict(preprocess_image_inception(img), verbose=0)
        idx    = int(np.argmax(pred)); conf = float(np.max(pred))
        elapsed = time.time() - start

    classes = nutrients_df["clase"].tolist()
    if idx >= len(classes):
        st.error(t("error_class_index")); st.stop()

    food_key = classes[idx]
    row = nutrients_df[nutrients_df["clase"] == food_key]
    if row.empty:
        st.error(t("error_nutrient_data")); st.stop()

    pdata = row.iloc[0]
    protein, fat = pdata["proteinas"], pdata["grasas"]
    carbs, kcal  = pdata["carbohidratos"], pdata["calorias"]
    food_name    = translate_food_name(food_key, lang)

    st.success(t("identified_food", food=food_name))
    st.caption(t("model_confidence", confidence=conf*100))
    st.info(t("analysis_time", time=elapsed))

    weight = st.number_input(t("weight_prompt"), min_value=1.0, value=100.0, step=1.0)
    factor = weight / 100.0

    st.markdown(t("nutritional_values"))
    st.markdown(f"- {t('proteins',  value=protein*factor)}")
    st.markdown(f"- {t('fats',      value=fat*factor)}")
    st.markdown(f"- {t('carbs',     value=carbs*factor)}")
    st.markdown(f"- {t('calories',  value=kcal*factor)}")

    if st.button(t("generate_pdf")):
        with st.spinner(t("generate_pdf")):
            pdf_bytes = generate_pdf_report(
                img, food_name,
                protein*factor, fat*factor, carbs*factor, kcal*factor,
                weight, conf, elapsed
            )
        st.download_button(
            label=t("download_pdf"),
            data=pdf_bytes,
            file_name=f"reporte_{food_key.lower()}.pdf",
            mime="application/pdf"
        )
        st.success(t("pdf_success"))
"""

with open("app_colab.py", "w") as f:
    f.write(app_code)

print("✅ Archivo 'app_colab.py' creado en tu proyecto")

# 4) Exponer con Ngrok y lanzar Streamlit
from pyngrok import ngrok

ngrok.kill()
ngrok.set_auth_token("2qDM7rrppnKWOWTihT6KfTPRlK8_3vGZRJGfZQvcGdCDecWz5")
public_url = ngrok.connect(8501)
print(f"🌐 Enlace público: {public_url}")

!streamlit run app_colab.py &