# convert_to_savedmodel.py
import os, sys, zipfile
import tensorflow as tf

# === RUTAS (ajusta si las tuyas son distintas) ===
H5_PATH  = "models/inceptionv3_food101.h5"                  # ruta a tu .h5
OUT_DIR  = "models/inceptionv3_food101_savedmodel"          # carpeta SavedModel a generar
ZIP_PATH = "models/inceptionv3_food101_savedmodel.zip"      # zip final a subir

def load_h5_any(h5_path):
    # 1) intenta con tf.keras (TF 2.x)
    try:
        m = tf.keras.models.load_model(h5_path, compile=False)
        print("[OK] Cargado con tf.keras")
        return m
    except Exception as e_tf:
        print("[WARN] tf.keras no pudo cargar:", e_tf)

    # 2) intenta con Keras 3
    try:
        import keras
        from keras.saving import load_model as kload
        m = kload(h5_path, compile=False)
        print("[OK] Cargado con keras (Keras 3)")
        return m
    except Exception as e_k:
        print("[ERR] keras (Keras 3) tampoco pudo:", e_k)

    raise RuntimeError("No se pudo cargar el .h5 con tf.keras ni con keras.")

def export_savedmodel(model, out_dir):
    # Keras 3 tiene .export(); tf-keras 2.x usa save_model(dir)
    try:
        if hasattr(model, "export"):
            model.export(out_dir)  # Keras 3
            print("[OK] Exportado con model.export(...) (Keras 3)")
        else:
            tf.keras.models.save_model(model, out_dir)  # TF 2.x
            print("[OK] Guardado con tf.keras.models.save_model(...) (SavedModel)")
    except Exception as e:
        print("[WARN] export/save_model falló, probando tf.saved_model.save:", e)
        tf.saved_model.save(model, out_dir)
        print("[OK] Guardado con tf.saved_model.save(...)")

def zip_dir(src_dir, zip_path):
    # Comprime el contenido de src_dir dentro de zip_path
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, src_dir)
                zf.write(fp, rel)
    print(f"[OK] ZIP creado: {zip_path}")

def main():
    if not os.path.exists(H5_PATH):
        print("No existe el archivo .h5:", H5_PATH)
        sys.exit(1)

    # Exportar SavedModel
    model = load_h5_any(H5_PATH)
    if os.path.isdir(OUT_DIR):
        # limpia si existía
        import shutil
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    export_savedmodel(model, OUT_DIR)

    # Comprimir
    zip_dir(OUT_DIR, ZIP_PATH)
    print("[DONE] SavedModel listo:", OUT_DIR)

if __name__ == "__main__":
    main()
