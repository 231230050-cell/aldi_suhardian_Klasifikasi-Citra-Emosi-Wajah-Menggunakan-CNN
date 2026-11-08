import os
import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="wide")

# ------------------------------------------------------------
# KONFIGURASI & PENCARIAN FOLDER OUTPUT
# ------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "final_fer_cnn_model.keras"

CANDIDATES = [
    os.path.join(BASE, "output"),
    os.path.join(os.getcwd(), "output"),
    r"E:\BIG DATA\praktikum_9\output",
    os.path.join(os.path.dirname(BASE), "output"),
]
OUT_DIR = next((c for c in CANDIDATES if os.path.exists(os.path.join(c, MODEL_NAME))), CANDIDATES[0])
P = lambda name: os.path.join(OUT_DIR, name)
MODEL_PATH = P(MODEL_NAME)
CONF_THRESHOLD = 50

EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
         "neutral": "😐", "sad": "😢", "surprise": "😲", "surprised": "😲",
         "contempt": "😒", "calm": "😌"}


def emo(name):
    return EMOJI.get(str(name).lower(), "🙂")


def show_img(src, **kw):
    """Kompatibel dengan versi Streamlit lama maupun baru."""
    try:
        st.image(src, use_container_width=True, **kw)
    except TypeError:
        st.image(src, width="stretch", **kw)


def load_json(name):
    path = P(name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
           padding: 1.8rem 1.5rem; border-radius: 18px; text-align: center;
           margin-bottom: 1.2rem; box-shadow: 0 6px 20px rgba(0,0,0,.15);}
    .hero h1 {margin: 0; color: white; font-size: 2.1rem;}
    .hero p {margin: .4rem 0 0; color: #eee;}
    .result-card {background: linear-gradient(135deg,#11998e 0%,#38ef7d 100%);
           color: white; padding: 1.2rem; border-radius: 16px; text-align: center;
           box-shadow: 0 6px 18px rgba(17,153,142,.35);}
    .result-card .emoji {font-size: 3.5rem;}
    .result-card .name {font-size: 2rem; font-weight: 700; text-transform: capitalize;}
    .result-card .conf {font-size: 1.1rem; opacity: .95;}
    </style>
    <div class="hero">
      <h1>🙂 Facial Emotion Recognition</h1>
      <p>Klasifikasi emosi wajah menggunakan Convolutional Neural Network (CNN) — Praktikum Analisis Big Data</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# DIAGNOSTIK (selalu tampil di sidebar)
# ------------------------------------------------------------
with st.sidebar:
    with st.expander("🛠 Diagnostik", expanded=not os.path.exists(MODEL_PATH)):
        st.write("**Folder app.py:**", BASE)
        st.write("**Folder output dipakai:**", OUT_DIR)
        st.write("**Folder ada?**", os.path.isdir(OUT_DIR))
        if os.path.isdir(OUT_DIR):
            st.write("**Isi folder:**")
            for fn in sorted(os.listdir(OUT_DIR)):
                st.write("•", fn)
        st.write("**Lokasi yang dicek:**")
        for c in CANDIDATES:
            st.write("•", c, "✅" if os.path.exists(os.path.join(c, MODEL_NAME)) else "❌")

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan: `{MODEL_PATH}`")
    st.info("Pastikan sel 4C (model.save) sudah dijalankan dan lihat panel **Diagnostik** di sidebar "
            "untuk mengetahui folder output mana yang terdeteksi. Jika file model ada di tempat lain, "
            "salin folder `output` ke sebelah `app.py`.")
    st.stop()


@st.cache_resource(show_spinner="Memuat model...")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


try:
    model = load_model(MODEL_PATH)
except Exception:
    st.error("Model gagal dimuat. Detail error:")
    st.code(traceback.format_exc())
    st.stop()

class_names = load_json("class_names.json")
if class_names is None:
    st.error("class_names.json tidak ditemukan di folder output. Jalankan sel 4C.")
    st.stop()

summary = load_json("dataset_summary.json")
history = load_json("history.json")
metrics = load_json("metrics.json")

IMG_SIZE = tuple(model.input_shape[1:3])


def preprocess(img: Image.Image):
    g = img.convert("L").resize(IMG_SIZE)
    return np.asarray(g, dtype="float32")[None, ..., None] / 255.0


def grad_cam(x, layer_name="last_conv"):
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    xt = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(xt, training=False)
        idx = int(tf.argmax(preds[0]))
        score = preds[:, idx]
    grads = tape.gradient(score, conv_out)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * w[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR),
                     dtype="float32") / 255.0
    return cam


# ------------------------------------------------------------
# SIDEBAR INFO MODEL
# ------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.write("**Arsitektur:** CNN 4 blok konvolusi (ReLU + BatchNorm + Dropout)")
    st.write(f"**Input:** {IMG_SIZE[0]}×{IMG_SIZE[1]} piksel (grayscale)")
    st.write(f"**Jumlah kelas:** {len(class_names)}")
    st.markdown("**Kelas yang dikenali:**")
    for c in class_names:
        st.write(f"{emo(c)} {c.capitalize()}")
    if metrics:
        st.divider()
        st.metric("Akurasi Test", f"{metrics['accuracy'] * 100:.2f}%")
    st.divider()
    st.caption("Tips: gunakan foto wajah menghadap depan, close-up, dan pencahayaan cukup.")

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_data, tab_train, tab_eval, tab_pred = st.tabs(
    ["📊 Dataset & EDA", "📈 Hasil Training", "🧪 Evaluasi Model", "🔮 Prediksi Gambar Baru"])

# ---------------- TAB 1: DATASET ----------------
with tab_data:
    st.subheader("Ringkasan Dataset")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah kelas", summary["n_classes"])
        c2.metric("Total gambar", f"{summary['n_total']:,}")
        c3.metric("Data latih", f"{summary['n_train']:,}")
        c4.metric("Data uji", f"{summary['n_test']:,}")
        st.caption(f"Proporsi latih : uji = {summary['split']} | Ukuran gambar model: "
                   f"{summary['img_size'][0]}×{summary['img_size'][1]} px | "
                   f"Rasio imbalance (maks/min): {summary['imbalance_ratio']:.2f}×")

        st.subheader("Distribusi Data per Kelas")
        dist = pd.DataFrame({
            "Total": summary["class_counts"],
            "Train": summary["train_counts"],
            "Test": summary["test_counts"],
        })
        cl, cr = st.columns([3, 2])
        with cl:
            st.bar_chart(dist[["Train", "Test"]])
        with cr:
            st.dataframe(dist)

        if summary["imbalance_ratio"] > 1.5:
            st.warning(f"Terdapat class imbalance (rasio {summary['imbalance_ratio']:.2f}×). "
                       "Pelatihan menggunakan class weight untuk menyeimbangkan.")
        else:
            st.success("Distribusi kelas relatif seimbang.")

        stt = summary.get("img_stats", {})
        if stt:
            st.subheader("Karakteristik Citra")
            st.write(f"- Ukuran dominan: **{stt['common_size'][0]}×{stt['common_size'][1]} px**\n"
                     f"- Mode warna: **{stt['modes']}** (L = grayscale)\n"
                     f"- Brightness rata-rata: **{stt['mean_brightness']:.1f}** | "
                     f"Kontras rata-rata: **{stt['mean_contrast']:.1f}** "
                     f"(dari {stt['n_sampled']:,} sampel)")
    else:
        st.warning("`dataset_summary.json` belum ada di folder output. Jalankan sel 4C di notebook.")

    if os.path.exists(P("1_sample_per_class.png")):
        st.subheader("Contoh Gambar per Emosi")
        show_img(P("1_sample_per_class.png"))
    else:
        st.info("`1_sample_per_class.png` belum ada (dibuat di Langkah 1).")

# ---------------- TAB 2: TRAINING ----------------
with tab_train:
    st.subheader("Akurasi & Loss selama Training")
    if history:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Accuracy**")
            st.line_chart(pd.DataFrame({"Train": history["accuracy"], "Validation": history["val_accuracy"]}))
        with c2:
            st.markdown("**Loss**")
            st.line_chart(pd.DataFrame({"Train": history["loss"], "Validation": history["val_loss"]}))
        best_ep = int(np.argmax(history["val_accuracy"])) + 1
        st.write(f"Total epoch berjalan: **{len(history['loss'])}** | "
                 f"Val accuracy terbaik: **{max(history['val_accuracy']) * 100:.2f}%** (epoch {best_ep})")
        gap = history["accuracy"][best_ep - 1] - history["val_accuracy"][best_ep - 1]
        st.caption("Selisih train–val accuracy pada epoch terbaik: "
                   f"{gap * 100:.1f} poin " + ("(indikasi overfitting)." if gap > 0.1 else "(relatif sehat)."))
    else:
        st.warning("`history.json` belum ada. Jalankan sel 4B (training) sampai selesai.")

# ---------------- TAB 3: EVALUASI ----------------
with tab_eval:
    st.subheader("Metrik Evaluasi (Data Uji)")
    if metrics:
        m = metrics["macro"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        c2.metric("Precision (macro)", f"{m['precision']:.3f}")
        c3.metric("Recall (macro)", f"{m['recall']:.3f}")
        c4.metric("F1-Score (macro)", f"{m['f1']:.3f}")

        rep = pd.DataFrame(metrics["per_class"]).T
        rep.columns = ["Precision", "Recall", "F1-Score", "Support"]
        st.markdown("**Laporan per kelas**")
        st.dataframe(rep.round(3))

        best, worst = rep["F1-Score"].idxmax(), rep["F1-Score"].idxmin()
        st.info(f"Kelas paling mudah dikenali: **{best}** (F1 {rep.loc[best, 'F1-Score']:.3f}). "
                f"Kelas tersulit: **{worst}** (F1 {rep.loc[worst, 'F1-Score']:.3f}).")
        st.bar_chart(rep["F1-Score"])
    else:
        st.warning("`metrics.json` belum ada. Jalankan Langkah 5 (evaluasi) di notebook.")

    st.subheader("Confusion Matrix")
    if os.path.exists(P("3_confusion_matrix.png")):
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            show_img(P("3_confusion_matrix.png"))
    else:
        st.warning("`3_confusion_matrix.png` belum ada. Jalankan Langkah 6.1.")

    st.subheader("Feature Importance (Grad-CAM)")
    st.caption("Area merah = bagian wajah yang paling memengaruhi keputusan model.")
    for fname, title in [("5_gradcam_samples.png", "Contoh Grad-CAM"),
                         ("6_gradcam_per_class.png", "Rata-rata area penting per emosi"),
                         ("7_sample_predictions.png", "Contoh prediksi vs label asli")]:
        if os.path.exists(P(fname)):
            st.markdown(f"**{title}**")
            show_img(P(fname))
        else:
            st.info(f"`{fname}` belum ada. Jalankan Langkah 6.")

# ---------------- TAB 4: PREDIKSI ----------------
with tab_pred:
    st.subheader("Uji Prediksi Gambar Baru")
    mode = st.radio("Sumber gambar", ["📁 Upload", "📷 Kamera"], horizontal=True)
    if mode == "📁 Upload":
        file = st.file_uploader("Pilih gambar wajah", type=["jpg", "jpeg", "png", "webp", "bmp"])
    else:
        file = st.camera_input("Ambil foto wajah")

    if file is not None:
        image = Image.open(file)
        x = preprocess(image)
        probs = model.predict(x, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        name, conf = class_names[top], float(probs[top]) * 100

        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            show_img(image, caption="Gambar input")
        with col2:
            st.markdown(
                f"""<div class="result-card">
                      <div class="emoji">{emo(name)}</div>
                      <div class="name">{name}</div>
                      <div class="conf">Keyakinan: {conf:.1f}%</div>
                    </div>""",
                unsafe_allow_html=True,
            )
            if conf < CONF_THRESHOLD:
                st.warning("Keyakinan rendah. Coba foto wajah yang lebih jelas/frontal.")
            else:
                st.success("Prediksi cukup meyakinkan.")
        with col3:
            try:
                cam = grad_cam(x)
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(x[0, ..., 0], cmap="gray")
                ax.imshow(cam, cmap="jet", alpha=0.45)
                ax.axis("off")
                ax.set_title("Grad-CAM", fontsize=10)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info(f"Grad-CAM tidak tersedia: {e}")

        st.subheader("Probabilitas semua kelas")
        st.bar_chart(pd.DataFrame({"Probabilitas (%)": probs * 100}, index=class_names))
        for i in order:
            st.write(f"{emo(class_names[i])} **{class_names[i].capitalize()}** — {probs[i] * 100:.1f}%")
            st.progress(float(min(max(probs[i], 0.0), 1.0)))
    else:
        st.info("Belum ada gambar. Upload foto wajah atau gunakan kamera untuk mulai.")

st.divider()
st.caption("Dibuat dengan TensorFlow & Streamlit — Praktikum Analisis Big Data")