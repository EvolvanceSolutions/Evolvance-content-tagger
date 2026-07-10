# app_p02.py -- EdTech Content Tagger
# Run with: streamlit run app_p02.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title = "EdTech Content Tagger",
    page_icon  = "EduTag",
    layout     = "centered"
)

CATEGORIES = ['chart', 'diagram', 'slide', 'photo', 'illustration']

DESCRIPTIONS = {
    'chart'        : 'Data visualization - bar, line, or pie chart',
    'diagram'      : 'Process or relationship diagram - flowchart, mindmap, hierarchy',
    'slide'        : 'Presentation slide - title, bullets, or section header',
    'photo'        : 'Real-world photograph - classroom, people, objects',
    'illustration' : 'Digital illustration - flat design, icons, artwork',
}

COLORS = ['#378ADD', '#1D9E75', '#D85A30', '#7B2FBE', '#854F0B']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("p02_content_tagger.keras")

model = load_model()

# ── Page header ───────────────────────────────────────────────────────────────

st.title("EdTech Content Tagger")
st.markdown(
    "*Automatically classify educational images into 5 categories "
    "using MobileNetV2 transfer learning -- 99.5% accurate*"
)
st.divider()

# ── File uploader ─────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload any educational image (PNG or JPG)",
    type = ['png', 'jpg', 'jpeg']
)

if uploaded is not None:

    # Display uploaded image
    img_display = Image.open(uploaded).convert("RGB")
    st.image(img_display, caption="Uploaded image", use_column_width=True)

    # Preprocess for model
    img_resized = img_display.resize((224, 224), Image.LANCZOS)
    arr         = np.array(img_resized, dtype=np.float32) / 255.0
    arr         = np.expand_dims(arr, axis=0)   # shape (1, 224, 224, 3)

    # Predict
    with st.spinner("Analysing content..."):
        probs   = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_cat = CATEGORIES[top_idx]
        top_pct = float(probs[top_idx]) * 100

    # ── Result display ────────────────────────────────────────────────────────

    st.divider()
    st.markdown(f"### Detected: **{top_cat.upper()}**")
    st.markdown(f"*{DESCRIPTIONS[top_cat]}*")
    st.markdown(f"**Confidence: {top_pct:.1f}%**")

    if top_pct >= 90:
        st.success("High confidence prediction -- ready for auto-tagging")
    elif top_pct >= 70:
        st.warning("Medium confidence -- human review recommended")
    else:
        st.error("Low confidence -- manual tagging required")

    # ── All category scores ───────────────────────────────────────────────────

    st.divider()
    st.markdown("**All category scores:**")
    for cat, prob, color in zip(CATEGORIES, probs, COLORS):
        pct  = float(prob) * 100
        col1, col2 = st.columns([3, 1])
        col1.progress(float(prob), text=cat)
        col2.markdown(f"**{pct:.1f}%**")

    # ── Suggested action ──────────────────────────────────────────────────────

    st.divider()
    st.markdown("**Suggested LMS action:**")
    actions = {
        'chart'        : 'Tag as data content. Enable chart accessibility alt-text.',
        'diagram'      : 'Tag as process content. Link to related flowchart templates.',
        'slide'        : 'Tag as slide content. Extract text for search indexing.',
        'photo'        : 'Tag as visual content. Add to media library.',
        'illustration' : 'Tag as graphic content. Check licensing metadata.',
    }
    st.info(actions[top_cat])

st.divider()
st.markdown(
    "*Powered by MobileNetV2 transfer learning | "
    "Built with TensorFlow + Streamlit | "
    "Your Company Name*"
)
