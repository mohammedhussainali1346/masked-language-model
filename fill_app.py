import os
import gdown
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Config

st.set_page_config(
    page_title="Masked Language Model",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS Styling

st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #00C6FF, #0072FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 30px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.prediction-box {
    background: rgba(0, 114, 255, 0.1);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.best-box {
    background: linear-gradient(90deg, #00C6FF, #0072FF);
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-weight: 500;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Load Model

@st.cache_resource
def load_resources():
    model_path = "bigru_mask_model.h5"

    if not os.path.exists(model_path):
        file_id = "1rKqfaYB8lu53DBJudFnz2x9ldZUIbPo6"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    model = load_model(model_path)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_resources()
max_sequence_len = model.input_shape[1]

# Prediction Logic

def predict_top_n(model, tokenizer, sentence, max_sequence_len, n=5):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_sequence_len, padding='pre')

    probs = model.predict(padded, verbose=0)[0]
    top_indices = np.argsort(probs)[-n*2:][::-1]

    results = []
    for i in top_indices:
        word = tokenizer.index_word.get(i, "")
        if word != "<OOV>" and word != "mask":
            results.append((word, float(probs[i])))
        if len(results) == n:
            break

    return results

# Header

st.markdown('<div class="main-title">Masked Language Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Context-Aware Word Prediction using Bidirectional GRU</div>', unsafe_allow_html=True)

# Layout

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter sentence with the word 'mask':",
        height=120,
        placeholder="Example: I am preparing to mask for the interview"
    )

    predict_btn = st.button("🚀 Generate Predictions")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔎 Instructions")
    st.markdown("""
    - Insert the word **mask** where prediction is needed  
    - Click **Generate Predictions**  
    - Review top semantic matches  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Predictions Section

if predict_btn:

    if "mask" not in user_input.lower():
        st.warning("⚠ Please include the word 'mask' in your sentence.")
    else:
        predictions = predict_top_n(
            model,
            tokenizer,
            user_input.lower(),
            max_sequence_len,
            n=5
        )

        st.markdown("## 🎯 Top Predictions")

        for word, prob in predictions:
            st.markdown(f"""
            <div class="prediction-box">
                <strong>{word}</strong>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(prob * 3, 1.0))  # amplify for visibility
            st.caption(f"Confidence Score: {prob:.4f}")

        # Best Completion
        best_word = predictions[0][0]
        completed_sentence = user_input.lower().replace("mask", best_word)

        st.markdown("## 🧠 Best Completion")
        st.markdown(f"""
        <div class="best-box">
        {completed_sentence}
        </div>
        """, unsafe_allow_html=True)