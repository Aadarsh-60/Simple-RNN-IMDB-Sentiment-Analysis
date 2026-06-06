import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit UI
st.set_page_config(page_title='Simple RNN IMDB', page_icon='🎬', layout='centered')

st.markdown("""
<style>
body {background: linear-gradient(90deg, #f8fafc, #eef2ff);}
.stApp { color: #0f172a }
.card {background: white; padding: 1.2rem; border-radius: 12px; box-shadow: 0 6px 18px rgba(15,23,42,0.08);}
.result-positive{color: #065f46; font-weight:700}
.result-negative{color: #991b1b; font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('# 🎬 IMDB Movie Review Sentiment')
st.markdown('A lightweight Simple RNN model to classify IMDB movie reviews as positive or negative.')

with st.expander('About this app'):
    st.write('The model was trained on the IMDB dataset. Enter a review below and click **Classify**.')

# Sidebar with examples
st.sidebar.header('Examples')
examples = [
    'That movie was absolutely fantastic — I loved it!',
    'The film was boring and too long. Not recommended.',
    'Great performances but the story was weak.',
    'An instant classic. Brilliant direction and acting.'
]
choice = st.sidebar.selectbox('Pick an example review', ['-- choose --'] + examples)

user_input = st.text_area('Enter a movie review', value=(choice if choice != '-- choose --' else ''))

col1, col2 = st.columns([3,1])

with col2:
    st.image('https://raw.githubusercontent.com/hjorturlarsen/IMDB-Dataset/master/images/imdb-logo.png', width=90)

with col1:
    threshold = st.slider('Positive threshold', 0.0, 1.0, 0.5)

if st.button('Classify'):
    if not user_input or user_input.strip() == '':
        st.warning('Please enter a movie review to classify.')
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)[0][0]
        sentiment = 'Positive' if prediction > threshold else 'Negative'

        if sentiment == 'Positive':
            st.markdown(f"<p class='result-positive'>Sentiment: ✅ {sentiment}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='result-negative'>Sentiment: ❌ {sentiment}</p>", unsafe_allow_html=True)

        st.write('Prediction score:')
        st.progress(min(max(float(prediction), 0.0), 1.0))
        st.caption(f'{prediction:.4f}  (threshold={threshold:.2f})')
        with st.expander('Show decoded words (approx)'):
            try:
                tokens = preprocess_text(user_input)[0]
                st.write(decode_review(tokens))
            except Exception:
                st.write('Unable to decode input.')

st.markdown('</div>', unsafe_allow_html=True)

