# 🎬 Simple RNN IMDB Sentiment Analysis

A deep learning project that uses a **Simple Recurrent Neural Network (Simple RNN)** to classify IMDB movie reviews as **Positive** or **Negative**.

The project includes:

* 📚 Model training notebooks
* 🤖 A pre-trained Keras model
* 🌐 A Streamlit web application for real-time sentiment prediction

---

## 📂 Project Structure

```
simple_rnn_imdb/
│
├── main.py                  # Streamlit web application
├── simple_rnn_imdb.h5       # Pre-trained Simple RNN model
├── requirements.txt         # Required Python packages
│
├── embedding.ipynb          # Word embedding exploration
├── simplernn.ipynb          # Model training notebook
├── prediction.ipynb         # Prediction and testing notebook
│
└── README.md
```

---

## 🚀 Features

* IMDB movie review sentiment classification
* Built using TensorFlow/Keras
* Interactive Streamlit interface
* Uses Keras IMDB dataset and word index
* Adjustable prediction threshold
* Ready-to-use pre-trained model

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <https://github.com/Aadarsh-60/Simple-RNN-IMDB-Sentiment-Analysis.git>
cd simple_rnn_imdb
```

### 2. Create a virtual environment

Linux / macOS:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If TensorFlow installation fails, make sure you are using a supported Python version (Python 3.10 or 3.11 is recommended).

---

## ▶️ Running the Application

Start the Streamlit server:

```bash
streamlit run main.py
```

After running the command, open your browser and visit:

```
http://localhost:8501
```

---

## 📖 How to Use

1. Launch the application.
2. Enter or paste a movie review.
3. Optionally select one of the sample reviews from the sidebar.
4. Click **Classify**.
5. View:

   * Predicted Sentiment
   * Confidence Score
   * Probability Bar

---

## 💡 Example

### Input

```
That movie was absolutely fantastic — I loved it!
```

### Output

```
Sentiment: Positive ✅
Confidence Score: 0.98
```

---

## 🧠 Model Information

| Property  | Value                           |
| --------- | ------------------------------- |
| Dataset   | IMDB Movie Reviews              |
| Model     | Simple RNN                      |
| Framework | TensorFlow / Keras              |
| Task      | Binary Sentiment Classification |
| Output    | Positive / Negative             |

---

## 📚 Notebooks

| Notebook           | Description                                |
| ------------------ | ------------------------------------------ |
| `embedding.ipynb`  | Understanding word embeddings              |
| `simplernn.ipynb`  | Building and training the Simple RNN model |
| `prediction.ipynb` | Testing and making predictions             |

---

## ⚙️ Text Preprocessing

The application uses the official **Keras IMDB word index** for tokenization.

* Unknown words are treated as `<UNK>`.
* Reviews are converted into integer sequences.
* Sequences are padded before being passed to the model.

---

## 📦 Requirements

Main libraries used:

* TensorFlow
* NumPy
* Streamlit
* Keras

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

* Add LSTM and GRU models for comparison.
* Deploy using Streamlit Cloud.
* Visualize word embeddings.
* Display prediction probabilities graphically.
* Add support for custom datasets.

---

## 📜 License

This project is intended for **learning and educational purposes**.

Feel free to fork, modify, and experiment with the code.
