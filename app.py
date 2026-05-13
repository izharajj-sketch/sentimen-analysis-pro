import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Library untuk Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

# --- Konfigurasi Resource NLTK & Sastrawi ---
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

download_nltk_resources()
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sentimen Analysis Produk",
    page_icon="🚀",
    layout="wide"
)

# --- Kamus Sentimen & Slang ---
positive_words = ["bagus", "lezat", "keren", "enak", "puas", "mantap", "cepat", "nikmat", "bersih", "rapi", "ramah", "love", "manis", "pas", "premium", "juara", "lembut", "renyah", "creamy", "halus", "harum", "segar", "asli", "pekat", "murni", "favorit", "langganan", "terbaik", "bahagia", "cantik", "mewah", "murah", "banyak", "yummy", "sweet"]
negative_words = ["hancur", "jelek", "rusak", "ramai", "kurang", "parah", "lambat", "kecewa", "kotor", "tidak", "lama", "marah", "pahit", "asam", "busuk", "kasar", "grainy", "crumbly", "lembek", "basah", "lummer", "eneg", "mahal", "basi", "kadaluarsa", "buatan", "apek", "aneh", "kesal", "lengket", "hambar", "rugi", "zonk", "penyok", "salah"]

slang_dict = {
    "jg": "juga", "gak": "tidak", "tdk": "tidak", "nggak": "tidak", "trus": "terus", "trs": "terus",
    "blm": "belum", "udah": "sudah", "udh": "sudah", "sdh": "sudah", "skrg": "sekarang", "ajh": "saja",
    "bgt": "banget", "gt": "begitu", "klo": "kalau", "kalo": "kalau", "kepengen": "ingin", "exp": "kadaluarsa",
    "dgn": "dengan", "brg": "barang", "msh": "masih", "aja": "saja", "ndak": "tidak", "ndk": "tidak",
    "tp": "tapi", "rekomended": "bagus", "mantul": "mantap", "skrg": "sekarang", "rapih": "rapi",
    "murce": "murah", "kece": "bagus", "gercep": "cepat", "yg": "yang", "tgl": "tanggal" , "krn": "karena",
    "dn": "dan", "d": "di", "ga": "tidak", "pait": "pahit", "ok": "okey", "enk": "enak", "kykny": "sepertinya"
}

# --- Tahap Preprocessing ---
def preprocess_step_by_step(text):

# --- 1. Fungsi Case Folding ---
    text = str(text).lower()

# --- 2. Fungsi Cleaning ---
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)

# --- 3. Fungsi Normalisasi ---
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]

# --- 4. Fungsi Tokenizing ---
    tokens = word_tokenize(" ".join(words))

# --- 5. Fungsi Stopword ---
    tokens = [w for w in tokens if w not in stop_words]

# --- 6. Fungsi Stemming ---
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

def calculate_polarity(clean_text):
    tokens = word_tokenize(clean_text)
    pos_count = sum(1 for w in tokens if w in positive_words)
    neg_count = sum(1 for w in tokens if w in negative_words)
    return pos_count - neg_count

def detect_sentiment(score):
    if score > 0: return "Positive"
    elif score < 0: return "Negative"
    else: return "Netral"

# --- Inisialisasi Session State ---
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["text", "text_hasil_preprocessed", "polarity_score", "sentimen"])

# --- Sidebar ---
st.sidebar.title("📊 Menu Utama Analisis Sentimen")
menu = st.sidebar.radio("Navigasi:", ["🏠 Dashboard", "📥 Input & Process", "📊 Visualisasi", "🔍 Detail Data", "🧠 Pelatihan Model", "💾 Export"])

# --- 1: Dashboard ---
if menu == "🏠 Dashboard":
    st.title("🚀 Analisis Sentimen Ulasan Produk Menggunakan Linear Regression")
    df = st.session_state.df
    if df.empty:
        st.info("Silahkan upload atau input data di menu **Input & Process**.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", len(df))
        col2.metric("Positive", len(df[df['sentimen'] == 'Positive']))
        col3.metric("Negative", len(df[df['sentimen'] == 'Negative']))
        col4.metric("Netral", len(df[df['sentimen'] == 'Netral']))
        st.write("### Preview Terakhir")
        st.dataframe(df.tail(5), use_container_width=True)

# --- 2: Input & Processing ---
elif menu == "📥 Input & Process":
    st.title("📥 Data Input & Preprocessing")
    tab1, tab2 = st.tabs(["Manual Text", "Upload File (CSV/Excel)"])
    input_data = []
    
    with tab1:
        manual = st.text_area("Input ulasan per baris:")
        if manual: input_data = manual.split("\n")
        
    with tab2:
        file = st.file_uploader("Pilih file CSV (hasil Excel) atau file Excel langsung", type=['csv', 'xlsx', 'xls'])
        if file:
            try:
                # Logika pembacaan file yang lebih fleksibel
                if file.name.endswith('.csv'):
                    # Mencoba membaca CSV dengan berbagai kemungkinan encoding dan separator (Excel sering pakai ';')
                    try:
                        df_up = pd.read_csv(file, sep=None, engine='python', encoding='utf-8')
                    except:
                        file.seek(0) # Reset pointer file
                        df_up = pd.read_csv(file, sep=None, engine='python', encoding='latin1')
                else:
                    df_up = pd.read_excel(file)
                
                st.success(f"Berhasil mengunggah: {file.name}")
                st.write("Preview Data:")
                st.dataframe(df_up.head(3))
                
                col = st.selectbox("Pilih kolom yang berisi teks ulasan:", df_up.columns)
                # Filter data agar tidak error saat preprocessing
                input_data = df_up[col].dropna().astype(str).tolist()
            except Exception as e: 
                st.error(f"Gagal membaca file: {e}")

    if st.button("🔥 Proses Dimulai", use_container_width=True):
        if input_data:
            with st.spinner('Memproses Data... (sedang loading harap tunggu)'):
                processed_list = []
                # Progress bar untuk user experience
                prog_bar = st.progress(0)
                for i, t in enumerate(input_data):
                    if t.strip() and t.lower() != 'nan':
                        cleaned = preprocess_step_by_step(t)
                        score = calculate_polarity(cleaned)
                        processed_list.append({
                            "text": t, 
                            "text_hasil_preprocessed": cleaned,
                            "polarity_score": score, 
                            "sentimen": detect_sentiment(score)
                        })
                    prog_bar.progress((i + 1) / len(input_data))
                
                st.session_state.df = pd.DataFrame(processed_list)
                st.success(f"✅ Selesai! {len(processed_list)} data berhasil diproses.")
        else: 
            st.warning("Data kosong! Silahkan input teks atau upload file terlebih dahulu.")

# --- 3: Visualisasi ---
elif menu == "📊 Visualisasi":
    st.title("📊 Visualisasi Data")
    df = st.session_state.df
    if not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribusi Sentimen")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='sentimen', palette='viridis', ax=ax)
            st.pyplot(fig)
        with c2:
            st.subheader("WordCloud")
            text_wc = " ".join(df['text_hasil_preprocessed'].astype(str))
            if text_wc.strip():
                wc = WordCloud(background_color='white', width=800, height=400).generate(text_wc)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc)
                ax2.axis('off')
                st.pyplot(fig2)
            else:
                st.write("Teks tidak cukup untuk membuat WordCloud.")
    else: 
        st.warning("Proses data dulu di menu Input & Process!")

# --- 4: Detail Data ---
elif menu == "🔍 Detail Data":
    st.title("🔍 Data Explorer")
    if not st.session_state.df.empty:
        st.dataframe(st.session_state.df, use_container_width=True)
    else: 
        st.warning("Belum ada data untuk ditampilkan.")

# --- 5: Pelatihan Model ---
elif menu == "🧠 Pelatihan Model":
    st.title("🧠 Pemodelan Linear Regression")
    df = st.session_state.df.copy()
    
    if len(df) < 5:
        st.error("Data terlalu sedikit untuk training (minimal butuh 5 ulasan).")
    else:
        if st.button("🚀 Train & Evaluate Model", use_container_width=True):
            s_map = {"Positive": 1, "Netral": 0, "Negative": -1}
            df['label'] = df['sentimen'].map(s_map)
            
            vec = CountVectorizer()
            X = vec.fit_transform(df['text_hasil_preprocessed'])
            y = df['label']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Thresholding
            y_pred_cat = [1 if x > 0.3 else (-1 if x < -0.3 else 0) for x in y_pred]
            
            st.divider()
            col_eval1, col_eval2 = st.columns([3, 2])
            
            with col_eval1:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(y_test, y_pred_cat, zero_division=0, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            with col_eval2:
                st.subheader("🎯 Confusion Matrix")
                fig, ax = plt.subplots()
                cm = confusion_matrix(y_test, y_pred_cat)
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                            xticklabels=['Neg', 'Neu', 'Pos'], 
                            yticklabels=['Neg', 'Neu', 'Pos'], ax=ax)
                plt.xlabel('Prediksi')
                plt.ylabel('Aktual')
                st.pyplot(fig)

# --- 6: Export ---
elif menu == "💾 Export":
    st.title("💾 Download Hasil")
    if not st.session_state.df.empty:
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="hasil_sentimen.csv", mime="text/csv", use_container_width=True)
        st.info("File CSV ini dapat dibuka kembali di Excel.")
    else: 
        st.warning("Tidak ada data untuk diunduh.")