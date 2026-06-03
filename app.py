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

# --- Kamus Sentimen Bawaan (Diperluas dengan sampel utama InSet Lexicon) ---
default_positive_words = [
    "bagus", "lezat", "keren", "enak", "puas", "mantap", "cepat", "nikmat", "bersih", "rapi", 
    "ramah", "love", "manis", "pas", "premium", "juara", "lembut", "renyah", "creamy", "halus", 
    "harum", "segar", "asli", "pekat", "murni", "favorit", "langganan", "terbaik", "bahagia", 
    "cantik", "mewah", "murah", "banyak", "yummy", "sweet", "aman", "amanah", "andal", "apresiasi", 
    "asri", "atensi", "autentik", "bakat", "bangga", "bantu", "bebas", "berhasil", "berkah", "bermutu", 
    "bisa", "bonus", "brilian", "cemerlang", "cerdas", "cinta", "cocok", "damai", "dampak", "dapat", 
    "disukai", "didukung", "efektif", "efisien", "ekonomis", "eksklusif", "elok", "empati", "energi", 
    "estetis", "faham", "fasilitas", "gembira", "gemilang", "gampang", "gairah", "gaya", "gratis", 
    "hebat", "hemat", "higienis", "hormat", "ideal", "impresif", "indah", "inovasi", "inspirasi", 
    "istimewa", "jelas", "jujur", "kagum", "kaya", "kelar", "kelebihan", "kemudahan", "keuntungan", 
    "kilat", "komit", "komplit", "kondusif", "kreatif", "kualitas", "lancar", "lapang", "lekat", 
    "lengkap", "luar biasa", "lurus", "mahir", "makmur", "maksimal", "mampu", "manfaat", "manjur", 
    "masuk akal", "matang", "memuaskan", "memikat", "menang", "menarik", "menerima", "menguntungkan", 
    "menonjol", "menyenangkan", "meriah", "mesra", "mempesona", "modern", "mudah", "mulia", "murah hati", 
    "mutu", "nyaman", "nyata", "optimal", "optimis", "orginal", "paham", "pakar", "paling", "pantas", 
    "pasti", "patut", "peduli", "pemberian", "pembaruan", "peningkatan", "penting", "penuh", "percaya", 
    "perhatian", "permata", "pilihan", "pikat", "pionir", "positif", "praktis", "prestasi", "pujian", 
    "puncak", "pulih", "rapih", "rasional", "raya", "rekomendasi", "resmi", "responsif", "riang", 
    "ringan", "salut", "sanjung", "santun", "sayang", "sejahtera", "sesuai", "setia", "siap", "simpatik", 
    "solusi", "spesial", "sukses", "sungguh", "supel", "surga", "tabah", "tahan", "takjub", "tanggap", 
    "tepat", "terang", "terbuka", "terbukti", "tercapai", "terkenal", "terpuji", "terjamin", "terjangkau", 
    "terhormat", "tertib", "tertarik", "tetap", "tidak mengecewakan", "tinggi", "top", "total", "tulus", 
    "unggul", "unik", "untung", "utama", "valid", "wajar", "wangi", "wibawa", "yakin", "yes"
]

default_negative_words = [
    "hancur", "jelek", "rusak", "ramai", "kurang", "parah", "lambat", "kecewa", "kotor", "lama", 
    "marah", "pahit", "asam", "busuk", "kasar", "grainy", "crumbly", "lembek", "basah", "lummer", 
    "eneg", "mahal", "basi", "kadaluarsa", "buatan", "apek", "aneh", "kesal", "lengket", "hambar", 
    "rugi", "zonk", "penyok", "salah", "abai", "acak", "aib", "alat", "alot", "ambles", "ambruk", 
    "ampas", "amuk", "ancam", "angker", "anjlok", "antre", "apes", "arogan", "asal-asalan", "asap", 
    "babar", "bacin", "bajakan", "bangkai", "banned", "banyak alasan", "batal", "bau", "beban", 
    "becek", "begal", "bengkok", "benjol", "berantakan", "berat", "berbau", "berbahaya", "berbohong", 
    "bercacat", "berdebu", "berdusta", "berkurang", "bermasalah", "berisik", "berkarat", "berlarut", 
    "berlendir", "berlumpur", "bermasalah", "berubah", "biadab", "biaya", "bingung", "bisa ular", 
    "bising", "buntu", "buruk", "buta", "cacat", "cadangan", "campur aduk", "capek", "carut", "cedera", 
    "cekcok", "celaka", "cemas", "cemberut", "cemburu", "cemooh", "cenderung", "cercah", "ceroboh", 
    "curang", "curiga", "dajal", "damprat", "danau", "dendam", "depresi", "derita", "desak", "diabaikan", 
    "diacak", "diakali", "dialihkan", "diancam", "diasingkan", "dibantah", "dibatalkan", "dibebani", 
    "dibegal", "dibenci", "diboikot", "dibongkar", "dibungkam", "dicaci", "dicat cacat", "dicemooh", 
    "dicurigai", "didenda", "diejek", "dihajar", "dihambat", "dihapus", "dihina", "dihukum", "diuji", 
    "dikadali", "dikeluhkan", "dikecam", "dikira", "dikritik", "dikucilkan", "dikurangi", "dilecehkan", 
    "dilema", "dilupakan", "dimanipulasi", "dimarahi", "dimiskinkan", "dimusuhi", "dinodai", "dipalak", 
    "diperas", "diperiksa", "diperkarakan", "dipermainkan", "dipersulit", "diprotes", "diragukan", 
    "dirampas", "dirampok", "dirugikan", "dirusak", "disabotase", "disalahkan", "disepelekan", "disesali", 
    "disita", "disabotase", "distorsi", "disudutkan", "disulitkan", "ditahan", "ditolak", "ditipu", 
    "dituntut", "dituduh", "dizalimi", "dongkol", "drop", "duka", "dusta", "ego", "egois", "ejek", 
    "eksploitasi", "emosi", "endapan", "fiktif", "fitnah", "gadungan", "gagal", "gagap", "gairah rendah", 
    "gila", "gundah", "gugat", "gundah gulana", "hambat", "hang", "hanyut", "hapus", "hargamahal", 
    "hasut", "haus", "heboh", "hina", "hitam", "hoax", "hujat", "hukum", "hutang", "iri", "iseng", 
    "istilah", "isu", "itikat buruk", "jahat", "jalang", "jam karet", "jangan", "janggal", "jatuh", 
    "jebakan", "jenuh", "jijik", "jinak", "jiplak", "jual mahal", "judes", "junjung", "kacau", 
    "kaku", "kalah", "kambing hitam", "kambumat", "kandas", "kandungan berbahaya", "kangen", "kanker", 
    "kapok", "karat", "kasihan", "kasus", "keder", "kelabakan", "kelalaian", "kelam", "kelat", 
    "keliru", "keluhan", "kemalingan", "kempes", "kendala", "kendor", "kering", "keruh", "kerugian", 
    "kesal", "kesalahan", "kesulitan", "keterbatasan", "keteteran", "ketinggalan", "khawatir", 
    "khianat", "khilaf", "kiamat", "kikir", "kikis", "klaim sepihak", "klise", "kolaps", "komplain", 
    "kontaminasi", "kontra", "korban", "korupsi", "krisis", "kritik", "kualat", "kualifikasi rendah", 
    "kumal", "kuman", "kuno", "kurang ajar", "kurang baik", "kurang memuaskan", "kusam", "kusut", 
    "labil", "lala", "lalai", "luntur", "lecek", "lelet", "lemah", "lemes", "letih", "lesu", 
    "liar", "licik", "licin", "lintah darat", "lupa", "luput", "macet", "mafia", "mager", "makian", 
    "malang", "malas", "malpraktik", "malu", "mampet", "mandeg", "manipulasi", "manja", "masalah", 
    "masam", "mati", "melecehkan", "meledak", "melelahkan", "melemahkan", "melenceng", "melengking", 
    "meluap", "memalukan", "memar", "mematikan", "membabi buta", "membalelo", "membantah", "membebani", 
    "membenci", "membingungkan", "membisu", "membohongi", "memboikot", "membosankan", "membual", 
    "membubarkan", "membusuk", "memicu", "memihak", "memisahkan", "memotong", "memaksa", "memalsukan", 
    "memanfaatkan", "memanas", "memancing", "memandang rendah", "memarahi", "memperburuk", "mempermainkan", 
    "mempersulit", "memprihatinkan", "memprotes", "memuakkan", "memukul", "memusnahkan", "memusuhi", 
    "menafikan", "menagih", "menahan", "menakutkan", "menambah beban", "menangis", "menanti", 
    "menantang", "menaruh curiga", "menaruh dendam", "menasihati", "mencaci", "mencemari", "mencemaskan", 
    "mencibir", "menciderai", "mencurangi", "mencurigakan", "mendakwa", "mendamprat", "menderita", 
    "mendesak", "mendiskreditkan", "mendiskriminasi", "mendominasi", "menduakan", "menentang", 
    "meneror", "menertawakan", "menewaskan", "mengabaikan", "mengacak", "mengakali", "mengancam", 
    "menganiaya", "mengantuk", "mengeluh", "mengecam", "mengecewakan", "mengejek", "mengekang", 
    "mengeksploitasi", "mengelak", "mengelabui", "mengeliminasi", "mengeluarkan", "mengemis", 
    "mengerikan", "menggugat", "menggulingkan", "mengguyur", "menghambat", "menghancurkan", "menghapus", 
    "menghasut", "menghebohkan", "menghina", "menghujat", "menghukum", "mengidap", "mengikat", 
    "mengintimidasi", "mengisolasi", "mengitari", "mengkritik", "mengorbankan", "mengurangi", 
    "mengurung", "mengusir", "meniduri", "menimbun", "menindas", "meninggalkan", "menipu", 
    "menodai", "menolak", "menonton", "menuntut", "menuduh", "menunggak", "menurun", "meraba", 
    "meragukan", "merajalela", "merana", "merampas", "merampok", "merana", "meremehkan", "merendahkan", 
    "merenggut", "meresahkan", "merugikan", "merusak", "merongrong", "merosot", "mesum", "miskin", 
    "misterius", "mitos", "modaran", "mokat", "molor", "monoton", "mubazir", "mudah rusak", 
    "muka dua", "mundur", "mungkir", "muntah", "murahan", "murka", "murung", "musibah", "musnah", 
    "musuh", "nakal", "najis", "nanar", "nanti dulu", "naas", "negatif", "nekat", "neraka", 
    "ngambek", "ngawur", "ngenes", "ngeri", "ngos-ngosan", "noda", "nol", "non-aktif", "onar", 
    "opini negatif", "oplosan", "opsi buruk", "otot kawat", "overprice", "pacaran", "pagar makan tanaman", 
    "pailit", "pajak", "pakar palsu", "paling jelek", "palsu", "panas", "pandir", "panik", "patah", 
    "patah hati", "patgulipat", "payah", "pecah", "pecat", "pedas", "pegal", "pelit", "pelaku", 
    "pelanggaran", "pelarian", "pelemahan", "pembajakan", "pembatalan", "pembatasan", "pembunuhan", 
    "pembusukan", "pemecatan", "pemerasan", "pemidanaan", "pemalsuan", "pencemaran", "pencopotan", 
    "pencurian", "penderitaan", "penebangan", "penembakan", "pengabaian", "pengalihan", "penganiayaan", 
    "pengangguran", "pengurangan", "penahanan", "penipuan", "penjara", "penjajahan", "penjinakan", 
    "penjiplakan", "penolakan", "penyanderaan", "penyensoran", "penyimpangan", "penyitaan", "penyusutan", 
    "perampasan", "perampokan", "perang", "perangkap", "perdebatan", "peretasan", "perkelahian", 
    "perlakuan buruk", "permasalahan", "perselisihan", "perseteruan", "pertikaian", "perusakan", 
    "pesimis", "pesta pora", "pusing", "piatu", "picing", "pindah", "pingsan", "pipih", "plagiat", 
    "plagiarisme", "preman", "premanisme", "protes", "provokasi", "pudar", "punah", "pupus", 
    "racun", "radikal", "ragu", "ragu-ragu", "rahasia", "raib", "rakus", "rampok", "ranjau", 
    "rapuh", "rekayasa", "rendah", "rentan", "resah", "resesi", "resiko", "retak", "riba", 
    "ribet", "ribut", "rindu", "risiko", "riskan", "roboh", "rongrong", "rugi", "rumit", 
    "runtuh", "rusak", "rusuh", "sabotase", "sakit", "sakit hati", "salah", "salah paham", 
    "salah tempat", "salam tempel", "saluran macet", "sama sekali tidak", "samar", "sampah", 
    "sandera", "sanksi", "sarat masalah", "sarang", "satire", "seadanya", "sebel", "sedih", 
    "sedikit", "segan", "sekarat", "selingkuh", "sembarangan", "sembunyi", "semu", "senyap", 
    "sepele", "seram", "serakah", "serang", "serius", "serobot", "sesat", "sesal", "setan", 
    "setengah-setengah", "setor", "shame", "sial", "siang bolong", "sindiran", "singkir", 
    "sinis", "sita", "skandal", "sobek", "sok", "stagnan", "stres", "suap", "subversif", 
    "sukacita palsu", "sulit", "sumpah serapah", "suram", "susah", "susut", "tabrakan", 
    "tahanan", "tahi", "tajam", "tak berdaya", "tak berguna", "tak pasti", "takut", "tamak", 
    "tumbang", "tumpul", "tunggakan", "tuntut", "tuduh", "udang di balik batu", "ulah", 
    "ulur", "ular berkepala dua", "umpatan", "undur diri", "unras", "untung buntung", 
    "utang", "vandal", "vandalisme", "virus", "vulgar", "wafat", "wanti-wanti", "waria", 
    "was-was", "wasiat buruk", "wong cilik dikebiri", "xenofobia", "yahudi", "yuran liar", 
    "zhalim", "zalim", "zero", "zina", "zonk"
]

# Slang bawaan script
slang_dict = {
    "pengemasan": "kemas", "kemasannya": "kemas", "packing": "kemas","jg": "juga", "gak": "tidak", "tdk": "tidak", "nggak": "tidak", "trus": "terus", "trs": "terus",
    "blm": "belum", "udah": "sudah", "udh": "sudah", "sdh": "sudah", "skrg": "sekarang", "ajh": "saja",
    "bgt": "banget", "gt": "begitu", "klo": "kalau", "kalo": "kalau", "kepengen": "ingin", "exp": "kadaluarsa", "expired": "kadaluarsa",
    "dgn": "dengan", "brg": "barang", "msh": "masih", "aja": "saja", "ndak": "tidak", "ndk": "tidak",
    "tp": "tapi", "rekomended": "bagus", "mantul": "mantap", "rapih": "rapi", "lbh": "lebih",
    "murce": "murah", "kece": "bagus", "gercep": "cepat", "yg": "yang", "tgl": "tanggal" , "krn": "karena",
    "dn": "dan", "d": "di", "ga": "tidak", "pait": "pahit", "ok": "okey", "enk": "enak", "kykny": "sepertinya"
}

# --- Sidebar Menu & Upload Kamus Kata Baku ---
st.sidebar.title("📊 Menu Utama Analisis Sentimen")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pengaturan Kamus Tambahan")

# 1. Upload Kamus Kata Baku
uploaded_kamus = st.sidebar.file_uploader("Upload 'kamuskatabaku.xlsx'", type=['xlsx', 'xls'])
if uploaded_kamus is not None:
    try:
        kamus_data = pd.read_excel(uploaded_kamus)
        if 'tidak_baku' in kamus_data.columns and 'kata_baku' in kamus_data.columns:
            kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
            slang_dict.update(kamus_tidak_baku)
            st.sidebar.success("✅ Kamus kata baku berhasil digabungkan!")
        else:
            st.sidebar.error("❌ Excel harus memiliki kolom 'tidak_baku' dan 'kata_baku'!")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memproses kamus kata baku: {e}")

# 2. Upload Kamus Positif (InSet Lexicon / Excel)
uploaded_positif = st.sidebar.file_uploader("Upload Kamus Kata Positif / InSet Positive (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
if uploaded_positif is not None:
    try:
        if uploaded_positif.name.endswith('.csv'):
            pos_data = pd.read_csv(uploaded_positif)
        else:
            pos_data = pd.read_excel(uploaded_positif)
            
        if not pos_data.empty:
            positive_words = pos_data.iloc[:, 0].dropna().astype(str).tolist()
            st.sidebar.success(f"✅ Kamus positif dimuat dari kolom: '{pos_data.columns[0]}'")
        else:
            st.sidebar.warning("⚠️ File positif kosong. Menggunakan kamus bawaan.")
            positive_words = default_positive_words
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memproses kamus positif: {e}")
        positive_words = default_positive_words
else:
    positive_words = default_positive_words

# 3. Upload Kamus Negatif (InSet Lexicon / Excel)
uploaded_negatif = st.sidebar.file_uploader("Upload Kamus Kata Negatif / InSet Negative (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
if uploaded_negatif is not None:
    try:
        if uploaded_negatif.name.endswith('.csv'):
            neg_data = pd.read_csv(uploaded_negatif)
        else:
            neg_data = pd.read_excel(uploaded_negatif)
            
        if not neg_data.empty:
            negative_words = neg_data.iloc[:, 0].dropna().astype(str).tolist()
            st.sidebar.success(f"✅ Kamus negatif dimuat dari kolom: '{neg_data.columns[0]}'")
        else:
            st.sidebar.warning("⚠️ File negatif kosong. Menggunakan kamus bawaan.")
            negative_words = default_negative_words
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memproses kamus negatif: {e}")
        negative_words = default_negative_words
else:
    negative_words = default_negative_words

st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigasi:", ["🏠 Dashboard", "📥 Input & Process", "📊 Visualisasi", "🔍 Detail Data", "🧠 Pelatihan Model", "💾 Export"])

# --- Tahap Preprocessing ---
def preprocess_step_by_step(text):
    # 1. Case Folding
    case_folding = str(text).lower()

    # 2. Cleaning
    cleaning = re.sub(r'https?://\S+|www\.\S+', '', case_folding)
    cleaning = re.sub(r'[-+]?[0-9]+', '', cleaning)
    cleaning = re.sub(r'[^\w\s]', ' ', cleaning)

    # 3. Normalisasi (Kamus Slang)
    words = cleaning.split()
    normalized_words = [slang_dict.get(w, w) for w in words]
    normalisasi = " ".join(normalized_words)

    # 4. Tokenizing
    tokens = word_tokenize(normalisasi)
    tokenize_res = ", ".join(tokens)

    # 5. Stopword Removal
    tokens_no_stop = [w for w in tokens if w not in stop_words]
    stopwords_res = ", ".join(tokens_no_stop)

    # 6. Stemming
    stemmed_tokens = [stemmer.stem(w) for w in tokens_no_stop]
    stemming_res = " ".join(stemmed_tokens)

    return {
        "text": text,
        "case_folding": case_folding,
        "cleaning": cleaning,
        "normalisasi": normalisasi,
        "tokenize": tokenize_res,
        "stopwords": stopwords_res,
        "stemming": stemming_res
    }

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
    st.session_state.df = pd.DataFrame(columns=[
        "text", "case_folding", "cleaning", "normalisasi", 
        "tokenize", "stopwords", "stemming", "polarity_score", "sentimen"
    ])

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
        st.dataframe(df[['text', 'stemming', 'sentimen']].tail(5), use_container_width=True)

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
                if file.name.endswith('.csv'):
                    try:
                        df_up = pd.read_csv(file, sep=None, engine='python', encoding='utf-8')
                    except:
                        file.seek(0)
                        df_up = pd.read_csv(file, sep=None, engine='python', encoding='latin1')
                else:
                    df_up = pd.read_excel(file)
                
                st.success(f"Berhasil mengunggah: {file.name}")
                st.write("Preview Data:")
                st.dataframe(df_up.head(3))
                
                col = st.selectbox("Pilih kolom yang berisi teks ulasan:", df_up.columns)
                input_data = df_up[col].dropna().astype(str).tolist()
            except Exception as e: 
                st.error(f"Gagal membaca file: {e}")

    if st.button("🔥 Proses Dimulai", use_container_width=True):
        if input_data:
            with st.spinner('Memproses Data... (sedang loading harap tunggu)'):
                processed_list = []
                prog_bar = st.progress(0)
                for i, t in enumerate(input_data):
                    if t.strip() and t.lower() != 'nan':
                        res = preprocess_step_by_step(t)
                        score = calculate_polarity(res["stemming"])
                        
                        res["polarity_score"] = score
                        res["sentimen"] = detect_sentiment(score)
                        processed_list.append(res)
                        
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
            text_wc = " ".join(df['stemming'].astype(str))
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
    st.title("🔍 Data Explorer (Detail Alur Preprocessing)")
    df = st.session_state.df
    
    if not df.empty:
        st.write("Tabel di bawah ini menampilkan hasil perubahan teks pada setiap tahap prapemrosesan data.")
        
        display_df = df[[
            "text", "case_folding", "cleaning", "normalisasi", 
            "tokenize", "stopwords", "stemming", "polarity_score", "sentimen"
        ]].copy()
        
        display_df.columns = [
            "Original Text", "Case Folding", "Cleaning", "Normalisasi", 
            "Tokenize", "Stopwords", "Stemming", "Skor Polarity", "Sentimen"
        ]
        
        st.dataframe(display_df, use_container_width=True)
    else: 
        st.warning("Belum ada data untuk ditampilkan. Silakan lakukan proses data terlebih dahulu pada menu **Input & Process**.")

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
            X = vec.fit_transform(df['stemming'])
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