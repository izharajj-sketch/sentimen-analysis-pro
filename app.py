import io
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

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

# --- Kamus Sentimen Bawaan ---
positive_words = [
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
    "tepat", "terang", "terbuka", "terbukti", "tercapai", "terkenal", "terpuji", "terjamnin", "terjangkau",
    "terhormat", "tertib", "tertarik", "tetap", "tidak mengecewakan", "tinggi", "top", "total", "tulus",
    "unggul", "unik", "untung", "utama", "valid", "wajar", "wangi", "wibawa", "yakin", "yes"
]

negative_words = [
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
    "cekcob", "celaka", "cemas", "cemberut", "cemburu", "cemooh", "cenderung", "cercah", "ceroboh",
    "curang", "curiga", "dajal", "danau", "dendam", "depresi", "derita", "desak", "diabaikan",
    "diacak", "diakali", "dialihkan", "diancam", "diasingkan", "dibantah", "dibatalkan", "dibebani",
    "dibegal", "dibenci", "diboikot", "dibongkar", "dibungkam", "dicaci", "dicat cacat", "dicemooh",
    "dicurigai", "didenda", "diejek", "dihajar", "dihambat", "dihapus", "dihina", "dihukum", "diuji",
    "dikadali", "dikeluhkan", "dikecam", "dikira", "dikritik", "dikucilkan", "dikurangi", "dilecehkan",
    "dilema", "dilupakan", "dimanipulasi", "dimarahi", "dimiskinkan", "dimusuhi", "dinodai", "dipalak",
    "diperas", "diperiksa", "diperkarakan", "dipermainkan", "dipersulit", "diprotes", "diragukan",
    "dirampas", "dirampok", "dirugikan", "dirusak", "disabotase", "disalahkan", "disepelekan", "disesali",
    "disita", "disabotase", "distorsi", "disudutkan", "disulitkan", "ditahan", "ditolak", "ditipu",
    "dituntut", "dituduh", "dizalimi", "dongkol", "drop", "duka", "dusta", "ego", "egois", "ejek",
    "eksploitasi", "emosi", "endapan", "fiktif", "fitnah", "gadungan", "gagal", "gagap", "gila",
    "gundah", "gugat", "gundah gulana", "hambat", "hang", "hanyut", "hapus", "hargamahal",
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
    "mencibir", "menciderai", "mencurangi", "mencurigaikan", "mendakwa", "mendamprat", "menderita",
    "mdesak", "mendiskreditkan", "mendiskriminasi", "mendominasi", "menduakan", "menentang",
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
    "penjiplakan", "penolakan", "penyanderaan", "penyensoran", "penyimpangan", "penyusutan",
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

slang_dict = {
    "pengemasan": "kemas", "kemasannya": "kemas", "packing": "kemas","jg": "juga", "gak": "tidak", "tdk": "tidak", "nggak": "tidak", "trus": "terus", "trs": "terus",
    "blm": "belum", "udah": "sudah", "udh": "sudah", "sdh": "sudah", "skrg": "sekarang", "ajh": "saja",
    "bgt": "banget", "gt": "begitu", "klo": "kalau", "kalo": "kalau", "kepengen": "ingin", "exp": "kadaluarsa", "expired": "kadaluarsa",
    "dgn": "dengan", "brg": "barang", "msh": "masih", "aja": "saja", "ndak": "tidak", "ndk": "tidak",
    "tp": "tapi", "rekomended": "bagus", "mantul": "mantap", "rapih": "rapi", "lbh": "lebih",
    "murah": "murah", "kece": "bagus", "gercep": "cepat", "yg": "yang", "tgl": "tanggal" , "krn": "karena",
    "dn": "dan", "d": "di", "ga": "tidak", "pait": "pahit", "ok": "okey", "enk": "enak", "kykny": "sepertinya"
}

# --- Sidebar Menu & Upload Kamus Kata Baku ---
st.sidebar.title("📊 Menu Utama Analisis Sentimen")

menu = st.sidebar.radio("Navigasi:", ["🏠 Dashboard", "📥 Input & Process", "📊 Visualisasi", "🔍 Detail Data", "🧠 Pelatihan Model", "📉 Evaluasi Model", "💾 Export"])

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Styling khusus agar label upload di sidebar berwarna hitam */
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1964&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] { background-color: #4B0082 !important; }
    [data-testid="stSidebar"] * { color:white !important; }
    .block-container {
        background-color: rgba(255, 255, 255, 0.82);
        padding: 4rem 3rem !important;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    h1, h2, h3, p, .stAlert, label, .stMarkdown { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# --- INJEKSI CSS DENGAN KONDISI BACKGROUND GAMBAR UNTUK SEMUA HALAMAN ---
bg_style = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1964&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Membuat card semi-transparan putih agar teks kontras dan terbaca jelas di semua halaman */
.block-container {
    background-color: rgba(255, 255, 255, 0.82);
    padding: 4rem 3rem !important;
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Memaksa warna font judul & teks utama di semua halaman agar stabil */
h1, h2, h3, p, .stAlert, label, .stMarkdown {
    color: #0f172a !important;
}

</style>
"""

# Injeksi style background dan style untuk tab manual/upload berwarna biru
st.markdown(bg_style, unsafe_allow_html=True)
st.markdown("""

<style>

    /* Styling agar tab manual text dan upload file berwarna biru */
    button[id^="tabs-bui"][aria-selected="true"] {
        background-color: #87CEEB !important;
        color: white !important;
        border-radius: 4px;
    }

    button[id^="tabs-bui"][aria-selected="false"] {

        background-color: #FFFFFF !important;
        color: #0369a1 !important;
    }

</style>

""", unsafe_allow_html=True)

# --- Tahap Preprocessing ---
def preprocess_step_by_step(text):
    case_folding = str(text).lower()
    cleaning = re.sub(r'https?://\S+|www\.\S+', '', case_folding)
    cleaning = re.sub(r'[-+]?[0-9]+', '', cleaning)
    cleaning = re.sub(r'[^\w\s]', ' ', cleaning)
    words = cleaning.split()
    normalized_words = [slang_dict.get(w, w) for w in words]
    normalisasi = " ".join(normalized_words)
    tokens = word_tokenize(normalisasi)
    tokenize_res = ", ".join(tokens)
    tokens_no_stop = [w for w in tokens if w not in stop_words]
    stopwords_res = ", ".join(tokens_no_stop)
    stmd_tokens = [stemmer.stem(w) for w in tokens_no_stop]
    stemming_res = " ".join(stmd_tokens)

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

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        "text", "case_folding", "cleaning", "normalisasi",
        "tokenize", "stopwords", "stemming", "polarity_score", "sentimen"
    ])

# --- 1: Dashboard ---
if menu == "🏠 Dashboard":
    # 1. Bagian Header dengan Logo dan Nama "PilPro"
    col1, col2 = st.columns([1, 10]) # Mengatur lebar kolom agar logo dan teks rapat
    with col1:
        # Ganti URL di bawah dengan link gambar logo Anda
        st.image("https://i.pinimg.com/736x/61/b4/31/61b4317bc2357dae01173a13b70796a1.jpg", width=80) 
    with col2:
        st.markdown("<h2 style='margin-top: 15px; color: #0f172a;'>SentiLear</h2>", unsafe_allow_html=True)

    # 2. Hero Section (Konten Utama)
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; border-radius: 20px; background: rgba(255, 255, 255, 0.85); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);">
        <h1 style="color: #0f172a !important; font-size: 2.8rem; margin-bottom: 1rem;">SELAMAT DATANG DI SENTILEAR</h1>
        <p style="font-size: 1.25rem; color: #334155 !important; margin-bottom: 2.5rem; line-height: 1.6;">
            SentiLear adalah platform analisis sentimen berbasis web yang dirancang untuk mengubah tumpukan ulasan produk yang tidak terstruktur menjadi data kuantitatif yang berharga.
            Dengan memanfaatkan algoritma Linear Regresi, aplikasi ini dapat menganalisis sentimen sebagai Positif, Netral atau Negatif, untuk memberikan pemahaman terhadap intensitas kepuasan pelanggan.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    df = st.session_state.df
    if df.empty:
        st.info("Silahkan upload atau input data di menu **Input & Process** untuk memulai analisis.")
    else:
        st.success(f"✅ Data telah siap! Terdapat {len(df)} ulasan yang telah diproses.")

# --- 2: Input & Processing ---
elif menu == "📥 Input & Process":
    st.title("📥 Data Input & Preprocessing")
    with st.expander("⚙️ Tambahkan Kamus Kata Baku"):
        uploaded_kamus = st.file_uploader("Upload file Excel (Kolom: 'tidak_baku', 'kata_baku'):", type=['xlsx', 'xls'])
        if uploaded_kamus is not None:
            try:
                kamus_data = pd.read_excel(uploaded_kamus)
                if 'tidak_baku' in kamus_data.columns and 'kata_baku' in kamus_data.columns:
                    kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
                    slang_dict.update(kamus_tidak_baku)
                    st.success("✅ Kamus kata baku berhasil digabungkan!")
                else:
                    st.error("❌ Excel harus memiliki kolom 'tidak_baku' dan 'kata_baku'!")
            except Exception as e:
                st.error(f"❌ Gagal memproses kamus kata baku: {e}")
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
    st.title("📊 Visualisasi Distribusi Sentimen Dan WordCloud")
    df = st.session_state.df

    if not df.empty and len(df) > 1:
        pos_count = len(df[df['sentimen'] == 'Positive'])
        neg_count = len(df[df['sentimen'] == 'Negative'])
        neu_count = len(df[df['sentimen'] == 'Netral'])
        total_data = len(df)
        st.info("📊 Menampilkan distribusi sentimen berdasarkan data yang telah diproses.")

    else:
        pos_count = 820
        neg_count = 590
        neu_count = 520
        total_data = pos_count + neg_count + neu_count

        st.warning("⚠️ Belum ada data baru yang diproses. Menampilkan grafik visualisasi dengan contoh skor default (Positif: 820, Negatif: 590, Netral: 520).")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Data", total_data)
    col_m2.metric("Positive Sentimen", pos_count)
    col_m3.metric("Negative Sentimen", neg_count)
    col_m4.metric("Netral Sentimen", neu_count)

    st.markdown("---")

    labels = ['Positive', 'Negative', 'Netral']
    sizes = [pos_count, neg_count, neu_count]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader(" Proporsi Sentimen ")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax_pie.pie(

            sizes,
            labels=labels,
            autopct=lambda p: '{:.1f}%\n({:,.0f})'.format(p, p * sum(sizes) / 100),
            startangle=140,
            colors=colors,
            textprops=dict(color="black")
        )

        plt.setp(autotexts, size=14, weight="bold")
        plt.setp(texts, size=12)
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

    with col_chart2:
        st.subheader(" Kuantitas Sentimen ")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4.8))
        bars = ax_bar.barh(labels, sizes, color=colors, height=0.5)
        ax_bar.tick_params(axis='y', labelsize=14)
        ax_bar.set_xlabel('Jumlah Data')
        ax_bar.set_xlim(0, max(sizes) * 1.15)

        for bar in bars:
            width = bar.get_width()
            ax_bar.text(
                width + (max(sizes) * 0.01),
                bar.get_y() + bar.get_height()/2,
                f'{int(width):,}',
                va='center',
                ha='left',
                fontsize=14,
                weight='bold'
            )

        sns.despine(ax=ax_bar, right=True, top=True)
        st.pyplot(fig_bar)

    st.markdown("---")

    st.subheader("☁️ WordCloud Kata Kunci Utama: ")

    if not df.empty:
        text_wc = " ".join(df['stemming'].astype(str))

        if text_wc.strip():
            wc = WordCloud(background_color='white', width=1000, height=400).generate(text_wc)
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
            ax_wc.imshow(wc)
            ax_wc.axis('off')
            st.pyplot(fig_wc)

        else:
            st.write("Teks tidak cukup untuk membuat WordCloud.")

    else:
        st.info("WordCloud kata kunci akan tampil di sini secara otomatis setelah Anda memproses data text/file ulasan.")

# --- 4: Detail Data ---

elif menu == "🔍 Detail Data":
    st.title("🔍 Data Explorer (Detail Alur Preprocessing)")
    df = st.session_state.df

    if not df.empty:
        st.write("Tabel di bawah ini menampilkan hasil tahap prapemrosesan data.")
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
    st.title("🧠 Pemodelan Linear Regresi")
    df = st.session_state.df.copy()

    if len(df) < 5:
        st.error("Data terlalu sedikit untuk training (minimal butuh 5 ulasan). Silakan isi data di menu **Input & Process**.")
    else:
        if st.button("🚀 Train Data & Simpan Model", use_container_width=True):
            s_map = {"Positive": 1, "Netral": 0, "Negative": -1}
            df['label'] = df['sentimen'].map(s_map)

            X = df[['polarity_score']].values
            y = df['label'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.X_all = X

            st.success("✅ Model Berhasil Dilatih dan Disimpan! Silakan menuju menu **📉 Evaluasi Model** untuk melihat hasil analisis.")

# --- 6: Menu Baru Evaluasi Regresi & Grafik ---

elif menu == "📉 Evaluasi Model":
    st.title("📉 Laporan Evaluasi & Grafik Linier")

    if "model" not in st.session_state:

        st.error("⚠️ **Akses Ditolak:** Anda belum melakukan pelatihan model!")
        st.info("Silakan buka menu **🧠 Pelatihan Model** terlebih dahulu, lalu klik tombol **Train & Simpan Model** untuk mengaktifkan halaman evaluasi ini.")

    else:
        model = st.session_state.model

        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        X = st.session_state.X_all

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.subheader("📋 Laporan Evaluasi Model Regresi Linear Sederhana")
        report_regresi = (
            f"      LAPORAN EVALUASI MODEL REGRESI LINEAR       \n"
            f"  \n"
            f" Persamaan Garis      : Y = {model.coef_[0]:.4f} * X + ({model.intercept_:.4f})\n"
            f" Jumlah Data Training : {X_train.shape[0]} ulasan\n"
            f" Jumlah Data Testing  : {X_test.shape[0]} ulasan\n\n"
            f" METRIK EVALUASI UTAMA:\n"
            f" \n"
            f"  Mean Squared Error (MSE)      : {mse:.4f}\n"
            f"  Root Mean Squared Error (RMSE) : {rmse:.4f}\n"
            f"  R² Score (Koefisien Determinasi): {r2:.4f}\n"
            f" \n"
        )

        st.code(report_regresi, language='text')

        with st.expander(" Penjelasan Laporan Evaluasi ", expanded=True):

            st.markdown(f"""
            Berikut adalah penjelasan sederhana mengenai performa model matematika Anda berdasarkan angka di atas:

            1. **Persamaan Garis ($Y = {model.coef_[0]:.2f} \\times X + {model.intercept_:.2f}$)**:
               * Ini adalah rumus pola prediksi yang ditemukan. Jika ulasan produk memiliki Skor Polarity ($X$) sebesar **0**, maka prediksi dasar nilai sentimennya ($Y$) adalah **{model.intercept_:.4f}**.
               * Setiap kenaikan **1 poin** pada skor kata positif/negatif ($X$), nilai sentimen ($Y$) akan meningkat sebesar **{model.coef_[0]:.4f}** ke arah positif.
            2. **Mean Squared Error (MSE) & RMSE ({mse:.4f})**:
               * Mengukur rata-rata kuadrat kesalahan prediksi model. Semakin nilainya **mendekati angka 0**, berarti tingkat kesalahan (error) model dalam menebak sentimen ulasan semakin super kecil/akurat.
            3. **R² Score / Koefisien Determinasi ({r2:.4f})**:
               * Menunjukkan seberapa besar variabel skor leksikon ($X$) mampu menjelaskan variasi label sentimen ($Y$). Nilainya berkisar antara 0 hingga 1.
               * Nilai **{r2:.4f}** berarti sekitar **{max(0.0, r2)*100:.1f}%** akurasi kecenderungan data ditentukan oleh kamus kata positif-negatif, sisanya dipengaruhi faktor variasi bahasa lain.
            """)

        st.markdown("---")

        st.subheader("📈 GRAFIK LINEAR REGRESI")

        X_line = np.linspace(X.min() - 1, X.max() + 1, 100).reshape(-1, 1)
        y_line = model.predict(X_line)

        fig_sub, ax_reg = plt.subplots(figsize=(10, 5))

        ax_reg.scatter(X_train, y_train, color='#3498db', alpha=0.5, s=60, label='Data Aktual (Train)')
        ax_reg.scatter(X_test, y_test, color='#2ecc71', alpha=0.7, s=60, label='Data Aktual (Test)', marker='X')
        ax_reg.plot(X_line, y_line, color='#e74c3c', linewidth=3, label=f'Garis Regresi (Y = {model.coef_[0]:.2f}X + {model.intercept_:.2f})')
        ax_reg.set_title(" Skor Polarity vs Label Sentimen ", fontsize=12)
        ax_reg.set_xlabel(" Skor Polarity Leksikon (X) ", fontsize=10)
        ax_reg.set_ylabel(" Label Numerik Sentimen (Y) ", fontsize=10)
        ax_reg.grid(True, linestyle='--', linewidth=0.5, color='#e0e0e0')
        ax_reg.legend(loc='upper left')

        st.pyplot(fig_sub)

        with st.expander(" Penjelasan Grafik Linear Regresi", expanded=True):

            st.markdown(f"""
            * **Titik Biru (Data Latih) & Hijau (Data Uji):** Merepresentasikan sebaran ulasan produk asli. Posisi vertikal menandakan posisi kelas riil sentimen (Atas = Positif [1], Tengah = Netral [0], Bawah = Negatif [-1]).
            * **Garis Merah Linier:** Ini adalah jalur kecenderungan prediksi model (*Trendline*).
            * **Interpretasi Arah Tren:** Garis merah yang **bergerak naik dari kiri bawah menuju kanan atas** membuktikan secara ilmiah bahwa semakin tinggi nilai polaritas kata hasil prapemrosesan ulasan ($X$), maka model secara otomatis memproyeksikan nilai sentimen ($Y$) bergerak naik linear secara konsisten ke area positif.
            """)

# --- 7: Export Data ---
elif menu == "💾 Export":
    st.title("💾 Download")
    df = st.session_state.df

    if not df.empty:
        st.subheader("Pilih Data untuk Diunduh")

        # Opsi pemilihan kategori
        opsi_export = st.multiselect(
            "Pilih kategori sentimen yang ingin diunduh:",
            options=["Positive", "Negative", "Netral"],
            default=["Positive", "Negative", "Netral"]
        )

        if opsi_export:
            # Filter data berdasarkan pilihan
            df_export = df[df['sentimen'].isin(opsi_export)]

            st.write(f"Total data yang akan diunduh: {len(df_export)} baris")

            # --- Perubahan di sini: Konversi ke Excel ---
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Hasil Analisis')
            
            excel_data = buffer.getvalue()

            st.download_button(
                label="📥 Silahkan Download",
                data=excel_data,
                file_name="hasil_analisis_sentimen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.warning("Silakan pilih minimal satu kategori untuk mendownload.")

        st.markdown("---")
        if st.checkbox("Tampilkan preview data yang akan diunduh"):
            st.dataframe(df[df['sentimen'].isin(opsi_export)])
    else:
        st.warning("Tidak ada data untuk diunduh.")