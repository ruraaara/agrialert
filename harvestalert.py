import math
import os
import requests
import xml.etree.ElementTree as ET
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from pathlib import Path
from streamlit_geolocation import streamlit_geolocation

# ----------------------------------------------------------------
#  KONFIGURASI
# ----------------------------------------------------------------
DATA_DIR      = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent)))
CSV_ENSO      = DATA_DIR / "el-nino-southern-oscillation-enso-el-nino-and-la-nina-events.csv"
CSV_NDVI      = DATA_DIR / "Data_NDVI_Indonesia_2022_2026.csv"
CSV_SENTINEL1 = DATA_DIR / "3_Data_Sentinel1_Indonesia.csv"
MODEL_PATH    = DATA_DIR / "model_agriwarn.onnx"
DB_TANIBOT    = DATA_DIR / "tanibot_offline.db"
RADIUS_KM     = 300

CLASS_TO_OPT = {
    'bacterial_leaf_blight': 'hdb',
    'brown_spot':            'brown_spot',
    'leaf_blast':            'blas',
    'leaf_scald':            'leaf_scald',
    'narrow_brown_spot':     'narrow_brown_spot',
    'rice_hispa':            'rice_hispa',
    'sheath_blight':         'sheath_blight',
    'tungro':                'tungro',
}

NAMA_KELAS = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast',
    'leaf_scald', 'narrow_brown_spot', 'rice_hispa', 'sheath_blight', 'tungro'
]

NAMA_INDO = {
    'bacterial_leaf_blight': 'Hawar Daun Bakteri (HDB/Kresek)',
    'brown_spot':            'Bercak Cokelat',
    'healthy':               'Sehat ✅',
    'leaf_blast':            'Blas Daun',
    'leaf_scald':            'Hawar Pelepah (Leaf Scald)',
    'narrow_brown_spot':     'Bercak Cokelat Sempit',
    'rice_hispa':            'Hispa Padi',
    'sheath_blight':         'Busuk Pelepah',
    'tungro':                'Tungro',
}

REKOMENDASI = {
    'bacterial_leaf_blight': (
        'Gunakan bakterisida berbahan aktif tembaga (copper hydroxide). '
        'Kurangi dosis pupuk nitrogen yang berlebihan. '
        'Pilih varietas tahan HDB seperti Inpari 13 atau Inpari 30.'
    ),
    'brown_spot': (
        'Semprotkan fungisida berbahan aktif mankozeb atau iprodion. '
        'Perbaiki drainase sawah dan pastikan pemupukan kalium (K) cukup. '
        'Hindari kekurangan air pada fase pengisian biji.'
    ),
    'healthy': (
        'Tanaman padi Anda dalam kondisi sehat! '
        'Lanjutkan perawatan rutin dan ikuti jadwal pemupukan NPK. '
        'Tetap pantau kondisi tanaman secara berkala.'
    ),
    'leaf_blast': (
        'Semprotkan fungisida sistemik berbahan aktif tricyclazole atau isoprothiolane. '
        'Kurangi kelembapan dengan mengatur jarak tanam lebih renggang. '
        'Hindari pemupukan nitrogen berlebih saat kondisi lembab.'
    ),
    'leaf_scald': (
        'Kurangi genangan air berlebih di sawah. '
        'Gunakan fungisida berbahan aktif propikonazol atau tebukonazol. '
        'Pastikan sirkulasi udara di sekitar tanaman baik.'
    ),
    'narrow_brown_spot': (
        'Perbaiki kesuburan tanah dengan pemupukan berimbang (NPK). '
        'Gunakan fungisida jika serangan sudah merata di seluruh petak. '
        'Pastikan tanaman tidak kekurangan silika dan kalium.'
    ),
    'rice_hispa': (
        'Potong dan musnahkan daun yang terserang untuk memutus siklus hama. '
        'Gunakan insektisida berbahan aktif karbofuran atau buprofezin jika serangan berat. '
        'Pantau jumlah hama — ambang pengendalian 1–2 ekor per rumpun.'
    ),
    'sheath_blight': (
        'Semprotkan fungisida berbahan aktif validamycin atau azoksistrobin. '
        'Jaga jarak tanam dan kurangi genangan air di sekitar batang. '
        'Hindari pemupukan nitrogen berlebih pada fase anakan.'
    ),
    'tungro': (
        'Tidak ada obat langsung untuk virus tungro. '
        'Kendalikan wereng hijau (vektor penular) dengan insektisida berbahan aktif imidakloprid. '
        'Cabut dan bakar tanaman yang terinfeksi. Tanam varietas tahan tungro seperti Tukad Petanu.'
    ),
}

BERBAHAYA = {'bacterial_leaf_blight', 'leaf_blast', 'sheath_blight', 'tungro', 'rice_hispa'}


# ----------------------------------------------------------------
#  KONFIGURASI HALAMAN
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AgriWarn 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Lexend', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.block-container {
    max-width: 1200px !important;
    padding: 1.25rem 1.5rem 4rem !important;
    margin: 0 auto !important;
}
.ha-header {
    background: #0a3d1f; border-radius: 20px; padding: 22px 28px;
    margin-bottom: 20px; border: 1.5px solid #1a6b38;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}
.ha-header-left { display: flex; align-items: center; gap: 14px; }
.ha-logo  { font-size: 2.4rem; line-height: 1; }
.ha-title { font-size: 1.6rem; font-weight: 800; color: #4ade80; margin: 0; letter-spacing: -0.5px; }
.ha-sub   { font-size: 0.85rem; color: #86efac; margin: 3px 0 0; }
.ha-badge {
    background: #052e16; border: 1px solid #16a34a; border-radius: 30px;
    padding: 6px 16px; font-size: 0.78rem; font-weight: 700; color: #4ade80; white-space: nowrap;
}
.gps-box {
    background: #0a1628; border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 12px 18px; margin-bottom: 18px; font-size: 0.83rem; color: #94a3b8;
    line-height: 1.8; display: flex; flex-wrap: wrap; gap: 4px 20px;
}
.gps-box b { color: #60a5fa; }
.big-card { border-radius: 18px; padding: 20px 24px; margin-bottom: 14px; border: 2px solid transparent; }
.big-card.aman    { background: #052e16; border-color: #16a34a; }
.big-card.waspada { background: #431407; border-color: #ea580c; }
.big-card.bahaya  { background: #450a0a; border-color: #dc2626; }
.big-card.info    { background: #0c1a3a; border-color: #2563eb; }
.bc-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: rgba(255,255,255,0.5); margin-bottom: 8px; }
.bc-row   { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.bc-icon  { font-size: 2rem; line-height: 1; flex-shrink: 0; }
.bc-value { font-size: 1.5rem; font-weight: 800; color: #fff; line-height: 1.1; }
.bc-sub   { font-size: 0.82rem; color: rgba(255,255,255,0.65); margin-top: 4px; line-height: 1.5; }
.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px;
}
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .metric-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; } }
.metric-card { border-radius: 14px; padding: 16px 18px; border: 1.5px solid transparent; min-width: 0; }
.metric-card.aman    { background: #052e16; border-color: #15803d; }
.metric-card.waspada { background: #431407; border-color: #c2410c; }
.metric-card.bahaya  { background: #450a0a; border-color: #b91c1c; }
.metric-card.info    { background: #0c1a3a; border-color: #1d4ed8; }
.mc-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.45); margin-bottom: 6px; }
.mc-val   { font-size: 1.3rem; font-weight: 800; color: #fff; line-height: 1.1; }
.mc-sub   { font-size: 0.7rem; color: rgba(255,255,255,0.55); margin-top: 4px; }
@media (max-width: 480px) { .metric-card { padding: 12px 14px; } .mc-val { font-size: 1.1rem; } }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
.rekom { background: #0a1628; border-radius: 0 14px 14px 0; border-left: 4px solid #22c55e; padding: 16px 18px; margin-bottom: 10px; }
.rekom.kuning { border-left-color: #f59e0b; }
.rekom.merah  { border-left-color: #ef4444; }
.rekom.biru   { border-left-color: #3b82f6; }
.rk-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.9px; color: #22c55e; margin-bottom: 8px; }
.rekom.kuning .rk-title { color: #f59e0b; }
.rekom.merah  .rk-title { color: #f87171; }
.rekom.biru   .rk-title { color: #60a5fa; }
.rk-text { font-size: 0.88rem; color: #d1d5db; line-height: 1.65; }
.hama-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
@media (max-width: 500px) { .hama-grid { grid-template-columns: 1fr; } }
.hama-card { background: #0a1628; border: 1.5px solid #1e3a5f; border-radius: 14px; padding: 14px 16px; display: flex; align-items: flex-start; gap: 12px; }
.hama-icon { font-size: 1.6rem; line-height: 1; flex-shrink: 0; margin-top: 2px; }
.hama-nama { font-size: 0.88rem; font-weight: 700; color: #f1f5f9; margin-bottom: 3px; }
.hama-ciri { font-size: 0.75rem; color: #94a3b8; line-height: 1.5; }
.sec-title { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin: 20px 0 12px; display: flex; align-items: center; gap: 8px; }
.divider { border: none; border-top: 1px solid #1e293b; margin: 18px 0; }
.status-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.chip { border-radius: 30px; padding: 5px 14px; font-size: 0.75rem; font-weight: 700; display: inline-flex; align-items: center; gap: 5px; }
.chip.aman    { background: #052e16; border: 1px solid #16a34a; color: #4ade80; }
.chip.waspada { background: #431407; border: 1px solid #ea580c; color: #fb923c; }
.chip.bahaya  { background: #450a0a; border: 1px solid #dc2626; color: #f87171; }
.chip.info    { background: #0c1a3a; border: 1px solid #1d4ed8; color: #60a5fa; }
div[data-baseweb="tab-list"] { background: #0a1628 !important; border-radius: 14px !important; padding: 4px !important; gap: 4px !important; margin-bottom: 20px; }
button[data-baseweb="tab"] { font-family: 'Lexend', sans-serif !important; font-size: 0.85rem !important; font-weight: 700 !important; border-radius: 10px !important; }
[data-testid="stSidebar"] { background: #060f1a !important; border-right: 1px solid #1e3a5f !important; }
.prob-bar-wrap { margin: 6px 0; }
.prob-label { font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px; display: flex; justify-content: space-between; }
.prob-bar-bg { background: #1e3a5f; border-radius: 6px; height: 10px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 6px; background: #22c55e; transition: width 0.4s; }
.prob-bar-fill.top { background: #4ade80; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 4px; }
@media (max-width: 480px) { .ha-title { font-size: 1.3rem; } .bc-value { font-size: 1.25rem; } .sec-title { font-size: 0.92rem; } .rk-text { font-size: 0.83rem; } .gps-box { font-size: 0.76rem; } }
</style>
""", unsafe_allow_html=True)


# ================================================================
#  LOAD DATA CSV
# ================================================================
@st.cache_data
def load_enso():
    if not CSV_ENSO.exists():
        return pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=52, freq="MS"),
            "ONI":  np.random.default_rng(42).uniform(-2.0, 2.0, 52)
        })
    df = pd.read_csv(CSV_ENSO)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date","ANOM"]].rename(columns={"ANOM":"ONI"})

@st.cache_data
def load_ndvi():
    if not CSV_NDVI.exists():
        return None
    df = pd.read_csv(CSV_NDVI)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df["NDVI"] = df["Rata_rata_NDVI"] / 10000.0
    df["EVI"]  = df["Rata_rata_EVI"]  / 10000.0
    return df[["Tanggal","NDVI","EVI"]]

@st.cache_data
def load_sar():
    if not CSV_SENTINEL1.exists():
        return None
    df = pd.read_csv(CSV_SENTINEL1)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    return df[["Tanggal","SAR_VH","SAR_VV"]]

df_enso = load_enso()
df_ndvi = load_ndvi()
df_sar  = load_sar()


# ================================================================
#  LOAD MODEL CNN (sekali saja, di-cache)
# ================================================================
@st.cache_resource
def load_model_cnn():
    if not MODEL_PATH.exists():
        return None
    try:
        import onnxruntime as ort
        return ort.InferenceSession(str(MODEL_PATH))
    except Exception:
        return None

model_cnn = load_model_cnn()


# ================================================================
#  LOAD TANIBOT ENGINE (sekali saja, di-cache)
# ================================================================
@st.cache_resource
def load_tanibot():
    if not DB_TANIBOT.exists():
        return None
    try:
        from tanibot_engine import TaniBotEngine
        return TaniBotEngine(str(DB_TANIBOT))
    except Exception:
        return None

tanibot = load_tanibot()


# ================================================================
#  PREDIKSI GAMBAR
# ================================================================
def prediksi_gambar(foto_bytes):
    if model_cnn is None:
        return None
    import cv2

    arr = np.frombuffer(foto_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img_batch = np.expand_dims(img.astype(np.float32), axis=0)

    input_name = model_cnn.get_inputs()[0].name
    probs = model_cnn.run(None, {input_name: img_batch})[0][0]

    idx   = int(np.argmax(probs))
    kelas = NAMA_KELAS[idx]

    return {
        "kelas":      kelas,
        "nama_indo":  NAMA_INDO[kelas],
        "confidence": float(probs[idx]),
        "rekomendasi": REKOMENDASI[kelas],
        "berbahaya":  kelas in BERBAHAYA,
        "probs":      {NAMA_KELAS[i]: float(probs[i]) for i in range(len(NAMA_KELAS))},
    }


# ================================================================
#  DATA 38 PROVINSI
# ================================================================
PROVINSI = {
    "Aceh":{"lat":-4.695,"lon":96.749},"Sumatera Utara":{"lat":2.115,"lon":99.545},
    "Sumatera Barat":{"lat":-0.740,"lon":100.800},"Riau":{"lat":0.293,"lon":101.707},
    "Jambi":{"lat":-1.485,"lon":102.438},"Sumatera Selatan":{"lat":-3.319,"lon":104.914},
    "Bengkulu":{"lat":-3.800,"lon":102.266},"Lampung":{"lat":-4.559,"lon":105.407},
    "Kepulauan Bangka Belitung":{"lat":-2.741,"lon":106.441},
    "Kepulauan Riau":{"lat":3.946,"lon":108.143},
    "DKI Jakarta":{"lat":-6.209,"lon":106.846},"Jawa Barat":{"lat":-6.918,"lon":107.619},
    "Jawa Tengah":{"lat":-7.150,"lon":110.140},"DI Yogyakarta":{"lat":-7.875,"lon":110.426},
    "Jawa Timur":{"lat":-7.536,"lon":112.238},"Banten":{"lat":-6.406,"lon":106.064},
    "Bali":{"lat":-8.410,"lon":115.189},"Nusa Tenggara Barat":{"lat":-8.653,"lon":117.362},
    "Nusa Tenggara Timur":{"lat":-8.657,"lon":121.079},
    "Kalimantan Barat":{"lat":0.000,"lon":109.333},
    "Kalimantan Tengah":{"lat":-1.681,"lon":113.382},
    "Kalimantan Selatan":{"lat":-3.093,"lon":115.284},
    "Kalimantan Timur":{"lat":0.539,"lon":116.419},
    "Kalimantan Utara":{"lat":3.073,"lon":116.041},
    "Sulawesi Utara":{"lat":0.625,"lon":123.975},
    "Sulawesi Tengah":{"lat":-1.430,"lon":121.446},
    "Sulawesi Selatan":{"lat":-3.669,"lon":119.974},
    "Sulawesi Tenggara":{"lat":-4.145,"lon":122.175},
    "Gorontalo":{"lat":0.700,"lon":122.447},"Sulawesi Barat":{"lat":-2.844,"lon":119.232},
    "Maluku":{"lat":-3.239,"lon":130.145},"Maluku Utara":{"lat":1.571,"lon":127.809},
    "Papua Barat":{"lat":-1.336,"lon":133.175},"Papua":{"lat":-4.270,"lon":138.080},
    "Papua Selatan":{"lat":-6.500,"lon":139.500},"Papua Tengah":{"lat":-4.000,"lon":136.000},
    "Papua Pegunungan":{"lat":-4.500,"lon":138.500},
    "Papua Barat Daya":{"lat":-1.500,"lon":131.500},
}


# ================================================================
#  GPS — HAVERSINE
# ================================================================
def jarak_km(la1,lo1,la2,lo2):
    R=6371.0; p1,p2=math.radians(la1),math.radians(la2)
    dp=math.radians(la2-la1); dl=math.radians(lo2-lo1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def prov_terdekat(lat,lon):
    return min(PROVINSI, key=lambda p: jarak_km(lat,lon,PROVINSI[p]["lat"],PROVINSI[p]["lon"]))


# ================================================================
#  BMKG
# ================================================================
@st.cache_data(ttl=1800)
def cuaca_bmkg(lat, lon):
    URL = ("https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/"
           "DigitalForecast-Indonesia.xml")
    FB = {"nama":"Estimasi","jarak_km":0,"tmax":33.0,"tmin":24.0,
          "kelembapan":80,"curah_hujan":10.0,"waktu":"-","ok":False,
          "pesan":"Gagal terhubung ke BMKG. Menampilkan estimasi."}
    try:
        r = requests.get(URL, timeout=8); r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return FB

    now_utc = datetime.now(timezone.utc)
    jmin = float("inf"); hasil = {}

    for area in root.findall(".//area"):
        try:
            alat = float(area.get("latitude","999"))
            alon = float(area.get("longitude","999"))
        except: continue
        if alat == 999: continue

        d = jarak_km(lat, lon, alat, alon)
        if d >= jmin: continue

        tmax=tmin=kel=rr=None; wlabel="-"; dmin=float("inf")
        for param in area.findall(".//parameter"):
            pid = param.get("id","")
            for tr in param.findall(".//timerange"):
                ds = tr.get("datetime", tr.get("day",""))
                try:
                    if len(ds)==12:
                        tdt = datetime(int(ds[:4]),int(ds[4:6]),int(ds[6:8]),
                                       int(ds[8:10]),int(ds[10:12]),tzinfo=timezone.utc)
                        dlt = abs((tdt-now_utc).total_seconds())
                        if dlt < dmin:
                            dmin=dlt; wlabel=tdt.strftime("%d %b %Y %H:%M")
                except: pass
                ve = tr.find("value")
                if ve is None or not ve.text: continue
                try:
                    v=float(ve.text)
                    if pid=="tmax": tmax=v
                    elif pid=="tmin": tmin=v
                    elif pid=="hu":   kel=v
                    elif pid=="rr":   rr=v
                except: pass

        jmin=d
        hasil={"nama":area.get("description",area.get("id","BMKG")),
               "jarak_km":round(d,1),"tmax":tmax or 33.0,"tmin":tmin or 24.0,
               "kelembapan":kel or 80,"curah_hujan":rr or 10.0,
               "waktu":wlabel,"ok":True,"pesan":""}

    if not hasil: return FB
    if hasil["jarak_km"] > RADIUS_KM:
        hasil["ok"]=False
        hasil["pesan"]=(f"⚠️ Stasiun terdekat {hasil['jarak_km']} km "
                        f"(batas wajar {RADIUS_KM} km).")
    return hasil


# ================================================================
#  KLASIFIKASI IKLIM & CUACA
# ================================================================
def oni_now():
    df = df_enso[df_enso["date"] <= datetime.now()].tail(1)
    return float(df.iloc[0]["ONI"]) if not df.empty else 0.0

def info_enso(oni):
    if oni >= 1.5:
        return {"fase":"El Niño Kuat","level":"🔴 AWAS KEKERINGAN","kelas":"bahaya","ikon":"☀️",
                "pesan":"Hujan sangat berkurang 40–60%. Siapkan irigasi cadangan. Pilih bibit tahan kering seperti Inpari 32."}
    elif oni >= 0.5:
        return {"fase":"El Niño","level":"🟡 WASPADA KERING","kelas":"waspada","ikon":"🌤️",
                "pesan":"Kemarau lebih lama dari biasanya. Hemat air irigasi dan pantau kondisi lahan."}
    elif oni >= -0.5:
        return {"fase":"Normal","level":"🟢 IKLIM NORMAL","kelas":"aman","ikon":"⛅",
                "pesan":"Pola hujan dan kemarau berjalan normal. Ikuti kalender tanam biasa."}
    elif oni >= -1.5:
        return {"fase":"La Niña","level":"🟡 WASPADA BANJIR","kelas":"waspada","ikon":"🌧️",
                "pesan":"Hujan lebih sering dan lebih deras. Periksa saluran air agar sawah tidak tergenang."}
    else:
        return {"fase":"La Niña Kuat","level":"🔴 AWAS BANJIR","kelas":"bahaya","ikon":"⛈️",
                "pesan":"Curah hujan sangat tinggi. Risiko banjir besar. Waspada serangan wereng dan tungro."}

def info_cuaca(tmax, ch):
    if tmax > 34.0 or ch > 50.0:
        return {"level":"🔴 CUACA EKSTREM","kelas":"bahaya",
                "ikon":"🌡️" if tmax>34 else "🌊",
                "pesan":"Cuaca sangat berbahaya! Segera panen jika sudah waktunya. Aktifkan pompa drainase."}
    elif tmax > 32.0 or ch > 30.0 or ch < 2.0:
        return {"level":"🟡 PERLU PERHATIAN","kelas":"waspada","ikon":"⚠️",
                "pesan":"Cek saluran air dan kondisi tanaman. Tambah pengairan jika hujan sangat sedikit."}
    return {"level":"🟢 CUACA BAIK","kelas":"aman","ikon":"🌤️",
            "pesan":"Cuaca hari ini bagus untuk padi. Waktu yang tepat untuk pemupukan NPK."}

def musim_bln(b):
    if b in [11,12,1,2,3]: return "🌧️ Musim Hujan"
    elif b in [6,7,8,9]:   return "☀️ Musim Kemarau"
    return "⛅ Musim Peralihan"

def saran_tanam(kelas_cuaca, kelas_enso):
    if kelas_cuaca=="bahaya" or kelas_enso=="bahaya":
        return ("merah",
                "🔴 <b>Tunda penanaman</b> sampai kondisi membaik. "
                "Kalau sudah tanam, pantau padi setiap hari dan pastikan saluran air tidak tersumbat.")
    elif kelas_cuaca=="waspada" or kelas_enso=="waspada":
        return ("kuning",
                "🟡 <b>Lanjut tanam tapi waspada.</b> Cek irigasi rutin. "
                "Kalau kering, tambah pengairan. Kalau terlalu basah, buka saluran drainase.")
    return ("",
            "🟢 <b>Kondisi bagus untuk bertanam!</b> Waktu tepat untuk pemupukan NPK. "
            "Ikuti jadwal tanam seperti biasa.")


# ================================================================
#  HELPER GRAFIK
# ================================================================
PLOT_STYLE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,22,40,1)",
    margin=dict(t=16,b=36,l=44,r=20),
    font=dict(family="Lexend, sans-serif", size=12),
)

def fig_oni(df, n=24):
    df_p = df.tail(n).copy()
    df_p["W"] = df_p["ONI"].apply(
        lambda x: "#ef4444" if x>=0.5 else ("#3b82f6" if x<=-0.5 else "#22c55e"))
    df_p["L"] = df_p["date"].dt.strftime("%b %Y")
    fig = go.Figure()
    fig.add_hrect(y0=0.5,  y1=3.0,  fillcolor="rgba(239,68,68,0.07)",  line_width=0)
    fig.add_hrect(y0=-3.0, y1=-0.5, fillcolor="rgba(59,130,246,0.07)", line_width=0)
    fig.add_hline(y= 0.5, line_dash="dot", line_color="#ef4444", line_width=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="#3b82f6", line_width=1)
    fig.add_trace(go.Bar(x=df_p["L"], y=df_p["ONI"], marker_color=df_p["W"],
                         hovertemplate="<b>%{x}</b><br>ONI: %{y:.2f}°C<extra></extra>"))
    fig.add_trace(go.Scatter(x=df_p["L"], y=df_p["ONI"], mode="lines+markers",
                             line=dict(color="white",width=2),
                             marker=dict(size=5,color=df_p["W"].tolist()),
                             hoverinfo="skip"))
    fig.update_layout(**PLOT_STYLE, height=280, showlegend=False,
                      yaxis_title="ONI (°C)", xaxis_tickangle=-40)
    return fig

def fig_ndvi(df):
    df_p = df.tail(24)
    fig = go.Figure()
    fig.add_hrect(y0=0.6, y1=0.8, fillcolor="rgba(34,197,94,0.1)",
                  annotation_text="Zona sehat (0.6–0.8)",
                  annotation_position="top left", line_width=0)
    fig.add_trace(go.Scatter(x=df_p["Tanggal"], y=df_p["NDVI"],
                             mode="lines+markers", name="NDVI",
                             line=dict(color="#4ade80",width=2.5),
                             marker=dict(size=5)))
    fig.update_layout(**PLOT_STYLE, height=260, showlegend=False,
                      yaxis_title="NDVI", yaxis_range=[0,1])
    return fig

def fig_sar(df):
    df_p = df.tail(24)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_p["Tanggal"], y=df_p["SAR_VH"],
                             mode="lines+markers", name="VH",
                             line=dict(color="#60a5fa",width=2.5),
                             marker=dict(size=5)))
    fig.update_layout(**PLOT_STYLE, height=260, showlegend=False,
                      yaxis_title="Backscatter VH (dB)")
    return fig


# ================================================================
#  ── TAMPILAN ──
# ================================================================

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("---")
    st.markdown("### 📍 Lokasi Anda")
    st.caption("Tekan tombol di bawah untuk deteksi otomatis.")
    gps = streamlit_geolocation()

    lat, lon, aktif = -7.8754, 110.4262, False
    if gps and gps.get("latitude") and gps.get("longitude"):
        lat, lon, aktif = float(gps["latitude"]), float(gps["longitude"]), True
        st.success(f"✅ GPS aktif\n📌 {round(lat,4)}°, {round(lon,4)}°\n"
                   f"🗺️ {prov_terdekat(lat,lon)}")
    else:
        st.warning("GPS belum aktif. Pilih manual.")
        lat = st.number_input("Lintang:", value=lat, format="%.4f")
        lon = st.number_input("Bujur:",   value=lon, format="%.4f")

    st.markdown("---")
    st.markdown("### 🗺️ Provinsi")
    daftar   = list(PROVINSI.keys())
    default  = prov_terdekat(lat, lon)
    provinsi = st.selectbox("", daftar, index=daftar.index(default))

    st.markdown("---")
    st.markdown("### 📊 Status Data")
    st.write(f"ENSO  : {'✅' if CSV_ENSO.exists() else '❌'} ({len(df_enso)} baris)")
    st.write(f"NDVI  : {'✅' if df_ndvi is not None else '❌'}")
    st.write(f"SAR   : {'✅' if df_sar  is not None else '❌'}")
    st.write(f"Model : {'✅ Siap' if model_cnn is not None else '❌ Tidak ditemukan'}")
    st.write(f"TaniBot: {'✅ Siap' if tanibot is not None else '❌ Bangun DB dulu'}")
    st.markdown("---")
    st.caption("Data: BMKG · NOAA · Sentinel-1/2\nHarvestAlert v1.0 · Satria Data 2026")


# ── DATA REAL-TIME ────────────────────────────────────────────────
cuaca  = cuaca_bmkg(lat, lon)
oni    = oni_now()
enso   = info_enso(oni)
status = info_cuaca(cuaca["tmax"], cuaca["curah_hujan"])
msmn   = musim_bln(datetime.now().month)
warna_rk, teks_rk = saran_tanam(status["kelas"], enso["kelas"])


# ── HEADER ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="ha-header">
  <div class="ha-header-left">
    <span class="ha-logo">🌾</span>
    <div>
      <p class="ha-title">HarvestAlert</p>
      <p class="ha-sub">Peringatan dini hama &amp; cuaca padi Indonesia</p>
    </div>
  </div>
  <span class="ha-badge">📍 {provinsi}</span>
</div>
""", unsafe_allow_html=True)


# ── 4 TAB ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Beranda", "🌤️  Cuaca & Iklim", "🔬  Cek Hama Padi", "ℹ️  Informasi"
])


# ════════════════════════════════════════════════════
#  TAB 1 — BERANDA
# ════════════════════════════════════════════════════
with tab1:
    if not cuaca["ok"] and cuaca["pesan"]:
        st.warning(cuaca["pesan"])

    st.markdown(f"""
    <div class="gps-box">
      <span>📡 <b>Stasiun BMKG:</b> {cuaca['nama']}</span>
      <span>📏 <b>Jarak:</b> {cuaca['jarak_km']} km</span>
      <span>🕐 <b>Data per:</b> {cuaca['waktu']}</span>
      <span>📌 <b>Koordinat:</b> {round(lat,4)}°, {round(lon,4)}°</span>
      <span>🗺️ <b>Provinsi:</b> {provinsi}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-row">
      <span class="chip {status['kelas']}">{status['ikon']} {status['level']}</span>
      <span class="chip {enso['kelas']}">{enso['ikon']} {enso['level']}</span>
      <span class="chip info">📅 {msmn}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="big-card {status['kelas']}">
      <div class="bc-label">Kondisi Cuaca Hari Ini — {provinsi}</div>
      <div class="bc-row">
        <span class="bc-icon">{status['ikon']}</span>
        <span class="bc-value">{status['level']}</span>
      </div>
      <div class="bc-sub">{status['pesan']}</div>
    </div>
    """, unsafe_allow_html=True)

    kel_int  = int(cuaca["kelembapan"])
    kls_kel  = "waspada" if kel_int > 85 else "aman"
    kls_ch   = "bahaya" if cuaca["curah_hujan"]>50 else "waspada" if cuaca["curah_hujan"]>30 else "aman"
    kls_suhu = "bahaya" if cuaca["tmax"]>34 else "waspada" if cuaca["tmax"]>32 else "aman"

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card {kls_suhu}">
        <div class="mc-label">🌡️ Suhu Maksimum</div>
        <div class="mc-val">{cuaca['tmax']}°C</div>
        <div class="mc-sub">Min: {cuaca['tmin']}°C &nbsp;|&nbsp; Ideal padi: 24–30°C</div>
      </div>
      <div class="metric-card {kls_kel}">
        <div class="mc-label">💧 Kelembapan Udara</div>
        <div class="mc-val">{kel_int}%</div>
        <div class="mc-sub">{'⚠️ Waspadai serangan wereng' if kel_int>85 else '✅ Dalam batas normal'}</div>
      </div>
      <div class="metric-card {kls_ch}">
        <div class="mc-label">🌧️ Curah Hujan</div>
        <div class="mc-val">{cuaca['curah_hujan']} mm</div>
        <div class="mc-sub">per jam &nbsp;|&nbsp; {'⚠️ Terlalu lebat' if cuaca['curah_hujan']>30 else '✅ Normal'}</div>
      </div>
      <div class="metric-card {enso['kelas']}">
        <div class="mc-label">{enso['ikon']} Kondisi Iklim</div>
        <div class="mc-val" style="font-size:1rem;">{enso['level']}</div>
        <div class="mc-sub">ONI: {oni:+.1f}°C &nbsp;|&nbsp; {enso['fase']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">💡 Yang Harus Dilakukan</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="two-col">
      <div class="rekom {warna_rk}">
        <div class="rk-title">🌱 Saran Tanam</div>
        <div class="rk-text">{teks_rk}</div>
      </div>
      <div class="rekom biru">
        <div class="rk-title">🌏 Kondisi Iklim — {enso['fase']}</div>
        <div class="rk-text">{enso['pesan']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    musim_tips = (
        "Musim hujan cocok untuk tanam padi. Pastikan drainase lancar dan waspadai blas."
        if "Hujan" in msmn else
        "Musim kemarau butuh irigasi cukup. Pilih varietas tahan kering seperti Inpari 32."
        if "Kemarau" in msmn else
        "Musim peralihan: curah hujan tidak menentu. Pantau cuaca setiap hari sebelum memutuskan tanam."
    )
    st.markdown(f"""
    <div class="rekom">
      <div class="rk-title">📅 {msmn}</div>
      <div class="rk-text">{musim_tips}</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  TAB 2 — CUACA & IKLIM
# ════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<p class="sec-title">🌤️ Cuaca Detail — {provinsi}</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡️ Suhu Maks",   f"{cuaca['tmax']} °C",
              delta=f"{round(cuaca['tmax']-30,1)} dari ideal 30°C", delta_color="inverse")
    c2.metric("🌡️ Suhu Min",    f"{cuaca['tmin']} °C")
    c3.metric("💧 Kelembapan",  f"{int(cuaca['kelembapan'])} %")
    c4.metric("🌧️ Curah Hujan", f"{cuaca['curah_hujan']} mm/hr")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">📊 Kondisi Iklim ENSO — Data Riil</p>', unsafe_allow_html=True)
    st.caption("🔴 El Niño = kemarau panjang  |  🔵 La Niña = hujan berlebih  |  🟢 Normal")
    st.plotly_chart(fig_oni(df_enso), use_container_width=True)

    if df_ndvi is not None:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-title">🌿 Kondisi Tanaman — NDVI Sentinel-2</p>', unsafe_allow_html=True)
        st.caption("Nilai 0.6–0.8 = tanaman sehat dan subur")
        st.plotly_chart(fig_ndvi(df_ndvi), use_container_width=True)

    if df_sar is not None:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-title">📡 Radar Satelit — SAR Sentinel-1 (VH)</p>', unsafe_allow_html=True)
        st.caption("Deteksi kelembapan tanah dan genangan air di lahan sawah")
        st.plotly_chart(fig_sar(df_sar), use_container_width=True)


# ════════════════════════════════════════════════════
#  TAB 3 — CEK HAMA
# ════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="sec-title">🔬 Cek Hama Padi — Foto Tanaman Anda</p>',
                unsafe_allow_html=True)

    if model_cnn is None:
        st.warning(f"⚠️ Model belum dimuat. Path: `{MODEL_PATH}` | Ada: `{MODEL_PATH.exists()}`")

    st.markdown("""
    <div class="rekom biru">
      <div class="rk-title">📷 Cara Menggunakan</div>
      <div class="rk-text">
        Ambil foto daun, batang, atau akar padi yang terlihat sakit, lalu unggah di sini.
        Sistem akan mendeteksi jenis penyakit secara otomatis dalam beberapa detik.
      </div>
    </div>
    """, unsafe_allow_html=True)

    foto = st.file_uploader("Pilih foto tanaman padi:", type=["jpg","jpeg","png"])

    if foto:
        from PIL import Image
        img = Image.open(foto)

        col_f, col_h = st.columns([1, 1])
        with col_f:
            st.image(img, use_column_width=True, caption="Foto yang diunggah")

        with col_h:
            st.markdown('<p class="sec-title">🤖 Hasil Deteksi</p>', unsafe_allow_html=True)

            if model_cnn is None:
                st.error("Model belum tersedia.")
            else:
                with st.spinner("Menganalisis foto..."):
                    hasil = prediksi_gambar(foto.getvalue())

                if hasil:
                    kls_card = "bahaya" if hasil["berbahaya"] else "waspada"
                    kls_rk   = "merah"  if hasil["berbahaya"] else "kuning"
                    conf_pct = round(hasil["confidence"] * 100, 1)

                    st.markdown(f"""
                    <div class="big-card {kls_card}">
                      <div class="bc-label">Penyakit Terdeteksi</div>
                      <div class="bc-value">{hasil['nama_indo']}</div>
                      <div class="bc-sub">Keyakinan model: <b>{conf_pct}%</b></div>
                    </div>
                    <div class="rekom {kls_rk}">
                      <div class="rk-title">💊 Cara Penanganan</div>
                      <div class="rk-text">{hasil['rekomendasi']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # TaniBot rekomendasi penanganan
                    if hasil["kelas"] != "healthy":
                        opt_id = CLASS_TO_OPT.get(hasil["kelas"])
                        if opt_id and tanibot:
                            st.markdown('<hr class="divider">', unsafe_allow_html=True)
                            st.markdown('<p class="sec-title">🌿 Rekomendasi Penanganan — TaniBot</p>',
                                        unsafe_allow_html=True)
                            fase = st.selectbox(
                                "Fase tanaman saat ini:",
                                ["pertanaman", "pesemaian", "pratanam"],
                                format_func=lambda x: {
                                    "pertanaman": "🌿 Pertanaman (sudah tanam)",
                                    "pesemaian":  "🌱 Pesemaian",
                                    "pratanam":   "🧺 Pratanam (belum tanam)",
                                }[x],
                                key="fase_sel"
                            )
                            recs_resp = tanibot.get_post_detection_recs(opt_id, fase)
                            WARNA_METODE = {
                                "Kultur Teknis": "",
                                "Biologis":      "biru",
                                "Kimiawi":       "merah",
                                "Mekanis":       "kuning",
                                "Sanitasi":      "",
                            }
                            for rec in recs_resp.recs:
                                warna  = WARNA_METODE.get(rec["metode"], "")
                                extras = ""
                                if rec.get("dosis"):
                                    extras += f"<br>💊 <b>Dosis:</b> {rec['dosis']}"
                                if rec.get("waktu_aplikasi"):
                                    extras += f"<br>⏰ <b>Waktu:</b> {rec['waktu_aplikasi']}"
                                st.markdown(f"""
                                <div class="rekom {warna}">
                                  <div class="rk-title">#{rec['prioritas']} [{rec['metode']}] {rec['langkah']}</div>
                                  <div class="rk-text">{rec['detail']}{extras}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            if recs_resp.sumber:
                                st.caption(f"📚 Sumber: {'; '.join(set(recs_resp.sumber))}")
                        elif tanibot is None:
                            st.info("TaniBot belum aktif — bangun database dulu dengan `tanibot_db_builder_v2.py`")

                    # Bar probabilitas semua kelas
                    st.markdown('<p class="sec-title" style="margin-top:16px;">📊 Probabilitas Semua Kelas</p>',
                                unsafe_allow_html=True)
                    probs_sorted = sorted(hasil["probs"].items(), key=lambda x: x[1], reverse=True)
                    bars_html = ""
                    for kls, prob in probs_sorted:
                        pct  = round(prob * 100, 1)
                        top  = "top" if kls == hasil["kelas"] else ""
                        nama = NAMA_INDO[kls]
                        bars_html += f"""
                        <div class="prob-bar-wrap">
                          <div class="prob-label">
                            <span>{nama}</span><span><b>{pct}%</b></span>
                          </div>
                          <div class="prob-bar-bg">
                            <div class="prob-bar-fill {top}" style="width:{pct}%"></div>
                          </div>
                        </div>"""
                    st.markdown(bars_html, unsafe_allow_html=True)

    else:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-title">🐛 Kenali Penyakit Padi yang Umum</p>',
                    unsafe_allow_html=True)

        HAMA = [
            ("🦠","Hawar Daun Bakteri (HDB)","Daun mengering dari tepi, seperti terbakar. Eksudat kuning di permukaan daun."),
            ("🍂","Bercak Cokelat",           "Bercak oval cokelat gelap di daun, sering di varietas kekurangan nutrisi."),
            ("🌿","Sehat",                     "Daun hijau segar, tidak ada bercak, tanaman tegak dan subur."),
            ("🍃","Blas Daun",                 "Bercak belah ketupat abu-abu dengan tepi cokelat, daun bisa mati."),
            ("📋","Hawar Pelepah",             "Bercak cokelat terang di pelepah daun bawah, berkembang ke atas."),
            ("🟤","Bercak Cokelat Sempit",     "Garis-garis sempit cokelat sejajar tulang daun."),
            ("🪲","Hispa Padi",                "Permukaan daun bergaris putih/transparan, ada bekas goresan larva."),
            ("🌾","Busuk Pelepah",             "Bercak oval abu-abu di pelepah, meluas dan batang bisa roboh."),
            ("🌡️","Tungro",                   "Daun kuning-oranye, tanaman kerdil, anakan berkurang."),
        ]
        st.markdown('<div class="hama-grid">', unsafe_allow_html=True)
        for ikon, nama, ciri in HAMA:
            st.markdown(f"""
            <div class="hama-card">
              <span class="hama-icon">{ikon}</span>
              <div>
                <div class="hama-nama">{nama}</div>
                <div class="hama-ciri">{ciri}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  TAB 4 — INFORMASI
# ════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="sec-title">ℹ️ Tentang HarvestAlert</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="rekom biru">
      <div class="rk-title">📖 Tentang Aplikasi</div>
      <div class="rk-text">
        <b>HarvestAlert</b> adalah sistem peringatan dini pertanian padi Indonesia yang
        menggabungkan data cuaca real-time BMKG, kondisi iklim global (ENSO),
        data satelit Sentinel-1/2, dan kecerdasan buatan untuk deteksi penyakit.<br><br>
        <b>Sumber data:</b> BMKG · NOAA/CPC · Sentinel-1 SAR · Sentinel-2 NDVI · BBPOPT<br>
        <b>Model AI:</b> CNN Agriwarn — 98% akurasi, 9 kelas penyakit daun padi<br>
        <b>Regulasi:</b> Permentan No. 39/2018 · UU PDP No. 27/2022<br>
        <b>Versi:</b> 1.0 · Satria Data 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">📊 Status Sistem</p>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Data BMKG",  "🟢 Live"     if cuaca["ok"]       else "🟡 Estimasi")
    s2.metric("Data ENSO",  f"🟢 {len(df_enso)} data")
    s3.metric("Data NDVI",  "🟢 Aktif"    if df_ndvi is not None else "❌ Tidak ada")
    s4.metric("Model CNN",  "🟢 Siap"     if model_cnn is not None else "❌ Tidak ada")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">🌦️ Panduan Warna Peringatan</p>', unsafe_allow_html=True)

    for warna, kls, arti, aksi in [
        ("🟢 Hijau",  "",       "Kondisi aman",  "Lanjutkan aktivitas normal"),
        ("🟡 Kuning", "kuning", "Perlu waspada", "Pantau lahan lebih sering"),
        ("🔴 Merah",  "merah",  "Berbahaya",     "Ambil tindakan, hubungi penyuluh"),
    ]:
        st.markdown(f"""
        <div class="rekom {kls}">
          <div class="rk-title">{warna}</div>
          <div class="rk-text"><b>{arti}</b> — {aksi}</div>
        </div>
        """, unsafe_allow_html=True)
