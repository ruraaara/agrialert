import math
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
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
    'healthy':               'Sehat',
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

AI_MODEL = "llama-3.3-70b-versatile"

# NOAA CPC ONI — gratis, tanpa API key, update bulanan
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
_SEAS_MONTH  = {
    "DJF":1, "JFM":2, "FMA":3, "MAM":4, "AMJ":5, "MJJ":6,
    "JJA":7, "JAS":8, "ASO":9, "SON":10, "OND":11, "NDJ":12,
}


# ----------------------------------------------------------------
#  IKON SVG (line-icon, profesional — pengganti emoji)
# ----------------------------------------------------------------
_ICON_PATHS = {
    "sprout":        '<path d="M12 21v-7"/><path d="M12 14c-4.5 0-8-3.5-8-8 4.5 0 8 3.5 8 8Zm0 0c4.5 0 8-3.5 8-8-4.5 0-8 3.5-8 8Z"/>',
    "pin":           '<path d="M12 22s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12Z"/><circle cx="12" cy="10" r="2.3"/>',
    "antenna":       '<path d="M12 3v5"/><path d="M8.5 6.5a5 5 0 0 1 7 0"/><path d="M5.5 9.5a8.5 8.5 0 0 1 13 0"/><circle cx="12" cy="15" r="2"/><path d="M9.5 22h5M12 17v5"/>',
    "ruler":         '<path d="M3 17 17 3l4 4-14 14-4-4Z"/><path d="m7.5 12.5 2 2M10.5 9.5l2 2M13.5 6.5l2 2"/>',
    "clock":         '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5l3 2.5"/>',
    "map":           '<path d="M9 4 4 6v14l5-2 6 2 5-2V4l-5 2-6-2Z"/><path d="M9 4v14M15 6v14"/>',
    "calendar":      '<rect x="3" y="5" width="18" height="16" rx="2.5"/><path d="M3 9.5h18M8 3v4M16 3v4"/>',
    "tag":           '<path d="M3 12 12 3h6a3 3 0 0 1 3 3v6l-9 9-9-9Z"/><circle cx="16" cy="8" r="1.3"/>',
    "phone":         '<path d="M5 4h3l2 5-2.2 1.3a11 11 0 0 0 5.9 5.9L15 14l5 2v3a2 2 0 0 1-2 2A17 17 0 0 1 3 6a2 2 0 0 1 2-2Z"/>',
    "sun":           '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v3M12 18.5v3M4.6 4.6l2.1 2.1M17.3 17.3l2.1 2.1M2.5 12h3M18.5 12h3M4.6 19.4l2.1-2.1M17.3 6.7l2.1-2.1"/>',
    "cloud":         '<path d="M7 18a4 4 0 1 1 .6-7.96A5 5 0 0 1 17 11a3.5 3.5 0 0 1-.5 7H7Z"/>',
    "cloud-sun":     '<circle cx="7.3" cy="7.3" r="2.1"/><path d="M7.3 3v1.6M3.9 4.9l1.1 1.1M3 8.3h1.6"/><path d="M9 19a4 4 0 1 1 .5-7.97A5 5 0 0 1 19 12a3.5 3.5 0 0 1-.5 7H9Z"/>',
    "cloud-rain":    '<path d="M7 15.5a4 4 0 1 1 .6-7.96A5 5 0 0 1 17 8.5a3.5 3.5 0 0 1-.5 7H7Z"/><path d="M8.5 19l-1 2.2M12.5 19l-1 2.2M16.5 19l-1 2.2"/>',
    "storm":         '<path d="M7 14a4 4 0 1 1 .6-7.96A5 5 0 0 1 17 7a3.5 3.5 0 0 1-.5 7H7Z"/><path d="M13.3 13.5 10.5 18h2l-1.8 3.7 4.6-5.2h-2l1.8-3Z"/>',
    "wave":          '<path d="M2 14.5c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0"/><path d="M2 19c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0"/>',
    "droplet":       '<path d="M12 2.5C9 7.2 5 11.3 5 15.3a7 7 0 0 0 14 0c0-4-4-8.1-7-12.8Z"/>',
    "wind":          '<path d="M3 8h11a3 3 0 1 0-3-3"/><path d="M3 16h15a3 3 0 1 1-3 3"/><path d="M3 12h9"/>',
    "thermometer":   '<path d="M12 14.2V4.5a2 2 0 1 0-4 0v9.7a4 4 0 1 0 4 0Z"/>',
    "alert-triangle":'<path d="M12 3 2 21h20L12 3Z"/><path d="M12 10v4M12 16.9h.01"/>',
    "alert-circle":  '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 15.9h.01"/>',
    "check-circle":  '<circle cx="12" cy="12" r="9"/><path d="m8 12.2 3 3 5-6"/>',
    "info-circle":   '<circle cx="12" cy="12" r="9"/><path d="M12 8.3h.01M12 11v5"/>',
    "bug":           '<rect x="8" y="9" width="8" height="9" rx="4"/><path d="M9 9V7a3 3 0 1 1 6 0v2M5 12h3M16 12h3M5.5 17l2.3-1.9M18.5 17l-2.3-1.9M5.5 7.2l2.8 1.9M18.5 7.2l-2.8 1.9M12 18v3"/>',
    "leaf":          '<path d="M21 3C10 3 3 10 3 21c11 0 18-7 18-18Z"/><path d="M3.5 20.5 14 10"/>',
    "microscope":    '<path d="M8.5 21h7M6.5 17h11M9.5 17v-3.3a3.3 3.3 0 1 1 4 0V17"/><circle cx="11.5" cy="7.6" r="2.8"/><path d="M9.4 5.7 7.8 4.1"/>',
    "gear":          '<circle cx="12" cy="12" r="3"/><path d="M19.4 13a7.6 7.6 0 0 0 0-2l1.9-1.5-1.9-3.3-2.3 1a7.6 7.6 0 0 0-1.7-1L15 4h-4l-.3 2.4a7.6 7.6 0 0 0-1.7 1l-2.3-1L4.7 9.5l1.9 1.5a7.6 7.6 0 0 0 0 2l-1.9 1.5 1.9 3.3 2.3-1a7.6 7.6 0 0 0 1.7 1L11 20h4l.3-2.4a7.6 7.6 0 0 0 1.7-1l2.3 1 1.9-3.3-1.9-1.3Z"/>',
    "chart":         '<path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/>',
    "bot":           '<rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 4v4M9 13.5h.01M15 13.5h.01"/><path d="M2 13h2M20 13h2"/>',
    "user":          '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.5-7 8-7s8 3 8 7"/>',
    "refresh":       '<path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 4v5h-5"/>',
    "trash":         '<path d="M5 7h14M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2m-9 0 1 13a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13"/>',
    "book":          '<path d="M4 5a2 2 0 0 1 2-2h12v17H6a2 2 0 0 0-2 2V5Z"/><path d="M6 17h12"/>',
    "pill":          '<rect x="2" y="9" width="20" height="6" rx="3"/><path d="M12 9v6"/>',
    "satellite":     '<rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 9 4 4M15 9l5-5M9 15l-5 5M15 15l5 5"/>',
    "send":          '<path d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7Z"/>',
    "lightbulb":     '<path d="M9 18h6M10 21h4"/><path d="M12 2a6 6 0 0 0-3.6 10.8c.4.3.6.8.6 1.3V15h6v-.9c0-.5.2-1 .6-1.3A6 6 0 0 0 12 2Z"/>',
    "camera":        '<rect x="2.5" y="6" width="19" height="14" rx="2.5"/><circle cx="12" cy="13" r="3.5"/><path d="M8.5 6 10 3.5h4L15.5 6"/>',
}


def ico(name, size="1.2em"):
    """Render ikon SVG line-style (pengganti emoji) sebagai string HTML inline."""
    paths = _ICON_PATHS.get(name)
    if not paths:
        return ""
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-0.22em;flex-shrink:0" '
        f'aria-hidden="true">{paths}</svg>'
    )


# ----------------------------------------------------------------
#  KONFIGURASI HALAMAN
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AgriWarn",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&display=swap');

:root {
    --c-primary: #1f7a4d;
    --c-primary-dark: #0f4429;
    --c-primary-darker: #0a331f;
    --c-primary-light: #e8f5ee;
    --c-sage: #6f8f78;

    --c-bg: #f6f8f5;
    --c-surface: #ffffff;
    --c-border: #dfe6e0;
    --c-text: #1c2620;
    --c-text-muted: #57685d;
    --c-text-on-light: #1c2620;
    --c-muted-on-surface: #57685d;
    --c-accent-text: #0f4429;

    --c-aman-bg: #eaf7ef;     --c-aman-border:#1f7a4d;   --c-aman-text:#14532d;
    --c-waspada-bg:#fef6e7;   --c-waspada-border:#d97706; --c-waspada-text:#92400e;
    --c-bahaya-bg:#fdeceb;    --c-bahaya-border:#dc2626; --c-bahaya-text:#991b1b;
    --c-info-bg:#eaf2fb;      --c-info-border:#2563eb;   --c-info-text:#1e40af;

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 3px rgba(15,68,41,0.07);
    --shadow-md: 0 4px 14px rgba(15,68,41,0.10);
}

/* Mode gelap (mengikuti preferensi sistem/browser) — menjaga kontras teks tetap
   tinggi karena widget asli Streamlit otomatis berganti ke teks terang saat gelap. */
@media (prefers-color-scheme: dark) {
    :root {
        --c-bg: #161b1f;
        --c-surface: #20262b;
        --c-border: #38423d;
        --c-text: #e9ede9;
        --c-muted-on-surface: #a3b3a8;
        --c-accent-text: #5fd897;
    }
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Lexend', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    font-size: 17px;
    color: var(--c-text);
}
.stApp { background: var(--c-bg) !important; }
.block-container {
    max-width: 1200px !important;
    padding: 1.25rem 1.5rem 4rem !important;
    margin: 0 auto !important;
}
.ha-header {
    background: var(--c-primary-dark); border-radius: var(--radius-lg); padding: 24px 28px;
    margin-bottom: 22px; border: none; box-shadow: var(--shadow-md);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}
.ha-header-left { display: flex; align-items: center; gap: 16px; }
.ha-logo  { font-size: 2.6rem; line-height: 1; }
.ha-title { font-size: 1.75rem; font-weight: 800; color: #ffffff; margin: 0; letter-spacing: -0.3px; }
.ha-sub   { font-size: 0.95rem; color: #cdeedb; margin: 4px 0 0; }
.ha-badge {
    background: rgba(255,255,255,0.14); border: 1.5px solid rgba(255,255,255,0.3); border-radius: 30px;
    padding: 8px 18px; font-size: 0.9rem; font-weight: 700; color: #ffffff; white-space: nowrap;
}
.gps-box {
    background: var(--c-surface); border: 1.5px solid var(--c-border); border-radius: var(--radius-md);
    padding: 14px 18px; margin-bottom: 20px; font-size: 0.92rem; color: var(--c-muted-on-surface);
    line-height: 1.9; display: flex; flex-wrap: wrap; gap: 6px 22px; box-shadow: var(--shadow-sm);
}
.gps-box b { color: var(--c-accent-text); }
.big-card { border-radius: var(--radius-lg); padding: 22px 26px; margin-bottom: 16px; border: 1.5px solid transparent; box-shadow: var(--shadow-sm); }
.big-card.aman    { background: var(--c-aman-bg);    border-color: var(--c-aman-border); }
.big-card.waspada { background: var(--c-waspada-bg); border-color: var(--c-waspada-border); }
.big-card.bahaya  { background: var(--c-bahaya-bg);  border-color: var(--c-bahaya-border); }
.big-card.info    { background: var(--c-info-bg);    border-color: var(--c-info-border); }
.bc-label { font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--c-text-muted); margin-bottom: 10px; }
.bc-row   { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.bc-icon  { font-size: 2.2rem; line-height: 1; flex-shrink: 0; }
.bc-value { font-size: 1.65rem; font-weight: 800; color: var(--c-text); line-height: 1.15; }
.bc-sub   { font-size: 0.92rem; color: var(--c-text-muted); margin-top: 6px; line-height: 1.6; }
.big-card.aman    .bc-value, .big-card.aman    .bc-label { color: var(--c-aman-text); }
.big-card.waspada .bc-value, .big-card.waspada .bc-label { color: var(--c-waspada-text); }
.big-card.bahaya  .bc-value, .big-card.bahaya  .bc-label { color: var(--c-bahaya-text); }
.big-card.info    .bc-value, .big-card.info    .bc-label { color: var(--c-info-text); }
.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 16px;
}
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .metric-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; } }
.metric-card { border-radius: var(--radius-md); padding: 18px 20px; border: 1.5px solid var(--c-border); background: var(--c-surface); box-shadow: var(--shadow-sm); min-width: 0; }
.metric-card.aman    { background: var(--c-aman-bg);    border-color: var(--c-aman-border); }
.metric-card.waspada { background: var(--c-waspada-bg); border-color: var(--c-waspada-border); }
.metric-card.bahaya  { background: var(--c-bahaya-bg);  border-color: var(--c-bahaya-border); }
.metric-card.info    { background: var(--c-info-bg);    border-color: var(--c-info-border); }
.mc-label { font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--c-text-muted); margin-bottom: 8px; }
.mc-val   { font-size: 1.5rem; font-weight: 800; color: var(--c-text); line-height: 1.15; }
.mc-sub   { font-size: 0.8rem; color: var(--c-text-muted); margin-top: 6px; }
.metric-card.aman    .mc-val { color: var(--c-aman-text); }
.metric-card.waspada .mc-val { color: var(--c-waspada-text); }
.metric-card.bahaya  .mc-val { color: var(--c-bahaya-text); }
.metric-card.info    .mc-val { color: var(--c-info-text); }
@media (max-width: 480px) { .metric-card { padding: 14px 16px; } .mc-val { font-size: 1.3rem; } }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
.rekom { background: var(--c-aman-bg); border-radius: 0 var(--radius-md) var(--radius-md) 0; border-left: 5px solid var(--c-primary); padding: 18px 20px; margin-bottom: 12px; box-shadow: var(--shadow-sm); }
.rekom.kuning { background: var(--c-waspada-bg); border-left-color: var(--c-waspada-border); }
.rekom.merah  { background: var(--c-bahaya-bg);  border-left-color: var(--c-bahaya-border); }
.rekom.biru   { background: var(--c-info-bg);    border-left-color: var(--c-info-border); }
.rk-title { font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--c-primary-dark); margin-bottom: 10px; }
.rekom.kuning .rk-title { color: var(--c-waspada-text); }
.rekom.merah  .rk-title { color: var(--c-bahaya-text); }
.rekom.biru   .rk-title { color: var(--c-info-text); }
.rk-text { font-size: 0.98rem; color: var(--c-text-on-light); line-height: 1.7; }
.hama-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
@media (max-width: 500px) { .hama-grid { grid-template-columns: 1fr; } }
.hama-card { background: var(--c-surface); border: 1.5px solid var(--c-border); border-radius: var(--radius-md); padding: 16px 18px; display: flex; align-items: flex-start; gap: 14px; box-shadow: var(--shadow-sm); }
.hama-icon { font-size: 1.8rem; line-height: 1; flex-shrink: 0; margin-top: 2px; }
.hama-nama { font-size: 0.98rem; font-weight: 700; color: var(--c-text); margin-bottom: 4px; }
.hama-ciri { font-size: 0.85rem; color: var(--c-muted-on-surface); line-height: 1.55; }
.sec-title { font-size: 1.15rem; font-weight: 700; color: var(--c-accent-text); margin: 24px 0 14px; display: flex; align-items: center; gap: 10px; }
.divider { border: none; border-top: 1.5px solid var(--c-border); margin: 20px 0; }
.status-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }
.chip { border-radius: 30px; padding: 8px 16px; font-size: 0.85rem; font-weight: 700; display: inline-flex; align-items: center; gap: 6px; border: 1.5px solid transparent; }
.chip.aman    { background: var(--c-aman-bg);    border-color: var(--c-aman-border);    color: var(--c-aman-text); }
.chip.waspada { background: var(--c-waspada-bg); border-color: var(--c-waspada-border); color: var(--c-waspada-text); }
.chip.bahaya  { background: var(--c-bahaya-bg);  border-color: var(--c-bahaya-border);  color: var(--c-bahaya-text); }
.chip.info    { background: var(--c-info-bg);    border-color: var(--c-info-border);    color: var(--c-info-text); }
div[data-baseweb="tab-list"] { background: var(--c-surface) !important; border: 1.5px solid var(--c-border) !important; border-radius: var(--radius-md) !important; padding: 6px !important; gap: 4px !important; margin-bottom: 22px; box-shadow: var(--shadow-sm); }
button[data-baseweb="tab"] { font-family: 'Lexend', sans-serif !important; font-size: 0.98rem !important; font-weight: 700 !important; border-radius: var(--radius-sm) !important; padding: 12px 10px !important; color: var(--c-muted-on-surface) !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: var(--c-primary) !important; color: #ffffff !important; }
[data-testid="stSidebar"] { background: var(--c-surface) !important; border-right: 1.5px solid var(--c-border) !important; }
.stButton > button, div[data-testid="stFormSubmitButton"] button {
    font-family: 'Lexend', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: var(--radius-md) !important;
    padding: 0.7rem 1.3rem !important;
    min-height: 48px !important;
    border: 1.5px solid var(--c-primary) !important;
    background: var(--c-primary) !important;
    color: #ffffff !important;
}
.stButton > button:hover {
    background: var(--c-primary-dark) !important;
    border-color: var(--c-primary-dark) !important;
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 800 !important; color: var(--c-accent-text) !important; }
[data-testid="stMetricLabel"] { font-size: 0.9rem !important; font-weight: 600 !important; }
.prob-bar-wrap { margin: 8px 0; }
.prob-label { font-size: 0.85rem; color: var(--c-muted-on-surface); margin-bottom: 4px; display: flex; justify-content: space-between; }
.prob-bar-bg { background: var(--c-border); border-radius: 8px; height: 12px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 8px; background: var(--c-sage); transition: width 0.4s; }
.prob-bar-fill.top { background: var(--c-primary); }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: var(--c-border); border-radius: 6px; }
@media (max-width: 480px) { .ha-title { font-size: 1.4rem; } .bc-value { font-size: 1.35rem; } .sec-title { font-size: 1.02rem; } .rk-text { font-size: 0.92rem; } .gps-box { font-size: 0.85rem; } }
.tbot-chat {
    background: var(--c-bg); border: 1.5px solid var(--c-border); border-radius: var(--radius-md);
    padding: 16px 14px; max-height: 400px; overflow-y: auto;
    margin: 12px 0; display: flex; flex-direction: column; gap: 12px;
}
.tbot-row { display: flex; align-items: flex-end; gap: 10px; }
.tbot-row.bot  { flex-direction: row; }
.tbot-row.user { flex-direction: row-reverse; }
.tbot-av { font-size: 1.3rem; flex-shrink: 0; }
.tbot-bbl { padding: 11px 15px; font-size: 0.92rem; line-height: 1.7; max-width: 84%; word-break: break-word; }
.tbot-bbl.bot  { background:var(--c-surface); border:1.5px solid var(--c-border); border-radius:4px 16px 16px 16px; color:var(--c-text); }
.tbot-bbl.user { background:var(--c-aman-bg); border:1.5px solid var(--c-aman-border); border-radius:16px 4px 16px 16px; color:var(--c-aman-text); }
.tbot-hint { font-size:0.82rem; color:var(--c-muted-on-surface); margin:8px 0 6px; }
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
#  DEEPSEEK AI CLIENT
# ================================================================
def get_ai_client():
    try:
        from openai import OpenAI
        api_key = None
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    except Exception:
        return None


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

# Kode wilayah adm4 (kelurahan/desa, Kepmendagri) di ibu kota tiap provinsi —
# dipakai sebagai titik referensi prakiraan cuaca BMKG (API publik mewajibkan kode adm4).
PROVINSI_ADM4 = {
    "Aceh": "11.71.01.2001",
    "Sumatera Utara": "12.71.01.1001",
    "Sumatera Barat": "13.71.01.1001",
    "Riau": "14.71.01.1002",
    "Jambi": "15.71.01.1001",
    "Sumatera Selatan": "16.71.01.1001",
    "Bengkulu": "17.71.01.1001",
    "Lampung": "18.71.01.1003",
    "Kepulauan Bangka Belitung": "19.71.01.1004",
    "Kepulauan Riau": "21.72.01.1001",
    "DKI Jakarta": "31.71.01.1001",
    "Jawa Barat": "32.73.01.1001",
    "Jawa Tengah": "33.74.01.1001",
    "DI Yogyakarta": "34.71.01.1001",
    "Jawa Timur": "35.78.01.1001",
    "Banten": "36.73.01.1001",
    "Bali": "51.71.01.1001",
    "Nusa Tenggara Barat": "52.71.01.1004",
    "Nusa Tenggara Timur": "53.71.01.1001",
    "Kalimantan Barat": "61.71.01.1002",
    "Kalimantan Tengah": "62.71.01.1001",
    "Kalimantan Selatan": "63.71.01.1001",
    "Kalimantan Timur": "64.72.01.1001",
    "Kalimantan Utara": "65.01.01.1001",
    "Sulawesi Utara": "71.71.01.1001",
    "Sulawesi Tengah": "72.71.01.1004",
    "Sulawesi Selatan": "73.71.01.1001",
    "Sulawesi Tenggara": "74.71.01.1005",
    "Gorontalo": "75.71.01.1001",
    "Sulawesi Barat": "76.02.01.1002",
    "Maluku": "81.71.01.1006",
    "Maluku Utara": "82.72.01.1001",
    "Papua Barat": "92.02.03.2001",
    "Papua": "91.71.01.1001",
    "Papua Selatan": "93.01.01.1002",
    "Papua Tengah": "94.01.01.1001",
    "Papua Pegunungan": "95.01.01.1001",
    "Papua Barat Daya": "96.71.01.1001",
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
BMKG_API_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"

@st.cache_data(ttl=1800)
def cuaca_bmkg(lat, lon):
    FB = {"nama":"Estimasi","jarak_km":0,"tmax":33.0,"tmin":24.0,
          "kelembapan":80,"curah_hujan":10.0,"waktu":"-","ok":False,
          "pesan":"Gagal terhubung ke BMKG. Menampilkan estimasi."}

    provinsi_terdekat = prov_terdekat(lat, lon)
    adm4 = PROVINSI_ADM4.get(provinsi_terdekat)
    if not adm4:
        return FB

    try:
        r = requests.get(BMKG_API_URL, params={"adm4": adm4}, timeout=8)
        r.raise_for_status()
        payload = r.json()
        lokasi = payload["lokasi"]
        flat = [e for hari in payload["data"][0]["cuaca"] for e in hari]
        if not flat:
            return FB
    except Exception:
        return FB

    def parse_utc(e):
        return datetime.strptime(e["utc_datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    try:
        flat.sort(key=parse_utc)
        now_utc = datetime.now(timezone.utc)
        mendatang = [e for e in flat if parse_utc(e) >= now_utc - timedelta(hours=3)] or flat
        jendela = mendatang[:8]
        terdekat = mendatang[0]

        try:
            wlabel = datetime.strptime(terdekat["local_datetime"], "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y %H:%M")
        except Exception:
            wlabel = terdekat.get("local_datetime", "-")

        d = jarak_km(lat, lon, lokasi["lat"], lokasi["lon"])
        hasil = {
            "nama": f"{lokasi['kecamatan']}, {lokasi['kotkab']}",
            "jarak_km": round(d, 1),
            "tmax": float(max(e["t"] for e in jendela)),
            "tmin": float(min(e["t"] for e in jendela)),
            "kelembapan": float(terdekat["hu"]),
            "curah_hujan": float(terdekat["tp"]),
            "waktu": wlabel,
            "ok": True,
            "pesan": "",
        }
    except Exception:
        return FB

    return hasil


# ================================================================
#  KLASIFIKASI IKLIM & CUACA
# ================================================================
def oni_now():
    df = df_enso[df_enso["date"] <= datetime.now()].tail(1)
    return float(df.iloc[0]["ONI"]) if not df.empty else 0.0

def info_enso(oni):
    if oni >= 1.5:
        return {"fase":"El Niño Kuat","level":"AWAS KEKERINGAN","kelas":"bahaya","ikon":"sun",
                "pesan":"Hujan sangat berkurang 40–60%. Siapkan irigasi cadangan. Pilih bibit tahan kering seperti Inpari 32."}
    elif oni >= 0.5:
        return {"fase":"El Niño","level":"WASPADA KERING","kelas":"waspada","ikon":"cloud-sun",
                "pesan":"Kemarau lebih lama dari biasanya. Hemat air irigasi dan pantau kondisi lahan."}
    elif oni >= -0.5:
        return {"fase":"Normal","level":"IKLIM NORMAL","kelas":"aman","ikon":"cloud",
                "pesan":"Pola hujan dan kemarau berjalan normal. Ikuti kalender tanam biasa."}
    elif oni >= -1.5:
        return {"fase":"La Niña","level":"WASPADA BANJIR","kelas":"waspada","ikon":"cloud-rain",
                "pesan":"Hujan lebih sering dan lebih deras. Periksa saluran air agar sawah tidak tergenang."}
    else:
        return {"fase":"La Niña Kuat","level":"AWAS BANJIR","kelas":"bahaya","ikon":"storm",
                "pesan":"Curah hujan sangat tinggi. Risiko banjir besar. Waspada serangan wereng dan tungro."}

def info_cuaca(tmax, ch):
    if tmax > 34.0 or ch > 50.0:
        return {"level":"CUACA EKSTREM","kelas":"bahaya",
                "ikon":"thermometer" if tmax>34 else "wave",
                "pesan":"Cuaca sangat berbahaya! Segera panen jika sudah waktunya. Aktifkan pompa drainase."}
    elif tmax > 32.0 or ch > 30.0 or ch < 2.0:
        return {"level":"PERLU PERHATIAN","kelas":"waspada","ikon":"alert-triangle",
                "pesan":"Cek saluran air dan kondisi tanaman. Tambah pengairan jika hujan sangat sedikit."}
    return {"level":"CUACA BAIK","kelas":"aman","ikon":"cloud-sun",
            "pesan":"Cuaca hari ini bagus untuk padi. Waktu yang tepat untuk pemupukan NPK."}

def musim_bln(b):
    if b in [11,12,1,2,3]: return "cloud-rain", "Musim Hujan"
    elif b in [6,7,8,9]:   return "sun", "Musim Kemarau"
    return "cloud-sun", "Musim Peralihan"

def saran_tanam(kelas_cuaca, kelas_enso):
    if kelas_cuaca=="bahaya" or kelas_enso=="bahaya":
        return ("merah", "alert-triangle",
                "<b>Tunda penanaman</b> sampai kondisi membaik. "
                "Kalau sudah tanam, pantau padi setiap hari dan pastikan saluran air tidak tersumbat.")
    elif kelas_cuaca=="waspada" or kelas_enso=="waspada":
        return ("kuning", "alert-circle",
                "<b>Lanjut tanam tapi waspada.</b> Cek irigasi rutin. "
                "Kalau kering, tambah pengairan. Kalau terlalu basah, buka saluran drainase.")
    return ("", "check-circle",
            "<b>Kondisi bagus untuk bertanam!</b> Waktu tepat untuk pemupukan NPK. "
            "Ikuti jadwal tanam seperti biasa.")


# ================================================================
#  HELPER GRAFIK
# ================================================================
PLOT_STYLE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(246,248,245,1)",
    margin=dict(t=16,b=36,l=44,r=20),
    font=dict(family="Lexend, sans-serif", size=13, color="#1c2620"),
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
#  NOAA CPC LIVE FETCH + PREDIKSI ENSO
# ================================================================
@st.cache_data(ttl=86400 * 7)   # refresh mingguan — NOAA update bulanan
def fetch_oni_noaa():
    """Fetch ONI langsung dari NOAA CPC. Returns (df|None, ok, fetch_time_str)."""
    try:
        r = requests.get(NOAA_ONI_URL, timeout=12)
        r.raise_for_status()
        rows = []
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0] in _SEAS_MONTH:
                try:
                    rows.append({
                        "date": datetime(int(parts[1]), _SEAS_MONTH[parts[0]], 1),
                        "ONI":  float(parts[3]),
                    })
                except (ValueError, IndexError):
                    pass
        if rows:
            df = (pd.DataFrame(rows)
                    .drop_duplicates("date")
                    .sort_values("date")
                    .reset_index(drop=True))
            return df, True, datetime.now().strftime("%d %b %Y %H:%M")
    except Exception:
        pass
    return None, False, None


def prediksi_oni_ar(df, p=3):
    """
    AR(p) model fitted on full NOAA historical data.
    y(t) = c + a1*y(t-1) + a2*y(t-2) + a3*y(t-3) + e
    Memanfaatkan autocorrelation kuat ENSO (event bertahan berbulan-bulan).
    """
    y = df["ONI"].values.astype(float)
    n = len(y)
    if n <= p:
        # Fallback ke persistence jika data terlalu sedikit
        last_date = pd.Timestamp(df["date"].iloc[-1])
        return {"date": last_date + pd.DateOffset(months=1),
                "oni": round(float(y[-1]), 2), "oni_low": round(float(y[-1]) - 0.3, 2),
                "oni_high": round(float(y[-1]) + 0.3, 2), "tren": "stabil →"}

    # Bangun lagged feature matrix dari seluruh data historis
    X = np.column_stack([[1.0] * (n - p)] +
                        [[y[i - j] for i in range(p, n)] for j in range(1, p + 1)])
    Y = y[p:]

    # OLS: koefisien AR belajar dari pola historis 75 tahun
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    intercept = coeffs[0]
    ar_coefs  = coeffs[1:]

    # Prediksi: y(T+1) = c + a1*y(T) + a2*y(T-1) + a3*y(T-2)
    last_vals = y[-p:][::-1]
    next_oni  = intercept + float(np.dot(ar_coefs, last_vals))

    # Ketidakpastian = std residual in-sample × 1.2
    std_err   = float(np.std(Y - X @ coeffs)) * 1.2

    last_date = pd.Timestamp(df["date"].iloc[-1])
    cur_oni   = float(y[-1])
    return {
        "date":     last_date + pd.DateOffset(months=1),
        "oni":      round(next_oni, 2),
        "oni_low":  round(next_oni - std_err, 2),
        "oni_high": round(next_oni + std_err, 2),
        "tren":     ("naik ↑"  if next_oni > cur_oni + 0.05
                     else "turun ↓" if next_oni < cur_oni - 0.05
                     else "stabil →"),
    }


def fig_oni_with_forecast(df, pred, n=24):
    """Bar chart ONI historis + diamond prediksi bulan depan."""
    df_p = df.tail(n).copy()
    df_p["W"] = df_p["ONI"].apply(
        lambda v: "#ef4444" if v >= 0.5 else ("#3b82f6" if v <= -0.5 else "#22c55e"))
    df_p["L"] = df_p["date"].dt.strftime("%b %Y")
    pred_lbl   = pred["date"].strftime("%b %Y") + " ▶"
    pred_color = ("#ef4444" if pred["oni"] >= 0.5
                  else "#3b82f6" if pred["oni"] <= -0.5 else "#22c55e")

    fig = go.Figure()
    fig.add_hrect(y0=0.5,  y1=3.0,  fillcolor="rgba(239,68,68,0.07)",  line_width=0)
    fig.add_hrect(y0=-3.0, y1=-0.5, fillcolor="rgba(59,130,246,0.07)", line_width=0)
    fig.add_hline(y= 0.5, line_dash="dot", line_color="#ef4444", line_width=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="#3b82f6", line_width=1)
    fig.add_trace(go.Bar(
        x=df_p["L"], y=df_p["ONI"], marker_color=df_p["W"],
        name="Historis",
        hovertemplate="<b>%{x}</b><br>ONI: %{y:.2f}°C<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=df_p["L"], y=df_p["ONI"], mode="lines+markers",
        line=dict(color="white", width=2),
        marker=dict(size=5, color=df_p["W"].tolist()),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pred_lbl], y=[pred["oni"]],
        mode="markers", name="Prediksi",
        marker=dict(size=14, color=pred_color, symbol="diamond",
                    line=dict(color="white", width=2)),
        error_y=dict(
            type="data",
            array=[max(0, pred["oni_high"] - pred["oni"])],
            arrayminus=[max(0, pred["oni"] - pred["oni_low"])],
            visible=True, color="rgba(255,255,255,0.45)", thickness=2, width=8,
        ),
        hovertemplate=(
            f"<b>Prediksi {pred['date'].strftime('%B %Y')}</b><br>"
            f"ONI: {pred['oni']:+.2f}°C<br>"
            f"Rentang: {pred['oni_low']:+.2f} s/d {pred['oni_high']:+.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **PLOT_STYLE, height=300, showlegend=True,
        yaxis_title="ONI (°C)", xaxis_tickangle=-40,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ================================================================
#  ── TAMPILAN ──
# ================================================================

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h2 style="display:flex;align-items:center;gap:8px;color:var(--c-accent-text);">{ico("gear")} Pengaturan</h2>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f'<h3 style="display:flex;align-items:center;gap:8px;color:var(--c-accent-text);">{ico("pin")} Lokasi Anda</h3>', unsafe_allow_html=True)
    st.caption("Tekan tombol di bawah untuk deteksi otomatis.")
    gps = streamlit_geolocation()

    lat, lon, aktif = -7.8754, 110.4262, False
    if gps and gps.get("latitude") and gps.get("longitude"):
        lat, lon, aktif = float(gps["latitude"]), float(gps["longitude"]), True
        st.success(f"GPS aktif\n{round(lat,4)}°, {round(lon,4)}°\n"
                   f"{prov_terdekat(lat,lon)}")
    else:
        st.warning("GPS belum aktif. Pilih manual.")
        lat = st.number_input("Lintang:", value=lat, format="%.4f")
        lon = st.number_input("Bujur:",   value=lon, format="%.4f")

    st.markdown("---")
    st.markdown(f'<h3 style="display:flex;align-items:center;gap:8px;color:var(--c-accent-text);">{ico("map")} Provinsi</h3>', unsafe_allow_html=True)
    daftar   = list(PROVINSI.keys())
    default  = prov_terdekat(lat, lon)
    provinsi = st.selectbox("", daftar, index=daftar.index(default))

    st.markdown("---")
    st.markdown(f'<h3 style="display:flex;align-items:center;gap:8px;color:var(--c-accent-text);">{ico("chart")} Status Data</h3>', unsafe_allow_html=True)
    st.write(f"ENSO  : {'Tersedia' if CSV_ENSO.exists() else 'Tidak ada'} ({len(df_enso)} baris)")
    st.write(f"NDVI  : {'Tersedia' if df_ndvi is not None else 'Tidak ada'}")
    st.write(f"SAR   : {'Tersedia' if df_sar  is not None else 'Tidak ada'}")
    st.write(f"Model : {'Siap' if model_cnn is not None else 'Tidak ditemukan'}")
    st.write(f"TaniBot: {'Siap' if tanibot is not None else 'Bangun DB dulu'}")
    st.markdown("---")
    st.caption("Data: BMKG · NOAA · Sentinel-1/2\nHarvestAlert v1.0 · Satria Data 2026")


# ── DATA REAL-TIME ────────────────────────────────────────────────
cuaca  = cuaca_bmkg(lat, lon)
oni    = oni_now()
enso   = info_enso(oni)
status = info_cuaca(cuaca["tmax"], cuaca["curah_hujan"])
musim_ikon, msmn = musim_bln(datetime.now().month)
warna_rk, ikon_rk, teks_rk = saran_tanam(status["kelas"], enso["kelas"])


# ── HEADER ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="ha-header">
  <div class="ha-header-left">
    <span class="ha-logo">{ico('sprout', '2rem')}</span>
    <div>
      <p class="ha-title">HarvestAlert</p>
      <p class="ha-sub">Peringatan dini hama &amp; cuaca padi Indonesia</p>
    </div>
  </div>
  <span class="ha-badge">{ico('pin')} {provinsi}</span>
</div>
""", unsafe_allow_html=True)


# ── 4 TAB ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab5, tab4 = st.tabs([
    "Beranda", "Cuaca & Iklim", "Cek Hama Padi", "Chat AI", "Informasi"
])


# ════════════════════════════════════════════════════
#  TAB 1 — BERANDA
# ════════════════════════════════════════════════════
with tab1:
    if not cuaca["ok"] and cuaca["pesan"]:
        st.warning(cuaca["pesan"])

    st.markdown(f"""
    <div class="gps-box">
      <span>{ico('antenna')} <b>Stasiun BMKG:</b> {cuaca['nama']}</span>
      <span>{ico('ruler')} <b>Jarak:</b> {cuaca['jarak_km']} km</span>
      <span>{ico('clock')} <b>Data per:</b> {cuaca['waktu']}</span>
      <span>{ico('pin')} <b>Koordinat:</b> {round(lat,4)}°, {round(lon,4)}°</span>
      <span>{ico('map')} <b>Provinsi:</b> {provinsi}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-row">
      <span class="chip {status['kelas']}">{ico(status['ikon'])} {status['level']}</span>
      <span class="chip {enso['kelas']}">{ico(enso['ikon'])} {enso['level']}</span>
      <span class="chip info">{ico(musim_ikon)} {msmn}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="big-card {status['kelas']}">
      <div class="bc-label">Kondisi Cuaca Hari Ini — {provinsi}</div>
      <div class="bc-row">
        <span class="bc-icon">{ico(status['ikon'], '2.2rem')}</span>
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
        <div class="mc-label">{ico('thermometer')} Suhu Maksimum</div>
        <div class="mc-val">{cuaca['tmax']}°C</div>
        <div class="mc-sub">Min: {cuaca['tmin']}°C &nbsp;|&nbsp; Ideal padi: 24–30°C</div>
      </div>
      <div class="metric-card {kls_kel}">
        <div class="mc-label">{ico('droplet')} Kelembapan Udara</div>
        <div class="mc-val">{kel_int}%</div>
        <div class="mc-sub">{'Waspadai serangan wereng' if kel_int>85 else 'Dalam batas normal'}</div>
      </div>
      <div class="metric-card {kls_ch}">
        <div class="mc-label">{ico('cloud-rain')} Curah Hujan</div>
        <div class="mc-val">{cuaca['curah_hujan']} mm</div>
        <div class="mc-sub">per jam &nbsp;|&nbsp; {'Terlalu lebat' if cuaca['curah_hujan']>30 else 'Normal'}</div>
      </div>
      <div class="metric-card {enso['kelas']}">
        <div class="mc-label">{ico(enso['ikon'])} Kondisi Iklim</div>
        <div class="mc-val" style="font-size:1rem;">{enso['level']}</div>
        <div class="mc-sub">ONI: {oni:+.1f}°C &nbsp;|&nbsp; {enso['fase']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("lightbulb")} Yang Harus Dilakukan</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="two-col">
      <div class="rekom {warna_rk}">
        <div class="rk-title">{ico('sprout')} Saran Tanam</div>
        <div class="rk-text">{ico(ikon_rk)} {teks_rk}</div>
      </div>
      <div class="rekom biru">
        <div class="rk-title">{ico('map')} Kondisi Iklim — {enso['fase']}</div>
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
      <div class="rk-title">{ico(musim_ikon)} {msmn}</div>
      <div class="rk-text">{musim_tips}</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  TAB 2 — CUACA & IKLIM
# ════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<p class="sec-title">{ico("cloud-sun")} Cuaca Detail — {provinsi}</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suhu Maks",   f"{cuaca['tmax']} °C",
              delta=f"{round(cuaca['tmax']-30,1)} dari ideal 30°C", delta_color="inverse")
    c2.metric("Suhu Min",    f"{cuaca['tmin']} °C")
    c3.metric("Kelembapan",  f"{int(cuaca['kelembapan'])} %")
    c4.metric("Curah Hujan", f"{cuaca['curah_hujan']} mm/hr")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("wave")} Kondisi ENSO — NOAA CPC Live</p>', unsafe_allow_html=True)

    # ── Fetch live NOAA ONI ──────────────────────────────────────
    df_oni_live, oni_ok, oni_time = fetch_oni_noaa()
    df_for_enso = (df_oni_live
                   if df_oni_live is not None and len(df_oni_live) >= 6
                   else df_enso)
    src_txt = (f"NOAA CPC · Diperbarui: {oni_time} · {len(df_for_enso)} data poin"
               if oni_ok
               else "Data lokal (NOAA tidak tersedia saat ini)")
    st.caption(src_txt)

    # ── Current & predicted ENSO ─────────────────────────────────
    oni_cur   = float(df_for_enso.iloc[-1]["ONI"])
    date_cur  = pd.Timestamp(df_for_enso.iloc[-1]["date"])
    enso_cur  = info_enso(oni_cur)
    pred      = prediksi_oni_ar(df_for_enso, p=3)
    enso_pred = info_enso(pred["oni"])

    st.markdown(f"""
    <div class="two-col">
      <div class="big-card {enso_cur['kelas']}">
        <div class="bc-label">{ico('wave')} ENSO Terkini — {date_cur.strftime('%B %Y')}</div>
        <div class="bc-row">
          <span class="bc-icon">{ico(enso_cur['ikon'], '2.2rem')}</span>
          <div>
            <span class="bc-value" style="font-size:1.15rem">{enso_cur['fase']}</span>
            <div class="bc-sub">ONI: <b>{oni_cur:+.2f}°C</b></div>
          </div>
        </div>
        <div class="bc-sub" style="margin-top:8px">{enso_cur['pesan']}</div>
      </div>
      <div class="big-card {enso_pred['kelas']}">
        <div class="bc-label">{ico('chart')} Prediksi — {pred['date'].strftime('%B %Y')} (bulan depan)</div>
        <div class="bc-row">
          <span class="bc-icon">{ico(enso_pred['ikon'], '2.2rem')}</span>
          <div>
            <span class="bc-value" style="font-size:1.15rem">{enso_pred['fase']}</span>
            <div class="bc-sub">ONI ≈ <b>{pred['oni']:+.2f}°C</b>
              &nbsp;({pred['oni_low']:+.1f} s/d {pred['oni_high']:+.1f})</div>
          </div>
        </div>
        <div class="bc-sub" style="margin-top:8px">
          Tren: <b>{pred['tren']}</b>. {enso_pred['pesan']}
        </div>
      </div>
    </div>
    <p style="font-size:0.7rem;color:#64748b;margin:-2px 0 12px">
      {ico('alert-triangle')} Prediksi menggunakan model AR(3) yang dilatih dari data historis NOAA 75 tahun — indikasi statistik, bukan model iklim resmi.
      Rujukan resmi: <b>bmkg.go.id</b> · <b>iri.columbia.edu</b>
    </p>
    """, unsafe_allow_html=True)

    st.caption("El Niño (ONI ≥ +0.5) · La Niña (ONI ≤ −0.5) · Normal · ◆ Prediksi bulan depan")
    st.plotly_chart(fig_oni_with_forecast(df_for_enso, pred), use_container_width=True)

    # ── Farming impact section ───────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("sprout")} Implikasi untuk Pertanian Padi Indonesia</p>',
                unsafe_allow_html=True)

    _DAMPAK = {
        "El Niño Kuat": [
            ("merah", "sun", "Kekeringan Ekstrem",
             "Produksi padi di Jawa, NTB, dan Sulawesi bisa turun 20–40%. "
             "Hentikan penanaman varietas sensitif kering. Prioritaskan lahan dekat irigasi teknis."),
            ("merah", "droplet", "Krisis Air Irigasi",
             "Terapkan AWD (Alternate Wetting & Drying): basah 2 hari → kering 3 hari → basah lagi. "
             "Tutup permukaan tanah dengan jerami untuk kurangi evaporasi."),
            ("merah", "bug", "Hama Tikus & Wereng",
             "Populasi tikus melonjak saat kemarau panjang. Koordinasi gropyokan bersama petani sekitar. "
             "Pasang bubu perangkap dan karet di pematang sawah."),
        ],
        "El Niño": [
            ("kuning", "droplet", "Hemat Air Irigasi",
             "Terapkan irigasi berselang. Tanam varietas toleran kering: "
             "Inpari 32, Inpago, atau Inpara 3. Jadwal tanam lebih awal dari biasanya."),
            ("kuning", "calendar", "Sesuaikan Kalender Tanam",
             "Maju jadwal tanam agar panen selesai sebelum puncak kemarau. "
             "Koordinasikan dengan kelompok tani dan penyuluh PPL."),
        ],
        "Normal": [
            ("", "check-circle", "Kondisi Iklim Ideal",
             "Ikuti kalender tanam yang sudah ditetapkan BPP. "
             "Waktu tepat untuk pemupukan NPK dan pengamatan rutin OPT."),
        ],
        "La Niña": [
            ("kuning", "cloud-rain", "Pantau Drainase",
             "Bersihkan saluran got sebelum hujan puncak. Perkuat pematang sawah untuk cegah jebol. "
             "Hindari genangan lebih dari 3 hari pada fase anakan."),
            ("kuning", "leaf", "Waspadai Penyakit Jamur",
             "Risiko blas dan busuk pelepah meningkat. Aplikasi fungisida profilaksis "
             "(mankozeb/validamycin) dianjurkan pada fase anakan aktif."),
        ],
        "La Niña Kuat": [
            ("merah", "storm", "Risiko Banjir Tinggi",
             "Di Kalimantan, Sumatera, dan pantai utara Jawa, bersiap untuk genangan panjang. "
             "Pindah ke varietas tahan banjir: Inpara 3, Ciherang Sub-1."),
            ("merah", "leaf", "Ledakan Penyakit",
             "Blas daun dan tungro melonjak drastis. Kendalikan wereng hijau (vektor tungro) "
             "dengan imidakloprid. Cabut dan bakar tanaman terinfeksi segera."),
            ("merah", "bug", "Wereng Cokelat Meledak",
             "Populasi wereng meledak di kondisi lembab berkepanjangan. "
             "Gunakan varietas tahan wereng (VUB BPT) dan hindari pemupukan N berlebih."),
        ],
    }

    fase_cur  = enso_cur["fase"]
    items_cur = _DAMPAK.get(fase_cur, _DAMPAK["Normal"])
    for warna, ikon_key, judul, teks in items_cur:
        st.markdown(f"""
        <div class="rekom {warna}">
          <div class="rk-title">{ico(ikon_key)} {judul}</div>
          <div class="rk-text">{teks}</div>
        </div>
        """, unsafe_allow_html=True)

    # Peringatan transisi jika prediksi berbeda dari kondisi sekarang
    if enso_pred["fase"] != enso_cur["fase"]:
        st.markdown(f"""
        <div class="rekom biru" style="margin-top:10px">
          <div class="rk-title">{ico('satellite')} Potensi Transisi ENSO Bulan Depan</div>
          <div class="rk-text">
            Tren 6 bulan terakhir menunjukkan kemungkinan pergeseran dari
            <b>{enso_cur['fase']}</b> → <b>{enso_pred['fase']}</b>
            pada {pred['date'].strftime('%B %Y')}.
            Mulai persiapkan langkah antisipasi sekarang.
          </div>
        </div>
        """, unsafe_allow_html=True)

    if df_ndvi is not None:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f'<p class="sec-title">{ico("leaf")} Kondisi Tanaman — NDVI Sentinel-2</p>', unsafe_allow_html=True)
        st.caption("Nilai 0.6–0.8 = tanaman sehat dan subur")
        st.plotly_chart(fig_ndvi(df_ndvi), use_container_width=True)

    if df_sar is not None:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f'<p class="sec-title">{ico("satellite")} Radar Satelit — SAR Sentinel-1 (VH)</p>', unsafe_allow_html=True)
        st.caption("Deteksi kelembapan tanah dan genangan air di lahan sawah")
        st.plotly_chart(fig_sar(df_sar), use_container_width=True)


# ════════════════════════════════════════════════════
#  TAB 3 — CEK HAMA
# ════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<p class="sec-title">{ico("microscope")} Cek Hama Padi — Foto Tanaman Anda</p>',
                unsafe_allow_html=True)

    if model_cnn is None:
        st.warning(f"Model belum dimuat. Path: `{MODEL_PATH}` | Ada: `{MODEL_PATH.exists()}`")

    st.markdown(f"""
    <div class="rekom biru">
      <div class="rk-title">{ico('camera')} Cara Menggunakan</div>
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
            st.markdown(f'<p class="sec-title">{ico("bot")} Hasil Deteksi</p>', unsafe_allow_html=True)

            if model_cnn is None:
                st.error("Model belum tersedia.")
            else:
                with st.spinner("Menganalisis foto..."):
                    hasil = prediksi_gambar(foto.getvalue())
                    if hasil:
                        st.session_state["ai_context"] = hasil

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
                      <div class="rk-title">{ico('pill')} Cara Penanganan</div>
                      <div class="rk-text">{hasil['rekomendasi']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # TaniBot rekomendasi penanganan
                    if hasil["kelas"] != "healthy":
                        opt_id = CLASS_TO_OPT.get(hasil["kelas"])
                        if opt_id and tanibot:
                            st.markdown('<hr class="divider">', unsafe_allow_html=True)
                            st.markdown(f'<p class="sec-title">{ico("bot")} TaniBot — Konsultasi Penanganan</p>',
                                        unsafe_allow_html=True)
                            fase = st.selectbox(
                                "Fase tanaman saat ini:",
                                ["pertanaman", "pesemaian", "pratanam"],
                                format_func=lambda x: {
                                    "pertanaman": "Pertanaman (sudah tanam)",
                                    "pesemaian":  "Pesemaian",
                                    "pratanam":   "Pratanam (belum tanam)",
                                }[x],
                                key="fase_sel"
                            )

                            chat_key = f"tbot_chat_{opt_id}_{fase}"
                            if chat_key not in st.session_state:
                                recs_resp = tanibot.get_post_detection_recs(opt_id, fase)
                                nav_opts  = [(n, l) for n, l in recs_resp.options if n != "root"]
                                # Kalau tidak ada nav_opts, jadikan tiap rec sebagai pilihan
                                if not nav_opts:
                                    nav_opts = [
                                        (f"_rec_{i}", f"#{rec['prioritas']} {rec['langkah']}")
                                        for i, rec in enumerate(recs_resp.recs)
                                    ]
                                greeting = (
                                    f"Halo! Saya <b>TaniBot</b>.<br>"
                                    f"Terdeteksi <b>{hasil['nama_indo']}</b> "
                                    f"pada fase <b>{fase}</b>.<br><br>"
                                    f"Pilih topik yang ingin kamu pelajari:"
                                )
                                st.session_state[chat_key] = {
                                    "history": [{"role": "bot", "text": greeting}],
                                    "opts":    nav_opts,
                                    "recs":    recs_resp.recs,
                                    "sumber":  recs_resp.sumber,
                                }

                            state = st.session_state[chat_key]

                            # Render chat bubbles
                            msgs_html = '<div class="tbot-chat">'
                            for msg in state["history"]:
                                if msg["role"] == "bot":
                                    msgs_html += (f'<div class="tbot-row bot">'
                                                  f'<span class="tbot-av">{ico("bot")}</span>'
                                                  f'<div class="tbot-bbl bot">{msg["text"]}</div>'
                                                  f'</div>')
                                else:
                                    msgs_html += (f'<div class="tbot-row user">'
                                                  f'<div class="tbot-bbl user">{msg["text"]}</div>'
                                                  f'<span class="tbot-av">{ico("user")}</span>'
                                                  f'</div>')
                            msgs_html += '</div>'
                            st.markdown(msgs_html, unsafe_allow_html=True)

                            # Tombol pilihan
                            if state["opts"]:
                                st.markdown('<p class="tbot-hint">Pilih topik:</p>',
                                            unsafe_allow_html=True)
                                for nid, lbl in state["opts"]:
                                    if st.button(lbl,
                                                 key=f"tbot_{nid}_{len(state['history'])}_{opt_id}",
                                                 use_container_width=True):
                                        state["history"].append({"role": "user", "text": lbl})
                                        if nid.startswith("_rec_"):
                                            # Tampilkan detail rec yang dipilih
                                            rec = state["recs"][int(nid.split("_")[2])]
                                            content = (f"<b>[{rec['metode']}]</b> "
                                                       f"{rec['detail']}")
                                            if rec.get("dosis"):
                                                content += f"<br>{ico('pill')} <b>Dosis:</b> {rec['dosis']}"
                                            if rec.get("waktu_aplikasi"):
                                                content += f"<br>{ico('clock')} <b>Waktu:</b> {rec['waktu_aplikasi']}"
                                            if state.get("sumber"):
                                                content += (
                                                    f"<br><br><span style='font-size:0.7rem;"
                                                    f"color:#64748b'>{ico('book')} "
                                                    f"{'; '.join(set(state['sumber']))}</span>"
                                                )
                                            state["history"].append({"role": "bot", "text": content})
                                            # Sisa rec yang belum dilihat
                                            idx = int(nid.split("_")[2])
                                            state["opts"] = [
                                                (f"_rec_{i}", f"#{r['prioritas']} {r['langkah']}")
                                                for i, r in enumerate(state["recs"]) if i != idx
                                            ]
                                        else:
                                            resp = tanibot.select(nid)
                                            state["history"].append({
                                                "role": "bot",
                                                "text": resp.text.replace("\n", "<br>"),
                                            })
                                            state["opts"] = [(n, l) for n, l in resp.options
                                                             if n != "root"]
                                        st.rerun()
                            else:
                                st.markdown(f'<p class="tbot-hint">{ico("check-circle")} Konsultasi selesai</p>',
                                            unsafe_allow_html=True)
                            if st.button("Mulai ulang",
                                         key=f"tbot_reset_{opt_id}",
                                         use_container_width=True):
                                del st.session_state[chat_key]
                                st.rerun()
                        elif tanibot is None:
                            st.info("TaniBot belum aktif — bangun database dulu dengan `tanibot_db_builder_v2.py`")

                    # Bar probabilitas semua kelas
                    st.markdown(f'<p class="sec-title" style="margin-top:16px;">{ico("chart")} Probabilitas Semua Kelas</p>',
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
        st.markdown(f'<p class="sec-title">{ico("bug")} Kenali Penyakit Padi yang Umum</p>',
                    unsafe_allow_html=True)

        HAMA = [
            ("leaf","Hawar Daun Bakteri (HDB)","Daun mengering dari tepi, seperti terbakar. Eksudat kuning di permukaan daun."),
            ("leaf","Bercak Cokelat",           "Bercak oval cokelat gelap di daun, sering di varietas kekurangan nutrisi."),
            ("check-circle","Sehat",            "Daun hijau segar, tidak ada bercak, tanaman tegak dan subur."),
            ("leaf","Blas Daun",                "Bercak belah ketupat abu-abu dengan tepi cokelat, daun bisa mati."),
            ("leaf","Hawar Pelepah",            "Bercak cokelat terang di pelepah daun bawah, berkembang ke atas."),
            ("leaf","Bercak Cokelat Sempit",    "Garis-garis sempit cokelat sejajar tulang daun."),
            ("bug","Hispa Padi",                "Permukaan daun bergaris putih/transparan, ada bekas goresan larva."),
            ("leaf","Busuk Pelepah",            "Bercak oval abu-abu di pelepah, meluas dan batang bisa roboh."),
            ("bug","Tungro",                    "Daun kuning-oranye, tanaman kerdil, anakan berkurang."),
        ]
        st.markdown('<div class="hama-grid">', unsafe_allow_html=True)
        for ikon, nama, ciri in HAMA:
            st.markdown(f"""
            <div class="hama-card">
              <span class="hama-icon">{ico(ikon, '1.6rem')}</span>
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
    st.markdown(f'<p class="sec-title">{ico("info-circle")} Tentang HarvestAlert</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rekom biru">
      <div class="rk-title">{ico('book')} Tentang Aplikasi</div>
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
    st.markdown(f'<p class="sec-title">{ico("chart")} Status Sistem</p>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Data BMKG",  "Live"  if cuaca["ok"]       else "Estimasi")
    s2.metric("Data ENSO",  f"{len(df_enso)} data")
    s3.metric("Data NDVI",  "Aktif" if df_ndvi is not None else "Tidak ada")
    s4.metric("Model CNN",  "Siap"  if model_cnn is not None else "Tidak ada")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("alert-triangle")} Panduan Warna Peringatan</p>', unsafe_allow_html=True)

    for ikon_key, warna, kls, arti, aksi in [
        ("check-circle", "Hijau",  "",       "Kondisi aman",  "Lanjutkan aktivitas normal"),
        ("alert-circle", "Kuning", "kuning", "Perlu waspada", "Pantau lahan lebih sering"),
        ("alert-triangle", "Merah", "merah", "Berbahaya",     "Ambil tindakan, hubungi penyuluh"),
    ]:
        st.markdown(f"""
        <div class="rekom {kls}">
          <div class="rk-title">{ico(ikon_key)} {warna}</div>
          <div class="rk-text"><b>{arti}</b> — {aksi}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  TAB 5 — CHAT AI (DEEPSEEK)
# ════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<p class="sec-title">{ico("bot")} Chat dengan AgriWarn AI</p>', unsafe_allow_html=True)

    ds_client = get_ai_client()

    if ds_client is None:
        st.markdown(f"""
        <div class="rekom kuning">
          <div class="rk-title">{ico('alert-triangle')} API Key Belum Dikonfigurasi</div>
          <div class="rk-text">
            Untuk mengaktifkan Chat AI, daftar gratis di <b>console.groq.com</b>
            → buat API key → tambahkan di Streamlit Cloud:<br>
            <b>App Settings → Secrets → Edit Secrets</b><br><br>
            <code>GROQ_API_KEY = "gsk_xxxxxxxxxxxx"</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Konteks dari hasil deteksi Tab 3
        ai_ctx = st.session_state.get("ai_context")

        if ai_ctx:
            nama_ctx  = ai_ctx.get("nama_indo", "")
            conf_ctx  = round(ai_ctx.get("confidence", 0) * 100, 1)
            kls_ctx   = ai_ctx.get("kelas", "")
            ctx_color = "merah" if ai_ctx.get("berbahaya") else "kuning"
            if kls_ctx == "healthy":
                ctx_color = ""
            st.markdown(f"""
            <div class="rekom biru">
              <div class="rk-title">{ico('camera')} Konteks Deteksi Aktif</div>
              <div class="rk-text">AI mengetahui hasil foto kamu: <b>{nama_ctx}</b>
              ({conf_ctx}% keyakinan). Kamu bisa langsung bertanya soal penanganan,
              gejala, dosis pestisida, atau pencegahan.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rekom">
              <div class="rk-title">{ico('lightbulb')} Tips</div>
              <div class="rk-text">Upload foto tanaman di tab <b>Cek Hama Padi</b> dulu,
              lalu kembali ke sini — AI akan otomatis tahu hasil deteksinya dan bisa
              memberi jawaban yang lebih spesifik.</div>
            </div>
            """, unsafe_allow_html=True)

        # Susun system prompt
        sys_prompt = (
            "Kamu adalah AgriWarn AI, asisten pertanian padi Indonesia yang ahli dan ramah. "
            "Bantu petani dengan pertanyaan seputar penyakit padi, hama, cara pemupukan, "
            "irigasi, dan pengendalian OPT (Organisme Pengganggu Tumbuhan). "
            "Jawab dalam Bahasa Indonesia yang mudah dipahami petani biasa. "
            "Berikan saran praktis dan berbasis ilmu pertanian yang valid."
        )
        if ai_ctx and ai_ctx.get("kelas") != "healthy":
            sys_prompt += (
                f"\n\nKonteks aktif: tanaman padi pengguna terdeteksi mengidap "
                f"{ai_ctx['nama_indo']} dengan keyakinan {round(ai_ctx['confidence']*100,1)}%. "
                f"Rekomendasi sistem: {ai_ctx.get('rekomendasi', '')}. "
                "Gunakan konteks ini untuk jawaban yang lebih relevan."
            )

        # Inisialisasi riwayat chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Tampilkan riwayat
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Input pengguna
        if prompt := st.chat_input("Tanya soal penyakit padi, penanganan, dosis pestisida..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                try:
                    messages = [{"role": "system", "content": sys_prompt}] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ]
                    stream = ds_client.chat.completions.create(
                        model=AI_MODEL,
                        messages=messages,
                        stream=True,
                    )
                    full_reply = st.write_stream(stream)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_reply,
                    })
                except Exception as e:
                    st.error(f"Gagal terhubung ke AI: {e}")

        if st.session_state.get("chat_history"):
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            if st.button("Hapus riwayat percakapan", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
