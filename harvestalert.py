import base64
import csv
import math
import os
import uuid
import requests
import streamlit as st
import streamlit.components.v1 as _stcv1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from pathlib import Path
from streamlit_geolocation import streamlit_geolocation
import io
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("agrialert.cnn")

# ----------------------------------------------------------------
#  KONFIGURASI
# ----------------------------------------------------------------
DATA_DIR      = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent)))
ASSET_DIR     = Path(__file__).parent   # mascot.png/gps.png/maps.png/sun.png ada di root repo, selalu di sebelah file ini — tidak ikut DATA_DIR yang bisa dioverride env var
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
}

NAMA_KELAS = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast',
    'leaf_scald', 'narrow_brown_spot'
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
        'Buang air genangan (drainase) agar lahan tidak lembap. '
        'Hentikan dulu pemberian pupuk Urea agar daun tidak terlalu lunak. '
        'Semprot bakterisida berbahan aktif Tembaga Hidroksida sesuai dosis di kemasan.'
    ),
    'brown_spot': (
        'Penyakit ini tanda tanah kurang subur. '
        'Berikan pupuk KCL untuk memperkuat batang tanaman. '
        'Semprot fungisida berbahan aktif Mankozeb jika bercak mulai menyebar.'
    ),
    'healthy': (
        'Kabar baik, tanaman padi Anda sehat!'
        'Lanjutkan perawatan rutin, jaga air agar teratur, dan penuhi kebutuhan pupuk sesuai jadwal. '
        'Tetap pantau kondisi tanaman secara berkala.'
    ),
    'leaf_blast': (
        'Tunda pemberian pupuk Urea agar jamur tidak makin cepat menyebar. '
        'Atur jarak tanam agak renggang agar angin masuk. '
        'Semprot fungisida berbahan aktif Trisiklazol (Tricyclazole) saat pagi/sore cerah.'
    ),
    'leaf_scald': (
        'Pastikan air sawah mengalir dan tidak menggenang terlalu tinggi. '
        'Bersihkan rumput liar agar udara lancar. '
        'Gunakan fungisida berbahan aktif Propikonazol untuk mencegah penularan.'
    ),
    'narrow_brown_spot': (
        'Berikan pemupukan NPK berimbang. '
        'Tanaman yang cukup gizi lebih tahan penyakit. '
        'Jika serangan sudah merata, semprot dengan fungisida yang umum tersedia di toko tani.'
    ),
    'rice_hispa': (
        'Potong ujung daun yang rusak lalu buang jauh dari sawah. '
        'Jika hama terlihat banyak, semprot insektisida berbahan aktif Karbofuran atau Buprofezin.'
    ),
    'sheath_blight': (
        'Kurangi air genangan agar pangkal batang tidak selalu basah. '
        'Jaga kebersihan lahan dari rumput. '
        'Semprotkan fungisida berbahan aktif Azoksistrobin atau Validamisin ke arah pelepah bawah.'
    ),
    'tungro': (
        'Penyakit ini disebabkan virus (bawaan wereng hijau). Tidak ada obat penyembuhnya. '
        'Segera cabut dan musnahkan tanaman kerdil agar tidak menular. '
        'Semprot insektisida berbahan aktif Imidakloprid untuk membasmi wereng hijau.'
    ),
}

BERBAHAYA = {'bacterial_leaf_blight', 'leaf_blast'}

AI_MODEL = "llama-3.3-70b-versatile"

# NOAA CPC ONI — gratis, tanpa API key, update bulanan
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
_SEAS_MONTH  = {
    "DJF":2, "JFM":3, "FMA":4, "MAM":5, "AMJ":6, "MJJ":7,
    "JJA":8, "JAS":9, "ASO":10, "SON":11, "OND":12, "NDJ":1,
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
    "wifi":          '<path d="M4 11.5a11.5 11.5 0 0 1 16 0"/><path d="M7.2 14.8a7 7 0 0 1 9.6 0"/><path d="M12 19.5h.01"/>',
    "wifi-off":      '<path d="M2 2l20 20"/><path d="M4 11.5a11.5 11.5 0 0 1 12.4-2.5"/><path d="M20 9a11.5 11.5 0 0 0-2-1.6"/><path d="M7.2 14.8a7 7 0 0 1 6-1.9"/><path d="M12 19.5h.01"/>',
    "flag":          '<path d="M5 21V4"/><path d="M5 4h13l-3 4 3 4H5"/>',
    "shield-check":  '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6Z"/><path d="m8.5 12 2.5 2.5L15.5 9"/>',
    "image":         '<rect x="3" y="3" width="18" height="18" rx="2.5"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m3 16 5-5 4 4 3-3 6 6"/>',
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
    page_title="AgriAlert",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

:root {
    color-scheme: light;
    /* ── Palet utama — sesuai desain Figma AgriAlert ── */
    --c-primary: #20965F;          /* hijau utama — header, tombol, shape gelap */
    --c-primary-dark: #20965F;
    --c-primary-darker: #176e47;
    --c-primary-light: #EEFFD3;    /* hijau muda — base card terang */
    --c-sage: #6f8f78;

    --c-tab-active-text: #EEFFD3;  /* font tab yang sedang aktif */
    --c-tab-inactive-text: #20965F;/* font tab yang tidak aktif */

    --c-bg: #EDEDED;               /* base warna web paling dasar */
    --c-surface: #ffffff;
    --c-border: #dfe6e0;
    --c-text: #1c2620;
    --c-text-muted: #57685d;
    --c-text-on-light: #20965F;
    --c-muted-on-surface: #57685d;
    --c-accent-text: #20965F;

    /* ── Status (kondisi cuaca / iklim) — hijau=aman, kuning=waspada, merah=bahaya ── */
    --c-aman-bg: #EEFFD3;     --c-aman-border:#20965F;   --c-aman-text:#20965F;   --c-aman-dot:#00EA7D;
    --c-waspada-bg:#EEFFD3;   --c-waspada-border:#20965F; --c-waspada-text:#20965F; --c-waspada-dot:#F3FF0F;
    --c-bahaya-bg:#EEFFD3;    --c-bahaya-border:#20965F; --c-bahaya-text:#20965F;  --c-bahaya-dot:#FF0505;
    --c-info-bg:#EEFFD3;      --c-info-border:#20965F;   --c-info-text:#20965F;

    --radius-sm: 8px;
    --radius-md: 15px;
    --radius-lg: 40px;
    --shadow-sm: 0 1px 3px rgba(32,150,95,0.08);
    --shadow-md: 0 4px 14px rgba(32,150,95,0.12);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
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
/* ── HEADER — mask group sesuai Figma ──
   Dua Rectangle hijau (#20965F, radius 40px) bertumpuk sebagai latar,
   mascot.png di-mask/diposisikan di kanan-atas, judul+sub di kiri. */
.ha-header {
    position: relative;
    width: 100%;
    min-height: 260px;
    border-radius: var(--radius-lg);
    background: #20965F;          /* Rectangle 29 / Rectangle 7 */
    overflow: hidden;
    margin-bottom: 32px;
    box-shadow: var(--shadow-md);
    padding: 24px 28px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}
.ha-mascot {
    position: absolute;
    width: 185px;
    height: 245px;
    right: 40px;
    bottom: 5px;
    background-image: var(--mascot-url);
    background-size: contain;
    background-repeat: no-repeat;
    background-position: bottom center;
    transform: rotate(-0.66deg);
    pointer-events: none;
    z-index: 1;
}
.ha-header-left { display: flex; align-items: center; gap: 16px; position: relative; z-index: 2; }
.ha-logo  { font-size: 2.6rem; line-height: 1; color: #ffffff; }
.ha-title { font-size: 5rem !important; font-weight: 800 !important; color: #ffffff !important; margin: 0 !important; letter-spacing: -1px !important; line-height: 1.05 !important; }
.ha-sub   { font-size: 0.95rem; color: #EEFFD3; margin: 4px 0 0; }
.ha-badge {
    display: inline-flex; align-items: center; gap: 6px;
    margin-top: 8px;
    background: rgba(255,255,255,0.14); border: 1.5px solid rgba(255,255,255,0.3); border-radius: 30px;
    padding: 6px 14px; font-size: 0.82rem; font-weight: 700; color: #ffffff; white-space: nowrap;
}
.gps-box {
    background: var(--c-surface); border: 1.5px solid var(--c-border); border-radius: var(--radius-md);
    padding: 14px 18px; margin-bottom: 16px; font-size: 0.92rem; color: var(--c-muted-on-surface);
    line-height: 1.9; display: flex; flex-wrap: wrap; gap: 6px 22px; box-shadow: var(--shadow-sm);
}
.gps-box b { color: var(--c-accent-text); }

/* ── Tombol GPS — Rectangle 16: 376x65, bg 20965F, radius 15 ──
   Catatan: streamlit_geolocation merender iframe pihak ketiga, jadi kita
   styling kontainernya saja (bentuk, warna dasar, radius); teks/tombol di
   dalam iframe mengikuti style bawaan komponen tersebut. */
.gps-trigger-wrap {
    margin-bottom: 14px;
    background: #20965F;
    border-radius: 15px;
    min-height: 65px;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}
.gps-trigger-wrap.aktif {
    background: #ffffff;
    border: 1.5px solid #20965F;
}
.gps-trigger-wrap iframe { width: 100%; }

/* ── Kotak instruksi maps — Rectangle 7: 376x118, bg EEFFD3, radius 15 ── */
.gps-info-row {
    display: flex; align-items: center; gap: 14px;
    background: #EEFFD3; border-radius: 15px; border: none;
    padding: 16px 20px; margin-bottom: 16px; box-shadow: var(--shadow-sm);
}
.gps-info-row.below-gps-btn { margin-top: -35px; }
.gps-info-icon { width: 32px; height: 32px; flex-shrink: 0; color: #20965F; }
.gps-info-icon img { width: 100%; height: 100%; object-fit: contain; }
.gps-info-text { font-family: 'Poppins', sans-serif; font-weight: 400; font-size: 12px; line-height: 18px; color: #20965F; }

/* ── Big-card "Kondisi Cuaca Hari Ini" — base EEFFD3, font 20965F ── */
.big-card { border-radius: var(--radius-lg); padding: 22px 26px; margin-bottom: 16px; border: none; box-shadow: var(--shadow-sm); background: #EEFFD3; }
.big-card.aman, .big-card.waspada, .big-card.bahaya, .big-card.info { background: #EEFFD3; border: none; }
.bc-label { font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #20965F; margin-bottom: 10px; }
.bc-row   { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.bc-icon  { font-size: 2.2rem; line-height: 1; flex-shrink: 0; color: #20965F; }
.bc-value { font-size: 1.65rem; font-weight: 900; color: #20965F; line-height: 1.15; }
.bc-sub   { font-size: 0.92rem; color: #20965F; margin-top: 6px; line-height: 1.6; }
.big-card .bc-value, .big-card .bc-label, .big-card .bc-sub { color: #20965F; }
/* Titik status — hijau aman / kuning waspada / merah bahaya */
.bc-dot { display:inline-block; width:24px; height:24px; border-radius:50%; margin-right:8px; vertical-align:middle; flex-shrink:0; }
.bc-dot.aman    { background:#00C96B; }
.bc-dot.waspada { background:#EAB308; }
.bc-dot.bahaya  { background:#FF0505; }

/* ── Metric cards (suhu/kelembapan/curah hujan/iklim) — selalu hijau 20965F, font putih ── */
.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 16px;
}
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .metric-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; } }
.metric-card { border-radius: var(--radius-md); padding: 18px 20px; border: none; background: #20965F; box-shadow: var(--shadow-sm); min-width: 0; }
.metric-card.aman, .metric-card.waspada, .metric-card.bahaya, .metric-card.info { background: #20965F; border: none; }
.mc-label { font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #ffffff; margin-bottom: 8px; }
.mc-val   { font-size: 1.5rem; font-weight: 800; color: #ffffff; line-height: 1.15; }
.mc-sub   { font-size: 0.8rem; color: #ffffff; margin-top: 6px; opacity: 0.9; }
.metric-card .mc-val, .metric-card .mc-label, .metric-card .mc-sub { color: #ffffff; }
@media (max-width: 480px) { .metric-card { padding: 14px 16px; } .mc-val { font-size: 1.3rem; } }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }

/* ── Rekom (Saran Tanam, Kondisi Iklim, dll) — base EEFFD3, font 20965F, tanpa border-left ── */
.rekom { background: #EEFFD3; border-radius: var(--radius-md); border: none; padding: 18px 20px; margin-bottom: 16px; box-shadow: var(--shadow-sm); }
.rekom.kuning, .rekom.merah, .rekom.biru { background: #EEFFD3; border: none; }
.rk-title { font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #20965F; margin-bottom: 10px; }
.rekom .rk-title, .rekom.kuning .rk-title, .rekom.merah .rk-title, .rekom.biru .rk-title { color: #20965F; }
.rk-text { font-size: 0.98rem; color: #20965F; line-height: 1.7; }
.hama-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
@media (max-width: 500px) { .hama-grid { grid-template-columns: 1fr; } }
.hama-card { background: #EEFFD3; border: none; border-radius: var(--radius-md); padding: 16px 18px; display: flex; align-items: flex-start; gap: 14px; box-shadow: var(--shadow-sm); }
.hama-icon { font-size: 1.8rem; line-height: 1; flex-shrink: 0; margin-top: 2px; color: #20965F; }
.hama-nama { font-size: 0.98rem; font-weight: 700; color: #20965F; margin-bottom: 4px; }
.hama-ciri { font-size: 0.85rem; color: #20965F; line-height: 1.55; }
.hama-card-v { background: #EEFFD3; border: none; border-radius: var(--radius-md); padding: 14px 16px 16px; display: flex; flex-direction: column; gap: 10px; box-shadow: var(--shadow-sm); }
.hama-card-v .hama-nama { display: flex; align-items: center; gap: 8px; margin-bottom: 0; color: #20965F; }
.hama-img-wrap { width: 100%; aspect-ratio: 4/3; border-radius: var(--radius-sm); overflow: hidden; border: none; }
.hama-img-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hama-img-ph { width: 100%; aspect-ratio: 4/3; border-radius: var(--radius-sm); border: 1.5px dashed #20965F; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; color: #20965F; text-align: center; padding: 10px; }
.hama-img-ph-path { font-family: monospace; font-size: 0.68rem; opacity: 0.75; word-break: break-all; color: #20965F; }
.sec-title { font-size: 1.15rem; font-weight: 700; color: #20965F; margin: 24px 0 14px; display: flex; align-items: center; gap: 10px; }
.divider { border: none; border-top: 1.5px solid #20965F; opacity: 0.25; margin: 20px 0; }
.status-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
.chip { border-radius: 30px; padding: 8px 16px; font-size: 0.85rem; font-weight: 700; display: inline-flex; align-items: center; gap: 6px; border: none; background: #EEFFD3; color: #20965F; }
.chip.aman, .chip.waspada, .chip.bahaya, .chip.info { background: #EEFFD3; border: none; color: #20965F; }
div[data-baseweb="tab-list"] { background: var(--c-surface) !important; border: 1.5px solid var(--c-border) !important; border-radius: var(--radius-md) !important; padding: 6px !important; gap: 4px !important; margin-bottom: 22px; box-shadow: var(--shadow-sm); }
button[data-baseweb="tab"] { font-family: 'Poppins', sans-serif !important; font-size: 0.98rem !important; font-weight: 700 !important; border-radius: var(--radius-sm) !important; padding: 12px 10px !important; color: #20965F !important; }
button[data-baseweb="tab"] p { color: #20965F !important; font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: #27B774 !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #EEFFD3 !important; }
[data-testid="stSidebar"] { background: var(--c-surface) !important; border-right: 1.5px solid var(--c-border) !important; }
/* ── Paksa teks sidebar tetap kebaca walau browser/OS pengunjung pakai dark mode ── */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label { color: var(--c-text) !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background: var(--c-primary-light) !important; border: 1.5px solid var(--c-border) !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] * { color: var(--c-text) !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] div[data-baseweb="select"] > div { background: var(--c-surface) !important; color: var(--c-text) !important; }
.stButton > button, div[data-testid="stFormSubmitButton"] button {
    font-family: 'Poppins', sans-serif !important;
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
@media (max-width: 600px) { .ha-mascot { width: 85px !important; height: 112px !important; right: 8px !important; bottom: 0 !important; } .ha-header { min-height: 150px !important; padding-right: 100px !important; } }
@media (max-width: 480px) { .ha-title { font-size: 2.4rem !important; } .bc-value { font-size: 1.35rem; } .sec-title { font-size: 1.02rem; } .rk-text { font-size: 0.92rem; } .gps-box { font-size: 0.85rem; } .bc-dot { width: 18px !important; height: 18px !important; } }
.tbot-chat {
    background: var(--c-bg); border: 1.5px solid var(--c-border); border-radius: var(--radius-md);
    padding: 16px 14px; max-height: 400px; overflow-y: auto;
    margin: 12px 0; display: flex; flex-direction: column; gap: 12px;
}
.tbot-row { display: flex; align-items: flex-end; gap: 10px; }
.tbot-row.bot  { flex-direction: row; }
.tbot-row.user { flex-direction: row-reverse; }
.tbot-av { font-size: 1.3rem; flex-shrink: 0; color: #20965F; }
.tbot-bbl { padding: 11px 15px; font-size: 0.92rem; line-height: 1.7; max-width: 84%; word-break: break-word; }
.tbot-bbl.bot  { background:var(--c-surface); border:1.5px solid var(--c-border); border-radius:4px 16px 16px 16px; color:var(--c-text); }
.tbot-bbl.user { background:var(--c-aman-bg); border:1.5px solid var(--c-aman-border); border-radius:16px 4px 16px 16px; color:var(--c-aman-text); }
.tbot-hint { font-size:0.82rem; color:var(--c-muted-on-surface); margin:8px 0 6px; }

/* ── Unggah foto (Cek Hama) — base EEFFD3, tombol 20965F font putih, tanpa stroke ── */
[data-testid="stFileUploader"] {
    background: #EEFFD3 !important; border: none !important; border-radius: var(--radius-md) !important;
    padding: 14px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #EEFFD3 !important; border: none !important; border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #20965F !important; color: #ffffff !important; border: none !important;
    border-radius: var(--radius-sm) !important; font-weight: 700 !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #20965F; }
[data-testid="stFileUploaderDropzone"] button * { color: #ffffff !important; }

/* ── Form & select widgets — selaraskan dengan palet, tanpa stroke tebal ── */
div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea, .stNumberInput input {
    border-color: #20965F !important; border-radius: var(--radius-sm) !important;
}
[data-testid="stForm"] { background: #EEFFD3; border: none; border-radius: var(--radius-md); padding: 18px 20px; }

/* ── DARK MODE OVERRIDE — paksa semua widget pakai warna light ── */
/* Input & textarea */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background-color: #ffffff !important;
    color: #1c2620 !important;
    caret-color: #1c2620 !important;
}
/* Select / dropdown */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div {
    background-color: #ffffff !important;
    color: #1c2620 !important;
}
div[data-baseweb="select"] svg { fill: #20965F !important; }
/* Popover dropdown options */
ul[role="listbox"] li,
div[data-baseweb="menu"] *,
div[data-baseweb="popover"] li {
    background-color: #ffffff !important;
    color: #1c2620 !important;
}
/* Widget labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] { color: #1c2620 !important; }
/* Checkbox label text */
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] p { color: #1c2620 !important; }
/* Chat input */
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #1c2620 !important;
}
/* st.info / st.warning / st.success inside dark mode */
[data-testid="stAlert"] p,
[data-testid="stAlert"] { color: #1c2620 !important; }
/* File uploader inner text */
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: #20965F !important; }
/* Probabilitas bar labels */
.prob-label span { color: #1c2620 !important; }

/* Caption text */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p { color: #57685d !important; }

/* Radio button labels */
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label div p,
[data-testid="stRadio"] span { color: #1c2620 !important; }

/* Chat input box */
[data-testid="stChatInput"] {
    background-color: #f0f7f4 !important;
    border: 1.5px solid #20965F !important;
    border-radius: 15px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #f0f7f4 !important;
    color: #1c2620 !important;
    caret-color: #1c2620 !important;
}
[data-testid="stChatInput"] button { background-color: #20965F !important; }
[data-testid="stChatInput"] button svg path { stroke: #ffffff !important; }

/* Chat messages */
[data-testid="stChatMessage"] { background: #ffffff !important; border: 1px solid #dfe6e0 !important; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessageContent"] * { color: #1c2620 !important; }

/* ── GPS button — container jadi tombol hijau, iframe di dalamnya jadi click target ── */
div:has(> [data-testid="stCustomComponentV1"]) {
    background-color: #20965F !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fff' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 22s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12Z'/%3E%3Ccircle cx='12' cy='10' r='2.3'/%3E%3C%2Fsvg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: 20px center !important;
    background-size: 26px 26px !important;
    border-radius: 15px !important;
    height: 65px !important;
    min-height: 65px !important;
    margin-bottom: 16px !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(32,150,95,0.18) !important;
    cursor: pointer !important;
}
div:has(> [data-testid="stCustomComponentV1"])::before {
    content: "KLIK UNTUK AKTIFKAN GPS";
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    align-items: center !important;
    padding: 18px 28px 18px 58px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.97rem !important;
    color: #ffffff !important;
    pointer-events: none !important;
    z-index: 1 !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stCustomComponentV1"] {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 65px !important;
    min-height: 65px !important;
    opacity: 0.001 !important;
    z-index: 2 !important;
    cursor: pointer !important;
    overflow: hidden !important;
    margin: 0 !important;
}
[data-testid="stCustomComponentV1"] iframe {
    display: block !important;
    width: 100% !important;
    height: 65px !important;
    min-height: 65px !important;
    cursor: pointer !important;
}
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
    """
    Load model CNN AgriWarn (ONNX).
    Input : (1, 224, 224, 3) float32, nilai piksel 0-255
            Rescaling 1/255 sudah ada DI DALAM graph — jangan normalisasi manual.
    Output: (1, 6) softmax probability → 6 kelas penyakit.
    """
    if not MODEL_PATH.exists():
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        _dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        sess.run(None, {sess.get_inputs()[0].name: _dummy})
        return sess
    except Exception as e:
        logger.error(f"Gagal load model ONNX: {e}")
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
#  PENGADUAN & CROWDSOURCING FOTO HAMA (consent-based)
# ================================================================
LAPORAN_CSV     = DATA_DIR / "laporan_pengaduan.csv"
CROWDSOURCE_DIR = DATA_DIR / "crowdsource_foto"
CROWDSOURCE_LOG = DATA_DIR / "crowdsource_log.csv"


def simpan_pengaduan(jenis, deskripsi, kontak, kelas_terdeteksi=""):
    baru = not LAPORAN_CSV.exists()
    with open(LAPORAN_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if baru:
            w.writerow(["waktu", "jenis_kesalahan", "kelas_terdeteksi", "deskripsi", "kontak"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), jenis, kelas_terdeteksi, deskripsi, kontak])


def simpan_crowdsource_foto(file_bytes, ext, jenis_hama, lokasi, catatan):
    folder = CROWDSOURCE_DIR / jenis_hama.split(" — ")[0].strip().replace(" ", "_").lower()
    folder.mkdir(parents=True, exist_ok=True)
    nama_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
    (folder / nama_file).write_bytes(file_bytes)

    baru = not CROWDSOURCE_LOG.exists()
    with open(CROWDSOURCE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if baru:
            w.writerow(["waktu", "jenis_hama", "lokasi", "catatan", "file"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"), jenis_hama, lokasi, catatan,
            str((folder / nama_file).relative_to(DATA_DIR)),
        ])
    return nama_file


# ================================================================
#  ASET UI (mascot, ikon gps/maps/sun) — dari folder assets/
# ================================================================
ASSET_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data
def ui_img_b64(filename):
    """Baca file dari assets/ lalu kembalikan sebagai data-URI base64.
    Kalau file belum ada (mis. belum diunggah ke repo), kembalikan string kosong
    supaya CSS/HTML yang memakainya tidak error — tinggal tampil tanpa gambar."""
    path = ASSET_DIR / filename
    if not path.exists():
        return ""
    ext  = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext == "jpg" else ext
    b64  = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{mime};base64,{b64}"


# ================================================================
#  ENSIKLOPEDIA HAMA & PENYAKIT (kartu judul-gambar-keterangan)
# ================================================================
ASSET_HAMA_DIR = DATA_DIR / "assets" / "hama"
ASSET_HAMA_DIR.mkdir(parents=True, exist_ok=True)


def gambar_hama_html(filename, alt=""):
    """<img> data-URI jika file foto ada di assets/hama/, kalau tidak tampilkan placeholder."""
    path = ASSET_HAMA_DIR / filename
    if path.exists():
        ext  = path.suffix.lstrip(".").lower()
        mime = "jpeg" if ext == "jpg" else ext
        b64  = base64.b64encode(path.read_bytes()).decode()
        return f'<div class="hama-img-wrap"><img src="data:image/{mime};base64,{b64}" alt="{alt}"/></div>'
    return (
        f'<div class="hama-img-ph">{ico("image", "1.6rem")}'
        f'<span>Foto referensi belum tersedia</span>'
        f'<span class="hama-img-ph-path">assets/hama/{filename}</span></div>'
    )


def render_hama_grid(items):
    st.markdown('<div class="hama-grid">', unsafe_allow_html=True)
    for img_file, ikon, nama, ciri in items:
        st.markdown(f"""
        <div class="hama-card-v">
          <div class="hama-nama">{ico(ikon, '1.3rem')} {nama}</div>
          {gambar_hama_html(img_file, nama)}
          <div class="hama-ciri">{ciri}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


HAMA = [
    ("hdb.jpg",                   "leaf",         "Hawar Daun Bakteri (HDB)", "Daun mengering dari tepi, seperti terbakar. Eksudat kuning di permukaan daun."),
    ("bercak_cokelat.jpg",        "leaf",         "Bercak Cokelat",           "Bercak oval cokelat gelap di daun, sering di varietas kekurangan nutrisi."),
    ("sehat.jpg",                 "check-circle", "Sehat",                    "Daun hijau segar, tidak ada bercak, tanaman tegak dan subur."),
    ("blas_daun.jpg",             "leaf",         "Blas Daun",                "Bercak belah ketupat abu-abu dengan tepi cokelat, daun bisa mati."),
    ("hawar_pelepah.jpg",         "leaf",         "Hawar Pelepah",            "Bercak cokelat terang di pelepah daun bawah, berkembang ke atas."),
    ("bercak_cokelat_sempit.jpg", "leaf",         "Bercak Cokelat Sempit",    "Garis-garis sempit cokelat sejajar tulang daun."),
    ("hispa.jpg",                 "bug",          "Hispa Padi",               "Permukaan daun bergaris putih/transparan, ada bekas goresan larva."),
    ("busuk_pelepah.jpg",         "leaf",         "Busuk Pelepah",            "Bercak oval abu-abu di pelepah, meluas dan batang bisa roboh."),
    ("tungro.jpg",                "bug",          "Tungro",                   "Daun kuning-oranye, tanaman kerdil, anakan berkurang."),
]

HAMA_CS = [
    ("penggerek_batang_sundep.jpg", "bug", "Penggerek Batang — Sundep",
     "Pucuk daun tengah (titik tumbuh) mengering, menguning, dan mudah tercabut pada fase vegetatif. "
     "Disebabkan larva penggerek batang (Scirpophaga spp.) yang menggerek titik tumbuh dari dalam batang."),
    ("penggerek_batang_beluk.jpg", "bug", "Penggerek Batang — Beluk",
     "Malai berwarna putih dan hampa, mudah tercabut pada fase generatif (berbunga/bermalai). Disebabkan "
     "larva penggerek batang yang merusak pangkal malai sehingga bulir tidak terbentuk."),
    ("tikus_sawah.jpg", "bug", "Tikus Sawah",
     "Batang padi terpotong atau roboh dekat permukaan tanah, anakan hilang, dan ditemukan lubang aktif di "
     "pematang/galangan sawah. Populasi melonjak saat fase vegetatif-generatif dan musim kering panjang."),
    ("wereng_batang_coklat.jpg", "bug", "Wereng Batang Coklat",
     "Serangga kecil cokelat berkerumun di pangkal batang dekat permukaan air. Serangan berat membuat "
     "tanaman mengering mendadak seperti terbakar (hopperburn), dan juga vektor virus kerdil rumput/kerdil hampa."),
]

# ================================================================
#  PREDIKSI GAMBAR
# ================================================================
@st.cache_resource
# ================================================================
#  PREDIKSI GAMBAR
#  Pipeline: resize 224x224 → float32 0-255 (tanpa /255, Rescaling
#  ada di dalam model) → ONNX inference → threshold entropy+conf
# ================================================================
def prediksi_gambar(foto_bytes: bytes) -> dict | None:
    if model_cnn is None:
        return None
 
    import math
 
    # ── Preprocessing ───────────────────────────────────────────────
    pil_img     = Image.open(io.BytesIO(foto_bytes)).convert("RGB")
    img_resized = pil_img.resize((224, 224), Image.LANCZOS)
    img_array   = np.array(img_resized, dtype=np.float32)  # nilai 0-255, TIDAK dibagi 255
    img_batch   = np.expand_dims(img_array, axis=0)        # (1, 224, 224, 3)
 
    # ── Inferensi ONNX ──────────────────────────────────────────────
    try:
        input_name  = model_cnn.get_inputs()[0].name   # "image_input" di model baru
        output_name = model_cnn.get_outputs()[0].name  # "dense_1"
        probs = model_cnn.run([output_name], {input_name: img_batch})[0][0]
    except Exception as e:
        logger.error(f"Inferensi ONNX gagal: {e}")
        return None
 
    idx        = int(np.argmax(probs))
    confidence = float(probs[idx])
    kelas      = NAMA_KELAS[idx]
 
    # ── Deteksi gambar non-padi via entropy ─────────────────────────
    # Dari kalibrasi empiris pada model ini:
    # • Random noise  : confidence 38-59%, entropy_ratio 0.527-0.677
    # • Foto padi asli: confidence >65%,   entropy_ratio <0.50
    # Threshold dipilih tepat di antara dua distribusi ini.
    entropy       = -sum(p * math.log(p + 1e-9) for p in probs)
    entropy_ratio = entropy / math.log(len(probs))  # 0=yakin, 1=sangat bingung
 
    # Gambar dianggap "bukan padi" jika KEDUANYA terpenuhi:
    # confidence rendah DAN entropy tinggi (model bingung dan tidak yakin)
    THRESHOLD_CONF    = 0.62   # dari test: noise max 0.588, padi asli >0.65
    THRESHOLD_ENTROPY = 0.50   # dari test: noise min 0.527, padi asli <0.50
 
    bukan_padi = (confidence < THRESHOLD_CONF) and (entropy_ratio > THRESHOLD_ENTROPY)
 
    if bukan_padi:
        return {
            "kelas":       "unknown",
            "nama_indo":   "Tidak Teridentifikasi",
            "confidence":  confidence,
            "rekomendasi": (
                "Foto tidak dapat dikenali sebagai tanaman padi atau penyakit padi. "
                "Pastikan foto diambil dari jarak dekat dengan fokus pada daun, "
                "batang, atau akar tanaman padi yang menunjukkan gejala. "
                "Hindari foto yang terlalu gelap, buram, atau berisi banyak objek lain."
            ),
            "berbahaya":   False,
            "probs":       {NAMA_KELAS[i]: float(probs[i]) for i in range(len(NAMA_KELAS))},
            "u2net_aktif": False,
        }
 
    return {
        "kelas":       kelas,
        "nama_indo":   NAMA_INDO[kelas],
        "confidence":  confidence,
        "rekomendasi": REKOMENDASI[kelas],
        "berbahaya":   kelas in BERBAHAYA,
        "probs":       {NAMA_KELAS[i]: float(probs[i]) for i in range(len(NAMA_KELAS))},
        "u2net_aktif": False,
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
    font=dict(family="Poppins, sans-serif", size=13, color="#1c2620"),
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
@st.cache_data(ttl=86400)   # refresh mingguan — NOAA update bulanan
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
                    _season = parts[0]
                    _year   = int(parts[1])
                    _month  = _SEAS_MONTH[_season]
                    if _season == "NDJ":
                        _year += 1
                    rows.append({
                        "date": datetime(_year, _month, 1),
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
    
@st.cache_data(ttl=86400)
def prediksi_oni_lstm(df, lookback=12):
    """
    Prediksi ONI 1 bulan ke depan menggunakan LSTM mini yang di-fit secara
    in-sample langsung dari data historis NOAA.

    Arsitektur:
        Input  → LSTM(32, return_sequences=False) → Dropout(0.1)
               → Dense(16, relu) → Dense(1)
    Loss       : MSE, Optimizer: Adam(lr=1e-3)
    Epochs     : 80 (cepat, data bulanan ±900 titik)
    Lookback   : 12 bulan (1 tahun musiman)

    Output dict (format identik dengan versi AR lama):
        {"date": Timestamp, "oni": float, "oni_low": float,
         "oni_high": float, "tren": str}
    """
    import warnings

    y_raw  = df["ONI"].values.astype("float32")
    n      = len(y_raw)

    # ── Fallback ke persistence jika data terlalu sedikit ──────────
    if n <= lookback + 2:
        last_date = pd.Timestamp(df["date"].iloc[-1])
        v = float(y_raw[-1])
        return {"date": last_date + pd.DateOffset(months=1),
                "oni": round(v, 2), "oni_low": round(v - 0.3, 2),
                "oni_high": round(v + 0.3, 2), "tren": "stabil →"}

    # ── Normalisasi Min-Max ke [0, 1] ───────────────────────────────
    y_min, y_max = y_raw.min(), y_raw.max()
    rng = y_max - y_min if y_max != y_min else 1.0
    y_scaled = (y_raw - y_min) / rng

    # ── Bangun sliding-window dataset ──────────────────────────────
    X_list, Y_list = [], []
    for i in range(lookback, n):
        X_list.append(y_scaled[i - lookback: i])
        Y_list.append(y_scaled[i])
    X = np.array(X_list, dtype="float32").reshape(-1, lookback, 1)
    Y = np.array(Y_list, dtype="float32")

    # ── Bangun & latih model LSTM (lazy import TF/Keras) ───────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import os as _os
            _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from tensorflow import keras                              # type: ignore
            from tensorflow.keras import layers                      # type: ignore

        # Kunci reproducibility
        keras.utils.set_random_seed(42)

        model = keras.Sequential([
            layers.Input(shape=(lookback, 1)),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.1),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        # EarlyStopping agar tidak overfit & tetap cepat
        cb_es = keras.callbacks.EarlyStopping(
            monitor="loss", patience=8, restore_best_weights=True, verbose=0
        )
        model.fit(
            X, Y,
            epochs=80,
            batch_size=16,
            verbose=0,
            callbacks=[cb_es],
        )

        # ── Inferensi: ambil window 12 bulan terakhir ───────────────
        last_window = y_scaled[-lookback:].reshape(1, lookback, 1).astype("float32")
        pred_scaled  = float(model.predict(last_window, verbose=0)[0, 0])

        # Hitung ketidakpastian dari residual in-sample
        y_pred_all  = model.predict(X, verbose=0).flatten()
        residuals   = Y - y_pred_all
        std_err_sc  = float(np.std(residuals)) * 1.3   # ×1.3 konservatif

    except Exception:
        # ── Fallback ke AR(3) jika TensorFlow tidak tersedia ────────
        p = 3
        n2 = len(y_raw)
        Xar = np.column_stack([[1.0] * (n2 - p)] +
                              [[y_raw[i - j] for i in range(p, n2)]
                               for j in range(1, p + 1)])
        Yar = y_raw[p:]
        coeffs, _, _, _ = np.linalg.lstsq(Xar, Yar, rcond=None)
        pred_raw  = coeffs[0] + float(np.dot(coeffs[1:], y_raw[-p:][::-1]))
        std_err   = float(np.std(Yar - Xar @ coeffs)) * 1.2
        last_date = pd.Timestamp(df["date"].iloc[-1])
        cur_oni   = float(y_raw[-1])
        return {
            "date":     last_date + pd.DateOffset(months=1),
            "oni":      round(pred_raw, 2),
            "oni_low":  round(pred_raw - std_err, 2),
            "oni_high": round(pred_raw + std_err, 2),
            "tren":     ("naik"  if pred_raw > cur_oni + 0.05
                         else "turun" if pred_raw < cur_oni - 0.05
                         else "stabil"),
        }

    # ── Denormalisasi kembali ke satuan °C ─────────────────────────
    pred_oni     = float(pred_scaled * rng + y_min)
    std_err_oni  = float(std_err_sc  * rng)          # skala kesalahan → °C

    cur_oni   = float(y_raw[-1])
    last_date = pd.Timestamp(df["date"].iloc[-1])

    return {
        "date":     last_date + pd.DateOffset(months=1),
        "oni":      round(pred_oni, 2),
        "oni_low":  round(pred_oni - std_err_oni, 2),
        "oni_high": round(pred_oni + std_err_oni, 2),
        "tren":     ("naik"  if pred_oni > cur_oni + 0.05
                     else "turun" if pred_oni < cur_oni - 0.05
                     else "stabil"),
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
    st.caption("Aktifkan GPS lewat tombol di tab Beranda. Atau atur manual di sini.")

    if "gps_lat" not in st.session_state:
        st.session_state["gps_lat"]   = -7.8754
        st.session_state["gps_lon"]   = 110.4262
        st.session_state["gps_aktif"] = False

    lat, lon, aktif = (
        st.session_state["gps_lat"], st.session_state["gps_lon"], st.session_state["gps_aktif"]
    )
    if aktif:
        st.success(f"GPS aktif\n{round(lat,4)}°, {round(lon,4)}°\n"
                   f"{prov_terdekat(lat,lon)}")
    else:
        st.warning("GPS belum aktif. Pilih manual.")
        lat = st.number_input("Lintang:", value=lat, format="%.4f")
        lon = st.number_input("Bujur:",   value=lon, format="%.4f")
        st.session_state["gps_lat"] = lat
        st.session_state["gps_lon"] = lon

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
    st.caption("Data: BMKG · NOAA · Sentinel-1/2\nAgriAlert v1.0 · Satria Data 2026")


# ── DATA REAL-TIME ────────────────────────────────────────────────
cuaca  = cuaca_bmkg(lat, lon)
oni    = oni_now()
enso   = info_enso(oni)
status = info_cuaca(cuaca["tmax"], cuaca["curah_hujan"])
musim_ikon, msmn = musim_bln(datetime.now().month)
warna_rk, ikon_rk, teks_rk = saran_tanam(status["kelas"], enso["kelas"])


# ── HEADER ───────────────────────────────────────────────────────
_mascot_url = ui_img_b64("mascot.png")
_mascot_style = f"--mascot-url: url('{_mascot_url}');" if _mascot_url else ""
st.markdown(f"""
<div class="ha-header" style="{_mascot_style}">
  <div class="ha-mascot"></div>
  <div class="ha-header-left">
    <div>
      <p class="ha-title">AgriAlert</p>
      <p class="ha-sub">Peringatan dini hama &amp; cuaca padi Indonesia</p>
      <span class="ha-badge">{ico('pin')} {provinsi}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── 5 TAB ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Beranda", "Cuaca & Iklim", "Kenali Hama & Penyakit", "Cek Hama Padi", "Informasi"
])


# ════════════════════════════════════════════════════
#  TAB 1 — BERANDA
# ════════════════════════════════════════════════════
with tab1:
    if not cuaca["ok"] and cuaca["pesan"]:
        st.warning(cuaca["pesan"])

    # ── Tombol GPS — klik langsung di Beranda (sesuai Figma) ──
    # Catatan teknis: streamlit_geolocation adalah komponen pihak ketiga yang
    # dirender di dalam iframe browser sendiri (untuk bisa minta izin lokasi).
    # Karena itu komponennya harus selalu ada di halaman (bukan di dalam if),
    # dan teks status Figma (merah/hijau) ditampilkan di luar iframe karena
    # CSS halaman tidak bisa menjangkau ke dalam iframe komponen pihak ketiga.
    _gps_aktif = st.session_state.get("gps_aktif", False)
    _label_teks = "GPS ANDA AKTIF" if _gps_aktif else "KLIK UNTUK AKTIFKAN GPS"
    _btn_bg = "#1e8a55" if _gps_aktif else "#20965F"
    st.markdown(f"""<style>
div:has(> [data-testid="stCustomComponentV1"])::before {{
    content: "{_label_teks}" !important;
}}
div:has(> [data-testid="stCustomComponentV1"]) {{
    background-color: {_btn_bg} !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fff' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 22s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12Z'/%3E%3Ccircle cx='12' cy='10' r='2.3'/%3E%3C%2Fsvg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: 20px center !important;
    background-size: 26px 26px !important;
}}
</style>""", unsafe_allow_html=True)
    hasil_gps = streamlit_geolocation()
    _stcv1.html("""<script>
(function(){
  function fix(){
    try{
      var pd=window.parent.document;
      /* stCustomComponentV1 IS the iframe element itself */
      var ifs=pd.querySelectorAll('[data-testid="stCustomComponentV1"]');
      for(var i=0;i<ifs.length;i++){
        try{
          var iframe=ifs[i];
          /* skip ourselves */
          if(iframe===window.frameElement) continue;
          var d=iframe.contentDocument||(iframe.contentWindow&&iframe.contentWindow.document);
          if(!d||!d.body) continue;
          if(d.getElementById('aa-gps-fix')) continue;
          var btn=d.querySelector('button');
          if(!btn) continue;
          var s=d.createElement('style');
          s.id='aa-gps-fix';
          s.textContent='html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden}button{position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;opacity:0.001!important;cursor:pointer!important;border:none!important;background:transparent!important}';
          (d.head||d.documentElement).appendChild(s);
        }catch(e){}
      }
    }catch(e){}
  }
  setTimeout(fix,300);setTimeout(fix,800);setTimeout(fix,1500);setTimeout(fix,3000);setTimeout(fix,5000);
})();
</script>""", height=1)

    if hasil_gps and hasil_gps.get("latitude") and hasil_gps.get("longitude"):
        _lat_baru, _lon_baru = float(hasil_gps["latitude"]), float(hasil_gps["longitude"])
        if not st.session_state.get("gps_aktif") or st.session_state.get("gps_lat") != _lat_baru:
            st.session_state["gps_lat"]   = _lat_baru
            st.session_state["gps_lon"]   = _lon_baru
            st.session_state["gps_aktif"] = True
            st.rerun()

    # ── Kotak instruksi maps ──
    # Pakai ikon "map" (bukan pin) supaya gak keliatan dobel sama pin GPS di tombol atas,
    # dan teksnya kondisional ikut status GPS biar gak nyuruh klik tombol yang udah diklik.
    _gps_info_text = ("Lokasi Anda sudah terdeteksi. Data cuaca otomatis mengikuti posisi Anda saat ini."
                       if _gps_aktif else
                       "Klik tombol di atas agar sistem kami bisa tahu kondisi cuaca di sawah Anda saat ini")
    st.markdown(f"""
    <div class="gps-info-row below-gps-btn">
      <div class="gps-info-text">{_gps_info_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="gps-box">
      <span> <b>Stasiun BMKG:</b> {cuaca['nama']}</span>
      <span> <b>Jarak:</b> {cuaca['jarak_km']} km</span>
      <span> <b>Data per:</b> {cuaca['waktu']}</span>
      <span> <b>Koordinat:</b> {round(lat,4)}°, {round(lon,4)}°</span>
      <span> <b>Provinsi:</b> {provinsi}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-row">
      <span class="chip {status['kelas']}">{ico(status['ikon'])} {status['level']}</span>
      <span class="chip {enso['kelas']}">{ico(enso['ikon'])} {enso['level']}</span>
      <span class="chip info">{ico(musim_ikon)} {msmn}</span>
    </div>
    """, unsafe_allow_html=True)

    _sun_url = ui_img_b64("sun.png")
    _sun_img_html = (f'<img src="{_sun_url}" style="position:absolute;width:80px;height:80px;right:24px;top:50%;transform:translateY(-50%);object-fit:contain;" alt="sun">'
                      if (_sun_url and status['kelas'] == 'aman') else "")
    st.markdown(f"""
    <div class="big-card {status['kelas']}" style="position:relative;">
      <div class="bc-label">Kondisi Cuaca Hari Ini - {provinsi}</div>
      <div class="bc-row">
        <span class="bc-dot {status['kelas']}"></span>
        <span class="bc-icon">{ico(status['ikon'], '2.2rem')}</span>
        <span class="bc-value">{status['level']}</span>
      </div>
      <div class="bc-sub">{status['pesan']}</div>
      {_sun_img_html}
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
        <div class="mc-sub">Min: {cuaca['tmin']}°C &nbsp;|&nbsp; Ideal padi: 24°-30°C</div>
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
    st.markdown("""
    <style>
    .metric-grid {
        display: flex !important;
        justify-content: center !important;
        flex-wrap: wrap !important;
        gap: 15px !important;
    }
    .metric-card {
        text-align: center !important;
        flex: 1 1 150px !important;
        max-width: 200px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("lightbulb")} Yang Harus Dilakukan</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="two-col">
      <div class="rekom {warna_rk}">
        <div class="rk-title"> Saran Tanam</div>
        <div class="rk-text">{ico(ikon_rk)} {teks_rk}</div>
      </div>
      <div class="rekom biru">
        <div class="rk-title"> Kondisi Iklim - {enso['fase']}</div>
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
      <div class="rk-title">{msmn}</div>
      <div class="rk-text">{musim_tips}</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  TAB 2 — CUACA & IKLIM  (VERSI 2 — bugfix length mismatch)
#  Ganti seluruh blok `with tab2:` yang lama dengan kode ini.
# ════════════════════════════════════════════════════
with tab2:

    # ================================================================
    #  HELPER: COMPOSITE SCORING ENGINE
    #  Indeks Lahan Stres Komposit (ILSK)
    # ================================================================
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def hitung_ilsk(df_ndvi_in, df_sar_in, df_enso_in):
        """
        Menggabungkan tiga sumber data satelit + iklim menjadi satu skor
        Indeks Lahan Stres Komposit (ILSK) skala 0-1.

        Metode: Rolling Z-score Anomaly Detection + Weighted Fusion
          Komponen A (bobot 45%) : NDVI Deficiency  — anomali vegetasi vs. baseline 4-bulan
          Komponen B (bobot 30%) : SAR Dryness      — anomali kelembapan tanah vs. baseline
          Komponen C (bobot 25%) : ENSO Pressure    — tekanan iklim makro dari ONI

        Seluruh join antar-dataset dilakukan via merge_asof (safe untuk
        dataset dengan panjang dan timestamp berbeda-beda).
        """
        WINDOW = 8        # biweekly x8 ~ 4 bulan rolling baseline
        W_NDVI = 0.45
        W_SAR  = 0.30
        W_ENSO = 0.25

        # ── Komponen A: NDVI Anomaly Score ──────────────────────────
        df_n = (df_ndvi_in[["Tanggal", "NDVI"]]
                .copy()
                .sort_values("Tanggal")
                .reset_index(drop=True))

        df_n["ndvi_baseline"] = (df_n["NDVI"]
                                  .rolling(WINDOW, min_periods=4).mean()
                                  .bfill())
        df_n["ndvi_roll_std"] = (df_n["NDVI"]
                                  .rolling(WINDOW, min_periods=4).std()
                                  .bfill()
                                  .replace(0, 0.01))
        df_n["ndvi_z"]       = ((df_n["NDVI"] - df_n["ndvi_baseline"])
                                 / df_n["ndvi_roll_std"])
        # z negatif = di bawah baseline = stres; skor 0=sehat, 1=kritis
        df_n["ndvi_stress"]  = ((-df_n["ndvi_z"]).clip(-3, 3) + 3) / 6

        # ── Komponen B: SAR Dryness Anomaly Score ───────────────────
        df_s = (df_sar_in[["Tanggal", "SAR_VH"]]
                .copy()
                .sort_values("Tanggal")
                .reset_index(drop=True))

        df_s["sar_baseline"] = (df_s["SAR_VH"]
                                 .rolling(WINDOW, min_periods=4).mean()
                                 .bfill())
        df_s["sar_roll_std"] = (df_s["SAR_VH"]
                                 .rolling(WINDOW, min_periods=4).std()
                                 .bfill()
                                 .replace(0, 0.5))
        df_s["sar_z"]        = ((df_s["SAR_VH"] - df_s["sar_baseline"])
                                 / df_s["sar_roll_std"])
        # SAR VH lebih positif (dB kurang negatif) = lahan lebih kering = stres lebih tinggi
        df_s["sar_stress"]   = (df_s["sar_z"].clip(-3, 3) + 3) / 6

        # ── Komponen C: ENSO Pressure Score ─────────────────────────
        df_e = df_enso_in.copy()
        date_col = "date" if "date" in df_e.columns else "Tanggal"
        df_e = df_e.rename(columns={date_col: "Tanggal"})
        df_e["Tanggal"]       = pd.to_datetime(df_e["Tanggal"])
        df_e["oni_drought"]   = df_e["ONI"].clip(0, 3) / 3.0
        df_e["oni_flood"]     = (-df_e["ONI"]).clip(0, 3) / 3.0
        df_e["enso_pressure"] = df_e[["oni_drought", "oni_flood"]].max(axis=1)

        # ── Gabung: semua via merge_asof (aman meski panjang berbeda) ─
        # Master timeline = NDVI (data terlengkap)
        df_merge = df_n[["Tanggal", "NDVI", "ndvi_baseline",
                          "ndvi_z", "ndvi_stress"]].copy()

        # Join SAR — toleransi ±30 hari agar tetap cocok meski jadwal akuisisi beda
        df_merge = pd.merge_asof(
            df_merge.sort_values("Tanggal"),
            df_s[["Tanggal", "SAR_VH", "sar_baseline",
                  "sar_stress"]].sort_values("Tanggal"),
            on="Tanggal",
            direction="nearest",
            tolerance=pd.Timedelta("30D")
        )

        # Isi NaN SAR (periode di luar jangkauan data) dengan skor netral
        sar_vh_median = df_s["SAR_VH"].median()
        df_merge["SAR_VH"]      = df_merge["SAR_VH"].fillna(sar_vh_median)
        df_merge["sar_baseline"] = df_merge["sar_baseline"].fillna(sar_vh_median)
        df_merge["sar_stress"]   = df_merge["sar_stress"].fillna(0.5)

        # Join ENSO — bulanan ke biweekly via backward fill
        df_merge = pd.merge_asof(
            df_merge.sort_values("Tanggal"),
            df_e[["Tanggal", "ONI", "enso_pressure",
                  "oni_drought", "oni_flood"]].sort_values("Tanggal"),
            on="Tanggal",
            direction="backward"
        )
        df_merge["ONI"]           = df_merge["ONI"].fillna(0.0)
        df_merge["enso_pressure"] = df_merge["enso_pressure"].fillna(0.0)
        df_merge["oni_drought"]   = df_merge["oni_drought"].fillna(0.0)
        df_merge["oni_flood"]     = df_merge["oni_flood"].fillna(0.0)

        # ── Skor Komposit ILSK ───────────────────────────────────────
        df_merge["ILSK"] = (
            W_NDVI * df_merge["ndvi_stress"]   +
            W_SAR  * df_merge["sar_stress"]    +
            W_ENSO * df_merge["enso_pressure"]
        ).clip(0, 1)

        # EWM smoothing untuk visualisasi tren
        df_merge["ILSK_smooth"] = (df_merge["ILSK"]
                                    .ewm(span=5, min_periods=2)
                                    .mean())

        # Klasifikasi status
        def _status(s):
            if s < 0.35:   return "Aman"
            elif s < 0.55: return "Waspada"
            return "Bahaya"

        df_merge["status"]        = df_merge["ILSK"].apply(_status)
        df_merge["status_smooth"] = df_merge["ILSK_smooth"].apply(_status)

        return df_merge.reset_index(drop=True)

    # ================================================================
    #  HITUNG ILSK (hanya jika kedua data satelit tersedia)
    # ================================================================
    data_lengkap = (df_ndvi is not None) and (df_sar is not None)

    if data_lengkap:
        df_ilsk = hitung_ilsk(df_ndvi, df_sar, df_enso)
        ilsk_terkini      = float(df_ilsk["ILSK"].iloc[-1])
        status_terkini    = df_ilsk["status"].iloc[-1]
        tanggal_terakhir  = df_ilsk["Tanggal"].iloc[-1].strftime("%d %b %Y")

        ndvi_val   = float(df_ilsk["NDVI"].iloc[-1])
        sar_val    = float(df_ilsk["SAR_VH"].iloc[-1])
        oni_val    = float(df_ilsk["ONI"].iloc[-1])
        ndvi_skor  = float(df_ilsk["ndvi_stress"].iloc[-1])
        sar_skor   = float(df_ilsk["sar_stress"].iloc[-1])
        enso_skor  = float(df_ilsk["enso_pressure"].iloc[-1])

        ilsk_4ago  = float(df_ilsk["ILSK"].iloc[-5]) if len(df_ilsk) >= 5 else ilsk_terkini
        delta_tren = ilsk_terkini - ilsk_4ago

    # ================================================================
    #  SECTION 1 — HEADER + LOKASI + CUACA BMKG
    # ================================================================
    st.markdown(
        f'<p class="sec-title">{ico("cloud-sun")} Cuaca Detail - {provinsi}</p>',
        unsafe_allow_html=True
    )

    _maps_icon_t2 = (
        f'<img src="{ui_img_b64("maps.png")}" alt="maps">'
        if ui_img_b64("maps.png") else ico("map", "1.6rem")
    )
    st.markdown(f"""
    <div class="gps-info-row">
      <div class="gps-info-icon">{_maps_icon_t2}</div>
      <div class="gps-info-text">
        Data cuaca dari koordinat: {round(lat, 4)}, {round(lon, 4)} — {provinsi}
      </div>
    </div>
    """, unsafe_allow_html=True)

    _delta_suhu = round(cuaca["tmax"] - 30, 1)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="mc-label">{ico("thermometer")} Suhu Maks</div>
        <div class="mc-val">{cuaca["tmax"]}°C</div>
        <div class="mc-sub">{_delta_suhu:+} dari ideal 30°C</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">{ico("thermometer")} Suhu Min</div>
        <div class="mc-val">{cuaca["tmin"]}°C</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">{ico("droplet")} Kelembapan</div>
        <div class="mc-val">{int(cuaca["kelembapan"])}%</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">{ico("cloud-rain")} Curah Hujan</div>
        <div class="mc-val">{cuaca["curah_hujan"]} mm/hr</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    #  SECTION 2 — ENSO LIVE
    # ================================================================
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sec-title" style="font-weight:900;color:#20965F;">'
        f'Pantauan Cuaca & Iklim (ENSO)</p>',
        unsafe_allow_html=True
    )

    df_oni_live, oni_ok, oni_time = fetch_oni_noaa()
    df_for_enso = (
        df_oni_live
        if df_oni_live is not None and len(df_oni_live) >= 6
        else df_enso
    )
    src_txt = (
        f"NOAA CPC · Diperbarui: {oni_time} · {len(df_for_enso)} data poin"
        if oni_ok else "Data lokal (NOAA tidak tersedia saat ini)"
    )
    st.caption(src_txt)

    oni_cur   = float(df_for_enso.iloc[-1]["ONI"])
    date_cur  = pd.Timestamp(df_for_enso.iloc[-1]["date"])
    enso_cur  = info_enso(oni_cur)
    pred = prediksi_oni_lstm(df_for_enso)
    enso_pred = info_enso(pred["oni"])

    st.markdown(f"""
    <div class="two-col">
      <div class="big-card {enso_cur['kelas']}">
        <div class="bc-label"> ENSO Terkini - {date_cur.strftime("%B %Y")}</div>
        <div class="bc-row">
          <span class="bc-icon">{ico(enso_cur["ikon"], "2.2rem")}</span>
          <div>
            <span class="bc-value" style="font-size:1.15rem">{enso_cur["fase"]}</span>
            <div class="bc-sub">ONI: <b>{oni_cur:+.2f}°C</b></div>
          </div>
        </div>
        <div class="bc-sub" style="margin-top:8px">{enso_cur["pesan"]}</div>
      </div>
      <div class="big-card {enso_pred['kelas']}">
        <div class="bc-label">{ico("chart")} Prediksi - {pred["date"].strftime("%B %Y")}</div>
        <div class="bc-row">
          <span class="bc-icon">{ico(enso_pred["ikon"], "2.2rem")}</span>
          <div>
            <span class="bc-value" style="font-size:1.15rem">{enso_pred["fase"]}</span>
            <div class="bc-sub">
              ONI kira-kira <b>{pred["oni"]:+.2f}°C</b>
              &nbsp;({pred["oni_low"]:+.1f} s/d {pred["oni_high"]:+.1f})
            </div>
          </div>
        </div>
        <div class="bc-sub" style="margin-top:8px">
          Tren: <b>{pred["tren"]}</b>. {enso_pred["pesan"]}
        </div>
      </div>
    </div>
    <p style="font-size:0.7rem;color:#64748b;margin:-2px 0 12px">
      {ico("alert-triangle")} Prediksi menggunakan model AR(3) dari data historis NOAA 75 tahun —
      indikasi statistik, bukan model iklim resmi.
      Rujukan: <b>bmkg.go.id</b> · <b>iri.columbia.edu</b>
    </p>
    """, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
          <div style= padding: 20px; border-radius: 12px; border: 1px solid #20965F;">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #20965F;">Grafik Tren Indeks Iklim (ONI)</h3>
            <p style="font-size: 0.9rem; color: #57685d; margin-bottom: 15px;">
                Grafik di bawah memantau suhu permukaan laut (ONI) dalam 2 tahun terakhir. 
                Ini membantu kita melihat apakah sedang terjadi El Niño (kemarau ekstrem) 
                atau La Niña (basah berlebih).
            </p>
            <div style="font-size: 0.8rem; color: #20965F; margin-bottom: 10px; border-bottom: 1px solid #20965F; padding-bottom: 10px;">
                El Niño (≥ +0.5) · La Niña (≤ -0.5) · Normal · Prediksi bulan depan
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        st.plotly_chart(
            fig_oni_with_forecast(df_for_enso, pred),
            use_container_width=True,
            theme=None
        )

    # ── Farming impact section ───────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sec-title">Implikasi untuk Pertanian Padi Indonesia</p>',
        unsafe_allow_html=True
    )

    _DAMPAK = {
        "El Nino Kuat": [
            ("merah","Kekeringan Ekstrem",
             "Produksi padi di Jawa, NTB, dan Sulawesi bisa turun 20-40%. "
             "Hentikan penanaman varietas sensitif kering. "
             "Prioritaskan lahan dekat irigasi teknis."),
            ("merah", "Krisis Air Irigasi",
             "Terapkan AWD (Alternate Wetting and Drying): basah 2 hari, "
             "kering 3 hari, basah lagi. "
             "Tutup permukaan tanah dengan jerami untuk kurangi evaporasi."),
            ("merah", "Hama Tikus dan Wereng",
             "Populasi tikus melonjak saat kemarau panjang. "
             "Koordinasi gropyokan bersama petani sekitar. "
             "Pasang bubu perangkap dan karet di pematang sawah."),
        ],
        "El Nino": [
            ("kuning", "Hemat Air Irigasi",
             "Terapkan irigasi berselang. Tanam varietas toleran kering: "
             "Inpari 32, Inpago, atau Inpara 3. "
             "Jadwal tanam lebih awal dari biasanya."),
            ("kuning", "Sesuaikan Kalender Tanam",
             "Maju jadwal tanam agar panen selesai sebelum puncak kemarau. "
             "Koordinasikan dengan kelompok tani dan penyuluh PPL."),
        ],
        "Normal": [
            ("", "Kondisi Iklim Ideal",
             "Ikuti kalender tanam yang sudah ditetapkan BPP. "
             "Waktu tepat untuk pemupukan NPK dan pengamatan rutin OPT."),
        ],
        "La Nina": [
            ("kuning", "Pantau Drainase",
             "Bersihkan saluran got sebelum hujan puncak. Perkuat pematang sawah. "
             "Hindari genangan lebih dari 3 hari pada fase anakan."),
            ("kuning", "Waspadai Penyakit Jamur",
             "Risiko blas dan busuk pelepah meningkat. "
             "Aplikasi fungisida profilaksis (mankozeb/validamycin) "
             "dianjurkan pada fase anakan aktif."),
        ],
        "La Nina Kuat": [
            ("merah", "Risiko Banjir Tinggi",
             "Di Kalimantan, Sumatera, dan pantai utara Jawa, "
             "bersiap untuk genangan panjang. "
             "Pindah ke varietas tahan banjir: Inpara 3, Ciherang Sub-1."),
            ("merah", "Ledakan Penyakit",
             "Blas daun dan tungro melonjak drastis. "
             "Kendalikan wereng hijau (vektor tungro) dengan imidakloprid. "
             "Cabut dan bakar tanaman terinfeksi segera."),
            ("merah", "Wereng Cokelat Meledak",
             "Populasi wereng meledak di kondisi lembab berkepanjangan. "
             "Gunakan varietas tahan wereng (VUB BPT) "
             "dan hindari pemupukan N berlebih."),
        ],
    }

    for warna, judul, teks in _DAMPAK.get(enso_cur["fase"], _DAMPAK["Normal"]):
        st.markdown(f"""
        <div class="rekom {warna}">
          <div class="rk-title">{judul}</div>
          <div class="rk-text">{teks}</div>
        </div>
        """, unsafe_allow_html=True)

    if enso_pred["fase"] != enso_cur["fase"]:
        st.markdown(f"""
        <div class="rekom biru" style="margin-top:10px">
          <div class="rk-title"> Potensi Transisi ENSO Bulan Depan</div>
          <div class="rk-text">
            Tren 6 bulan terakhir menunjukkan kemungkinan pergeseran dari
            {enso_cur["fase"]} ke {enso_pred["fase"]}
            pada {pred["date"].strftime("%B %Y")}.
            Mulai persiapkan langkah antisipasi sekarang.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    #  SECTION 3 — ILSK: INDEKS LAHAN STRES KOMPOSIT
    # ================================================================
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sec-title" style="font-weight:900;color:#20965F;">'
        f' Indeks Lahan Stres Komposit (ILSK)</p>',
        unsafe_allow_html=True
    )

    if not data_lengkap:
        st.warning(
            "Data satelit NDVI atau SAR belum tersedia. "
            "ILSK akan tampil setelah kedua dataset dimuat."
        )
    else:
        # ── Banner Peringatan Utama ─────────────────────────────
        tren_txt = (
            "meningkat dalam 2 bulan terakhir"  if delta_tren >  0.04 else
            "menurun dalam 2 bulan terakhir"    if delta_tren < -0.04 else
            "stabil dalam 2 bulan terakhir"
        )

        if status_terkini == "Bahaya":
            st.error(
                f"PERINGATAN DINI - STRES LAHAN TINGGI\n\n"
                f"Berdasarkan data satelit per {tanggal_terakhir}, "
                f"Indeks Lahan Stres Komposit (ILSK) menunjukkan nilai "
                f"{ilsk_terkini:.2f} (Kategori: BAHAYA). "
                f"Tingkat stres lahan {tren_txt}. "
                f"Segera lakukan evaluasi kondisi irigasi dan "
                f"kelembapan tanah di lapangan."
            )
        elif status_terkini == "Waspada":
            st.warning(
                f"PERLU PERHATIAN - STRES LAHAN SEDANG\n\n"
                f"Berdasarkan data satelit per {tanggal_terakhir}, "
                f"Indeks Lahan Stres Komposit (ILSK) bernilai "
                f"{ilsk_terkini:.2f} (Kategori: WASPADA). "
                f"Tingkat stres lahan {tren_txt}. "
                f"Pantau kondisi lahan lebih sering dan "
                f"siapkan tindakan preventif."
            )
        else:
            st.success(
                f"KONDISI LAHAN NORMAL\n\n"
                f"Berdasarkan data satelit per {tanggal_terakhir}, "
                f"Indeks Lahan Stres Komposit (ILSK) bernilai "
                f"{ilsk_terkini:.2f} (Kategori: AMAN). "
                f"Tingkat stres lahan {tren_txt}. "
                f"Tidak ada indikasi tekanan signifikan pada "
                f"lahan sawah saat ini."
            )

        # ── Kartu Breakdown 3 Komponen ──────────────────────────
        def _warna_skor(s):
            if s < 0.35:   return "aman"
            elif s < 0.55: return "waspada"
            return "bahaya"

        def _label_skor(s):
            if s < 0.35:   return "Rendah"
            elif s < 0.55: return "Sedang"
            return "Tinggi"
        st.markdown("""
        <style>
            .metric-card {
                text-align: center !important;
            }
        </style>
        """, unsafe_allow_html=True)


        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns:repeat(3,1fr);margin-top:14px">

          <div class="metric-card {_warna_skor(ndvi_skor)}">
            <div class="mc-label"> Stres Vegetasi (NDVI)</div>
            <div class="mc-val">{ndvi_skor:.2f}</div>
            <div class="mc-sub">
              NDVI: {ndvi_val:.3f}<br>
              Tekanan: <b>{_label_skor(ndvi_skor)}</b> - bobot 45%
            </div>
          </div>

          <div class="metric-card {_warna_skor(sar_skor)}">
            <div class="mc-label"> Anomali Kelembapan (SAR)</div>
            <div class="mc-val">{sar_skor:.2f}</div>
            <div class="mc-sub">
              SAR VH: {sar_val:.1f} dB<br>
              Tekanan: <b>{_label_skor(sar_skor)}</b> - bobot 30%
            </div>
          </div>

          <div class="metric-card {_warna_skor(enso_skor)}">
            <div class="mc-label"> Tekanan Iklim (ENSO)</div>
            <div class="mc-val">{enso_skor:.2f}</div>
            <div class="mc-sub">
              ONI: {oni_val:+.2f}°C<br>
              Tekanan: <b>{_label_skor(enso_skor)}</b> - bobot 25%
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rekom" style="margin-top:10px">
          <div class="rk-title">{ico("info-circle")} Cara Membaca ILSK</div>
          <div class="rk-text">
            ILSK adalah angka antara 0 sampai 1 — semakin tinggi, semakin besar tekanan
            yang sedang dialami lahan sawah Anda.<br><br>
            <b>Aman</b> (di bawah 0.35): kondisi lahan normal, tidak ada indikasi gangguan berarti.<br>
            <b>Waspada</b> (0.35–0.55): mulai ada tanda stres, pantau lahan lebih sering.<br>
            <b>Bahaya</b> (di atas 0.55): tekanan lahan tinggi, segera periksa irigasi dan kondisi tanaman.<br><br>
            Angka ini digabung dari tiga sumber data satelit:
            kesehatan tanaman via NDVI Sentinel-2 (bobot 45%),
            kadar air tanah via SAR Sentinel-1 (bobot 30%),
            dan kondisi iklim global via indeks ENSO-ONI NOAA (bobot 25%).
            Perhitungannya pakai z-score rolling 4 bulan supaya perubahan musiman
            tidak dianggap sebagai ancaman — yang dilihat adalah tren jangka panjangnya.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── GRAFIK 1: ILSK Timeline ─────────────────────────────
        st.markdown(
            f'<p class="sec-title" style="margin-top:20px">'
            f'{ico("chart")} Tren ILSK — Seluruh Periode Data</p>',
            unsafe_allow_html=True
        )

        fig_ilsk = go.Figure()

        # Background bands
        fig_ilsk.add_hrect(y0=0.55, y1=1.0,
                           fillcolor="rgba(239,68,68,0.13)", line_width=0,
                           annotation_text="Bahaya",
                           annotation_position="top left",
                           annotation_font_color="#ef4444")
        fig_ilsk.add_hrect(y0=0.35, y1=0.55,
                           fillcolor="rgba(234,179,8,0.11)", line_width=0,
                           annotation_text="Waspada",
                           annotation_position="top left",
                           annotation_font_color="#ca8a04")
        fig_ilsk.add_hrect(y0=0.0, y1=0.35,
                           fillcolor="rgba(34,197,94,0.09)", line_width=0,
                           annotation_text="Aman",
                           annotation_position="bottom left",
                           annotation_font_color="#16a34a")

        # Garis ambang batas
        fig_ilsk.add_hline(y=0.55, line_dash="dot",
                           line_color="#ef4444", line_width=1.2)
        fig_ilsk.add_hline(y=0.35, line_dash="dot",
                           line_color="#ca8a04", line_width=1.2)

        # Area fill
        fig_ilsk.add_trace(go.Scatter(
            x=df_ilsk["Tanggal"], y=df_ilsk["ILSK_smooth"],
            fill="tozeroy", fillcolor="rgba(32,150,95,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip"
        ))

        # Kurva ILSK raw — titik diwarnai sesuai status
        ilsk_colors = df_ilsk["status"].map(
            {"Aman": "#22c55e", "Waspada": "#eab308", "Bahaya": "#ef4444"}
        ).tolist()

        fig_ilsk.add_trace(go.Scatter(
            x=df_ilsk["Tanggal"], y=df_ilsk["ILSK"],
            mode="lines+markers",
            name="ILSK (nilai aktual)",
            line=dict(color="#20965F", width=1.5),
            marker=dict(size=6, color=ilsk_colors,
                        line=dict(color="white", width=1)),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>ILSK: %{y:.3f}<extra></extra>"
        ))

        # Kurva tren halus (EWM)
        fig_ilsk.add_trace(go.Scatter(
            x=df_ilsk["Tanggal"], y=df_ilsk["ILSK_smooth"],
            mode="lines",
            name="ILSK (tren halus)",
            line=dict(color="#176e47", width=2.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Tren: %{y:.3f}<extra></extra>"
        ))

        # Titik terkini
        dot_color = (
            "#ef4444" if status_terkini == "Bahaya" else
            "#eab308" if status_terkini == "Waspada" else
            "#22c55e"
        )
        fig_ilsk.add_trace(go.Scatter(
            x=[df_ilsk["Tanggal"].iloc[-1]],
            y=[ilsk_terkini],
            mode="markers",
            name="Posisi Terkini",
            marker=dict(size=14, symbol="diamond", color=dot_color,
                        line=dict(color="white", width=2)),
            hovertemplate=(
                f"<b>Terkini: {tanggal_terakhir}</b><br>"
                f"ILSK: {ilsk_terkini:.3f}<br>"
                f"Status: {status_terkini}<extra></extra>"
            )
        ))

        fig_ilsk.update_layout(
            **PLOT_STYLE, height=340,
            yaxis=dict(title="Indeks Lahan Stres (ILSK)", range=[0, 1]),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1)
        )
        st.plotly_chart(fig_ilsk, use_container_width=True, theme=None)

        # ── GRAFIK 2: Dekomposisi 3 Komponen ───────────────────
        st.markdown(
            f'<p class="sec-title">'
            f'{ico("chart")} Dekomposisi Tiga Komponen ILSK</p>',
            unsafe_allow_html=True
        )

        fig_decomp = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=[
                "Komponen A — Skor Stres Vegetasi (NDVI Sentinel-2)",
                "Komponen B — Skor Anomali Kelembapan Tanah (SAR Sentinel-1 VH)",
                "Komponen C — Skor Tekanan Iklim Makro (ENSO-ONI NOAA)"
            ]
        )

        # Panel A: NDVI Stress Score + baseline
        fig_decomp.add_trace(go.Scatter(
            x=df_ilsk["Tanggal"], y=df_ilsk["ndvi_stress"],
            mode="lines+markers", name="Skor Stres NDVI",
            line=dict(color="#4ade80", width=2),
            marker=dict(size=4),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Skor: %{y:.3f}<extra></extra>"
        ), row=1, col=1)

        fig_decomp.add_trace(go.Scatter(
            x=df_ilsk["Tanggal"], y=df_ilsk["NDVI"],
            mode="lines", name="NDVI (nilai asli)",
            line=dict(color="#86efac", width=1.2, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>NDVI: %{y:.3f}<extra></extra>"
        ), row=1, col=1)

        fig_decomp.add_trace(go.Scatter(
            x=df_ilsk["Tanggal"], y=df_ilsk["ndvi_baseline"],
            mode="lines", name="Baseline NDVI (rolling 4-bulan)",
            line=dict(color="#16a34a", width=1, dash="longdash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Baseline: %{y:.3f}<extra></extra>"
        ), row=1, col=1)

        # Panel B: SAR Stress Score (bar)
        sar_bar_colors = [
            "#ef4444" if s > 0.55 else ("#eab308" if s > 0.35 else "#60a5fa")
            for s in df_ilsk["sar_stress"]
        ]
        fig_decomp.add_trace(go.Bar(
            x=df_ilsk["Tanggal"], y=df_ilsk["sar_stress"],
            name="Skor Anomali SAR",
            marker_color=sar_bar_colors,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Skor: %{y:.3f}<extra></extra>"
        ), row=2, col=1)

        # Panel C: ENSO breakdown — drought vs flood
        fig_decomp.add_trace(go.Bar(
            x=df_ilsk["Tanggal"], y=df_ilsk["oni_drought"],
            name="Tekanan Kekeringan (El Nino)",
            marker_color="rgba(239,68,68,0.65)",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Drought: %{y:.3f}<extra></extra>"
        ), row=3, col=1)

        fig_decomp.add_trace(go.Bar(
            x=df_ilsk["Tanggal"], y=df_ilsk["oni_flood"],
            name="Tekanan Banjir (La Nina)",
            marker_color="rgba(59,130,246,0.65)",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Flood: %{y:.3f}<extra></extra>"
        ), row=3, col=1)

        fig_decomp.update_layout(
            **PLOT_STYLE, height=600,
            showlegend=True, barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1, font=dict(size=11))
        )
        for r in [1, 2, 3]:
            fig_decomp.update_yaxes(range=[0, 1], title_text="Skor [0-1]",
                                    row=r, col=1)

        st.plotly_chart(fig_decomp, use_container_width=True, theme=None)

        # ── GRAFIK 3: Distribusi Status ─────────────────────────
        st.markdown(
            f'<p class="sec-title">'
            f'{ico("flag")} Distribusi Status Lahan — Seluruh Periode</p>',
            unsafe_allow_html=True
        )

        status_counts = (df_ilsk["status"]
                         .value_counts()
                         .reindex(["Aman", "Waspada", "Bahaya"], fill_value=0))
        total_obs = len(df_ilsk)

        fig_dist = go.Figure(go.Bar(
            x=status_counts.index,
            y=status_counts.values,
            marker_color=["#22c55e", "#eab308", "#ef4444"],
            text=[f"{v} obs ({v / total_obs * 100:.0f}%)"
                  for v in status_counts.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y} observasi<extra></extra>"
        ))
        fig_dist.update_layout(
            **PLOT_STYLE, height=260, showlegend=False,
            yaxis_title="Jumlah Observasi", xaxis_title="Status ILSK"
        )
        st.plotly_chart(fig_dist, use_container_width=True, theme=None)

        # ── Tabel Top-5 Periode Risiko Tertinggi ───────────────
        st.markdown(
            f'<p class="sec-title">'
            f'{ico("alert-triangle")} 5 Periode Risiko Tertinggi</p>',
            unsafe_allow_html=True
        )

        top5 = (df_ilsk
                .nlargest(5, "ILSK")
                [["Tanggal", "NDVI", "SAR_VH", "ONI", "ILSK", "status"]]
                .copy())
        top5["Tanggal"] = top5["Tanggal"].dt.strftime("%d %b %Y")
        top5 = top5.rename(columns={
            "Tanggal": "Tanggal Observasi",
            "NDVI":    "NDVI",
            "SAR_VH":  "SAR VH (dB)",
            "ONI":     "ONI",
            "ILSK":    "Skor ILSK",
            "status":  "Status"
        })

        def _style_status(val):
            if val == "Bahaya":
                return "background-color:#fef2f2;color:#b91c1c;font-weight:700"
            elif val == "Waspada":
                return "background-color:#fefce8;color:#92400e;font-weight:700"
            return "background-color:#f0fdf4;color:#15803d;font-weight:700"

        st.dataframe(
            top5.style
                .format({
                    "NDVI":        "{:.3f}",
                    "SAR VH (dB)": "{:.1f}",
                    "ONI":         "{:+.2f}",
                    "Skor ILSK":   "{:.3f}"
                })
                .map(_style_status, subset=["Status"])
                .set_properties(**{"font-size": "0.88rem"}),
            use_container_width=True,
            hide_index=True
        )

    # ================================================================
    #  SECTION 4 — DATA MENTAH (di dalam expander)
    # ================================================================
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.expander("Lihat Data Mentah Satelit"):
        st.markdown(
            f'<p class="sec-title" style="margin-top:0">'
            f'{ico("leaf")} NDVI Sentinel-2 — Nilai Asli</p>',
            unsafe_allow_html=True
        )
        if df_ndvi is not None:
            st.caption("Rentang nilai sehat padi: 0.60 - 0.80")
            st.plotly_chart(fig_ndvi(df_ndvi),
                            use_container_width=True, theme=None)
        else:
            st.info("Data NDVI belum tersedia.")

        st.markdown(
            f'<p class="sec-title">'
            f'{ico("satellite")} SAR Sentinel-1 VH — Nilai Asli (dB)</p>',
            unsafe_allow_html=True
        )
        if df_sar is not None:
            st.caption(
                "Nilai lebih tinggi (kurang negatif) menandakan lahan lebih kering. "
                "Tipikal sawah normal: -17 hingga -12 dB."
            )
            st.plotly_chart(fig_sar(df_sar),
                            use_container_width=True, theme=None)
        else:
            st.info("Data SAR Sentinel-1 belum tersedia.")

        st.markdown(
            f'<p class="sec-title">'
            f'{ico("wave")} ENSO-ONI — Nilai Asli</p>',
            unsafe_allow_html=True
        )
        st.caption(
            "Data historis ONI dari NOAA/CPC sebelum diolah "
            "menjadi skor tekanan iklim."
        )
        st.plotly_chart(fig_oni(df_for_enso, n=36),
                        use_container_width=True, theme=None)

# ════════════════════════════════════════════════════
#  TAB 3 — KENALI HAMA & PENYAKIT
# ════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<p class="sec-title">{ico("bug")} Kenali Penyakit Padi yang Umum</p>',
                unsafe_allow_html=True)
    render_hama_grid(HAMA)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("camera")} Hama yang Butuh Bantuan Foto Anda</p>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="rekom biru">
      <div class="rk-title">{ico('shield-check')} Kenapa Ini Penting?</div>
      <div class="rk-text">
        Data foto untuk hama-hama di bawah ini masih sangat terbatas di Indonesia, padahal
        prakiraan serangannya termasuk yang terluas. Kenali ciri-cirinya, lalu kirim foto
        kondisi tanaman Anda lewat tab <b>Informasi → Bantu Kumpulkan Foto Hama</b> untuk
        membantu AI kami belajar mengenalinya.
      </div>
    </div>
    """, unsafe_allow_html=True)
    render_hama_grid(HAMA_CS)


# ════════════════════════════════════════════════════
#  TAB 4 — CEK HAMA
# ════════════════════════════════════════════════════
with tab4:
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
                    conf_pct = round(hasil["confidence"] * 100, 1)

                    # ── Foto tidak teridentifikasi sebagai padi ──────────────
                    if hasil["kelas"] == "unknown":
                        st.markdown(f"""
                        <div class="big-card waspada">
                          <div class="bc-label">Foto Tidak Teridentifikasi</div>
                          <div class="bc-row">
                            <span class="bc-dot waspada"></span>
                            <span class="bc-value" style="font-size:1.2rem;">Bukan Foto Padi</span>
                          </div>
                          <div class="bc-sub">
                                Keyakinan tertinggi model: <b>{conf_pct}%</b> —
                                terlalu rendah untuk diagnosis yang dapat dipercaya.
                          </div>
                        </div>
                        <div class="rekom kuning">
                          <div class="rk-title">{ico('camera')} Tips Foto yang Baik</div>
                          <div class="rk-text">{hasil['rekomendasi']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Hasil deteksi normal ─────────────────────────────────
                    else:
                        kls_card = "bahaya" if hasil["berbahaya"] else "waspada"
                        kls_rk   = "merah"  if hasil["berbahaya"] else "kuning"

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

                        # ── Konsultasi AI — hanya muncul kalau bukan unknown ──
                        st.markdown('<hr class="divider">', unsafe_allow_html=True)
                        st.markdown(f'<p class="sec-title">{ico("bot")} Konsultasi AI — Penanganan</p>',
                                    unsafe_allow_html=True)

                        opt_id    = CLASS_TO_OPT.get(hasil["kelas"])
                        ds_client = get_ai_client()
                        # ... sisa kode konsultasi AI yang sudah ada tetap di sini ...

                    # ── Konsultasi AI — gabungan Offline (TaniBot) & Online (AgriAlert AI) ──
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.markdown(f'<p class="sec-title">{ico("bot")} Konsultasi AI — Penanganan</p>',
                                unsafe_allow_html=True)

                    opt_id    = CLASS_TO_OPT.get(hasil["kelas"])
                    ds_client = get_ai_client()
                    mode_pilihan = []
                    if opt_id and tanibot:
                        mode_pilihan.append("offline")
                    if ds_client is not None:
                        mode_pilihan.append("online")

                    if not mode_pilihan:
                        st.info("Konsultasi AI belum aktif — bangun database TaniBot dengan "
                                "`tanibot_db_builder_v2.py` atau atur `GROQ_API_KEY` di Secrets.")
                    else:
                        mode_key = f"ai_mode_{hasil['kelas']}"
                        if st.session_state.get(mode_key) not in mode_pilihan:
                            st.session_state[mode_key] = mode_pilihan[0]

                        if len(mode_pilihan) > 1:
                            mode = st.radio(
                                "Pilih mode konsultasi:",
                                mode_pilihan,
                                format_func=lambda m: "Offline — TaniBot (tanpa internet)"
                                    if m == "offline" else "Online — AgriAlert AI (lebih detail)",
                                horizontal=True,
                                index=mode_pilihan.index(st.session_state[mode_key]),
                                key=f"{mode_key}_radio",
                            )
                            st.session_state[mode_key] = mode
                        else:
                            mode = mode_pilihan[0]

                        # ---- MODE OFFLINE: TaniBot (pohon keputusan lokal) ----
                        if mode == "offline" and opt_id and tanibot:
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

                        # ---- MODE ONLINE: AgriAlert AI (Groq, butuh internet) ----
                        elif mode == "online" and ds_client is not None:
                            conf_ctx = round(hasil.get("confidence", 0) * 100, 1)
                            kls_ctx  = hasil.get("kelas", "")
                            st.markdown(f"""
                            <div class="rekom biru">
                              <div class="rk-title">{ico('camera')} Konteks Deteksi Aktif</div>
                              <div class="rk-text">AI mengetahui hasil foto kamu: <b>{hasil['nama_indo']}</b>
                              ({conf_ctx}% keyakinan). Kamu bisa langsung bertanya soal penanganan,
                              gejala, dosis pestisida, atau pencegahan.</div>
                            </div>
                            """, unsafe_allow_html=True)

                            sys_prompt = (
                                "Kamu adalah Penyuluh Pertanian Lapangan (PPL) senior yang sangat berpengalaman, ramah, dan membumi. "
                                "Bantu petani padi di Indonesia dengan bahasa yang santai, tidak kaku, dan mudah dipahami (jangan gunakan gaya bahasa AI atau robot). "
                                "Berikan saran praktis yang bisa langsung diterapkan di sawah berbasis Pengendalian Hama Terpadu (PHT). "
                                "Jika menyarankan pestisida/kimia, selalu sarankan pencegahan alami atau mekanis terlebih dahulu."
                                "Jawab dalam Bahasa Indonesia yang mudah dipahami petani biasa. "
                                "Berikan saran praktis dan berbasis ilmu pertanian yang valid."
                            )
                            if kls_ctx != "healthy":
                                sys_prompt += (
                                    f"\n\nKonteks aktif: tanaman padi pengguna terdeteksi mengidap "
                                    f"{hasil['nama_indo']} dengan keyakinan {conf_ctx}%. "
                                    f"Rekomendasi sistem: {hasil.get('rekomendasi', '')}. "
                                    "Gunakan konteks ini untuk jawaban yang lebih relevan."
                                )

                            if "chat_history" not in st.session_state:
                                st.session_state.chat_history = []

                            for msg in st.session_state.chat_history:
                                with st.chat_message(msg["role"]):
                                    st.write(msg["content"])

                            if prompt := st.chat_input(
                                "Tanya soal penyakit padi, penanganan, dosis pestisida...",
                                key=f"chat_input_{opt_id or kls_ctx}",
                            ):
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
                                if st.button("Hapus riwayat percakapan", type="secondary",
                                             key=f"clear_chat_{opt_id or kls_ctx}"):
                                    st.session_state.chat_history = []
                                    st.rerun()

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
        st.info("Belum tahu ciri-ciri hama & penyakit padi? Lihat tab "
                 "**Kenali Hama & Penyakit** untuk referensi lengkap sebelum mengunggah foto.")


# ════════════════════════════════════════════════════
#  TAB 5 — INFORMASI
# ════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<p class="sec-title">{ico("info-circle")} Tentang AgriAlert</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rekom biru">
      <div class="rk-title">{ico('book')} Tentang Aplikasi</div>
      <div class="rk-text">
        <b>AgriAlert</b> adalah sistem peringatan dini pertanian padi Indonesia yang
        menggabungkan data cuaca real-time BMKG, kondisi iklim global (ENSO),
        data satelit Sentinel-1/2, dan kecerdasan buatan untuk deteksi penyakit.<br><br>
        <b>Sumber data:</b> BMKG · NOAA/CPC · Sentinel-1 SAR · Sentinel-2 NDVI · BBPOPT<br>
        <b>Model AI:</b> CNN AgriAlert — 98% akurasi, 9 kelas penyakit daun padi<br>
        <b>Regulasi:</b> Permentan No. 39/2018 · UU PDP No. 27/2022<br>
        <b>Versi:</b> 1.0 · Satria Data 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("chart")} Status Sistem</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="mc-label">{ico('antenna')} Data BMKG</div>
        <div class="mc-val">{"Live" if cuaca["ok"] else "Estimasi"}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">{ico('wave')} Data ENSO</div>
        <div class="mc-val">{len(df_enso)} data</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">{ico('leaf')} Data NDVI</div>
        <div class="mc-val">{"Aktif" if df_ndvi is not None else "Tidak ada"}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">{ico('microscope')} Model CNN</div>
        <div class="mc-val">{"Siap" if model_cnn is not None else "Tidak ada"}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

    # ── Kotak Pengaduan — laporkan kesalahan deteksi AI ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("flag")} Laporkan Kesalahan Deteksi</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="rekom kuning">
      <div class="rk-title">{ico('alert-triangle')} Hasil Deteksi AI Tidak Sesuai?</div>
      <div class="rk-text">
        Jika AI salah mengenali penyakit atau hama dari foto Anda, beri tahu kami di sini.
        Laporan Anda dipakai untuk memperbaiki ketepatan model ke depannya.
      </div>
    </div>
    """, unsafe_allow_html=True)

    ctx_lapor = st.session_state.get("ai_context", {})
    with st.form("form_pengaduan", clear_on_submit=True):
        jenis_lapor = st.selectbox(
            "Jenis kesalahan:",
            [
                "Salah jenis penyakit / hama",
                "Tanaman sehat terdeteksi sakit",
                "Hasil tidak akurat / meragukan",
                "Lainnya",
            ],
        )
        deskripsi_lapor = st.text_area(
            "Ceritakan kesalahannya (opsional):",
            placeholder="Contoh: Foto daun sehat tapi hasilnya terdeteksi Blas Daun...",
        )
        kontak_lapor = st.text_input("Nomor HP / Email (opsional, agar bisa kami tindak lanjuti):")
        kirim_lapor  = st.form_submit_button("Kirim Laporan", use_container_width=True)

    if kirim_lapor:
        simpan_pengaduan(jenis_lapor, deskripsi_lapor, kontak_lapor, ctx_lapor.get("nama_indo", ""))
        st.success("Terima kasih, laporan Anda sudah kami terima dan akan ditinjau tim AgriAlert.")

    # ── Crowdsourcing foto hama (consent-based) ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{ico("camera")} Bantu Kumpulkan Foto Hama</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="rekom biru">
      <div class="rk-title">{ico('shield-check')} Kenapa Foto Anda Penting?</div>
      <div class="rk-text">
        Data foto untuk <b>penggerek batang</b> (gejala sundep dan beluk), <b>tikus</b>, dan
        <b>wereng batang coklat</b> masih sangat terbatas, padahal ketiganya termasuk hama
        dengan prakiraan serangan terluas di Indonesia. Kirim foto kondisi tanaman yang
        terserang agar AI AgriAlert bisa belajar mengenali hama ini lebih baik untuk semua petani.
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_crowdsource", clear_on_submit=True):
        jenis_hama_cs = st.selectbox(
            "Jenis hama pada foto:",
            [
                "Penggerek Batang — Sundep (pucuk kering, fase vegetatif)",
                "Penggerek Batang — Beluk (malai putih hampa, fase generatif)",
                "Tikus — kerusakan tanaman",
                "Wereng Batang Coklat — serangan pada batang/rumpun",
                "Lainnya",
            ],
        )
        foto_cs    = st.file_uploader("Unggah foto kondisi tanaman:", type=["jpg", "jpeg", "png"])
        catatan_cs = st.text_area("Catatan tambahan (umur tanaman, lokasi sawah, dll — opsional):")
        setuju_cs  = st.checkbox(
            "Saya setuju foto ini disimpan dan dipakai AgriAlert untuk melatih ulang model AI "
            "agar lebih akurat mengenali hama ini. Foto tidak akan disebarluaskan untuk tujuan lain "
            "di luar pengembangan sistem ini."
        )
        kirim_cs = st.form_submit_button("Kirim Foto", use_container_width=True)

    if kirim_cs:
        if not foto_cs:
            st.warning("Pilih foto terlebih dahulu sebelum mengirim.")
        elif not setuju_cs:
            st.warning("Mohon centang persetujuan penggunaan foto sebelum mengirim.")
        else:
            ext = foto_cs.name.rsplit(".", 1)[-1].lower()
            simpan_crowdsource_foto(foto_cs.getvalue(), ext, jenis_hama_cs, provinsi, catatan_cs)
            st.success("Terima kasih! Foto Anda sudah tersimpan dan akan membantu melatih AI kami.")
