"""
Smart Crop Disease Detection & Advisory System - v2
SIH26131 - Government of Maharashtra

New in v2:
- Camera capture OR file upload
- Blur detection with limited re-upload prompts
- Confidence-tiered advice (high/moderate/low)
- Full single-language interface (not just advice text)
- Voice-over (text-to-speech) for advice
- Local-language disease names
- Approximate medicine cost range
- GIS hotspot map (DEMO data - clearly labeled)
- Government dashboard sync (DEMO/mockup - clearly labeled)

Run:  streamlit run app.py
"""

import json
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import subprocess
    import shutil as _shutil
    TTS_AVAILABLE = _shutil.which("espeak") is not None
    PICO_AVAILABLE = _shutil.which("pico2wave") is not None
except ImportError:
    TTS_AVAILABLE = False
    PICO_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


@st.cache_data(ttl=60)
def is_online() -> bool:
    """Quick connectivity check (cached for 60s so we're not re-checking on
    every rerun). Used to decide: online workflow (better voice via gTTS,
    live weather auto-fetch) vs offline workflow (local espeak/pico2wave,
    manual weather selection) - app works fully either way."""
    import socket
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_PATH = "model/crop_disease_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"
DISEASE_INFO_PATH = "disease_info.json"
IMG_SIZE = (224, 224)
BLUR_THRESHOLD = 100       # Laplacian variance below this = "blurry"
MAX_RETRY_PHOTOS = 3       # limit on how many re-uploads we ask for


st.set_page_config(page_title="AgroShield", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# AGROSHIELD BRAND / ACCESSIBILITY STYLING
# ---------------------------------------------------------------------------
# Pure local CSS: no external fonts, images, JS, or network dependency.
st.markdown("""
<style>
    :root {
        --agro-green: #176b3a;
        --agro-green-dark: #0f4f2b;
        --agro-leaf: #2f8f4e;
        --agro-mint: #eaf6ee;
        --agro-cream: #f8fbf7;
        --agro-ink: #173024;
        --agro-border: #d7e6da;
        --agro-shadow: 0 8px 24px rgba(23, 107, 58, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(47,143,78,.07), transparent 28%),
            linear-gradient(180deg, #fbfdfb 0%, #f4f9f5 100%);
        color: var(--agro-ink);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f8f2 0%, #e7f3ea 100%);
        border-right: 1px solid var(--agro-border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--agro-green-dark);
    }

    h1 {
        color: var(--agro-green-dark) !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: var(--agro-green-dark);
        font-weight: 750;
    }

    [data-testid="stTabs"] button[role="tab"] {
        font-weight: 700;
        color: #42614d;
        border-radius: 10px 10px 0 0;
        padding: 0.65rem 1rem;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--agro-green-dark);
        background: var(--agro-mint);
    }

    /* Cohesive branded buttons, including the icon-selection controls. */
    div[data-testid="stButton"] > button {
        border-radius: 14px;
        border: 1px solid var(--agro-border);
        min-height: 74px;
        background: rgba(255,255,255,.92);
        color: var(--agro-ink);
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(23,107,58,.05);
        transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        border-color: #9cc8a8;
        box-shadow: 0 7px 18px rgba(23,107,58,.11);
    }

    div[data-testid="stButton"] > button:focus {
        border-color: var(--agro-leaf);
        box-shadow: 0 0 0 3px rgba(47,143,78,.18);
    }

    div[data-testid="stButton"] > button p {
        white-space: pre-line !important;
        line-height: 1.25 !important;
        text-align: center !important;
    }

    /* Selected icon card. */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(180deg, #e8f6ec 0%, #d9efdf 100%);
        color: var(--agro-green-dark);
        border: 2px solid #4b9c63;
        box-shadow: 0 6px 18px rgba(47,143,78,.14);
    }

    /* Keep compact action buttons compact even though icon cards are tall. */
    div[data-testid="stButton"] > button[aria-label="🔊"],
    div[data-testid="stButton"] > button[aria-label="🔇"] {
        min-height: 48px;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--agro-border);
        border-radius: 16px;
        background: rgba(255,255,255,.70);
        box-shadow: var(--agro-shadow);
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--agro-border);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: var(--agro-shadow);
    }

    [data-testid="stFileUploader"],
    [data-testid="stCameraInput"] {
        border-radius: 14px;
    }

    .agro-section-label {
        margin: .45rem 0 .7rem 0;
        font-size: 1.02rem;
        font-weight: 750;
        color: var(--agro-green-dark);
    }

    .agro-hint {
        padding: .7rem .9rem;
        border-radius: 12px;
        background: var(--agro-mint);
        border: 1px solid var(--agro-border);
        color: #355643;
        font-size: .92rem;
    }

    footer { visibility: hidden; }

    /* Compact camera capture area */
    [data-testid="stCameraInput"] {
        max-width: 430px !important;
        margin: 0 auto !important;
    }
    [data-testid="stCameraInput"] > div {
        padding: 0 !important;
    }
    [data-testid="stCameraInput"] button {
        min-height: 54px !important;
        height: 54px !important;
        border-radius: 12px !important;
        padding: 0.45rem 0.8rem !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stCameraInput"] button svg {
        display: none !important;
    }
    [data-testid="stCameraInput"] button p::before {
        content: "📸  ";
        font-size: 1.25rem;
    }

    /* Match camera + upload box heights, and put icons INSIDE each field
       (rather than as separate labels above them). */
    [data-testid="stCameraInput"],
    [data-testid="stFileUploaderDropzone"] {
        min-height: 300px !important;
    }
    [data-testid="stCameraInput"] > div:first-child {
        min-height: 300px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"]::before {
        content: "🖼️";
        font-size: 2.6rem;
        display: block;
        margin-bottom: 0.4rem;
    }
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img {
        border-radius: 12px !important;
    }
    /* Replace the default video-camera placeholder icon (shown before the
       browser grants camera permission) with a photo-camera icon instead.
       :has(svg) scopes this to ONLY the placeholder state - once permission
       is granted and the live video feed replaces the placeholder (removing
       the svg), this rule naturally stops applying on its own. */
    [data-testid="stCameraInput"] > div:first-child:has(svg) svg {
        display: none !important;
    }
    [data-testid="stCameraInput"] > div:first-child:has(svg)::before {
        content: "📷";
        font-size: 3rem;
        display: block;
        text-align: center;
    }

    /* Reduce vertical gaps so the diagnosis page feels like a mobile-first app. */
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 1.5rem !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.45rem;
    }
    [data-testid="stExpander"] {
        border-radius: 16px !important;
        border: 1px solid var(--agro-border) !important;
        background: rgba(255,255,255,.72) !important;
    }
    .agro-section {
        padding: 12px 16px;
        border: 1px solid var(--agro-border);
        border-radius: 16px;
        background: rgba(255,255,255,.82);
        box-shadow: var(--agro-shadow);
        margin: 8px 0 12px 0;
    }
    .agro-section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--agro-green-dark);
        margin-bottom: 2px;
    }
    .agro-section-subtitle {
        color: #587062;
        font-size: .88rem;
        margin-bottom: 8px;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FULL UI TRANSLATION DICTIONARY (single-language interface, not just advice)
# ---------------------------------------------------------------------------
UI = {
    "English": {
        "title": "🌱 AgroShield",
        "sidebar_header": "How to use",
        "sidebar_body": "1. Take or upload a leaf photo\n2. Get instant diagnosis\n3. See advice, local name & cost\n4. Listen to advice (audio)\n5. View disease hotspots",
        "lang_label": "Language",
        "tab_diagnose": "🔍 Diagnose",
        "tab_hotspots": "🗺️ Disease Hotspots (Demo)",
        "tab_dashboard": "🏛️ Government Dashboard Sync",
        "upload_option": "Upload a photo",
        "camera_option": "Take a photo (camera)",
        "input_method": "How would you like to provide the image?",
        "upload_prompt": "Upload a leaf image",
        "camera_prompt": "Take a photo of the leaf",
        "context_header": "🧭 Field context (optional, improves advice)",
        "soil_label": "🧪 Soil type",
        "location_label": "📍 District",
        "state_label": "🗺️ State",
        "weather_label": "🌦️ Current weather",
        "weather_live_detected": "🌐 Live weather detected: {weather} (online)",
        "weather_override": "Override with manual selection",
        "weather_offline_note": "📡 Offline — select weather manually",
        "context_disclaimer": "This context adjusts the advisory text below — the AI diagnosis itself is based only on the leaf photo.",
        "crop_stage_label": "🌾 Crop growth stage",
        "variety_label": "🌱 Crop variety (optional)",
        "variety_placeholder": "e.g. Roma, Pusa Ruby...",
        "variety_voice_prompt": "🎤 Or record variety name",
        "variety_voice_unavailable": "(voice input needs a newer Streamlit version)",
        "variety_voice_recorded": "🎤 Voice note recorded and attached to your record (not transcribed — saved as audio for the agriculture officer to review).",
        "pest_history_label": "🐛 Local pest history at this field",
        "blurry_warning": "⚠️ This photo looks blurry. Please retake or upload a clearer photo.",
        "blurry_limit": "You've reached the retry limit. Proceeding with the best available photo — results may be less accurate.",
        "diagnosis_header": "Diagnosis",
        "prediction_label": "Prediction",
        "local_name_label": "Local name",
        "confidence_label": "Confidence",
        "confidence_high": "✅ High confidence diagnosis.",
        "confidence_moderate": "⚠️ Moderate confidence — advice below is a reasonable guide, but consider confirming with a second photo or local agri officer.",
        "confidence_low": "❗ Low confidence — please upload a clearer, closer photo of a single affected leaf for a reliable diagnosis.",
        "advice_header": "💊 Recommended action",
        "advice_header_soft": "🤔 Possible action (low confidence — verify with a clearer photo)",
        "severity_label": "⚠️ Severity",
        "cost_label": "💰 Approximate treatment cost",
        "cost_disclaimer": "(indicative range — actual prices vary by location and dealer; verify locally)",
        "cost_hidden_low_confidence": "💰 Treatment cost hidden — diagnosis confidence is too low for a reliable cost estimate. Get a clearer photo first.",
        "weather_risk_note": "🌧️ Current humid/rainy conditions increase disease spread risk — act promptly.",
        "soil_drainage_note": "🧪 In heavy/clay soils, improve drainage to reduce fungal recurrence.",
        "yield_risk_note": "🌾 Crop is in flowering/fruiting stage — disease at this stage directly risks yield loss; prioritize treatment.",
        "seedling_note": "🌱 Young seedlings are more vulnerable — monitor closely and treat early to protect plant establishment.",
        "pest_history_note": "🐛 This field has a history of frequent pest issues — consider preventive monitoring and rotating treatment types to avoid resistance buildup.",
        "variety_logged_note": "🌱 Variety noted for your records. Variety-specific resistance guidance is on our roadmap.",
        "listen_button": "🔊 Listen to advice",
        "stop_voice_button": "Stop voice-over",
        "replay_voice_button": "Play voice-over again",
        "no_model": "⚠️ No trained model found yet — this is an interface preview.",
        "no_advisory": "No advisory available for this class yet.",
        "gradcam_header": "🔍 Why the model thinks this (Explainable AI)",
        "qa_header": "❓ Ask about your result",
        "qa_btn_why": "🤔 Why this?",
        "qa_btn_severity": "⚠️ What's severity?",
        "qa_btn_confidence": "📊 What's confidence?",
        "qa_btn_disagree": "🙅 I disagree",
        "qa_q_why": "The photo was matched to '{cls}' with {pct:.0f}% confidence, based on visible patterns in the leaf compared to thousands of reference photos. Higher confidence means the pattern is a closer match.",
        "qa_q_severity": "Severity ({sev}) estimates how urgently this needs treatment: None means healthy, Moderate means treat soon, High means act immediately to prevent spread or yield loss.",
        "qa_q_confidence": "Confidence ({pct:.0f}%) shows how sure the AI is about this specific diagnosis. Below 50%, we hide cost estimates and soften the advice, since the result may not be reliable.",
        "qa_q_disagree": "If this doesn't match what you're seeing, try a clearer, closer photo in good light, or show the leaf to your local agriculture officer for a second opinion — AI diagnosis is a helpful first check, not a final verdict.",
        "hotspot_header": "Disease Hotspot Map",
        "hotspot_note": "⚠️ DEMO DATA — this shows simulated outbreak clusters for illustration. In production, this would pull live, aggregated detections from all app users across Maharashtra, geo-tagged by district.",
        "dashboard_header": "Sync to Government Agriculture Dashboard",
        "dashboard_note": "⚠️ DEMO / MOCKUP — no real government system is connected yet. This shows how a detection could be reported to the Maharashtra Krishi Vibhag (Agriculture Department) dashboard in a production deployment.",
        "dashboard_button": "Report this detection (Demo)",
        "dashboard_success": "✅ (Demo) Detection logged — in production this would sync to the district agriculture office dashboard for outbreak tracking.",
        "footer": "Not a substitute for professional agronomist advice.",
        "welcome_voice_text": "Welcome to AgroShield. First, add a leaf photo using the camera or upload button. Then, you can optionally fill in field information for better advice.",
    },
    "हिंदी (Hindi)": {
        "title": "🛡️🌱 AgroShield",
        "sidebar_header": "उपयोग कैसे करें",
        "sidebar_body": "1. पत्ती की फोटो लें या अपलोड करें\n2. तुरंत निदान प्राप्त करें\n3. सलाह, स्थानीय नाम और लागत देखें\n4. सलाह सुनें (ऑडियो)\n5. रोग हॉटस्पॉट देखें",
        "lang_label": "भाषा",
        "tab_diagnose": "🔍 निदान करें",
        "tab_hotspots": "🗺️ रोग हॉटस्पॉट (डेमो)",
        "tab_dashboard": "🏛️ सरकारी डैशबोर्ड सिंक",
        "upload_option": "फोटो अपलोड करें",
        "camera_option": "फोटो लें (कैमरा)",
        "input_method": "आप छवि कैसे प्रदान करना चाहेंगे?",
        "upload_prompt": "पत्ती की छवि अपलोड करें",
        "camera_prompt": "पत्ती की फोटो लें",
        "context_header": "🧭 खेत की जानकारी (वैकल्पिक, सलाह बेहतर बनाता है)",
        "soil_label": "🧪 मिट्टी का प्रकार",
        "location_label": "📍 जिला",
        "state_label": "🗺️ राज्य",
        "weather_label": "🌦️ वर्तमान मौसम",
        "weather_live_detected": "🌐 लाइव मौसम पता चला: {weather} (ऑनलाइन)",
        "weather_override": "मैन्युअल चयन से बदलें",
        "weather_offline_note": "📡 ऑफलाइन — मौसम मैन्युअल रूप से चुनें",
        "context_disclaimer": "यह जानकारी नीचे दी गई सलाह को समायोजित करती है — AI निदान केवल पत्ती की फोटो पर आधारित है।",
        "crop_stage_label": "🌾 फसल वृद्धि अवस्था",
        "variety_label": "🌱 फसल किस्म (वैकल्पिक)",
        "variety_placeholder": "जैसे रोमा, पूसा रूबी...",
        "variety_voice_prompt": "🎤 या किस्म का नाम रिकॉर्ड करें",
        "variety_voice_unavailable": "(आवाज़ इनपुट के लिए नया Streamlit संस्करण चाहिए)",
        "variety_voice_recorded": "🎤 आवाज़ नोट रिकॉर्ड की गई और आपके रिकॉर्ड से जोड़ी गई (लिखित नहीं की गई — कृषि अधिकारी की समीक्षा के लिए ऑडियो के रूप में सहेजी गई)।",
        "pest_history_label": "🐛 इस खेत में स्थानीय कीट इतिहास",
        "blurry_warning": "⚠️ यह फोटो धुंधली लग रही है। कृपया दोबारा लें या स्पष्ट फोटो अपलोड करें।",
        "blurry_limit": "आपने पुनः प्रयास सीमा तक पहुंच गए हैं। उपलब्ध सर्वश्रेष्ठ फोटो के साथ आगे बढ़ रहे हैं — परिणाम कम सटीक हो सकते हैं।",
        "diagnosis_header": "निदान",
        "prediction_label": "पूर्वानुमान",
        "local_name_label": "स्थानीय नाम",
        "confidence_label": "विश्वास स्तर",
        "confidence_high": "✅ उच्च विश्वास निदान।",
        "confidence_moderate": "⚠️ मध्यम विश्वास — नीचे दी गई सलाह एक उचित मार्गदर्शन है, लेकिन दूसरी फोटो या स्थानीय कृषि अधिकारी से पुष्टि करने पर विचार करें।",
        "confidence_low": "❗ कम विश्वास — विश्वसनीय निदान के लिए कृपया एक स्पष्ट, नज़दीकी फोटो अपलोड करें।",
        "advice_header": "💊 अनुशंसित कार्रवाई",
        "advice_header_soft": "🤔 संभावित कार्रवाई (कम विश्वास — स्पष्ट फोटो से सत्यापित करें)",
        "severity_label": "⚠️ गंभीरता",
        "cost_label": "💰 अनुमानित उपचार लागत",
        "cost_disclaimer": "(सांकेतिक सीमा — वास्तविक कीमतें स्थान और डीलर के अनुसार भिन्न होती हैं; स्थानीय रूप से सत्यापित करें)",
        "cost_hidden_low_confidence": "💰 उपचार लागत छिपाई गई — विश्वसनीय लागत अनुमान के लिए निदान विश्वास बहुत कम है। पहले एक स्पष्ट फोटो लें।",
        "weather_risk_note": "🌧️ वर्तमान नम/बारिश की स्थिति रोग फैलने का खतरा बढ़ाती है — तुरंत कार्रवाई करें।",
        "soil_drainage_note": "🧪 भारी/चिकनी मिट्टी में, फफूंद की पुनरावृत्ति कम करने के लिए जल निकासी में सुधार करें।",
        "yield_risk_note": "🌾 फसल फूल/फल अवस्था में है — इस अवस्था में रोग सीधे उपज हानि का खतरा बढ़ाता है; उपचार को प्राथमिकता दें।",
        "seedling_note": "🌱 युवा पौध अधिक संवेदनशील होते हैं — बारीकी से निगरानी करें और पौधों की स्थापना की सुरक्षा के लिए जल्दी उपचार करें।",
        "pest_history_note": "🐛 इस खेत में बार-बार कीट समस्याओं का इतिहास है — निवारक निगरानी और प्रतिरोध से बचने के लिए उपचार प्रकार बदलने पर विचार करें।",
        "variety_logged_note": "🌱 किस्म आपके रिकॉर्ड के लिए नोट की गई। किस्म-विशिष्ट प्रतिरोध मार्गदर्शन हमारी योजना में है।",
        "listen_button": "🔊 सलाह सुनें",
        "stop_voice_button": "आवाज़ रोकें",
        "replay_voice_button": "आवाज़ फिर से चलाएं",
        "no_model": "⚠️ अभी तक कोई प्रशिक्षित मॉडल नहीं मिला — यह इंटरफ़ेस पूर्वावलोकन है।",
        "no_advisory": "इस श्रेणी के लिए अभी सलाह उपलब्ध नहीं है।",
        "gradcam_header": "🔍 मॉडल ऐसा क्यों सोचता है (व्याख्या योग्य AI)",
        "qa_header": "❓ अपने परिणाम के बारे में पूछें",
        "qa_btn_why": "🤔 ऐसा क्यों?",
        "qa_btn_severity": "⚠️ गंभीरता क्या है?",
        "qa_btn_confidence": "📊 विश्वास क्या है?",
        "qa_btn_disagree": "🙅 मैं असहमत हूं",
        "qa_q_why": "फोटो को '{cls}' से {pct:.0f}% विश्वास के साथ मिलाया गया, पत्ती में दिखाई देने वाले पैटर्न की हजारों संदर्भ फोटो से तुलना के आधार पर। अधिक विश्वास का मतलब है करीबी मिलान।",
        "qa_q_severity": "गंभीरता ({sev}) बताती है कि इसका इलाज कितनी तत्काल जरूरत है: कोई नहीं का मतलब स्वस्थ, मध्यम का मतलब जल्द इलाज करें, उच्च का मतलब फैलाव या उपज हानि रोकने के लिए तुरंत कार्रवाई करें।",
        "qa_q_confidence": "विश्वास ({pct:.0f}%) दिखाता है कि AI इस निदान के बारे में कितना निश्चित है। 50% से नीचे, हम लागत अनुमान छिपाते हैं और सलाह को नरम करते हैं, क्योंकि परिणाम विश्वसनीय नहीं हो सकता।",
        "qa_q_disagree": "यदि यह आपको दिख रही चीज़ से मेल नहीं खाता, तो अच्छी रोशनी में एक स्पष्ट, नज़दीकी फोटो आज़माएं, या दूसरी राय के लिए अपने स्थानीय कृषि अधिकारी को पत्ती दिखाएं — AI निदान एक सहायक पहली जांच है, अंतिम फैसला नहीं।",
        "hotspot_header": "रोग हॉटस्पॉट मानचित्र",
        "hotspot_note": "⚠️ डेमो डेटा — यह उदाहरण के लिए सिम्युलेटेड प्रकोप समूह दिखाता है। उत्पादन में, यह महाराष्ट्र भर के सभी ऐप उपयोगकर्ताओं से लाइव, एकत्रित पहचान खींचेगा।",
        "dashboard_header": "सरकारी कृषि डैशबोर्ड से सिंक करें",
        "dashboard_note": "⚠️ डेमो / मॉकअप — अभी कोई वास्तविक सरकारी प्रणाली जुड़ी नहीं है। यह दिखाता है कि उत्पादन में एक पहचान को महाराष्ट्र कृषि विभाग डैशबोर्ड को कैसे रिपोर्ट किया जा सकता है।",
        "dashboard_button": "इस पहचान की रिपोर्ट करें (डेमो)",
        "dashboard_success": "✅ (डेमो) पहचान दर्ज की गई — उत्पादन में यह प्रकोप ट्रैकिंग के लिए जिला कृषि कार्यालय डैशबोर्ड से सिंक होगा।",
        "footer": "पेशेवर कृषि विशेषज्ञ सलाह का विकल्प नहीं।",
        "welcome_voice_text": "AgroShield में आपका स्वागत है। पहले, कैमरा या अपलोड बटन का उपयोग करके पत्ती की फोटो जोड़ें। फिर, बेहतर सलाह के लिए वैकल्पिक रूप से खेत की जानकारी भरें।",
    },
    "मराठी (Marathi)": {
        "title": "🛡️🌱 AgroShield",
        "sidebar_header": "कसे वापरावे",
        "sidebar_body": "1. पानाचा फोटो घ्या किंवा अपलोड करा\n2. त्वरित निदान मिळवा\n3. सल्ला, स्थानिक नाव व खर्च पहा\n4. सल्ला ऐका (ऑडिओ)\n5. रोग हॉटस्पॉट पहा",
        "lang_label": "भाषा",
        "tab_diagnose": "🔍 निदान करा",
        "tab_hotspots": "🗺️ रोग हॉटस्पॉट (डेमो)",
        "tab_dashboard": "🏛️ शासकीय डॅशबोर्ड सिंक",
        "upload_option": "फोटो अपलोड करा",
        "camera_option": "फोटो घ्या (कॅमेरा)",
        "input_method": "तुम्ही प्रतिमा कशी द्याल?",
        "upload_prompt": "पानाची प्रतिमा अपलोड करा",
        "camera_prompt": "पानाचा फोटो घ्या",
        "context_header": "🧭 शेताची माहिती (ऐच्छिक, सल्ला सुधारते)",
        "soil_label": "🧪 मातीचा प्रकार",
        "location_label": "📍 जिल्हा",
        "state_label": "🗺️ राज्य",
        "weather_label": "🌦️ सध्याचे हवामान",
        "weather_live_detected": "🌐 थेट हवामान आढळले: {weather} (ऑनलाइन)",
        "weather_override": "मॅन्युअल निवडीने बदला",
        "weather_offline_note": "📡 ऑफलाइन — हवामान मॅन्युअली निवडा",
        "context_disclaimer": "ही माहिती खालील सल्ला समायोजित करते — AI निदान फक्त पानाच्या फोटोवर आधारित आहे.",
        "crop_stage_label": "🌾 पिकाची वाढीची अवस्था",
        "variety_label": "🌱 पिकाची जात (ऐच्छिक)",
        "variety_placeholder": "उदा. रोमा, पुसा रुबी...",
        "variety_voice_prompt": "🎤 किंवा जातीचे नाव रेकॉर्ड करा",
        "variety_voice_unavailable": "(आवाज इनपुटसाठी नवीन Streamlit आवृत्ती आवश्यक)",
        "variety_voice_recorded": "🎤 आवाज नोंद रेकॉर्ड केली आणि तुमच्या नोंदीशी जोडली (लिखित केलेली नाही — कृषी अधिकाऱ्याच्या पुनरावलोकनासाठी ऑडिओ म्हणून जतन केली).",
        "pest_history_label": "🐛 या शेतातील स्थानिक कीड इतिहास",
        "blurry_warning": "⚠️ हा फोटो अस्पष्ट दिसत आहे. कृपया पुन्हा घ्या किंवा स्पष्ट फोटो अपलोड करा.",
        "blurry_limit": "तुम्ही पुन्हा-प्रयत्न मर्यादेपर्यंत पोहोचला आहात. उपलब्ध सर्वोत्तम फोटोसह पुढे जात आहोत — निकाल कमी अचूक असू शकतात.",
        "diagnosis_header": "निदान",
        "prediction_label": "अंदाज",
        "local_name_label": "स्थानिक नाव",
        "confidence_label": "विश्वास पातळी",
        "confidence_high": "✅ उच्च विश्वासार्ह निदान.",
        "confidence_moderate": "⚠️ मध्यम विश्वास — खालील सल्ला योग्य मार्गदर्शक आहे, परंतु दुसऱ्या फोटोने किंवा स्थानिक कृषी अधिकाऱ्याकडून खात्री करून घ्या.",
        "confidence_low": "❗ कमी विश्वास — विश्वसनीय निदानासाठी कृपया स्पष्ट, जवळचा फोटो अपलोड करा.",
        "advice_header": "💊 शिफारस केलेली कृती",
        "advice_header_soft": "🤔 संभाव्य कृती (कमी विश्वास — स्पष्ट फोटोने खात्री करा)",
        "severity_label": "⚠️ तीव्रता",
        "cost_label": "💰 अंदाजे उपचार खर्च",
        "cost_disclaimer": "(सूचक श्रेणी — प्रत्यक्ष किंमती ठिकाण व विक्रेत्यानुसार बदलतात; स्थानिक पातळीवर खात्री करा)",
        "cost_hidden_low_confidence": "💰 उपचार खर्च लपवला — विश्वसनीय खर्च अंदाजासाठी निदान विश्वास खूप कमी आहे. आधी स्पष्ट फोटो घ्या.",
        "weather_risk_note": "🌧️ सध्याची दमट/पावसाळी स्थिती रोग पसरण्याचा धोका वाढवते — त्वरित कृती करा.",
        "soil_drainage_note": "🧪 जड/चिकणमाती जमिनीत, बुरशीची पुनरावृत्ती कमी करण्यासाठी निचरा सुधारा.",
        "yield_risk_note": "🌾 पीक फुलोरा/फळधारणा अवस्थेत आहे — या अवस्थेत रोगामुळे उत्पन्न हानीचा थेट धोका असतो; उपचारास प्राधान्य द्या.",
        "seedling_note": "🌱 लहान रोपे अधिक संवेदनशील असतात — बारकाईने निरीक्षण करा आणि रोपांच्या स्थापनेचे संरक्षण करण्यासाठी लवकर उपचार करा.",
        "pest_history_note": "🐛 या शेतात वारंवार कीड समस्यांचा इतिहास आहे — प्रतिबंधात्मक निरीक्षण आणि प्रतिकार टाळण्यासाठी उपचार प्रकार बदलण्याचा विचार करा.",
        "variety_logged_note": "🌱 जात तुमच्या नोंदीसाठी नमूद केली. जात-विशिष्ट प्रतिकार मार्गदर्शन आमच्या योजनेत आहे.",
        "listen_button": "🔊 सल्ला ऐका",
        "stop_voice_button": "आवाज थांबवा",
        "replay_voice_button": "आवाज पुन्हा ऐका",
        "no_model": "⚠️ अद्याप प्रशिक्षित मॉडेल सापडले नाही — हे इंटरफेस पूर्वावलोकन आहे.",
        "no_advisory": "या वर्गासाठी अद्याप सल्ला उपलब्ध नाही.",
        "gradcam_header": "🔍 मॉडेल असे का विचार करते (स्पष्टीकरणीय AI)",
        "qa_header": "❓ तुमच्या निकालाबद्दल विचारा",
        "qa_btn_why": "🤔 असे का?",
        "qa_btn_severity": "⚠️ तीव्रता म्हणजे काय?",
        "qa_btn_confidence": "📊 विश्वास म्हणजे काय?",
        "qa_btn_disagree": "🙅 मी असहमत आहे",
        "qa_q_why": "फोटो '{cls}' शी {pct:.0f}% विश्वासाने जुळवला गेला, पानावरील दिसणाऱ्या नमुन्यांची हजारो संदर्भ फोटोंशी तुलना करून. जास्त विश्वास म्हणजे जवळचे जुळणे.",
        "qa_q_severity": "तीव्रता ({sev}) सांगते की उपचाराची किती तातडीने गरज आहे: काहीही नाही म्हणजे निरोगी, मध्यम म्हणजे लवकर उपचार करा, उच्च म्हणजे प्रसार किंवा उत्पन्न हानी टाळण्यासाठी त्वरित कृती करा.",
        "qa_q_confidence": "विश्वास ({pct:.0f}%) दाखवतो की AI या निदानाबद्दल किती खात्री आहे. 50% पेक्षा कमी असल्यास, आम्ही खर्च अंदाज लपवतो आणि सल्ला सौम्य करतो, कारण निकाल विश्वसनीय नसू शकतो.",
        "qa_q_disagree": "जर हे तुम्हाला दिसत असलेल्याशी जुळत नसेल, तर चांगल्या प्रकाशात स्पष्ट, जवळचा फोटो घ्या, किंवा दुसऱ्या मताासाठी तुमच्या स्थानिक कृषी अधिकाऱ्याला पान दाखवा — AI निदान ही उपयुक्त पहिली तपासणी आहे, अंतिम निर्णय नाही.",
        "hotspot_header": "रोग हॉटस्पॉट नकाशा",
        "hotspot_note": "⚠️ डेमो डेटा — हे उदाहरणासाठी सिम्युलेटेड उद्रेक गट दाखवते. उत्पादनात, हे संपूर्ण महाराष्ट्रातील सर्व अ‍ॅप वापरकर्त्यांकडून थेट, एकत्रित शोध घेईल.",
        "dashboard_header": "शासकीय कृषी डॅशबोर्डशी सिंक करा",
        "dashboard_note": "⚠️ डेमो / मॉकअप — अद्याप कोणतीही वास्तविक शासकीय प्रणाली जोडलेली नाही. उत्पादनात एक शोध महाराष्ट्र कृषी विभाग डॅशबोर्डला कसा कळवला जाऊ शकतो हे हे दाखवते.",
        "dashboard_button": "हा शोध कळवा (डेमो)",
        "dashboard_success": "✅ (डेमो) शोध नोंदवला — उत्पादनात हे उद्रेक ट्रॅकिंगसाठी जिल्हा कृषी कार्यालय डॅशबोर्डशी सिंक होईल.",
        "footer": "व्यावसायिक कृषी सल्लागाराच्या सल्ल्याचा पर्याय नाही.",
        "welcome_voice_text": "AgroShield मध्ये आपले स्वागत आहे. प्रथम, कॅमेरा किंवा अपलोड बटण वापरून पानाचा फोटो जोडा. नंतर, चांगल्या सल्ल्यासाठी ऐच्छिकपणे शेताची माहिती भरा.",
    },
}

# ---------------------------------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


@st.cache_data
def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_disease_info():
    # NOT cached, so a restarted server always picks up the latest file.
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


MAX_DIMENSION = 1600  # auto-downscale any side larger than this


def auto_resize_image(pil_image: Image.Image) -> Image.Image:
    """Automatically downscales large photos instead of asking the user to
    compress them first - no size constraint from the user's side. Keeps
    aspect ratio. Does nothing if the image is already small enough."""
    img = pil_image.copy()
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    return img


def is_blurry(pil_image: Image.Image) -> bool:
    """Laplacian-variance blur check. Falls back to 'not blurry' if OpenCV
    isn't installed, so the app still works without it."""
    if not CV2_AVAILABLE:
        return False
    img = np.array(pil_image.convert("L"))  # grayscale
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    import matplotlib.cm as cm
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(pil_image.size)
    colored = cm.get_cmap("jet")(np.array(heatmap_img) / 255.0)[:, :, :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(pil_image.convert("RGB"), colored, alpha)


def text_to_speech_bytes(text: str, lang_code: str) -> tuple:
    """Hybrid online/offline TTS. Returns (audio_bytes, mime_type) or
    (None, None) on failure.
    - ONLINE workflow: tries gTTS first (Google's TTS) - noticeably clearer
      and more natural for Hindi/Marathi than the offline engines.
    - OFFLINE workflow (or if gTTS fails): falls back to pico2wave for
      English (clear, natural) and espeak for Hindi/Marathi (more robotic
      but works with zero internet).
    This means voice quality quietly improves when you have signal, and
    never breaks when you don't."""
    import tempfile, os as _os

    if is_online() and GTTS_AVAILABLE:
        try:
            tts = gTTS(text=text, lang=lang_code)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
            tts.save(temp_path)
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
            _os.remove(temp_path)
            if audio_bytes:
                return audio_bytes, "audio/mp3"
        except Exception:
            pass  # fall through to offline path below

    if not TTS_AVAILABLE:
        return None, None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        if lang_code == "en" and PICO_AVAILABLE:
            result = subprocess.run(
                ["pico2wave", "-l", "en-US", "-w", temp_path, text],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                # pico2wave failed (e.g. text too long/special chars) - fall back to espeak
                subprocess.run(["espeak", "-w", temp_path, text], capture_output=True, timeout=30)
        else:
            espeak_voice = {"en": "en", "hi": "hi", "mr": "mr"}.get(lang_code, "en")
            result = subprocess.run(
                ["espeak", "-v", espeak_voice, "-w", temp_path, text],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                subprocess.run(["espeak", "-w", temp_path, text], capture_output=True, timeout=30)

        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        _os.remove(temp_path)
        return (audio_bytes, "audio/wav") if audio_bytes else (None, None)
    except Exception:
        return None, None


def advice_to_bullets(advice: str) -> list[str]:
    """Split an advice string into clean bullet points. Splits on sentence
    boundaries (period + space) so existing disease_info.json entries work
    as-is, no need to rewrite all 38 entries into list format."""
    if not advice:
        return []
    raw_parts = advice.replace("।", ".").split(". ")
    bullets = [p.strip().rstrip(".") for p in raw_parts if p.strip()]
    return bullets


def severity_bar_html(severity: str) -> str:
    """Renders a horizontal severity scale with a marker line positioned
    according to severity - pure CSS, no images/fonts needed, works fully
    offline. (Kept as an alternate visual; the app now primarily uses
    severity_dots_html below.)"""
    position = {"None": 5, "Moderate": 50, "High": 90, "Unknown": 50}.get(severity, 50)
    color = {"None": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c", "Unknown": "#95a5a6"}.get(severity, "#95a5a6")
    return f"""
    <div style="width:100%; margin-top:6px;">
      <div style="position:relative; height:10px; border-radius:5px;
                  background:linear-gradient(to right, #2ecc71, #f39c12, #e74c3c);">
        <div style="position:absolute; left:{position}%; top:-4px; transform:translateX(-50%);
                    width:4px; height:18px; background:{color}; border:1px solid #333; border-radius:2px;"></div>
      </div>
    </div>
    """


def severity_dots_html(severity: str) -> str:
    """Three dots (green/yellow/red) with the active severity level
    enlarged and glowing - offline, pure CSS/HTML."""
    active = {"None": "green", "Moderate": "yellow", "High": "red"}.get(severity, None)
    colors = {"green": "#2ecc71", "yellow": "#f1c40f", "red": "#e74c3c"}
    dots_html = ""
    for key, color in colors.items():
        is_active = key == active
        style = (
            f"width:18px; height:18px; border-radius:50%; background:{color}; display:inline-block; margin-right:10px;"
            + (" opacity:1; transform:scale(1.3); box-shadow:0 0 6px rgba(0,0,0,0.3); border:2px solid #333;"
               if is_active else " opacity:0.25;")
        )
        dots_html += f'<span style="{style}"></span>'
    return f'<div style="margin-top:4px;">{dots_html}<span style="font-weight:600; margin-left:6px;">{severity}</span></div>'


def confidence_pie_html(confidence: float) -> str:
    """Small pie/donut chart showing confidence % as a green fill - pure
    CSS conic-gradient, no JS/images needed, works fully offline."""
    pct = max(0, min(100, confidence))
    return f"""
    <div style="width:64px; height:64px; border-radius:50%;
                background: conic-gradient(#2ecc71 0% {pct}%, #e8e8e8 {pct}% 100%);
                display:flex; align-items:center; justify-content:center;">
      <div style="width:46px; height:46px; border-radius:50%; background:white;
                  display:flex; align-items:center; justify-content:center;
                  font-size:13px; font-weight:700; color:#2ecc71;">{pct:.0f}%</div>
    </div>
    """


def icon_button_row(
    state_key: str,
    options: list[tuple],
    default: str,
    lang_for_voice: str = None,
    spoken_labels: dict = None,
) -> str:
    """Render accessible icon cards with the option name underneath.

    Each option can be (value, icon, display_label). For backward
    compatibility, (value, icon) is also accepted and the value is used
    as the display label.

    When a farmer taps a card, the selected option is spoken aloud in the
    current interface language. The audio is generated before the rerun and
    stored in session state, so it survives Streamlit's rerun and actually
    plays after the click.
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = default

    cols = st.columns(len(options))
    just_clicked_value = None

    for col, option in zip(cols, options):
        value, icon = option[:2]
        display_label = option[2] if len(option) >= 3 else value

        with col:
            is_selected = st.session_state[state_key] == value
            button_label = f"{icon}\n{display_label}"
            if st.button(
                button_label,
                key=f"{state_key}_{value}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state[state_key] = value
                just_clicked_value = value

    if just_clicked_value and TTS_AVAILABLE and lang_for_voice:
        spoken_text = (spoken_labels or {}).get(just_clicked_value, just_clicked_value)
        audio_bytes, mime_type = text_to_speech_bytes(
            spoken_text, LANG_TTS_CODE.get(lang_for_voice, "en")
        )
        if audio_bytes:
            st.session_state[f"selection_audio_{state_key}"] = (audio_bytes, mime_type)

    # Render the just-generated audio in the SAME run. Do not rerun here:
    # rerunning immediately can interrupt browser autoplay before it starts.
    selection_audio_key = f"selection_audio_{state_key}"
    if selection_audio_key in st.session_state:
        audio_bytes, mime_type = st.session_state[selection_audio_key]
        if audio_bytes:
            import base64
            b64 = base64.b64encode(audio_bytes).decode()
            st.markdown(
                f'<audio autoplay style="display:none"><source src="data:{mime_type};base64,{b64}" type="{mime_type}"></audio>',
                unsafe_allow_html=True,
            )
        # It will be regenerated on the next click, but we remove it now so
        # ordinary reruns (language changes, uploads, etc.) don't replay it.
        del st.session_state[selection_audio_key]

    return st.session_state[state_key]


LANG_TTS_CODE = {"English": "en", "हिंदी (Hindi)": "hi", "मराठी (Marathi)": "mr"}

# Options for the optional field-context dropdowns (offline, no live weather
# API call - farmer selects current conditions manually).
SOIL_TYPES = ["Black soil (Regur)", "Red soil", "Alluvial soil", "Laterite soil", "Sandy soil", "Clay soil"]
WEATHER_OPTIONS = ["Sunny/Dry", "Cloudy", "Humid", "Rainy"]
CROP_STAGES = ["Seedling", "Vegetative", "Flowering", "Fruiting/Maturity"]
PEST_HISTORY_OPTIONS = ["None reported", "Occasional", "Frequent"]

# Localized spoken labels for the voice-on-click confirmation - keyed by
# language, then by the option's internal English value.
SPOKEN_LABELS = {
    "English": {
        "Sunny/Dry": "Sunny, dry", "Cloudy": "Cloudy", "Humid": "Humid", "Rainy": "Rainy",
        "Black soil (Regur)": "Black soil", "Red soil": "Red soil", "Alluvial soil": "Alluvial soil",
        "Laterite soil": "Laterite soil", "Sandy soil": "Sandy soil", "Clay soil": "Clay soil",
        "Seedling": "Seedling", "Vegetative": "Vegetative", "Flowering": "Flowering", "Fruiting/Maturity": "Fruiting or maturity",
        "None reported": "None reported", "Occasional": "Occasional", "Frequent": "Frequent",
    },
    "हिंदी (Hindi)": {
        "Sunny/Dry": "धूप, सूखा", "Cloudy": "बादल", "Humid": "नम", "Rainy": "बारिश",
        "Black soil (Regur)": "काली मिट्टी", "Red soil": "लाल मिट्टी", "Alluvial soil": "जलोढ़ मिट्टी",
        "Laterite soil": "लैटेराइट मिट्टी", "Sandy soil": "रेतीली मिट्टी", "Clay soil": "चिकनी मिट्टी",
        "Seedling": "अंकुर अवस्था", "Vegetative": "वानस्पतिक अवस्था", "Flowering": "फूल अवस्था", "Fruiting/Maturity": "फल या पकने की अवस्था",
        "None reported": "कोई नहीं", "Occasional": "कभी-कभी", "Frequent": "बार-बार",
    },
    "मराठी (Marathi)": {
        "Sunny/Dry": "ऊन, कोरडे", "Cloudy": "ढगाळ", "Humid": "दमट", "Rainy": "पाऊस",
        "Black soil (Regur)": "काळी माती", "Red soil": "लाल माती", "Alluvial soil": "गाळाची माती",
        "Laterite soil": "जांभी माती", "Sandy soil": "वालुकामय माती", "Clay soil": "चिकण माती",
        "Seedling": "रोपटी अवस्था", "Vegetative": "वाढीची अवस्था", "Flowering": "फुलोरा अवस्था", "Fruiting/Maturity": "फळधारणा किंवा पक्वता",
        "None reported": "काहीही नाही", "Occasional": "अधूनमधून", "Frequent": "वारंवार",
    },
}
# Labels shown under the icons. These intentionally mirror the
# app's existing localized terminology so the spoken and visible names agree.
DISPLAY_LABELS = {
    "English": {
        "Sunny/Dry": "Sunny / Dry", "Cloudy": "Cloudy", "Humid": "Humid", "Rainy": "Rainy",
        "Black soil (Regur)": "Black soil", "Red soil": "Red soil", "Alluvial soil": "Alluvial soil",
        "Laterite soil": "Laterite soil", "Sandy soil": "Sandy soil", "Clay soil": "Clay soil",
        "Seedling": "Seedling", "Vegetative": "Vegetative", "Flowering": "Flowering",
        "Fruiting/Maturity": "Fruiting / Maturity",
        "None reported": "None reported", "Occasional": "Occasional", "Frequent": "Frequent",
    },
    "हिंदी (Hindi)": {
        "Sunny/Dry": "धूप / सूखा", "Cloudy": "बादल", "Humid": "नम", "Rainy": "बारिश",
        "Black soil (Regur)": "काली मिट्टी", "Red soil": "लाल मिट्टी", "Alluvial soil": "जलोढ़ मिट्टी",
        "Laterite soil": "लैटेराइट मिट्टी", "Sandy soil": "रेतीली मिट्टी", "Clay soil": "चिकनी मिट्टी",
        "Seedling": "अंकुर अवस्था", "Vegetative": "वानस्पतिक अवस्था", "Flowering": "फूल अवस्था",
        "Fruiting/Maturity": "फल / पकने की अवस्था",
        "None reported": "कोई नहीं", "Occasional": "कभी-कभी", "Frequent": "बार-बार",
    },
    "मराठी (Marathi)": {
        "Sunny/Dry": "ऊन / कोरडे", "Cloudy": "ढगाळ", "Humid": "दमट", "Rainy": "पाऊस",
        "Black soil (Regur)": "काळी माती", "Red soil": "लाल माती", "Alluvial soil": "गाळाची माती",
        "Laterite soil": "जांभी माती", "Sandy soil": "वालुकामय माती", "Clay soil": "चिकण माती",
        "Seedling": "रोपटी अवस्था", "Vegetative": "वाढीची अवस्था", "Flowering": "फुलोरा अवस्था",
        "Fruiting/Maturity": "फळधारणा / पक्वता",
        "None reported": "काहीही नाही", "Occasional": "अधूनमधून", "Frequent": "वारंवार",
    },
}

MIN_CONFIDENCE_FOR_COST = 50  # below this, hide cost estimate - showing a
                               # confident price when the model itself
                               # isn't confident could mislead the farmer

# State -> district mapping. Maharashtra is fully listed (the target state
# for this PS); other states have a representative starter list - expand
# as needed if you want full national coverage.
STATE_DISTRICTS = {
    "Maharashtra": ["Nashik", "Pune", "Nagpur", "Aurangabad", "Kolhapur", "Amravati", "Solapur", "Satara", "Other"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Other"],
    "Karnataka": ["Bengaluru", "Belagavi", "Mysuru", "Hubballi", "Other"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior", "Other"],
    "Other": ["Other"],
}

# Approximate district coordinates, for the ONLINE workflow's live weather
# fetch. Districts not listed here (or "Other") fall back to the OFFLINE
# workflow (manual weather selection) automatically.
DISTRICT_COORDS = {
    "Nashik": (20.0059, 73.7910), "Pune": (18.5204, 73.8567), "Nagpur": (21.1458, 79.0882),
    "Aurangabad": (19.8762, 75.3433), "Kolhapur": (16.7050, 74.2433), "Amravati": (20.9374, 77.7796),
    "Solapur": (17.6599, 75.9064), "Satara": (17.6805, 74.0183),
}


@st.cache_data(ttl=1800)
def fetch_live_weather(lat: float, lon: float) -> str:
    """ONLINE workflow: fetch real current weather via Open-Meteo (free,
    no API key needed) and map it to one of our 4 weather categories.
    Returns None if the fetch fails for any reason - caller should fall
    back to the OFFLINE workflow (manual icon selection) in that case."""
    try:
        import urllib.request, json as _json
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,relative_humidity_2m,cloud_cover"
        with urllib.request.urlopen(url, timeout=4) as response:
            data = _json.loads(response.read())
        current = data.get("current", {})
        precip = current.get("precipitation", 0)
        humidity = current.get("relative_humidity_2m", 0)
        cloud = current.get("cloud_cover", 0)
        if precip and precip > 0.5:
            return "Rainy"
        elif humidity and humidity > 75:
            return "Humid"
        elif cloud and cloud > 60:
            return "Cloudy"
        else:
            return "Sunny/Dry"
    except Exception:
        return None

# Demo hotspot data - Maharashtra district centroids with SIMULATED case counts.
# Clearly not real - for interface demonstration only.
DEMO_HOTSPOTS = pd.DataFrame({
    "district": ["Nashik", "Pune", "Nagpur", "Aurangabad", "Kolhapur", "Amravati"],
    "lat": [20.0059, 18.5204, 21.1458, 19.8762, 16.7050, 20.9374],
    "lon": [73.7910, 73.8567, 79.0882, 75.3433, 74.2433, 77.7796],
    "simulated_cases": [42, 28, 35, 19, 23, 31],
})

# ---------------------------------------------------------------------------
# APP STATE
# ---------------------------------------------------------------------------
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0
if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

with st.sidebar:
    lang = st.selectbox("Language / भाषा", list(UI.keys()))
    t = UI[lang]
    state = st.selectbox(t["state_label"], list(STATE_DISTRICTS.keys()))
    district = st.selectbox(t["location_label"], STATE_DISTRICTS[state])
    st.divider()
    st.header(t["sidebar_header"])
    st.write(t["sidebar_body"])

st.title(t["title"])

model = load_model()
class_names = load_class_names()
disease_info = load_disease_info()

tab1, tab2, tab3 = st.tabs([t["tab_diagnose"], t["tab_hotspots"], t["tab_dashboard"]])

# ---------------------------------------------------------------------------
# TAB 1: DIAGNOSE
# ---------------------------------------------------------------------------
with tab1:
    # Voice-guided welcome instruction - plays automatically once per
    # session so a user who can't read still knows what to do. This is the
    # "walkthrough" voice, separate from the diagnosis-advice voice.
    # Cache key includes a hash of the actual text+language, so editing the
    # welcome message (or switching language) always produces fresh audio
    # instead of silently reusing a stale cached clip from earlier in the
    # session.
    if TTS_AVAILABLE:
        welcome_text_hash = hashlib.md5((t["welcome_voice_text"] + lang).encode()).hexdigest()
        welcome_played_key = f"welcome_played_{welcome_text_hash}"
        if welcome_played_key not in st.session_state:
            st.session_state[welcome_played_key] = True
            welcome_key = f"welcome_audio_{welcome_text_hash}"
            if welcome_key not in st.session_state:
                st.session_state[welcome_key] = text_to_speech_bytes(t["welcome_voice_text"], LANG_TTS_CODE.get(lang, "en"))
            w_audio, w_mime = st.session_state[welcome_key]
            if w_audio:
                import base64
                b64_w = base64.b64encode(w_audio).decode()
                st.markdown(
                    f'<audio autoplay style="display:none"><source src="data:{w_mime};base64,{b64_w}" type="{w_mime}"></audio>',
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------
    # CAPTURE / UPLOAD (shown first)
    # Compact, centered camera area with a clear secondary upload option.
    # -------------------------------------------------------------------
    st.markdown(
        '<div class="agro-section">'
        '<div class="agro-section-title">📷 Add a leaf photo</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    camera_col, upload_col = st.columns(2, gap="medium")
    with camera_col:
        camera_file = st.camera_input("Camera", label_visibility="visible")
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )

    pil_image = None
    if camera_file is not None:
        pil_image = Image.open(camera_file)
    elif uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

    # -------------------------------------------------------------------
    # FIELD INFORMATION (shown second, after the photo)
    # -------------------------------------------------------------------
    st.markdown(
        '<div class="agro-section-title">🧭 Field information</div>',
        unsafe_allow_html=True,
    )
    with st.expander(t["context_header"], expanded=False):
        st.caption(t["context_disclaimer"])

        st.write(t["weather_label"])
        # Hybrid workflow: ONLINE + known district coordinates -> auto-fetch
        # live weather and show it read-only with a manual-override option.
        # OFFLINE (or unknown district / fetch fails) -> manual icon buttons.
        live_weather = None
        if is_online() and district in DISTRICT_COORDS:
            lat, lon = DISTRICT_COORDS[district]
            live_weather = fetch_live_weather(lat, lon)

        if live_weather:
            st.success(t["weather_live_detected"].format(weather=live_weather))
            use_manual_weather = st.checkbox(t["weather_override"], key="weather_override_check")
            if use_manual_weather:
                weather = icon_button_row("weather_choice", [
                    ("Sunny/Dry", "☀️", DISPLAY_LABELS[lang]["Sunny/Dry"]),
                    ("Cloudy", "☁️", DISPLAY_LABELS[lang]["Cloudy"]),
                    ("Humid", "💧", DISPLAY_LABELS[lang]["Humid"]),
                    ("Rainy", "🌧️", DISPLAY_LABELS[lang]["Rainy"]),
                ], default=None, lang_for_voice=lang, spoken_labels=SPOKEN_LABELS.get(lang, {}))
            else:
                weather = live_weather
        else:
            st.caption(t["weather_offline_note"])
            weather = icon_button_row("weather_choice", [
                ("Sunny/Dry", "☀️", DISPLAY_LABELS[lang]["Sunny/Dry"]),
                ("Cloudy", "☁️", DISPLAY_LABELS[lang]["Cloudy"]),
                ("Humid", "💧", DISPLAY_LABELS[lang]["Humid"]),
                ("Rainy", "🌧️", DISPLAY_LABELS[lang]["Rainy"]),
            ], default=None, lang_for_voice=lang, spoken_labels=SPOKEN_LABELS.get(lang, {}))

        st.write(t["soil_label"])
        soil_type = icon_button_row("soil_choice", [
            ("Black soil (Regur)", "⬛", DISPLAY_LABELS[lang]["Black soil (Regur)"]),
            ("Red soil", "🟥", DISPLAY_LABELS[lang]["Red soil"]),
            ("Alluvial soil", "🌊", DISPLAY_LABELS[lang]["Alluvial soil"]),
            ("Laterite soil", "🧱", DISPLAY_LABELS[lang]["Laterite soil"]),
            ("Sandy soil", "🏖️", DISPLAY_LABELS[lang]["Sandy soil"]),
            ("Clay soil", "🟫", DISPLAY_LABELS[lang]["Clay soil"]),
        ], default=None, lang_for_voice=lang, spoken_labels=SPOKEN_LABELS.get(lang, {}))

        st.write(t["crop_stage_label"])
        crop_stage = icon_button_row("crop_stage_choice", [
            ("Seedling", "🌱", DISPLAY_LABELS[lang]["Seedling"]),
            ("Vegetative", "🌿", DISPLAY_LABELS[lang]["Vegetative"]),
            ("Flowering", "🌸", DISPLAY_LABELS[lang]["Flowering"]),
            ("Fruiting/Maturity", "🍅", DISPLAY_LABELS[lang]["Fruiting/Maturity"]),
        ], default=None, lang_for_voice=lang, spoken_labels=SPOKEN_LABELS.get(lang, {}))

        st.write(t["pest_history_label"])
        pest_history = icon_button_row("pest_history_choice", [
            ("None reported", "✅", DISPLAY_LABELS[lang]["None reported"]),
            ("Occasional", "⚠️", DISPLAY_LABELS[lang]["Occasional"]),
            ("Frequent", "🐛", DISPLAY_LABELS[lang]["Frequent"]),
        ], default=None, lang_for_voice=lang, spoken_labels=SPOKEN_LABELS.get(lang, {}))

        st.write(t["variety_label"])
        variety_col1, variety_col2 = st.columns(2)
        with variety_col1:
            variety = st.text_input(t["variety_label"], placeholder=t["variety_placeholder"], label_visibility="collapsed")
        with variety_col2:
            variety_voice = None
            if hasattr(st, "audio_input"):
                variety_voice = st.audio_input(t["variety_voice_prompt"], label_visibility="collapsed")
            else:
                st.caption(t["variety_voice_unavailable"])
        if variety_voice is not None:
            st.caption(t["variety_voice_recorded"])

    if pil_image is not None:
        st.markdown(
            '<div class="agro-section-title">🔎 AI diagnosis</div>'
            '<div class="agro-section-subtitle">AgroShield analyzes the leaf image and shows the result below.</div>',
            unsafe_allow_html=True,
        )
        pil_image = auto_resize_image(pil_image)  # no size constraint on the user's end
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="", use_container_width=True)

        # Blur check with limited retry prompts. We hash the image bytes so
        # that re-running the script (e.g. when you click "Listen to advice"
        # or switch language) on the SAME photo doesn't wrongly count as a
        # new attempt - only a genuinely new upload/capture increments it.
        current_hash = hashlib.md5(pil_image.tobytes()).hexdigest()
        is_new_image = current_hash != st.session_state.last_image_hash

        blurry = is_blurry(pil_image)
        if blurry and is_new_image and st.session_state.retry_count < MAX_RETRY_PHOTOS:
            st.session_state.last_image_hash = current_hash
            st.session_state.retry_count += 1
            st.warning(f"{t['blurry_warning']} ({st.session_state.retry_count}/{MAX_RETRY_PHOTOS})")
            st.stop()
        elif blurry and st.session_state.retry_count >= MAX_RETRY_PHOTOS:
            st.info(t["blurry_limit"])
        else:
            st.session_state.last_image_hash = current_hash
            st.session_state.retry_count = 0  # good photo, reset counter

        if model is None or class_names is None:
            st.warning(t["no_model"])
        else:
            img_array = preprocess_image(pil_image)
            preds = model.predict(img_array)[0]
            top_idx = int(np.argmax(preds))
            confidence = float(preds[top_idx]) * 100
            predicted_class = class_names[top_idx]
            info = disease_info.get(predicted_class, {})

            with col2:
                st.subheader(t["diagnosis_header"])
                st.metric(t["prediction_label"], predicted_class.replace("_", " "))

                local_name = info.get("local_name_" + LANG_TTS_CODE.get(lang, "en"), None)
                if local_name:
                    st.write(f"**{t['local_name_label']}:** {local_name}")

                # Confidence shown as a small pie/donut chart, "confirmation" framing
                conf_col1, conf_col2 = st.columns([1, 2])
                with conf_col1:
                    st.markdown(confidence_pie_html(confidence), unsafe_allow_html=True)
                with conf_col2:
                    st.write("")
                    if confidence >= 85:
                        st.success(t["confidence_high"])
                    elif confidence >= 60:
                        st.warning(t["confidence_moderate"])
                    else:
                        st.error(t["confidence_low"])

                # Severity shown as 3 dots (green/yellow/red), active one highlighted
                severity = info.get("severity", "Unknown")
                st.write(f"**{t['severity_label']}:**")
                st.markdown(severity_dots_html(severity), unsafe_allow_html=True)

        advice_key_map = {"English": "advice_en", "हिंदी (Hindi)": "advice_hi", "मराठी (Marathi)": "advice_mr"}

        if pil_image is not None and model is not None and class_names is not None:
            advice_key = advice_key_map[lang]
            advice = info.get(advice_key, info.get("advice_en", t["no_advisory"]))

            voice_on_key = f"voice_on_{current_hash}_{lang}"
            if voice_on_key not in st.session_state:
                st.session_state[voice_on_key] = True  # voice on by default

            st.divider()

            # Recommended action - full width, below the picture row
            adv_col1, adv_col2 = st.columns([5, 1])
            with adv_col1:
                st.subheader(t["advice_header"] if confidence >= MIN_CONFIDENCE_FOR_COST else t["advice_header_soft"])
            with adv_col2:
                if TTS_AVAILABLE and advice != t["no_advisory"]:
                    voice_btn_col1, voice_btn_col2 = st.columns(2)
                    with voice_btn_col1:
                        if st.button("🔊", key=f"voice_on_btn_{current_hash}_{lang}",
                                     type="primary" if st.session_state[voice_on_key] else "secondary",
                                     help=t["stop_voice_button"]):
                            st.session_state[voice_on_key] = True
                            st.rerun()
                    with voice_btn_col2:
                        if st.button("🔇", key=f"voice_off_btn_{current_hash}_{lang}",
                                     type="primary" if not st.session_state[voice_on_key] else "secondary",
                                     help=t["replay_voice_button"]):
                            st.session_state[voice_on_key] = False
                            st.rerun()

            bullets = advice_to_bullets(advice)
            if bullets:
                for b in bullets:
                    st.markdown(f"- {b}.")
            else:
                st.write(advice)

            # Context-aware notes (rule-based, NOT part of the trained model -
            # the image is the only real diagnostic input). These adjust the
            # advisory tone based on farmer-provided field conditions.
            if severity in ("Moderate", "High") and weather in ("Humid", "Rainy"):
                st.warning(t["weather_risk_note"])
            if severity in ("Moderate", "High") and soil_type and "Clay" in soil_type:
                st.info(t["soil_drainage_note"])
            if severity in ("Moderate", "High") and crop_stage in ("Flowering", "Fruiting/Maturity"):
                st.warning(t["yield_risk_note"])
            if crop_stage == "Seedling" and severity != "None":
                st.warning(t["seedling_note"])
            if pest_history == "Frequent":
                st.info(t["pest_history_note"])
            if variety.strip():
                st.caption(t["variety_logged_note"])

            # Cost shown last, right before "Ask about your result"
            cost = info.get("cost_estimate_inr")
            if cost and confidence >= MIN_CONFIDENCE_FOR_COST:
                st.write(f"**{t['cost_label']}:** ₹{cost}")
                st.caption(t["cost_disclaimer"])
            elif cost and confidence < MIN_CONFIDENCE_FOR_COST:
                st.caption(t["cost_hidden_low_confidence"])

            # Voice-over: no visible player box - controlled entirely by the
            # 🔊/🔇 buttons above. Hidden HTML <audio> element with autoplay,
            # since Streamlit's st.audio always shows visible controls.
            if TTS_AVAILABLE and advice != t["no_advisory"] and st.session_state[voice_on_key]:
                audio_cache_key = f"audio_{current_hash}_{lang}"
                if audio_cache_key not in st.session_state:
                    with st.spinner("..."):
                        st.session_state[audio_cache_key] = text_to_speech_bytes(advice, LANG_TTS_CODE.get(lang, "en"))
                audio_bytes, audio_mime = st.session_state[audio_cache_key]
                if audio_bytes:
                    import base64
                    b64_audio = base64.b64encode(audio_bytes).decode()
                    st.markdown(
                        f'<audio autoplay style="display:none"><source src="data:{audio_mime};base64,{b64_audio}" type="{audio_mime}"></audio>',
                        unsafe_allow_html=True,
                    )

            # "Ask about your result" - honest scripted Q&A, NOT a general
            # open-ended AI (that would need internet or a heavy local LLM,
            # neither realistic offline in this timeframe). Answers are
            # dynamically templated from the REAL diagnosis data (confidence,
            # severity, prediction) - not fake/canned text.
            st.divider()
            st.subheader(t["qa_header"])
            qa_questions = {
                "why_diagnosis": t["qa_q_why"].format(pct=confidence, cls=predicted_class.replace("_", " ")),
                "what_severity": t["qa_q_severity"].format(sev=severity),
                "what_confidence": t["qa_q_confidence"].format(pct=confidence),
                "disagree": t["qa_q_disagree"],
            }
            qa_selected_key = f"qa_selected_{current_hash}_{lang}"
            qa_cols = st.columns(4)
            qa_buttons = [
                ("why_diagnosis", t["qa_btn_why"]),
                ("what_severity", t["qa_btn_severity"]),
                ("what_confidence", t["qa_btn_confidence"]),
                ("disagree", t["qa_btn_disagree"]),
            ]
            for col, (qkey, label) in zip(qa_cols, qa_buttons):
                with col:
                    if st.button(label, key=f"qa_{qkey}_{current_hash}_{lang}", use_container_width=True):
                        st.session_state[qa_selected_key] = qkey
                        st.rerun()

            if qa_selected_key in st.session_state:
                answer = qa_questions[st.session_state[qa_selected_key]]
                st.info(answer)
                if TTS_AVAILABLE:
                    qa_audio_key = f"qa_audio_{st.session_state[qa_selected_key]}_{current_hash}_{lang}"
                    if qa_audio_key not in st.session_state:
                        with st.spinner("..."):
                            st.session_state[qa_audio_key] = text_to_speech_bytes(answer, LANG_TTS_CODE.get(lang, "en"))
                    qa_audio, qa_mime = st.session_state[qa_audio_key]
                    if qa_audio:
                        import base64
                        b64_qa = base64.b64encode(qa_audio).decode()
                        st.markdown(
                            f'<audio autoplay style="display:none"><source src="data:{qa_mime};base64,{b64_qa}" type="{qa_mime}"></audio>',
                            unsafe_allow_html=True,
                        )

            st.divider()
            st.subheader(t["gradcam_header"])
            try:
                heatmap = make_gradcam_heatmap(img_array, model)
                overlay = overlay_heatmap(pil_image, heatmap)
                st.image(overlay, width=400)
            except Exception as e:
                st.caption(f"Heatmap unavailable ({e}).")
# ---------------------------------------------------------------------------
# TAB 2: GIS HOTSPOTS (DEMO)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader(t["hotspot_header"])
    st.warning(t["hotspot_note"])

    # Offline scatter plot instead of tile-based map (st.map/folium need
    # internet to fetch map tiles - this works with zero connectivity,
    # important for a live demo where venue wifi isn't guaranteed).
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = DEMO_HOTSPOTS["simulated_cases"] * 15
    scatter = ax.scatter(DEMO_HOTSPOTS["lon"], DEMO_HOTSPOTS["lat"], s=sizes, c=DEMO_HOTSPOTS["simulated_cases"], cmap="YlOrRd", edgecolors="black", alpha=0.8)
    for _, row in DEMO_HOTSPOTS.iterrows():
        ax.annotate(row["district"], (row["lon"], row["lat"]), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Simulated disease case density by district (offline view)")
    plt.colorbar(scatter, label="Simulated case count")
    st.pyplot(fig)

    st.dataframe(DEMO_HOTSPOTS.rename(columns={
        "district": "District", "simulated_cases": "Simulated case count"
    })[["District", "Simulated case count"]], hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: GOVERNMENT DASHBOARD SYNC (DEMO)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(t["dashboard_header"])
    st.warning(t["dashboard_note"])
    if st.button(t["dashboard_button"]):
        st.success(t["dashboard_success"])

st.divider()
st.caption(t["footer"])
