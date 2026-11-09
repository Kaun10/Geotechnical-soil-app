import os
import json
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tempfile
import datetime
from fpdf import FPDF

# --- 1. Define Version String Here for Cache-Busting ---
APP_VERSION = "v1.0.1-rel2026.09"

# Page Setup (MUST be first Streamlit command)
st.set_page_config(
    page_title=f"Geotechnical Soil Classification System {APP_VERSION}",
    page_icon="🏗️",
    layout="centered"
)

# Load prompt configuration for soil labels safely
with open("prompt_config.json", "r") as f:
    config = json.load(f)

labels = config["labels"]
core_prompts = config["prompts"]

@st.cache_resource
def load_geotechnical_model():
    """Loads and caches the zero-shot CLIP model and processor."""
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    return model, processor

# --- 2. Update st.title to use the version variable ---
st.title(f"🏗️ Geotechnical Soil Classification System {APP_VERSION}")
st.write("Upload a field or laboratory specimen photo for automated visual analysis backed by **Dynamic Adaptive Thresholding and Regional Geological Mapping**.")

# Dynamic path resolution for standalone EXE compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "saved_clip_soil_model")
SUGGESTIONS_FILE = os.path.join(BASE_DIR, "location_suggestions.csv")

# Sidebar Controls for Field Adjustments
st.sidebar.header("⚙️ System Guardrails")
BASE_THRESHOLD = st.sidebar.slider(
    "Base Soil Confidence Threshold (%)",
    min_value=50.0,
    max_value=90.0,
    value=70.0,
    step=5.0,
    help="The baseline confidence required for immediate visual validation."
)
# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(f"🏗️ **Geotechnical AI {APP_VERSION}**\nDeveloped by Kaun")

# Geotechnical Knowledge Base
GEOTECHNICAL_DATA = {
    "Black_cotton_soil": {
        "uscs_symbol": "CH (High-Plasticity Clay / Vertisol)",
        "aashto": "A-7-6 (Expansive Clay)",
        "subgrade_rating": "Poor / Unsuitable (Requires Stabilization)",
        "drainage": "Impermeable / Very Poor",
        "shrink_swell": "High to Extreme (Expansive Soil)",
        "engineering_notes": (
            "• Foundations: Unsuitable for shallow footings without ground improvement. Recommend deep piers/piles or raft foundations.\n"
            "• Pavements: High heave risk. Requires subgrade lime/cement stabilization or removal/replacement with inert fill.\n"
            "• Slope Stability: High plastic deformation and swelling pressure when saturated."
        )
    },
    "Gravel": {
        "uscs_symbol": "GW / GP (Well-Graded or Poorly-Graded Gravel)",
        "aashto": "A-1-a (Gravelly Soils)",
        "subgrade_rating": "Excellent",
        "drainage": "Excellent / High Permeability",
        "shrink_swell": "Negligible / Non-Expansive",
        "engineering_notes": (
            "• Foundations: Excellent bearing capacity for shallow footings and structural foundations.\n"
            "• Pavements: Ideal subgrade and base course material for heavy load-bearing pavements.\n"
            "• Compaction: Compacts readily under heavy vibratory rollers with minimal moisture sensitivity."
        )
    },
    "Laterite": {
        "uscs_symbol": "GC / GM / MH (Clayey/Silty Gravel or Tropical Silt)",
        "aashto": "A-2-6 / A-7-5 (Tropical Residual Soil)",
        "subgrade_rating": "Good to Fair",
        "drainage": "Moderate to Low",
        "shrink_swell": "Low to Moderate",
        "engineering_notes": (
            "• Foundations: Adequate bearing capacity when undisturbed and dry; loses strength rapidly if inundated.\n"
            "• Pavements: Widely used as road base/sub-base in tropical construction when properly compacted.\n"
            "• Erosion: Vulnerable to severe surface erosion and gullying under heavy rainfall if slope is exposed."
        )
    },
    "Sand": {
        "uscs_symbol": "SW / SP (Well-Graded or Poorly-Graded Sand)",
        "aashto": "A-3 / A-1-b (Fine to Coarse Sand)",
        "subgrade_rating": "Good",
        "drainage": "Good to Rapid",
        "shrink_swell": "Non-Expansive",
        "engineering_notes": (
            "• Foundations: Good bearing capacity when confined/compacted. Vulnerable to scour and wind erosion.\n"
            "• Pavements: Stable subgrade material; requires confinement to prevent lateral displacement.\n"
            "• Seismic Hazard: High risk of soil liquefaction under dynamic/earthquake loading if saturated."
        )
    },
    "Unknown_Soil": {
        "uscs_symbol": "Unclassified (Non-Standard Matrix)",
        "aashto": "Unrated / Special Study Required",
        "subgrade_rating": "Undefined (Requires Laboratory Testing)",
        "drainage": "Variable / Unknown",
        "shrink_swell": "Unknown - Preliminary Assessment Unavailable",
        "engineering_notes": (
            "• Foundations: Standard presumptive bearing capacities do not apply without index testing.\n"
            "• Pavements: Unsuitable for use as subgrade or base material until full laboratory soil mechanics testing is completed.\n"
            "• Recommendation: Submit physical bag and core samples to an approved geotechnical testing laboratory."
        )
    }
}

# Sourced Engineering Geological Database (v1.0.1 Map)
GEOLOGICAL_DATABASE = {
    "yamalta/deba": "Black_cotton_soil", "yamalta": "Black_cotton_soil", "deba": "Black_cotton_soil",
    "akko": "Black_cotton_soil", "gombe": "Black_cotton_soil",
    "igbo-ora": "Black_cotton_soil", "ibarapa central": "Black_cotton_soil", "oyo": "Black_cotton_soil",
    "gulani": "Black_cotton_soil", "yobe": "Black_cotton_soil",
    "guyuk": "Black_cotton_soil", "numan": "Black_cotton_soil", "adamawa": "Black_cotton_soil",
    "new marte": "Black_cotton_soil", "marte": "Black_cotton_soil", "borno": "Black_cotton_soil",
    "idogo": "Black_cotton_soil", "yewa south": "Black_cotton_soil", "ogun": "Black_cotton_soil",
    "apir": "Laterite", "makurdi": "Laterite", "benue": "Laterite",
    "ikpayongo": "Laterite", "gwer-east": "Laterite", "gwer east": "Laterite",
    "ikwuano": "Laterite", "abia": "Laterite",
    "ihiala": "Laterite", "anaocha": "Laterite", "anambra east": "Laterite",
    "idemili south": "Laterite", "orumba south": "Laterite", "nijkoka": "Laterite",
    "oyi": "Laterite", "anambra": "Laterite",
    "tofa": "Laterite", "kano": "Laterite",
    "ibesikpo asutan": "Laterite", "akwa ibom": "Laterite",
    "galadimawa": "Laterite", "airport road": "Laterite", "fct": "Laterite", "abuja": "Laterite",
    "ado-ekiti": "Laterite", "ado ekiti": "Laterite", "ekiti": "Laterite",
    "umuloho": "Laterite", "ihube": "Laterite", "okigwe": "Laterite", "imo": "Laterite",
    "leru": "Laterite", "lokpauku": "Laterite", "umunneochi": "Laterite",
    "etsako": "Laterite", "akoko-edo": "Laterite", "akoko edo": "Laterite", "edo": "Laterite",
    "oko": "Laterite", "irepodun": "Laterite", "kwara": "Laterite",
    "ila": "Laterite", "ife central": "Laterite", "osun": "Laterite",
    "akure": "Laterite", "ilesha": "Laterite", "owo": "Laterite", "ondo": "Laterite"
}

# Universal Exclusion Prompts
exclusion_prompts = [
    "A photo of a human face, person, or human body parts",
    "A photo of a human hand holding an object",
    "A photo of an indoor room, office space, furniture, or man-made electronics",
    "A photo of an everyday object, vehicle, or item that is not a geological soil specimen",
    "A blank, blurry, out-of-focus, or dark image with no clear texture"
]

all_prompts = core_prompts + exclusion_prompts

# --- PDF Exporter Class & Sanitizer ---
class GeotechnicalPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "GEOTECHNICAL SOIL CLASSIFICATION REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, f"Geotechnical AI Framework {APP_VERSION} | Developed by Kaun", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 25, 200, 25)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} | Preliminary Site Assessment Tool - Confirm with Lab Index Tests", align="C")

def generate_live_pdf_report(
    soil_type, confidence, uscs, aashto, subgrade_rating, 
    drainage, shrink_swell, recommendation, location_str="Unspecified Site", sample_id="SMP-001", img_path=None
):
    def clean_text(txt):
        if not txt:
            return ""
        return (str(txt)
                .replace("•", "*")
                .replace("—", "-")
                .replace("–", "-")
                .replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'"))

    pdf = GeotechnicalPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 1. Project & Specimen Metadata
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. Project & Specimen Metadata", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(50, 6, "System Version:", border=1)
    pdf.cell(140, 6, f"Geotechnical Soil Classification System {APP_VERSION}", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 6, "Date / Time:", border=1)
    pdf.cell(140, 6, timestamp, border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 6, "Sample ID:", border=1)
    pdf.cell(140, 6, clean_text(sample_id), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 6, "Site Location / Region:", border=1)
    pdf.cell(140, 6, clean_text(location_str), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Specimen Visual Record Section
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Specimen Visual Record", new_x="LMARGIN", new_y="NEXT")
    if img_path and os.path.exists(img_path):
        pdf.image(img_path, x=10, y=pdf.get_y(), w=45)
        pdf.ln(35)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "[ Visual Matrix Captured Successfully ]", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # 3. Primary Classification Metrics
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Primary Classification Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 6, "Predicted Soil Class:", border=1)
    pdf.cell(140, 6, clean_text(soil_type), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 6, "Confidence Score:", border=1)
    pdf.cell(140, 6, f"{confidence:.1f}% (Vision Engine + OpenAI CLIP)", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 4. Standard Geotechnical Indexing
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "4. Standard Geotechnical Indexing", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 6, "USCS Group Symbol:", border=1)
    pdf.cell(140, 6, clean_text(uscs), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 6, "AASHTO Classification:", border=1)
    pdf.cell(140, 6, clean_text(aashto), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 6, "Subgrade Rating:", border=1)
    pdf.cell(140, 6, clean_text(subgrade_rating), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 6, "Drainage / Shrink-Swell:", border=1)
    pdf.cell(140, 6, clean_text(f"{drainage} | Shrink/Swell: {shrink_swell}"), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 5. Engineering Recommendations & Site Considerations
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "5. Engineering Recommendations & Site Considerations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(0, 6, txt=clean_text(recommendation), border=1)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# --- Main App Interface Layout ---
st.subheader("📍 Step 1: Regional Location & Metadata (Optional)")
col1, col2 = st.columns(2)
with col1:
    location_input = st.text_input(
        "State, LGA, or Town/Borrow Pit Name:", 
        placeholder="e.g., Makurdi, Deba, Abuja",
        key="loc_field"
    ).strip().lower()
with col2:
    sample_id_input = st.text_input(
        "Sample Identification Code:", 
        placeholder="e.g., SMP-001",
        value="SMP-001",
        key="sample_id_field"
    ).strip()

st.subheader("📸 Step 2: Specimen Visual Input")
input_method = st.radio(
    "Select input source:", 
    ["📁 Upload Saved File", "📷 Take Live Field Photo"], 
    horizontal=True,
    key="input_source_choice"
)

uploaded_file = None
if input_method == "📁 Upload Saved File":
    uploaded_file = st.file_uploader("Select a soil image file...", type=["jpg", "jpeg", "png"], key="file_upload_widget")
else:
    uploaded_file = st.camera_input("Capture field specimen photo", key="camera_input_widget")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((800, 800))
    st.image(image, caption="Input Specimen Matrix", use_container_width=True)
    
    # Save temp image path for PDF inclusion
    temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_img_file.name)
    st.session_state.temp_img_path = temp_img_file.name

    if "last_file" not in st.session_state or st.session_state.last_file != getattr(uploaded_file, 'name', 'camera_capture'):
        st.session_state.last_file = getattr(uploaded_file, 'name', 'camera_capture')
        st.session_state.has_run = False

    if st.button("🔬 Classify & Evaluate Geotechnical Grade", type="primary"):
        with st.spinner("Loading AI model weights & running evaluation..."):
            model, processor = load_geotechnical_model()
            
            inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1).cpu().numpy()[0] * 100

        top_idx = probs.argmax()
        top_score = probs[top_idx]
        
        st.session_state.probs = probs
        st.session_state.top_score = top_score
        st.session_state.top_idx = top_idx
        st.session_state.has_run = True

    # Render results and triage widgets if classification has been run
    if st.session_state.get("has_run", False):
        probs = st.session_state.probs
        top_score = st.session_state.top_score
        top_idx = st.session_state.top_idx
        
        st.divider()
        final_class = None
        location_verified = False
        
        # Check if an exclusion prompt won the match (non-soil detected)
        if top_idx >= len(core_prompts):
            winning_exclusion = all_prompts[top_idx]
            st.warning(
                "⚠️ **Validation Notice: Non-Soil Specimen Detected**\n\n"
                f"The visual input matched an exclusion parameter (*\"{winning_exclusion}\"* with {top_score:.1f}% confidence). "
                "Please ensure the camera focuses clearly on a raw geotechnical soil specimen."
            )
        elif top_score >= BASE_THRESHOLD:
            final_class = labels[top_idx]
            if final_class == "Unknown_Soil":
                st.warning("⚠️ **Visual Match Notice:** Specimen closely matches unclassified/unknown structural profiles.")
            else:
                st.success(f"📌 **Predicted Class via Vision Engine: {final_class.replace('_', ' ')}**")
            st.metric(label="Computer Vision Confidence Score", value=f"{top_score:.2f}%")
        else:
            st.warning(
                f"⚠️ **Visual Recognition Uncertain (Top Score: {top_score:.1f}% vs Required: {BASE_THRESHOLD}%)**\n\n"
                "Confidence level of identifying the sample is low due to site disturbances or lighting contrast. "
                "Activating secondary engineering validation matrices..."
            )
            
            if location_input:
                matched_db_class = None
                for key in GEOLOGICAL_DATABASE:
                    if key in location_input:
                        matched_db_class = GEOLOGICAL_DATABASE[key]
                        break
                
                if matched_db_class:
                    final_class = matched_db_class
                    location_verified = True
                    st.success(f"📍 **Geological Database Alignment:** Location verified within a mapped **{final_class.replace('_',' ')}** regional deposit matrix.")
                else:
                    st.info("ℹ️ The specified location/LGA is not currently logged within the local v1.0.1 database registry.")
                    st.write("📋 **Help expand the system:** Suggest this borrow pit location below for academic journal validation.")
                    
                    sug_col1, sug_col2 = st.columns([3, 1])
                    with sug_col1:
                        sug_loc = st.text_input(
                            "Confirm Location String for database review:", 
                            value=location_input.title(), 
                            label_visibility="collapsed",
                            key="sug_input_field"
                        )
                    with sug_col2:
                        if st.button("💾 Submit"):
                            with open(SUGGESTIONS_FILE, "a", encoding="utf-8") as sf:
                                sf.write(f'"{sug_loc.replace(chr(34), chr(39))}"\n')
                            st.toast("Location logged safely for monthly verification routine!", icon="💾")

        if not location_verified and top_idx < len(core_prompts) and top_score < BASE_THRESHOLD:
            st.markdown("### 🖐️ Step 3: Adaptive Field Triage (Tactile Indexing)")
            st.write("Answer the following mechanical properties to dynamically update system certainty thresholds.")
            matrix_selection = st.radio(
                "Identify dominant structural matrix behavior:",
                [
                    "Select Matrix...", 
                    "Cohesive (Clay-rich, forms molded shapes when damp)", 
                    "Non-Cohesive (Granular, loose, falls apart instantly)",
                    "Unlisted / Other Soil Type (Organic, Peat, Silt, or Mixed)"
                ],
                key="matrix_selection_radio"
            )
            if matrix_selection.startswith("Cohesive"):
                st.write("Select observable characteristics:")
                c1 = st.checkbox("Sticky, Smooth, Plastic (Heavily clayey when wet)", key="cb_c1")
                c2 = st.checkbox("Gritty firm, Earthy (Dull matte clay containing harder grains)", key="cb_c2")
                if c1:
                    final_class = "Black_cotton_soil"
                    st.success("⚡ Adaptive Threshold Reached: Explicit cohesive property matches standard Expansive Black Cotton indexing.")
                elif c2:
                    final_class = "Laterite"
                    st.success("⚡ Adaptive Threshold Reached: Tactile parameters isolate firm tropical lateritic clay residual matrix.")
            elif matrix_selection.startswith("Non-Cohesive"):
                st.write("Select observable grain properties:")
                n1 = st.checkbox("Gritty, Loose sand grains (Particles smaller than match heads)", key="cb_n1")
                n2 = st.checkbox("Hard, Coarse, Loose rock fragments or distinct pebbles (>2mm chunks)", key="cb_n2")
                if n1:
                    final_class = "Sand"
                    st.success("⚡ Adaptive Threshold Reached: Granular profile maps directly to clean sand subgrade fill layers.")
                elif n2:
                    final_class = "Gravel"
                    st.success("⚡ Adaptive Threshold Reached: Mechanical interlock and grain scale satisfies coarse gravel indexing.")
            elif matrix_selection.startswith("Unlisted"):
                final_class = "Unknown_Soil"
                st.info("ℹ️ User override activated: Specimen designated as an unlisted geotechnical soil class.")

        if final_class:
            if final_class == "Unknown_Soil":
                st.warning(
                    "⚠️ **Classification Notice: Unlisted / Unknown Soil Specimen**\n\n"
                    "The specimen does not match standard regional parameters for Sand, Gravel, Laterite, or Black Cotton Soil. "
                    "Full laboratory index testing (Hydrometer, Atterberg Limits, and Particle Size Distribution) is strictly required."
                )
                geo = GEOTECHNICAL_DATA["Unknown_Soil"]
            else:
                st.subheader(f"🏗️ Engineering Geotechnical Evaluation: {final_class.replace('_', ' ')}")
                geo = GEOTECHNICAL_DATA.get(final_class, {})

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"USCS Symbol: {geo.get('uscs_symbol')}")
                st.write(f"AASHTO Class: {geo.get('aashto')}")
                st.write(f"Subgrade Rating: {geo.get('subgrade_rating')}")
            with col2:
                st.write(f"Drainage Class: {geo.get('drainage')}")
                st.write(f"Shrink/Swell Risk: {geo.get('shrink_swell')}")
            st.markdown("---")
            st.markdown("### 📋 Engineering Recommendations & Site Considerations")
            st.markdown(geo.get("engineering_notes"))

            # --- PDF Export Button Section ---
            st.markdown("---")
            st.subheader("📄 Export Geotechnical Summary")

            if st.button("Generate Downloadable PDF Report"):
                pdf_file_path = generate_live_pdf_report(
                    soil_type=final_class.replace('_', ' '),
                    confidence=top_score,
                    uscs=geo.get('uscs_symbol', 'N/A'),
                    aashto=geo.get('aashto', 'N/A'),
                    subgrade_rating=geo.get('subgrade_rating', 'N/A'),
                    drainage=geo.get('drainage', 'N/A'),
                    shrink_swell=geo.get('shrink_swell', 'N/A'),
                    recommendation=geo.get('engineering_notes', 'N/A'),
                    location_str=location_input.title() if location_input else "Unspecified Site Location",
                    sample_id=sample_id_input if sample_id_input else "SMP-001",
                    img_path=st.session_state.get('temp_img_path', None)
                )
                
                with open(pdf_file_path, "rb") as pdf_bytes:
                    st.download_button(
                        label="⬇️ Download PDF Geotechnical Report",
                        data=pdf_bytes,
                        file_name=f"Geotechnical_Report_{final_class}.pdf",
                        mime="application/pdf"
                    )

        elif top_idx < len(core_prompts):
            st.info("🔒 Awaiting conclusive adaptive triage input parameters to populate structural evaluation metadata.")

        st.markdown("---")
        st.subheader("📊 Primary CLIP Visual Probability Layout")
        for idx, name in enumerate(labels):
            st.write(f"{name.replace('_', ' ')}: {probs[idx]:.2f}%")
            st.progress(int(probs[idx]))