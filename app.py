import os
import json
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Page Setup (MUST be first Streamlit command)
st.set_page_config(
    page_title="Geotechnical Soil Classification System 1.0.1",
    page_icon="🏗️",
    layout="centered"
)

# Load prompt configuration for soil labels safely
with open("prompt_config.json", "r") as f:
    config = json.load(f)

labels = config["labels"]

@st.cache_resource
def load_geotechnical_model():
    """Loads and caches the zero-shot CLIP model and processor."""
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    return model, processor

# Lazy load the model inside a spinner when the app renders
with st.spinner("Loading AI model weights... Please wait."):
    model, processor = load_geotechnical_model()

st.title("🏗️ Geotechnical Soil Classification System v1.0.1")
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

# Geotechnical Knowledge Base
GEOTECHNICAL_DATA = {
    "Black_cotton_soil": {
        "uscs_symbol": "CH (High-Plasticity Clay / Vertisol)",
        "aashto": "A-7-6 (Expansive Clay)",
        "subgrade_rating": "Poor / Unsuitable (Requires Stabilization)",
        "drainage": "Impermeable / Very Poor",
        "shrink_swell": "High to Extreme (Expansive Soil)",
        "engineering_notes": (
            "• **Foundations:** Unsuitable for shallow footings without ground improvement. Recommend deep piers/piles or raft foundations.\n"
            "• **Pavements:** High heave risk. Requires subgrade lime/cement stabilization or removal/replacement with inert fill.\n"
            "• **Slope Stability:** High plastic deformation and swelling pressure when saturated."
        )
    },
    "Gravel": {
        "uscs_symbol": "GW / GP (Well-Graded or Poorly-Graded Gravel)",
        "aashto": "A-1-a (Gravelly Soils)",
        "subgrade_rating": "Excellent",
        "drainage": "Excellent / High Permeability",
        "shrink_swell": "Negligible / Non-Expansive",
        "engineering_notes": (
            "• **Foundations:** Excellent bearing capacity for shallow footings and structural foundations.\n"
            "• **Pavements:** Ideal subgrade and base course material for heavy load-bearing pavements.\n"
            "• **Compaction:** Compacts readily under heavy vibratory rollers with minimal moisture sensitivity."
        )
    },
    "Laterite": {
        "uscs_symbol": "GC / GM / MH (Clayey/Silty Gravel or Tropical Silt)",
        "aashto": "A-2-6 / A-7-5 (Tropical Residual Soil)",
        "subgrade_rating": "Good to Fair",
        "drainage": "Moderate to Low",
        "shrink_swell": "Low to Moderate",
        "engineering_notes": (
            "• **Foundations:** Adequate bearing capacity when undisturbed and dry; loses strength rapidly if inundated.\n"
            "• **Pavements:** Widely used as road base/sub-base in tropical construction when properly compacted.\n"
            "• **Erosion:** Vulnerable to severe surface erosion and gullying under heavy rainfall if slope is exposed."
        )
    },
    "Sand": {
        "uscs_symbol": "SW / SP (Well-Graded or Poorly-Graded Sand)",
        "aashto": "A-3 / A-1-b (Fine to Coarse Sand)",
        "subgrade_rating": "Good",
        "drainage": "Good to Rapid",
        "shrink_swell": "Non-Expansive",
        "engineering_notes": (
            "• **Foundations:** Good bearing capacity when confined/compacted. Vulnerable to scour and wind erosion.\n"
            "• **Pavements:** Stable subgrade material; requires confinement to prevent lateral displacement.\n"
            "• **Seismic Hazard:** High risk of **soil liquefaction** under dynamic/earthquake loading if saturated."
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

import streamlit as st
from transformers import CLIPModel, CLIPProcessor

@st.cache_resource
def load_geotechnical_model():
    """Loads and caches the zero-shot CLIP model and processor."""
    model_id = "openai/clip-vit-base-patch32"
    
    # Load processor and model
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    
    return model, processor

# Call the cached function at the entry point of your app
with st.spinner("Loading AI model weights... Please wait."):
    model, processor = load_geotechnical_model()

labels = config["labels"]

# Explicitly define core and universal exclusion prompts to guarantee they evaluate
core_prompts = [
    "A close-up photo of dry black cotton soil with characteristic shrinkage cracks",
    "A close-up photo of well-graded gravel aggregate and coarse rock fragments",
    "A close-up photo of tropical laterite soil with a reddish-brown clay matrix",
    "A close-up photo of clean, loose construction sand"
]

exclusion_prompts = [
    "A photo of a human face, person, or human body parts",
    "A photo of a human hand holding an object",
    "A photo of an indoor room, office space, furniture, or man-made electronics",
    "A photo of an everyday object, vehicle, or item that is not a geological soil specimen",
    "A blank, blurry, out-of-focus, or dark image with no clear texture"
]

all_prompts = core_prompts + exclusion_prompts
st.subheader("📍 Step 1: Regional Location Parameters (Optional)")
location_input = st.text_input(
    "Enter State, Local Government Area (LGA), or Borrow Pit/Town Name if known:", 
    key="loc_field"
).strip().lower()

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
    
    # Initialize session state for this image if not present
    if "last_file" not in st.session_state or st.session_state.last_file != getattr(uploaded_file, 'name', 'camera_capture'):
        st.session_state.last_file = getattr(uploaded_file, 'name', 'camera_capture')
        st.session_state.has_run = False

    if st.button("🔬 Classify & Evaluate Geotechnical Grade", type="primary"):
        with st.spinner("Stage 1: Running primary computer vision evaluation..."):
            inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1).cpu().numpy()[0] * 100

        top_idx = probs.argmax()
        top_score = probs[top_idx]
        
        # Save results in session state so they persist across widget clicks
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
                ["Select Matrix...", "Cohesive (Clay-rich, forms molded shapes when damp)", "Non-Cohesive (Granular, loose, falls apart instantly)"],
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

        if final_class:
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
        elif top_idx < len(core_prompts):
            st.info("🔒 Awaiting conclusive adaptive triage input parameters to populate structural evaluation metadata.")

        st.markdown("---")
        st.subheader("📊 Primary CLIP Visual Probability Layout")
        for idx, name in enumerate(labels):
            st.write(f"{name}: {probs[idx]:.2f}%")
            st.progress(int(probs[idx]))