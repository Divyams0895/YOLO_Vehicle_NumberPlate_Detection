
import streamlit as st
from ultralytics import YOLO
import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
from PIL import Image

# PAGE CONFIGURATION

st.set_page_config(
    page_title = "AI Vehicle Number Plate Detection",
    page_icon="🚘",
    layout = "wide"
)

model_path = 'best.pt'
owners_data = 'owner_details.csv'


tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# LOAD MODEL / DATA

@st.cache_resource
def load_model():
    return YOLO(model_path)

@st.cache_resource
def load_owners_data():
    return pd.read_csv(owners_data)


# HELPER FUNCTIONS

def clean_plate(text):
    text = str(text).upper()
    text = text.replace(" ","")
    text = text.replace("\n","")

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def search_vehicle(num_plate, df):
    if not num_plate:
        return None

    result = df[
        df["registration_number"]
        .astype(str)
        .str.upper()
        .str.strip()
        == num_plate.upper().strip()
    ]

    if not result.empty:
        return result.iloc[0]

    return None


def process_image(image_bgr, model):
    """
    Run YOLO -> crop plate -> preprocess -> OCR -> database lookup.
    Returns:
        annotated_image, plate_results
    """

    annotated = image_bgr.copy()
    results_data = []

    results = model(image_bgr)

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image_bgr.shape[1], int(x2))
            y2 = min(image_bgr.shape[0], int(y2))

            plate = image_bgr[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # Resize
            plate_resized = cv2.resize(
                plate,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            # Grayscale
            gray = cv2.cvtColor(
                plate_resized,
                cv2.COLOR_BGR2GRAY
            )

            # Blur
            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )

            # OTSU threshold
            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Tesseract
            config = (
                "--psm 7 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            text = pytesseract.image_to_string(
                thresh,
                config=config
            )

            plate_number = clean_plate(text)

            results_data.append({
                "plate": plate_number,
                "confidence": confidence,
                "crop": plate_resized,
                "threshold": thresh,
                "bbox": (x1, y1, x2, y2),
            })

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            # Put plate number
            if plate_number:
                cv2.putText(
                    annotated,
                    plate_number,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    return annotated, results_data


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# SIDEBAR
# ============================================================


# with st.sidebar:

    # st.header("⚙️ Settings")

    # st.subheader("Model")
    # st.caption(f"YOLO model: `{model_path}`")

    # confidence_threshold = st.slider(
    #     "Detection confidence",
    #     min_value=0.10,
    #     max_value=0.95,
    #     value=0.50,
    #     step=0.05
    # )

    # st.subheader("OCR")

    # tesseract_path = st.text_input(
    #     "Tesseract executable",
    #     value=tesseract_path
    # )

    # st.divider()

    # st.info(
    #     "For production use, protect vehicle-owner information "
    #     "with authentication and appropriate access controls."
    # )

# ============================================================
# HEADER
# ============================================================

st.header(
    "🚘 Vehicle Number Plate Detection"
)

st.caption(
    "Upload a vehicle image to detect the number plate,read it using OCR, and look up the corresponding record."
)

confidence_threshold = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload vehicle image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image containing a visible vehicle number plate."
)

if uploaded_file is None:

    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.markdown(
    #         '<div class="metric-box">'
    #         '<div class="small-label">Step 1</div>'
    #         '<div class="small-value">Upload Image</div>'
    #         '</div>',
    #         unsafe_allow_html=True
    #     )

    # with col2:
    #     st.markdown(
    #         '<div class="metric-box">'
    #         '<div class="small-label">Step 2</div>'
    #         '<div class="small-value">Detect + OCR</div>'
    #         '</div>',
    #         unsafe_allow_html=True
    #     )

    # with col3:
    #     st.markdown(
    #         '<div class="metric-box">'
    #         '<div class="small-label">Step 3</div>'
    #         '<div class="small-value">View Record</div>'
    #         '</div>',
    #         unsafe_allow_html=True
    #     )

    st.stop()

# ============================================================
# LOAD TESSERACT
# ============================================================

try:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # Check that Tesseract can be called.
    pytesseract.get_tesseract_version()

except Exception:
    st.error(
        "Tesseract OCR could not be initialized. "
        "Check the Tesseract executable path in the sidebar."
    )
    st.stop()

# ============================================================
# LOAD MODEL AND CSV
# ============================================================

try:
    with st.spinner("Loading YOLO model..."):
        model = load_model()

    owner_df = load_owners_data()

except FileNotFoundError as e:
    st.error(
        f"Required file not found: `{e.filename}`. "
        "Make sure `best.pt` and `owner_details.csv` are in the same "
        "folder as `app.py`."
    )
    st.stop()

except Exception as e:
    st.error(f"Could not load application resources: {e}")
    st.stop()

# ============================================================
# READ IMAGE
# ============================================================

pil_image = Image.open(uploaded_file).convert("RGB")
image_rgb = np.array(pil_image)
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# ============================================================
# RUN DETECTION
# ============================================================

if st.button("🔍 Detect Number Plate", type="primary", use_container_width=True):

    with st.spinner("Detecting plate and reading OCR..."):

        # Temporarily apply sidebar threshold.
        # The helper itself uses 0.5, so filter the model boxes here
        # by changing the model prediction threshold.
        original_model = model

        results = original_model(image_bgr, conf=confidence_threshold)

        annotated = image_bgr.copy()
        results_data = []

        for result in results:

            for box in result.boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image_bgr.shape[1], int(x2))
                y2 = min(image_bgr.shape[0], int(y2))

                plate = image_bgr[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                plate_resized = cv2.resize(
                    plate,
                    None,
                    fx=4,
                    fy=4,
                    interpolation=cv2.INTER_CUBIC
                )

                gray = cv2.cvtColor(
                    plate_resized,
                    cv2.COLOR_BGR2GRAY
                )

                gray = cv2.GaussianBlur(
                    gray,
                    (3, 3),
                    0
                )

                _, thresh = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                config = (
                    "--psm 7 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )

                raw_text = pytesseract.image_to_string(
                    thresh,
                    config=config
                )

                plate_number = clean_plate(raw_text)

                vehicle_data = search_vehicle(
                    plate_number,
                    owner_df
                )

                results_data.append({
                    "plate": plate_number,
                    "confidence": confidence,
                    "raw_text": raw_text,
                    "crop": plate_resized,
                    "threshold": thresh,
                    "vehicle_data": vehicle_data,
                })

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                if plate_number:
                    cv2.putText(
                        annotated,
                        plate_number,
                        (x1, max(35, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.subheader("📸 Detection Result")

    img_col, info_col = st.columns([1.7, 1])

    with img_col:
        st.image(
            bgr_to_rgb(annotated),
            caption="Detected number plate",
            use_container_width=True
        )

    with info_col:

        if not results_data:
            st.warning(
                "No number plate was detected above the selected "
                "confidence threshold."
            )

        for i, item in enumerate(results_data, start=1):

            plate_number = item["plate"]
            confidence = item["confidence"]
            vehicle_data = item["vehicle_data"]

            st.markdown(
                f'<div class="result-card">'
                f'<div class="small-label">Detected Plate #{i}</div>'
                f'<div class="plate">'
                f'{plate_number if plate_number else "NOT READ"}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.metric(
                "YOLO Confidence",
                f"{confidence * 100:.1f}%"
            )

            if plate_number:
                st.success(f"OCR Result: **{plate_number}**")
            else:
                st.warning("OCR could not read the number plate.")

            if vehicle_data is not None:

                st.markdown("### 👤 Vehicle Details")

                st.write(
                    f"**Registration:** "
                    f"{vehicle_data['registration_number']}"
                )

                st.write(
                    f"**Owner:** "
                    f"{vehicle_data['owner_name']}"
                )

                st.write(
                    f"**House Name:** "
                    f"{vehicle_data['house_name']}"
                )

                st.write(
                    f"**Place:** "
                    f"{vehicle_data['place']}"
                )

                st.write(
                    f"**Phone:** "
                    f"{vehicle_data['phone']}"
                )

            elif plate_number:
                st.error(
                    "Vehicle not found in the owner database."
                )

            with st.expander("🔎 OCR Details"):

                st.write(
                    "**Raw OCR output:**",
                    repr(item["raw_text"])
                )

                st.image(
                    bgr_to_rgb(item["crop"]),
                    caption="Cropped plate"
                )

                st.image(
                    item["threshold"],
                    caption="Thresholded OCR image"
                )

    # ========================================================
    # DATABASE PREVIEW
    # ========================================================

    with st.expander("📋 Owner database"):

        # Avoid exposing the complete database by default.
        st.caption(
            "Database loaded successfully. "
            "The table below is shown only for local development."
        )

        st.dataframe(
            owner_df,
            use_container_width=True,
            hide_index=True
        )
