import os
import time
from datetime import datetime
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import base64

# Streamlit Setup
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# Constants
LABELS_LIST = ['Real', 'Fake']
DEEPFAKE_THRESHOLD = 0.5

# Initialize Session State
if "posted_photos" not in st.session_state:
    st.session_state.posted_photos = []
if "deepfake_photo" not in st.session_state:
    st.session_state.deepfake_photo = None

# Model Loading Function
@st.cache_resource
def load_mesonet_model(weights_path):
    try:
        model = tf.keras.models.load_model(weights_path)
        return model
    except Exception as e:
        st.error(f"Error loading MesoNet model: {e}")
        return None

# Image Processing Function
def preprocess_image_meso(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((256, 256))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Inference Function
def is_deepfake_meso(image_path):
    if not st.session_state.mesonet_model:
        st.warning("MesoNet model is not loaded.")
        return None, 0
    try:
        image = preprocess_image_meso(image_path)
        predictions = st.session_state.mesonet_model.predict(image)
        confidence = predictions[0][0]
        return confidence >= DEEPFAKE_THRESHOLD, confidence
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None, 0

# Watermarking Function
def add_watermark(image_path, username):
    try:
        # Open initial image
        img = Image.open(image_path).convert("RGBA")
        
        # Calculate new size maintaining aspect ratio with 300x300 minimum
        min_size = 300
        ratio = max(min_size / img.size[0], min_size / img.size[1])
        if ratio > 1:
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        watermark_text = f"Deepfake, created by {username} on {datetime.now().strftime('%Y-%m-%d')}"

        txt = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt)

        # Calculate font size based on image dimensions
        font_size = int(img.size[1] * 0.04)
        try:
            font = ImageFont.truetype("Arial", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_position = (10, img.size[1] - text_size[1] - 10)

        # Add semi-transparent background with proportional padding
        padding = int(font_size * 0.5)  # Padding proportional to font size
        rect_position = [
            text_position[0] - padding,
            text_position[1] - padding,
            text_position[0] + text_size[0] + padding,
            text_position[1] + text_size[1] + padding
        ]
        draw.rectangle(rect_position, fill=(0, 0, 0, 180))
        draw.text(text_position, watermark_text, fill=(255, 255, 255, 255), font=font)

        return Image.alpha_composite(img, txt).convert("RGB")
    except Exception as e:
        st.error(f"Error adding watermark: {e}")
        return None

# Clear Detection State
def clear_detection_state():
    """Clear the detection state and related session variables"""
    st.session_state.deepfake_photo = None
    if 'last_selected_photo' in st.session_state:
        del st.session_state.last_selected_photo

# Load Model
if "mesonet_model" not in st.session_state:
    st.session_state.mesonet_model = load_mesonet_model("./model/MesoNet_model.h5")

# Main UI
def main():
    # Sidebar
    username = st.sidebar.text_input("Username", value="user123")
    st.sidebar.image("profile.jpg", width=100, caption=f"{username}'s Profile")

    # Main layout
    post_col, view_col = st.columns([3, 2])

    # Left Column: Photo Selection and Posting
    with post_col:
        st.title("Instagram - Post a Photo")
        
        # Photo Selection
        photo_dir = "images"
        os.makedirs(photo_dir, exist_ok=True)
        photos = [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.endswith((".jpg", ".png"))]

        if not photos:
            st.warning("No photos available in the 'images' directory.")
        else:
            selected_photo = st.selectbox("Choose a photo to post:", photos)

            # Clear detection state if photo selection changes
            if 'last_selected_photo' not in st.session_state or st.session_state.last_selected_photo != selected_photo:
                clear_detection_state()
                st.session_state.last_selected_photo = selected_photo

            if selected_photo:
                # Display selected photo
                with open(selected_photo, "rb") as file:
                    img_data = file.read()
                encoded_img = "data:image/jpeg;base64," + base64.b64encode(img_data).decode()

                st.markdown(f"""
                    <div style="position: relative; width: 100%; padding-top: 56.25%; background: black;">
                        <img src="{encoded_img}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
                    </div>
                """, unsafe_allow_html=True)

                # Perform deepfake detection
                is_fake, confidence = is_deepfake_meso(selected_photo)

                if is_fake is not None:
                    st.info(f"Confidence Score: {confidence:.2f}")
                    
                    if is_fake:
                        st.warning("This photo is identified as a deepfake.")
                        st.session_state.deepfake_photo = selected_photo
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Post with Watermark"):
                                watermarked_img = add_watermark(selected_photo, username)
                                if watermarked_img:
                                    st.session_state.posted_photos.append({
                                        "image": watermarked_img,
                                        "caption": f"Deepfake (Watermarked)"
                                    })
                                    st.success("Photo Posted with Watermark!")
                                    clear_detection_state()
                                    time.sleep(1)
                                    st.rerun()
                        
                        with col2:
                            if st.button("Cancel Post"):
                                st.info("Post Cancelled.")
                                clear_detection_state()
                                time.sleep(1)
                                st.rerun()
                    else:
                        if st.button("Post Photo"):
                            # Ensure minimum size for normal posts too
                            img = Image.open(selected_photo)
                            min_size = 300
                            ratio = max(min_size / img.size[0], min_size / img.size[1])
                            if ratio > 1:
                                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            st.session_state.posted_photos.append({
                                "image": img,
                                "caption": "Normal Post"
                            })
                            st.success("Photo Posted Successfully!")
                            clear_detection_state()
                            time.sleep(1)
                            st.rerun()

    # Right Column: Posted Photos
    with view_col:
        st.title("Posted Photos")
        if st.session_state.posted_photos:
            for i in range(0, len(st.session_state.posted_photos), 3):
                cols = st.columns(3)
                for col, post in zip(cols, st.session_state.posted_photos[i:i+3]):
                    with col:
                        if isinstance(post["image"], Image.Image):
                            # Maintain minimum size while preserving aspect ratio
                            img = post["image"]
                            min_size = 300
                            ratio = max(min_size / img.size[0], min_size / img.size[1])
                            if ratio > 1:
                                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                            st.image(img, caption=post["caption"], use_container_width=True)
        else:
            st.write("No photos posted yet.")

if __name__ == "__main__":
    main()
