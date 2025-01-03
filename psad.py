# In your imports, add Dialog components
from streamlit.components.v1 import html

# Modify the main UI part in your code:
def main():
    required_dirs = ["images", "model"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            st.warning(f"Created missing directory: {dir_name}")

    if not os.path.exists("profile.jpg"):
        st.warning("Profile image not found. Please add a profile.jpg file.")

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
        photo_files = sorted([f for f in os.listdir(photo_dir) if f.endswith((".jpg", ".png"))])
        photos = [os.path.join(photo_dir, f) for f in photo_files]

        if not photos:
            st.warning("No photos available in the 'images' directory.")
        else:
            selected_index = st.selectbox("Choose a photo to post:", range(len(photos)), format_func=lambda x: photo_files[x])
            selected_photo = photos[selected_index]

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
                    
                    if st.button("Post Photo"):
                        if is_fake:
                            # Show dialog for deepfake
                            dialog_key = "deepfake_dialog"
                            if dialog_key not in st.session_state:
                                st.session_state[dialog_key] = True
                            
                            if st.session_state[dialog_key]:
                                with st.dialog("Deepfake Detection"):
                                    st.write("This photo is identified as a deepfake.")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("Post with Watermark", key="watermark"):
                                            watermarked_img = add_watermark(selected_photo, username)
                                            if watermarked_img:
                                                st.session_state.posted_photos.append({
                                                    "image": watermarked_img,
                                                    "caption": f"Deepfake (Watermarked)"
                                                })
                                                st.success("Photo Posted with Watermark!")
                                                st.session_state[dialog_key] = False
                                                st.experimental_rerun()
                                    
                                    with col2:
                                        if st.button("Cancel Post", key="cancel"):
                                            st.session_state[dialog_key] = False
                                            st.experimental_rerun()
                        else:
                            # Direct posting for non-deepfake images
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
                            st.experimental_rerun()

    # Right Column: Posted Photos
    with view_col:
        st.title("Posted Photos")
        if st.session_state.posted_photos:
            for i in range(0, len(st.session_state.posted_photos), 3):
                cols = st.columns(3)
                for col, post in zip(cols, st.session_state.posted_photos[i:i+3]):
                    with col:
                        if isinstance(post["image"], Image.Image):
                            img = post["image"]
                            min_size = 300
                            ratio = max(min_size / img.size[0], min_size / img.size[1])
                            if ratio > 1:
                                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                                img = img.resize(new_size, Image.Resampling.LANCZOS)
                            st.image(img, caption=post["caption"], use_container_width=True)
        else:
            st.write("No photos posted yet.")