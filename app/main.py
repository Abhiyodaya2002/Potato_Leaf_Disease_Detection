import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# Define the remedies dictionary
remedies = {
    "Potato___Early_blight": "Containing copper and pyrethrins, Bonide® Garden Dust is a safe, one-step control for many insect attacks and fungal problems. For best results, cover both the tops and undersides of leaves with a thin uniform film or dust. Depending on foliage density, 10 oz will cover 625 sq ft. Repeat applications every 7-10 days, as needed.",
    "Potato___Late_blight": "Used as a foliar spray, Organocide® Plant Doctor will work its way through the entire plant to prevent fungal problems from occurring and attack existing many problems. Mix 2 tsp/ gallon of water and spray at transplant or when direct seeded crops are at 2-4 true leaf, then at 1-2 week intervals as required to control disease.",
    # Add remedies for other diseases as needed
}

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    remedy_info = remedies.get(predicted_class_name, "Remedy is not required.")
    return predicted_class_name, remedy_info

def prediction_page():
    st.title('🥬 Potato Disease Classifier')
    uploaded_images = st.file_uploader("Upload images for classification...", accept_multiple_files=True)

    if uploaded_images:
        for i, uploaded_image in enumerate(uploaded_images):
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((150, 150))
                st.image(resized_img)

            with col2:
                button_label = f'Classify {i}'  # Unique label for each button
                if st.button(button_label):
                    # Preprocess the uploaded image, predict the class, and get remedy information
                    predicted_class, remedy_info = predict_image_class(model, uploaded_image, class_indices)
                    st.success(f'Prediction: {str(predicted_class)}')
                    st.info(f'Remedy: {remedy_info}')

def contact_page():
    st.image("static/foldedhand.jpeg", width=50)
    st.header("कृषक सहायता फॉर्म")
    contact_form = """
    <form action="https://formsubmit.co/abhiyodayapandey@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name*" required>
         <input type="email" name="email" placeholder="Your email">
         <input type="text" name=mobile no." placeholder="Your mobile no.*" required>
         <textarea name="message" placeholder="Your message here*"></textarea>
         <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Go to", ["Welcome", "How to Use Web Application", "Potato Disease Classifier", "Contact"])

    if page == "Welcome":
        st.markdown(
            """
            <style>
            /* CSS for styling the title */
            .title {
                font-size: 36px;
                font-weight: bold;
                color: #333333;
                text-align: center;
                padding-top: 10px;
            }

            /* CSS for styling the image */
            .image {
                width: 80%;
                margin: auto;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            </style>
            """
            , unsafe_allow_html=True
        )

        # Display the title with the specified style
        st.markdown('<p class="title">Welcome to Potato Leaf Disease Identifier</p>', unsafe_allow_html=True)

        # Display the image with the specified style
        st.image("static/potatoheart2.jpg", caption="potato", use_column_width=True, output_format='auto')

    elif page == "How to Use Web Application":
        st.markdown(
            """
            <div>
                <h1 style="text-align: center; font-size: 36px; font-weight: bold; color: #333333;">How to Use This Web Application</h1>
                <ol style="font-size: 20px; color: #333333;">
                    <li>Upload an image of a potato leaf that you suspect is infected with a disease.</li>
                    <li>Click on the "Classify" button to let the model predict the disease based on the uploaded image.</li>
                    <li>View the predicted disease and suggested remedy provided by the system.</li>
                    <li>If necessary, take appropriate actions to manage the disease based on the provided remedy information.</li>
                </ol>
            </div>
            """
            , unsafe_allow_html=True
        )
        st.image("static/mobileleaf3.jpg",caption="mobileleaf")

    elif page == "Potato Disease Classifier":
        prediction_page()

    elif page == "Demo Video":
        st.title('Demo Video')
        # Replace 'your_video_url' with the actual URL of your video file
        st.video('video/demo.mp4')

    elif page == "Contact":
        contact_page()

if __name__ == "__main__":
    main()