import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char

st.set_page_config(layout='wide')

# Sidebar content
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy - Application that reads lips')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipBuddy')

# Define the correct paths
base_path = os.path.abspath(os.path.join('..'))  # Move one level up from the 'app' folder
data_path = os.path.join(base_path, 'data', 's1')  # Path to the data folder
model_path = os.path.join(base_path, 'checkpoints1.weights.h5')  # Path to the model file

# Load available video options
if os.path.exists(data_path):
    options = os.listdir(data_path)
    selected_video = st.selectbox('Choose video', options)
else:
    st.error(f"Data path not found: {data_path}")
    options = []

col1, col2 = st.columns(2)

if options: 
    # Column 1: Render original video
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join(data_path, selected_video)
        if os.path.exists(file_path):
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            # Rendering video inside the app
            with open('test_video.mp4', 'rb') as video:
                video_bytes = video.read() 
                st.video(video_bytes)
        else:
            st.error(f"Video file not found: {file_path}")

    # Column 2: Display processed frames and predictions
    with col2: 
        if os.path.exists(file_path):
            st.info('This is all the machine learning model sees when making a prediction')
            video, annotations = load_data(tf.convert_to_tensor(file_path))

            # Process video frames for animation
            frames = (video.numpy() * 255).astype(np.uint8)  # Scale to [0, 255]
            if frames.ndim == 4 and frames.shape[-1] == 1:  # Handle single-channel frames
                frames = np.repeat(frames, 3, axis=-1)  # Convert grayscale to RGB
            
            # Save frames as an animation
            animation_path = 'animation.gif'
            imageio.mimsave(animation_path, frames, fps=10)
            st.image(animation_path, width=400)

            # Load and import the model
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                st.info('Model loaded successfully!')
            else:
                st.error(f"Model file not found: {model_path}")
                model = None

            if model:
                # Display raw model predictions
                st.info('This is the output of the machine learning model as tokens')
                yhat = model.predict(tf.expand_dims(video, axis=0))
                decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
                st.text(decoder)

                # Convert raw predictions to readable text
                st.info('Decode the raw tokens into words')
                converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
                st.text(converted_prediction)
        else:
            st.error(f"Unable to process video file: {file_path}")
