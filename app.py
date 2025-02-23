import streamlit as st
from PIL import Image
import cv2
import numpy as np
from deepface import DeepFace
import random
import requests

# Define playlists based on sentiments
happy_playlist = [
    "Happy Song 1",
    "Happy Song 2",
    "Happy Song 3",
    "Happy Song 4",
    "Happy Song 5"
]

sad_playlist = [
    "Sad Song A",
    "Sad Song B",
    "Sad Song C",
    "Sad Song D",
    "Sad Song E"
]

# Function to run face detection and sentiment analysis
def detect_sentiment(image):
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        st.write(f"Error detecting face: {e}")
        return None


# Function to display the face image and playlist based on sentiment
def display_face_and_playlist(image):
    # Get detected sentiment
    emotion = detect_sentiment(image)
    
    if emotion is None:
        st.write("Could not detect emotion or face!")
        return

    # Display image and sentiment on the left
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Detected Emotion: {emotion}")
    
    # Display playlist based on detected emotion
    if emotion == "happy":
        playlist = random.sample(happy_playlist, 5)
    elif emotion == "sad":
        playlist = random.sample(sad_playlist, 5)
    else:
        playlist = ["Neutral Song 1", "Neutral Song 2", "Neutral Song 3", "Neutral Song 4", "Neutral Song 5"]
    
    # Display the playlist on the right
    st.write("Recommended Playlist:")
    for song in playlist:
        st.write(song)


# Function to fetch playlist from Shazam API based on the detected sentiment
def fetch_shazam_playlist(emotion):
    # Example Shazam API endpoint (you would need an actual API key and proper integration)
    # For now, I will simulate it with a static playlist based on emotion
    if emotion == "happy":
        return ["Happy Song 1", "Happy Song 2", "Happy Song 3", "Happy Song 4", "Happy Song 5"]
    elif emotion == "sad":
        return ["Sad Song A", "Sad Song B", "Sad Song C", "Sad Song D", "Sad Song E"]
    else:
        return ["Neutral Song 1", "Neutral Song 2", "Neutral Song 3", "Neutral Song 4", "Neutral Song 5"]


# Main function to manage both image upload and webcam feed
def main():
    st.title("MoodSync: Emotion-Based Playlist")

    # Sidebar for image upload
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file)
        display_face_and_playlist(image)
    else:
        # Display instructions when no image is uploaded
        st.write("Please upload an image to detect emotion and get a playlist!")

    # Option to switch to live webcam feed
    use_webcam = st.sidebar.checkbox("Use Webcam Feed")
    
    if use_webcam:
        st.write("Live Webcam Feed")
        run_webcam_feed()


# Function to handle live webcam feed and display dynamic playlist
def run_webcam_feed():
    st.title("Live Webcam Feed with Emotion Detection")
    
    # Set up webcam capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            st.write("Failed to capture image!")
            break
        
        # Convert frame to RGB for DeepFace analysis
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get sentiment analysis from the frame
        emotion = detect_sentiment(frame_rgb)
        
        if emotion is None:
            st.write("Could not detect emotion or face!")
            continue

        # Display frame with sentiment detected
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption=f"Detected Emotion: {emotion}", use_column_width=True)
        
        # Fetch playlist from Shazam API
        playlist = fetch_shazam_playlist(emotion)
        
        st.write("Recommended Playlist:")
        for song in playlist:
            st.write(song)

        # Break the loop when the user closes the app
        if st.button('Stop Webcam Feed'):
            cap.release()
            cv2.destroyAllWindows()
            break


# Run the app
if __name__ == "__main__":
    main()
