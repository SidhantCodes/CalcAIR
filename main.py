import numpy as np
# import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector

import google.generativeai as genai
import os
from dotenv import load_dotenv

from PIL import Image

import streamlit as st

import pyttsx3
import threading

load_dotenv()

engine = pyttsx3.init()

st.set_page_config(page_title="AI Math Solver", page_icon="✏️", layout="wide")



# Create a header
st.title("✏️ AI Math Problem Solver")
st.markdown("Draw your math problem and get instant solutions!")


col1, col2 = st.columns([1.5,1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
    st.markdown("### 🖐️ Hand Gestures:")
    st.markdown("- ☝️ Index finger: Draw")
    st.markdown("- 🤏 Index + Pinky: Stop drawing")
    st.markdown("- 🖐️ All fingers: Clear canvas")
    st.markdown("- 🤘 All except pinky: Send to AI")
with col2:
    st.title("Answer")
    st_output = st.empty()

gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3, 1024)
cap.set(4, 768)


# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
def getHandInfo(img):
    # Find hands in the current frame
    # Set draw=False to prevent drawing the bounding box and hand type
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # bbox = hand["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        # center = hand['center']  # Center coordinates of the first hand
        # handType = hand["type"]  # Type of the first hand ("Left" or "Right")

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        # print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up
        print(fingers)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmlist = info
    curr_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        curr_pos = lmlist[8][0:2] #skip z-axis
        if prev_pos is None:
            prev_pos = curr_pos
        cv2.line(canvas, curr_pos, prev_pos, (255,0,255), 12)
    elif fingers == [0, 1, 0, 0, 1]:
        # Stop drawing when index and pinky fingers are up
        curr_pos = None
        prev_pos = None
    elif fingers == [1, 1, 1, 1, 1]:
        # Clear the entire canvas when all fingers are up
        canvas.fill(0)  # Fill the canvas with black (assuming black background)
        curr_pos = None
        prev_pos = None
    return curr_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)
        res = model.generate_content(["Solve this math problem, and respond with the correct asnwer along with proper explanation!", pil_image])
        return res.text

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

prev_pos = None
canvas = None
image_combined = None
answer = "Please Ask a question, I'd be happy to help you!😁"
prev_ans = None
# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()

    if canvas is None:
        canvas = np.zeros_like(img)
        # image_combined = img.copy()

    img = cv2.flip(img, 1)
    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        # print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        answer = sendToAI(model,canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
    st_output.markdown(f"### Solution:\n{answer if answer else 'Waiting for the question...'}") 

    # Global variable to store the previous answer
    previous_answer = None

    if answer and answer != "Please Ask a question, I'd be happy to help you!😁":
        if answer != previous_answer:
            st_output.markdown(f"### Solution:\n{answer}")
            # Start a new thread to speak the answer
            threading.Thread(target=speak_text, args=(answer,)).start()
            previous_answer = answer
    else:
        st_output.info("Draw a math problem and use the 'Send to AI' gesture to get an answer.")

    # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("image_canvas_combined", image_combined)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)


