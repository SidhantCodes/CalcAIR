import numpy as np
# import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector

import google.generativeai as genai
import os
from dotenv import load_dotenv

from PIL import Image

load_dotenv()

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
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

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
        res = model.generate_content(["Solve this math problem", pil_image])
        print(res.text)

prev_pos = None
canvas = None
image_combined = None

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
        sendToAI(model,canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    cv2.imshow("image_canvas_combined", image_combined)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)


