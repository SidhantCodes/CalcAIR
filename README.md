# CalcAI

## Description
This project is an interactive AI-powered math problem solver that uses computer vision and hand gestures to input math problems and provide solutions in real-time.

## Features
- Real-time hand tracking and gesture recognition
- Draw math problems using hand gestures
- AI-powered problem-solving using Google's Gemini model
- Text-to-speech solution narration
- User-friendly Streamlit interface

## Requirements
- Python 3.7+
- OpenCV
- NumPy
- cvzone
- Google GenerativeAI
- Streamlit
- pyttsx3
- python-dotenv

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up a `.env` file with your Gemini API key: `GEMINI_API_KEY=your_api_key_here`
4. Run the app: `streamlit run main.py`

## Usage
- Use your index finger to draw
- Index + Pinky fingers to stop drawing
- All fingers up to clear the canvas
- All fingers except pinky to send to AI for solving


