import streamlit as st
import numpy as np
import pickle
import speech_recognition as sr
import pyttsx3
import time

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Mobile Health AI", layout="centered")
st.title("🩺 Smart Breast Cancer Health App")
st.write("Manual or hands-free voice-guided input with summary and realistic limits")

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def voice_to_text(prompt="Speak now"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak(prompt)
        st.info(prompt)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            return None

fields = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry",
    "mean fractal dimension","radius error","texture error","perimeter error",
    "area error","smoothness error","compactness error","concavity error",
    "concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area",
    "worst smoothness","worst compactness","worst concavity",
    "worst concave points","worst symmetry","worst fractal dimension"
]

limits = {
    "mean radius": (6.0, 30.0),
    "mean texture": (9.0, 40.0),
    "mean perimeter": (40.0, 190.0),
    "mean area": (140.0, 2500.0),
    "mean smoothness": (0.05, 0.16),
    "mean compactness": (0.02, 0.35),
    "mean concavity": (0.0, 0.43),
    "mean concave points": (0.0, 0.2),
    "mean symmetry": (0.1, 0.3),
    "mean fractal dimension": (0.05, 0.1),
    "radius error": (0.1, 2.0),
    "texture error": (0.36, 4.0),
    "perimeter error": (0.8, 21.0),
    "area error": (6.0, 542.0),
    "smoothness error": (0.001, 0.03),
    "compactness error": (0.002, 0.135),
    "concavity error": (0.0, 0.4),
    "concave points error": (0.0, 0.05),
    "symmetry error": (0.01, 0.08),
    "fractal dimension error": (0.001, 0.03),
    "worst radius": (7.0, 40.0),
    "worst texture": (12.0, 50.0),
    "worst perimeter": (50.0, 250.0),
    "worst area": (150.0, 3500.0),
    "worst smoothness": (0.07, 0.2),
    "worst compactness": (0.02, 1.0),
    "worst concavity": (0.0, 1.0),
    "worst concave points": (0.0, 0.3),
    "worst symmetry": (0.1, 0.4),
    "worst fractal dimension": (0.05, 0.2)
}

if 'features' not in st.session_state:
    st.session_state['features'] = [0.0]*31

st.subheader("Enter Medical Features")
input_method = st.radio("Choose Input Method", ["Manual Input", "Voice-Guided Input"])

if input_method == "Manual Input":
    st.info("💡 Enter numeric values for all features")
    cols = st.columns(3)
    for i, field in enumerate(fields):
        with cols[i % 3]:
            val = st.number_input(
                field,
                value=st.session_state['features'][i],
                format="%.4f"
            )
            min_val, max_val = limits[field]
            if val < min_val or val > max_val:
                st.warning(f"{field} is out of realistic range ({min_val}–{max_val})")
            st.session_state['features'][i] = val
else:
    if st.button("🎤 Start Hands-Free Voice Input"):
        st.session_state['features'] = []
        for idx, field in enumerate(fields):
            valid_input = False
            while not valid_input:
                spoken = voice_to_text(prompt=f"Feature {idx+1} of 31: {field}")
                if spoken:
                    try:
                        value = float(spoken)
                        min_val, max_val = limits[field]
                        if value < min_val or value > max_val:
                            speak(f"Value out of range ({min_val} to {max_val}). Please repeat.")
                            st.error(f"{field}: {value} is out of realistic range ({min_val}–{max_val}). Try again.")
                        else:
                            st.session_state['features'].append(value)
                            st.success(f"{field}: {value} recorded")
                            valid_input = True
                            time.sleep(0.3)
                    except:
                        speak("Could not recognize a number. Please repeat.")
                        st.error("Could not recognize a number. Please try again.")
                else:
                    speak("No voice detected. Please repeat.")
                    st.error("No voice detected. Please try again.")
        speak("All features recorded. Here is a summary.")

features = st.session_state['features']

if len(features) == 31:
    st.subheader("📝 Feature Summary")
    for i, field in enumerate(fields):
        st.write(f"{field}: {features[i]}")
        speak(f"{field}: {features[i]}")
        time.sleep(0.05)

    st.write("✅ If all values are correct, press the button below to predict.")
    if st.button("🔍 Confirm and Predict"):
        features_array = np.array(features).reshape(1, -1)
        try:
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0][1] * 100
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        if prediction == 1:
            st.error("⚠️ Result: Malignant (Cancer Detected)")
            speak("Result: Malignant. Cancer detected.")
        else:
            st.success("✅ Result: Benign (No Cancer)")
            speak("Result: Benign. No cancer detected.")

        st.subheader("📊 Cancer Risk Level")
        st.progress(int(probability))
        st.write(f"Risk Percentage: **{probability:.2f}%**")
        speak(f"Cancer risk is {int(probability)} percent.")

        st.subheader("🤖 AI Explanation")
        if probability > 70:
            explanation = "High risk detected. Immediate medical consultation recommended."
        elif probability > 40:
            explanation = "Moderate risk. Regular monitoring advised."
        else:
            explanation = "Low risk. Maintain a healthy lifestyle."
        st.write(explanation)
        speak(explanation)

        st.subheader("🚨 Emergency Contact")
        speak("If this is an emergency, please contact your doctor immediately.")
        st.warning("Dialing emergency contact... (integrate with mobile API)")







 



