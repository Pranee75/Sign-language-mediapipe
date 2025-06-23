# Sign-language-mediapipe
This is a beginner-level machine learning project that uses MediaPipe and OpenCV to detect basic sign language gestures via webcam.

---

## 📹 What It Does
- Tracks hand landmarks in real-time using MediaPipe
- Matches them to predefined gesture patterns
- Displays the detected letter (A, B, etc.) or "Unidentified"

---

## 🧠 My Learning Journey

I first attempted to build this using TensorFlow, but faced several issues.  
So I restarted the project using MediaPipe — and finally got it working!

Through this, I learned:
- Basics of hand tracking
- Real-time gesture recognition
- How to debug, fail, restart, and keep going!

Even though it still has limitations (not all signs work perfectly), I'm proud that I saw it through from start to finish.

---

## 🔧 Tools & Libraries Used
- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Matplotlib

---

## ⚠️ Known Limitations
- Limited to only a few signs
- Does not recognize signs that require motion or dynamic movement
- Only supports **one hand** — cannot detect two-hand gestures
- Accuracy drops under poor lighting or fast hand movement
- Takes a few seconds to initialize the webcam and MediaPipe model
- May fail if hand is partially out of frame or blurred

---

## 🙏 Acknowledgements

This project is inspired by Felipe’s GitHub repository:   https://github.com/computervisioneng/sign-language-detector-python

I studied the code, modified parts of it, experimented, and made it my own.  
Thanks to their open-source work and the MIT License, I was able to build, break, and learn freely.

---

## 📈 Future Improvements
- Add more sign classes
- Improve accuracy
- Add support for detecting both hands
- Enable recognition of dynamic signs that involve motion
- Optimize webcam load time and model startup delay
- Train my own model with TensorFlow or Keras
- Build a GUI using Streamlit or Tkinter

---

## 💡 Final Note

This is my first ML-based project — it may not be perfect, but it’s real.  
I’ve learned more from struggling with this than from any tutorial.

