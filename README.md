# 👁️ Eye Blink Morse Code Communicator

An assistive communication system that converts **eye blinks into Morse code using computer vision**.

This project uses **OpenCV + MediaPipe Face Mesh** to detect eye movement and translate blink durations into Morse code, allowing a user to **communicate using only eye blinks**.

---

# 📜 Inspiration

This project is inspired by a real historical incident during the Vietnam War.

Jeremiah Denton, a U.S. Navy pilot captured as a prisoner of war, was forced to appear in a propaganda video claiming he was being treated well.

While answering questions, he secretly **blinked in Morse code** to communicate the word:

```
T O R T U R E
```

This revealed to the world that American prisoners were being tortured.

This moment became one of the most powerful examples of **covert Morse code communication**.

This project recreates that idea using **modern computer vision**, allowing Morse communication through eye blinks.

---

# 🧠 How It Works

The system detects **eye closure using facial landmarks** and measures blink duration.

Eye state is determined using the **Eye Aspect Ratio (EAR)**:

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 * ||p1−p4||)
```

When EAR drops below a threshold, the eye is considered **closed**.

Blink durations are interpreted as Morse signals.

| Blink Duration | Morse Signal |
|---|---|
| 200ms – 500ms | DOT (.) |
| 600ms – 1200ms | DASH (-) |
| Eyes open > 1500ms | Character break |
| Eyes open > 3000ms | Word break |

The Morse signals are then decoded into **readable text in real time**.

---

# ⚙️ Technologies Used

- Python  
- OpenCV  
- MediaPipe Face Mesh  
- NumPy  
- SciPy  

These tools allow real-time **facial landmark detection and blink analysis**.

---

# 📦 Installation

Clone the repository:

```
git clone https://github.com/krishshastri01/eye-morse-code-decoder.git
cd eye-morse-code-decoder
```

Install dependencies:

```
pip install opencv-python mediapipe numpy scipy
```

---

# ▶️ Running the Project

Run the program:

```
python eye_morse.py
```

Your webcam will open and begin detecting eye blinks.

Blink patterns will be translated into Morse code and decoded into text.

---

# 💡 Potential Applications

This system can be expanded for:

- Assistive communication for paralyzed patients
- Communication for locked-in syndrome
- Accessibility tools
- Hands-free text input
- Emergency signaling

---

# 🚀 Future Improvements

Possible upgrades include:

- GUI interface for decoded text
- Speech output (Text-to-Speech)
- Personalized blink calibration
- Faster Morse decoding
- Mobile deployment

# 👨‍💻 Author

**Krish Shastri**  
Electronics & Communication Engineering Student

GitHub:  
https://github.com/krishshastri01
