# 🦴 OsteoAI - Osteoarthritis Prediction Web App

A modern Flask web application for AI-powered osteoarthritis detection from medical images.  
Styled to match the look and feel of the [Atlas of Inflammation Resolution (AIR)](https://air.bio.informatik.uni-rostock.de) website.

---

## 🚀 Features

- **Single-page prediction app** (no homepage)
- Upload X-ray, MRI, or CT images for instant AI analysis
- Clean, AIR-inspired UI with responsive design
- Confidence score and downloadable report
- Secure file upload (max 16MB, image formats only)
- No patient data stored

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd predicitionOsteo
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your trained model**
   - Place your `my_model.h5` file in the `models/` directory.

5. **Run the app**
   ```bash
   python app.py
   ```
   Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📁 Project Structure

```
predicitionOsteo/
├── app.py
├── requirements.txt
├── models/
│   └── my_model.h5
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── prediction.js
├── templates/
│   └── predict.html
└── uploads/
```

---

## 🖼️ Usage

1. Open the app in your browser.
2. Click the upload area or drag-and-drop a medical image.
3. Click **Analyze Image**.
4. View the AI’s diagnosis and confidence.
5. Download a report if needed.

---

## ⚙️ Model Requirements

- **Format:** Keras `.h5` file
- **Input:** RGB image, 224x224 pixels
- **Output:** Array of probabilities for three classes: `healthy`, `moderate`, `severe`
- **Architecture:** CNN-based classification model (e.g., ResNet, VGG, etc.)
- **Class Labels (order expected by the app):**
  - `healthy`
  - `moderate`
  - `severe`

---

## 🧑‍⚕️ Disclaimer

This tool is for **educational and research purposes only**.  
It is **not a substitute for professional medical advice, diagnosis, or treatment**.  
Always consult a qualified healthcare provider for medical concerns.

---

## 📄 License

This project is for academic and research use.  
Contact the author for other uses.

---

## 🙏 Acknowledgements

- [Atlas of Inflammation Resolution (AIR)](https://air.bio.informatik.uni-rostock.de) for UI inspiration
- [Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pillow](https://python-pillow.org/)

---

*Made with ❤️ for advancing digital health.*