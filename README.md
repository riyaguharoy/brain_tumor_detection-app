# 🧠 Brain Tumor Detection App

A deep learning web app that classifies brain MRI scans as **Healthy** or **Brain Tumor**, powered by a Convolutional Neural Network (CNN) built with TensorFlow/Keras and served through a Streamlit interface.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is **not** a certified medical diagnostic tool and must never be used to make real clinical decisions. Always consult a qualified radiologist or physician for actual diagnosis.

---

## Application Demo 

Upload an MRI image and the app returns a prediction with a confidence score.

<img width="1075" height="837" alt="image" src="https://github.com/user-attachments/assets/c1fdddcd-39bc-47bc-8b2d-7502b354a44c" />

---

## How it works

The model is a CNN trained on labeled brain MRI images (`Brain Tumor` / `Healthy`):

| Layer | Output Shape | Params |
|---|---|---|
| Conv2D (32 filters, 3x3) | (62, 62, 32) | 896 |
| MaxPooling2D | (31, 31, 32) | 0 |
| Conv2D (32 filters, 3x3) | (29, 29, 32) | 9,248 |
| MaxPooling2D | (14, 14, 32) | 0 |
| Flatten | (6272,) | 0 |
| Dense | (128,) | 802,944 |
| Dense | (10,) | 1,290 |
| Dense (sigmoid output) | (1,) | 11 |

- **Input:** 64×64×3 RGB MRI images
- **Output:** a single sigmoid probability - closer to 1 means Healthy, closer to 0 means Brain Tumor
- **Training:** images normalized to [0, 1], with data augmentation (rotation, flip, zoom, shift) applied during training

## Project structure

```
.
├── brain_tumor_detection.ipynb   # training notebook (data prep, model building, training)
├── streamlit_app.py              # web app for running predictions
├── brain_tumor_model.h5          # trained model weights
├── requirements.txt              # Python dependencies
└── README.md
```

## Running locally

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/brain-tumor-detection-app.git
   cd brain-tumor-detection-app
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`) and upload an MRI image.

## Retraining the model

Open `brain_tumor_detection.ipynb` in Jupyter, Colab, or Kaggle to retrain from scratch or fine-tune on new data. After training, save the model with:

```python
model.save("brain_tumor_model.h5")
```

and replace the existing `brain_tumor_model.h5` in this repo.

## Tech stack

- TensorFlow / Keras - model training and inference
- Streamlit - web app interface
- Pillow / NumPy - image preprocessing
