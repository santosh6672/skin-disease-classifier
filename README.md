# 🧠 Skin Disease Classifier using FastAPI

This project is a **Deep Learning-based web application** that classifies **skin diseases** from user-uploaded images using a **VGG-based Convolutional Neural Network (CNN)** model.  
It provides an **AI-powered diagnosis** and **treatment suggestions**, hosted on a **FastAPI** backend.

---

## 🚀 Features

- 🩺 Detects multiple types of skin diseases using a trained **VGG CNN model**
- 🌐 FastAPI backend for inference and API endpoints

---
## 📊 Dataset Description

**Dataset Name:** Multiple Skin Disease Detection and Classification  
**Source:** [ISIC Archive (via Kaggle)](https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification)  
**About:**  
The dataset contains **nine different classes** of skin diseases or conditions collected from the **ISIC (International Skin Imaging Collaboration)** archive.  

### Classes Included:
1. Melanoma  
2. Actinic Keratosis  
3. Basal Cell Carcinoma  
4. Dermatofibroma  
5. Nevus  
6. Pigmented Benign Keratosis  
7. Seborrheic Keratosis  
8. Squamous Cell Carcinoma  
9. Vascular Lesion  

This dataset is widely used for building **Deep Learning** and **Computer Vision** models to assist dermatologists in **automated skin disease detection** and **clinical decision-making**.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/santosh6672/skin-disease-classifier.git
cd skin-disease-classifier
````

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the FastAPI app

```bash
uvicorn web.main:app --reload
```

### 5️⃣ Open the app in your browser

```
http://127.0.0.1:8000
```

---

## 🧠 Model Information

* **Architecture:** VGG-based CNN
* **Input:** Skin disease image (JPEG/PNG)
* **Output:** Predicted disease label + confidence score
* **Framework:** TensorFlow / Keras



## 🧪 API Endpoints

| Method | Endpoint    | Description                                   |
| ------ | ----------- | --------------------------------------------- |
| `POST` | `/predict/` | Upload an image and get disease prediction    |
| `GET`  | `/`         | Root route showing homepage or status message |

Example request using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@sample.jpg"
```

---

## 📦 Dependencies

Major packages used:

* **FastAPI** – web framework
* **TensorFlow / Keras** – deep learning model
* **Pillow** – image handling
* **Uvicorn** – ASGI server
* **Python-dotenv** – environment configuration
* **Git LFS** – for large model storage

Install them with:

```bash
pip install fastapi tensorflow pillow uvicorn python-dotenv
```
