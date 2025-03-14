# 📌 DistilBERT Text Classification Web App
This project fine-tunes **DistilBERT** on the **20 Newsgroups dataset** for text classification and deploys a **Flask web application** to classify text into one of 20 categories.

## 🔹 Live Demo on Colab
Test the project in **Google Colab**: [Click Here](https://colab.research.google.com/drive/144aaBAgmt-elK7ikpWgPy038qiQOXrSR#scrollTo=H0B_dYPRFdIp)

---
## 🚀 Setup Instructions
You can run this project **on Google Colab** (recommended) or **locally on your machine**.

## 📌 Running on Google Colab
### **1️⃣ Open the Colab Notebook**
- Use the provided Colab link above.

### **2️⃣ Enable GPU for Faster Training**
- Go to **Runtime > Change runtime type**
- Set **Hardware Accelerator** to **GPU**

### **3️⃣ Run Training and Evaluation Cells**
- Execute all the cells in the notebook to:
  ✅ Train DistilBERT
  ✅ Evaluate model performance
  ✅ Deploy the web app

### **4️⃣ Run the Web Application**
- Execute the **Flask Web App** cell
- It will display a **public ngrok URL**
- Open the URL and test text classification!

---
## 📌 Running Locally (Manual Setup)

### **1️⃣ Install Dependencies**
Run the following in your terminal:
```bash
pip install torch transformers flask datasets pandas scikit-learn pyngrok
```

### **2️⃣ Train the Model**
Run the Python training script to fine-tune DistilBERT:
```bash
python train.py  # Ensure you have this script or modify accordingly
```
This saves the model as `distilbert_newsgroup_model/`.

### **3️⃣ Run the Web App Locally**
```bash
python app.py
```
- Open `http://127.0.0.1:5000/` in your browser.
- Enter text, click **"Classify"**, and view predictions!

### **4️⃣ Deploying via ngrok (Optional)**
If testing from an external device, run:
```bash
ngrok http 5000
```
Copy the **ngrok URL** and access the app online.

---
## 📊 Model Performance
| Metric     | Score |
|------------|------|
| Accuracy   | 92.22% |
| Precision  | 92.34% |
| Recall     | 92.22% |
| F1 Score   | 92.25% |

