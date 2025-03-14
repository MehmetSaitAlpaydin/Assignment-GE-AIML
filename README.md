# 📌 DistilBERT Text Classification Web App
This project fine-tunes **DistilBERT** on the **20 Newsgroups dataset** for text classification and deploys a **Flask web application** to classify text into one of 20 categories.

## 🔹 Live Demo on Colab
Test the project in **Google Colab**: [Click Here](https://colab.research.google.com/drive/144aaBAgmt-elK7ikpWgPy038qiQOXrSR#scrollTo=H0B_dYPRFdIp)

---
## 📌 Mini Report

### **1️⃣ Dataset Choice & Preprocessing**
The **20 Newsgroups dataset** is a well-known benchmark dataset for text classification. It consists of **newsgroup posts** from 20 different categories, covering topics such as politics, religion, sports, and technology.

**Preprocessing Steps:**
- **Data Loading:** The dataset was loaded using the Hugging Face `datasets` library.
- **Label Assignment:** Each category was assigned a unique numeric label.
- **Train/Test Split:** The dataset was split into **80% training** and **20% validation**.
- **Tokenization:** The **DistilBERT tokenizer** was used to convert text into tokenized inputs for the model.

---
### **2️⃣ Model Selection & Training**
DistilBERT was chosen due to its **efficiency and strong performance** in NLP tasks. DistilBERT is a **lighter version of BERT**, making it suitable for real-time applications.

**Training Details:**
- **Base Model:** `distilbert-base-uncased`
- **Training Hardware:** Google Colab with **T4 GPU**
- **Batch Size:** 16
- **Epochs:** 3
- **Optimizer:** AdamW with weight decay
- **Loss Function:** Cross-entropy loss
- **Evaluation Strategy:** Performance was logged after each epoch

The trained model was saved as `distilbert_newsgroup_model/` for later use.

---
### **3️⃣ Evaluation Results & Analysis**
After training, the model was evaluated on the validation set. Here are the key metrics:

| Metric     | Score |
|------------|------|
| **Accuracy**   | 92.22% |
| **Precision**  | 92.34% |
| **Recall**     | 92.22% |
| **F1 Score**   | 92.25% |

**Analysis:**
- The model achieved **high accuracy (92.22%)**, showing strong classification ability.
- Precision and recall are **well-balanced**, indicating that the model is correctly identifying categories without excessive false positives or false negatives.
- The F1-score of **92.25%** confirms that the model generalizes well to unseen data.

---
### **4️⃣ Web Application Overview**
A **Flask web application** was built to allow users to classify text in real time. 

**Features:**
✅ **User Input:** A text box for entering text
✅ **Inference Button:** A button to send input for classification
✅ **Output Display:** The app shows the **top 3 predicted categories** with confidence scores.

**How It Works:**
1. The user enters text in the input box.
2. The text is sent to the `/predict` endpoint via an AJAX request.
3. The trained **DistilBERT model** predicts the **most likely categories**.
4. The response is displayed with confidence scores.

The web app is deployed using **ngrok** for external access.

---
## 🚀 Setup Instructions
This project can be run **on Google Colab** (recommended) or **locally on a machine**.

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
Run the following in a terminal:
```bash
pip install torch transformers flask datasets pandas scikit-learn pyngrok
```

### **2️⃣ Train the Model**
Run the Python training script to fine-tune DistilBERT:
```bash
python train.py  # Ensure this script exists or modify accordingly
```
This saves the model as `distilbert_newsgroup_model/`.

### **3️⃣ Run the Web App Locally**
```bash
python app.py
```
- Open `http://127.0.0.1:5000/` in a browser.
- Enter text, click **"Classify"**, and view predictions!

### **4️⃣ Deploying via ngrok (Optional)**
If testing from an external device, run:
```bash
ngrok http 5000
```
Copy the **ngrok URL** and access the app online.

