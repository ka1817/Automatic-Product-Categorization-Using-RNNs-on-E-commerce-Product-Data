## 🛍️ Automatic Product Categorization Using LSTM RNN

---

## 📚 Overview

This project presents a deep learning-based solution to automatically categorize e-commerce products using their descriptions. Leveraging an LSTM (Long Short-Term Memory) neural network, the system classifies product descriptions into multiple categories like **Household**, **Books**, **Electronics**, and **Clothing & Accessories**.

---

## 🧩 Problem Statement

Manual classification of products is labor-intensive, error-prone, and not scalable. With thousands of new products listed every day, e-commerce platforms face challenges in maintaining consistent taxonomy and accurate product discovery.

**🎯 Objective:** Build an automated, scalable, and accurate multi-class text classification model to categorize product descriptions based on natural language.

---

## 🎯 Goals

* ✅ Automate product categorization using product descriptions
* 🔍 Enhance user search experience through accurate classification
* 💼 Reduce operational overhead by minimizing manual labeling

---

## 📂 Dataset

The dataset consists of labeled product descriptions from four categories:

| Category               | Count  |
| ---------------------- | ------ |
| Household              | 19,312 |
| Books                  | 11,820 |
| Electronics            | 10,621 |
| Clothing & Accessories | 8,671  |

* Format: `.csv`
* Columns: `product_description`, `category`

---

## 🧠 Solution Approach

We trained an **LSTM-based RNN model** using TensorFlow and Keras to learn sequential patterns in product descriptions. The model is wrapped inside a **FastAPI** backend and deployed via **Docker** on an **AWS EC2** instance.

* Text Preprocessing: Tokenization, Padding, Stopword Removal
* Model Architecture: Embedding → LSTM → Dense → Softmax
* Optimizer: Adam | Loss: Categorical Crossentropy
* Deployment: FastAPI + Docker + EC2
* CI/CD: GitHub Actions

---

## 📈 Model Performance

| Metric              | Value |
| ------------------- | ----- |
| Test Accuracy       | 97    |
| Test Loss           | 0.14  |

---

## 🏗️ Project Structure

```
project-root/
│
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions for CI/CD
├── data/
│   └── ecommerceDataset.csv    # Input dataset
├── logs/                       # Log files
├── models/                     # Trained model artifacts
├── static/                     # Static assets (confusion matrix, screenshots, etc.)
├── templates/
│   └── index.html              # Frontend template for UI
├── src/
│   ├── data_ingestion.py       # Load raw dataset
│   ├── data_preprocessing.py   # Clean and prepare text
│   ├── model_training.py       # Build, train and save model
│   ├── logging.py              # Logger utility
│   └── api/
│       ├── model_loader.py     # Load model/tokenizer/labels
│       └── predict.py          # Predict class of new description
├── main.py                     # FastAPI app entrypoint
├── requirements.txt            # Project dependencies
├── Dockerfile                  # Container configuration
└── README.md                   # Project documentation
```

---

## 🧪 Example API Usage

### **POST** `/predict`

**Request:**

```json
{
  "description": "Bluetooth enabled noise-cancelling headphones"
}
```

**Response:**

```json
{
  "predicted_category": "Electronics"
}
```

---

## 🐳 Docker Usage

### Build the image

```bash
docker build -t pranavreddy123/text-classification-app .
```

### Run the container

```bash
docker run -d -p 8000:8000 pranavreddy123/text-classification-app
```

Access the app at: [http://localhost:8000/](http://localhost:8000/)

---

## 🧪 Running Locally

```bash
# Clone the repository
git clone https://github.com/ka1817/Automatic-Product-Categorization-Using-RNNs-on-E-commerce-Product-Data.git
cd Automatic-Product-Categorization-Using-RNNs-on-E-commerce-Product-Data

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
uvicorn main:app --reload
```

---

## 🔁 CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration and deployment:

* ✅ Code push to `main` triggers pipeline
* 🧪 Runs unit tests using `pytest`
* 🐳 Builds and pushes Docker image to **DockerHub**
* 🚀 SSH-deploys container to **AWS EC2**

**Secrets used:**

* `DOCKER_USERNAME`, `DOCKER_PASSWORD`
* `EC2_SSH_KEY`, `EC2_HOST`, `EC2_USER`

---

## 🧰 Tech Stack

* 🐍 Python 3.10
* 🐼 Pandas  
* 🧠 TensorFlow / Keras
* 🧮 Scikit-learn
* ⚡ FastAPI
* 🐳 Docker
* ☁️ AWS EC2
* 🔁 GitHub Actions
* 📊 Matplotlib, Seaborn
* 📈 MLflow


---

## 🔗 Useful Links

* 🧠 GitHub Repo: [View Here](https://github.com/ka1817/Automatic-Product-Categorization-Using-RNNs-on-E-commerce-Product-Data)
* 🐳 DockerHub: [View Here](https://hub.docker.com/u/pranavreddy123)

---

## 🙋‍♂️ Author

**Katta Sai Pranav Reddy**
📧 Email: [kattapranavreddy@gmail.com](mailto:kattapranavreddy@gmail.com)
🔗 LinkedIn: [https://www.linkedin.com/in/pranav-reddy-katta/](https://www.linkedin.com/in/pranav-reddy-katta/)

---

