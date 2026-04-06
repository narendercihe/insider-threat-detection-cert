# 🔍 Insider Threat Detection using CERT Dataset

## 📌 Overview

This project implements an **unsupervised machine learning pipeline** to detect potential insider threats using the **CERT Insider Threat Dataset**.
It analyzes user logon behavior and (optionally) psychometric traits to identify anomalous activity patterns.

The system is designed to detect:

* Unusual login timings (after-hours access)
* Abnormal system usage patterns
* Suspicious user-day behavioral anomalies

---

## 🎯 Objectives

* Build a complete **data pipeline** from raw logs to anomaly detection
* Engineer meaningful behavioral features from logon data
* Apply **Isolation Forest** for anomaly detection
* Generate interpretable outputs and visualizations
* Create a **clean, reproducible ML project structure**

---

## 📂 Dataset

This project uses the **CERT Insider Threat Dataset**, specifically:

* `logon.csv` → user login/logout activity
* `psychometric.csv` → personality traits (optional)

> ⚠️ Note: Dataset files are **not included** in this repository due to size constraints.

---

## 🏗️ Project Structure

```text
insider-threat-detection-cert/
├── data/
│   ├── raw/                # Input CSV files (not tracked)
│   └── processed/          # Feature-engineered data
├── outputs/
│   ├── figures/            # Visualizations
│   ├── predictions/        # Model results
│   └── models/             # Saved models
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
├── main.py                 # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/narendercihe/insider-threat-detection-cert.git
cd insider-threat-detection-cert
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1: Place dataset files

Put your dataset files inside:

```text
data/raw/
```

Required:

* `logon.csv`

Optional:

* `psychometric.csv`

---

### Step 2: Run the project

```bash
python main.py
```

---

## 📊 Outputs

After running, the project generates:

### 📁 Processed Data

* `data/processed/daily_user_features.csv`

### 📁 Predictions

* `outputs/predictions/anomaly_results.csv`
* `outputs/predictions/top_anomalies.csv`

### 📁 Model

* `outputs/models/isolation_forest.joblib`

### 📁 Visualizations

* `outputs/figures/hour_distribution.png`
* `outputs/figures/top_suspicious_users.png`

---

## 🤖 Model Details

### Isolation Forest

* Unsupervised anomaly detection algorithm
* Detects outliers based on feature isolation
* Works well for high-dimensional behavioral data

### Features Used

* Total login events per day
* After-hours activity
* Number of unique machines used
* Behavioral ratios (e.g., after-hours ratio)

---

## 📈 Example Output

Top suspicious user-day records:

```
user     day        anomaly_score
DNS1758  2010-03-13   -0.1339
BPD2437  2010-03-27   -0.1312
```

---

## ⚠️ Limitations

* No labeled data (unsupervised learning)
* Psychometric data may be unavailable
* Model performance depends on feature quality

---

## 🚀 Future Improvements

* Add supervised models (if labeled data available)
* Incorporate additional logs (email, file access)
* Build a dashboard for visualization
* Deploy as a web application

---

## 🧠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Narender Kumar

---
