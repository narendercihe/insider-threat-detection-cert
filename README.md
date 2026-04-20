# 🔍 Insider Threat Detection using CERT Dataset

## 📌 Overview

This project implements an unsupervised machine learning pipeline to detect potential insider threats using the **CERT Insider Threat Dataset**. It analyzes user logon behavior and (optionally) psychometric traits to identify anomalous activity patterns.  
The system is designed to detect:
- Unusual login timings (after-hours access)
- Abnormal system usage patterns
- Suspicious user-day behavioral anomalies

---

## 🎯 Objectives

- Build a complete data pipeline from raw logs to anomaly detection
- Engineer meaningful behavioral features from logon data
- Apply **Isolation Forest** for anomaly detection
- Generate interpretable outputs and visualizations
- Create a clean, reproducible ML project structure

---

## 📂 Dataset

This project uses the **CERT Insider Threat Dataset**, specifically:
- `logon.csv` → user login/logout activity
- `device.csv` → device used for login
- `users.csv`: User information
- `psychometric.csv`: Psychometric data
- `file.csv` → file access logs
- `http.csv` → web access logs
- `email.csv` (optional) → email logs
- `ldap.csv` → LDAP authentication logs
- `answers.csv` → true labels (for model evaluation)

You can download the dataset from Kaggle:

[**Dataset Link**](https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research?utm_source=chatgpt.com&select=file.csv)

> ⚠️ **Note**: Dataset files are **not included** in this repository due to size constraints. Please download them from the provided link.

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
