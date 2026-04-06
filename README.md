# Insider Threat Detection using CERT Dataset

This project detects suspicious user behavior from the CERT insider threat dataset using logon activity and psychometric traits.

## Project Objective

The goal is to identify anomalous user-day behavior patterns that may indicate insider threats. The project uses:

- `logon.csv` for user logon/logoff activity
- `psychometric.csv` for Big Five personality traits
- `Isolation Forest` for anomaly detection

## Project Structure

```text
insider-threat-detection-cert/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── predictions/
├── src/
├── .gitignore
├── requirements.txt
├── main.py
└── README.md
