# Insider Threat Detection using CERT Dataset

## Overview

This project builds a comparative insider threat detection pipeline using the CERT dataset. It combines user logon behavior, device activity, psychometric traits, and optional user metadata to detect suspicious user-day patterns.

The project compares three models:

- Isolation Forest
- Autoencoder (AE)
- Variational Autoencoder (VAE)

## Objective

The objective is to identify suspicious insider-like behavior and compare anomaly detection models using:

- Accuracy
- Precision
- Recall
- F1-score

## Dataset Used

Place these files in `data/raw/`:

- `logon.csv`
- `device.csv`
- `psychometric.csv`
- `users.csv` (optional but recommended)

## Project Workflow

1. Load and preprocess datasets
2. Create daily behavioral features
3. Merge psychometric and optional user data
4. Generate pseudo labels using behavioral rules
5. Train Isolation Forest, Autoencoder, and VAE
6. Evaluate models with classification metrics
7. Save figures, predictions, and comparison tables

## Features Used

### Logon features
- total_events
- total_logons
- total_logoffs
- unique_pcs
- after_hours_events
- weekend_events
- first_hour
- last_hour
- activity_span
- after_hours_ratio
- weekend_ratio

### Device features
- device_events
- after_hours_device_events
- weekend_device_events
- unique_device_pcs
- connect_like_events
- after_hours_device_ratio
- weekend_device_ratio

### Historical features
- user_avg_total_events
- user_avg_after_hours
- user_avg_unique_pcs
- user_avg_device_events
- deviation_total_events
- deviation_after_hours
- deviation_unique_pcs
- deviation_device_events

### Psychometric features
- O, C, E, A, N

## Pseudo Labeling

Because official answer labels were not available in the chosen subset, pseudo labels are generated from strong behavioral rules such as:

- high after-hours activity
- unusual device usage
- multi-PC access
- weekend activity with device events
- large deviation from personal baseline

A user-day is marked suspicious when multiple such conditions are met.

## Models

### 1. Isolation Forest
Used as a baseline anomaly detection method.

### 2. Autoencoder
Learns normal behavioral patterns and flags records with high reconstruction error.

### 3. Variational Autoencoder (VAE)
Probabilistic version of autoencoder used for anomaly detection via reconstruction error.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
