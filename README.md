# 🏥 Smart Hospital Queue Management – ML Module

This repository contains the **Machine Learning and Rule-Based Triage logic** for a Smart Hospital Queue Management system built during a hackathon.

The goal is to **prioritize patients** based on symptoms or disease severity, enabling faster response for critical cases.

---

## 🚀 Project Overview

The system works in **two stages**:

### 1️⃣ Symptom → Disease (ML Model)
- Patients select symptoms from the frontend
- Symptoms are converted into a binary vector (1 = present, 0 = absent)
- A trained ML model predicts the most likely disease

### 2️⃣ Disease → Severity (Rule-Based)
- Disease severity is determined using curated Excel lists:
  - `EMERGENCY`
  - `MEDIUM`
  - `NORMAL`
- This ensures **explainable and safe triage decisions**

---

## 🧠 Why Rule-Based Severity?
In healthcare triage, deterministic and explainable logic is preferred over probabilistic ML models.  
Hence, disease severity is determined using predefined medical priority lists.

---

## 📂 Project Structure

