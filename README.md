# 🎓 Student Academic Success, Subject Performance & Career Readiness Analytics Platform

> An early-intervention intelligence platform that stitches fragmented student data into actionable academic and career guidance.

---

## Judge Pitch

Educational institutions already collect exam records, subject marks, attendance, prior academic performance, and behavioral or lifestyle data. The problem is that these signals usually live in separate files or systems, so faculty, mentors, and academic leaders often react only after students have already accumulated backlogs, low CGPA, or disengagement.

This platform solves that gap by **stitching multiple student datasets into one unified view** and turning them into **early-warning insights, subject-level analytics, and intervention-oriented predictions**. Instead of showing only charts, the system helps institutions answer three high-value questions:

1. **Which students are likely to struggle, and why?**
2. **Which subjects or semesters create the biggest learning gaps?**
3. **What support actions should faculty, mentors, or academic administrators prioritize?**

---

## Role-Based Problem Statement

Institutions miss timely intervention opportunities because student performance, behavior, and background data are fragmented.

- **Faculty challenge:** difficult subjects and weak learning clusters are hard to detect early.
- **Mentor challenge:** student-level warning signals are scattered, so support is delayed.
- **Academic leader challenge:** decision-making lacks a unified institutional risk view.

- Academic performance is analyzed separately from behavioral or lifestyle signals.
- Subject difficulty and backlog patterns are not connected to student-level risk.
- Faculty often identify weak students too late, after CGPA decline or repeated failures.
- Career readiness discussions happen without a complete view of academic consistency and risk factors.

---

## Our Solution

We built a **Student Success & Career Readiness Analytics Platform** that combines data engineering, analytics, and machine learning into one end-to-end workflow:

1. **Data Stitching Layer**
	Integrates multiple raw datasets into a unified student-level dataset.

2. **Analytics Layer**
	Reveals subject difficulty, SGPA/CGPA trends, backlog burden, attendance patterns, and lifestyle-performance relationships.

3. **Prediction Layer**
	Predicts academic performance and classifies at-risk students using machine learning.

4. **Decision Support Layer**
	Enables faculty, mentors, and academic leaders to identify learning gaps early and prioritize interventions.

---

## Why This Matters

This project is not just a dashboard. It is a **decision-support system for academic intervention**.

- **For faculty:** identify difficult subjects and weak performance clusters.
- **For mentors:** detect at-risk students before they fail multiple courses.
- **For academic leaders:** monitor institutional risk patterns and support planning.
- **For career guidance teams:** connect academic stability with readiness indicators.

---

## Demo Value For Judges

This project demonstrates a complete hackathon-worthy pipeline:

- **Real problem relevance:** fragmented student data and late intervention.
- **Technical depth:** multi-source data stitching, feature engineering, and ML.
- **Usable output:** interactive dashboard instead of static analysis.
- **Actionable insights:** identifies who is struggling, where the gaps are, and what patterns drive risk.

---

## One-Line Pitch

**We turn disconnected student records into a unified early-warning and academic intelligence platform that helps institutions detect risk, understand learning gaps, and improve student outcomes.**

---

## 🗂️ Project Structure

```
student-analytics-platform/
├── data/
│   ├── raw/                        # Original 4 datasets
│   └── processed/
│       ├── final_student_dataset.csv   # Unified stitched dataset (514 students × 59 features)
│       └── subject_summary.csv         # Subject-level analytics
│
├── src/
│   ├── data_loader.py              # Load all 4 raw datasets
│   ├── data_cleaning.py            # Clean & normalise each dataset
│   ├── data_stitching.py           # Merge → final_student_dataset.csv
│   ├── feature_engineering.py      # Derived features + risk flags + encoding
│   └── ml_models.py                # CGPA regression + risk classification models
│
├── dashboard/
│   └── app.py                      # Streamlit 5-page interactive dashboard
│
├── charts/                         # Pre-generated EDA charts (PNG)
├── requirements.txt
└── README.md
```

---

## 🔗 Data Stitching Strategy

### Current Streamlit App Layout

The platform now runs as a multi-page Streamlit app:

- `app.py` -> Overview page (entrypoint)
- `pages/1_Student_Dashboard.py`
- `pages/2_Faculty_Analytics.py`
- `pages/3_Mentor_Risk_Monitoring.py`
- `pages/4_Career_Placement_Insights.py`
- `pages/5_Lifestyle_Impact_Analytics.py`
- `pages/6_AI_Academic_Advisor.py`
- `pages/7_Data_Quality_ETL_Monitoring.py`
- `pages/8_Admin_Panel.py`
- `utils/` for shared data prep, scoring, and UI helpers

| Dataset | Rows | Key |
|---|---|---|
| Student Performance | 25,510 (514 students × semesters) | `Rollno` → pivot to 1 row/student |
| Attitude & Behaviour | 235 | Proportional mapping (no shared ID) |
| Student Behaviour | 235 | Averaged with Attitude (complementary source) |
| Research Student | 220 | Fuzzy join on `10th + 12th marks` |

**Final dataset: 514 students × 59 features**

---

## 🔧 Derived Features

| Feature | Formula |
|---|---|
| `average_marks` | mean(10th, 12th, college marks) |
| `academic_risk_score` | weighted: CGPA(40%) + backlogs(30%) + stress(15%) + attendance(15%) |
| `study_efficiency` | mean_cgpa / study_hours_per_day |
| `at_risk_student` | CGPA<5 OR backlogs>2 OR attendance<60% OR (high stress + low study) |
| `cgpa_band` | Fail / Poor / Average / Good / Excellent |

---

## 🤖 ML Models & Results

### CGPA Prediction (Regression)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.1048 | 0.0776 | **0.9834** |
| Decision Tree | 0.0941 | 0.0647 | **0.9866** |
| **Random Forest** | **0.0850** | **0.0508** | **0.9891** |

### Risk Classification
| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 99.03% | 0.995 |
| Decision Tree | 100.00% | 1.000 |
| Random Forest | 99.03% | 0.995 |

### Top Risk Factors (Feature Importance)
1. Academic Risk Score — 34.3%
2. Total Backlogs — 20.6%
3. Mean SGPA — 11.9%
4. Practical Marks — 7.3%

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the data pipeline (generates processed data)
cd src
python data_stitching.py
python feature_engineering.py
python warehouse_pipeline.py
python data_simulation.py --batches 3 --batch-size 40 --interval-seconds 1

# Optional: connect MySQL backend
set DATABASE_URL=mysql://root:9963@localhost:3306/studentanalyticdashboard

# 3. Launch the multi-page platform
streamlit run app.py

# 4. Dockerized deployment (app + mysql)
docker compose up --build
```

---

## 📊 Dashboard Pages

| Page | Contents |
|---|---|
| Overview | Institution KPIs, CGPA/subject/stress analytics, cross-persona insights |
| Student Dashboard | Student selector, predicted CGPA, readiness and personalized recommendations |
| Faculty Analytics | Subject difficulty heatmap, pass/fail trends, class performance trend |
| Mentor / Risk Monitoring | At-risk KPIs, risk reason table, attendance-stress-performance analysis |
| Career & Placement Insights | Readiness distribution, skill-gap analysis, career path recommendation |
| Lifestyle Impact Analytics | Study/sleep/stress/social impact on academic outcomes |
| AI Academic Advisor | Chat-style Q&A for risk, subjects, placement and student-specific insights |
| Data Quality & ETL Monitoring | Missing data heatmap, data distribution checks, ETL log visibility |
| Admin Panel | Upload data, trigger ETL, retrain models, refresh dashboard cache |

---

## 🏆 Key Insights

- **96.7% of students** are flagged at-risk under composite scoring — highlighting systemic issues
- **Backlogs** (20.6%) and **low attendance** are the strongest individual risk drivers
- **Study efficiency** drops sharply when social media exceeds 2 hours/day
- **Financial stress** correlates with 0.8 CGPA drop on average
- Students in early semesters (1-2) perform significantly worse than mid-program semesters
