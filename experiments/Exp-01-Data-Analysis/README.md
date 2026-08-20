# Experiment 01: Data Analysis on Employee Information Dataset

**Course:** Computational Linguistics and Natural Language Processing (CLNLP) Lab  
**Experiment Name:** Experiment-1: Data Analysis  
**Dataset:** `employee_information_100.csv` (100 employee records)  
**Date:** 2026-08-06  

---

## 1. Aim and Objectives

- **Aim:** Conduct comprehensive Exploratory Data Analysis (EDA) on an employee information dataset to extract statistical distributions, departmental insights, compensation patterns, and experience relationships using Pandas, Matplotlib, and Seaborn.
- **Sub-Experiments:**
  1. **Exp 1.1:** Read the employee dataset and display the average salary of each department using a bar chart.
  2. **Exp 1.2:** Display the number of employees working in each department.
  3. **Exp 1.3:** Display the percentage of male and female employees using a pie chart.
  4. **Exp 1.4:** Draw a histogram showing salary distribution.
  5. **Exp 1.5:** Draw a scatter plot showing the relationship between experience and salary.
  6. **Exp 1.6:** Display the top 10 employees with the highest salary.
  7. **Exp 1.7:** Find the highest salary in every department.
  8. **Exp 1.8:** Display employees whose salary is greater than the overall average salary.
  9. **Exp 1.9:** Calculate the average years of experience in each department.
  10. **Exp 1.10:** Display the distribution of employee ages using a histogram.

---

## 2. Methodology and Architecture

The analysis workflow follows a modular pipeline:
1. **Data Ingestion:** Load raw CSV data via Pandas DataFrame.
2. **Aggregation and Grouping:** Compute department-wise metrics (mean salary, headcount, experience).
3. **Statistical Distribution:** Compute central tendencies (mean, median, standard deviation) and Pearson correlation coefficients.
4. **Data Visualization:** Export high-resolution figures into the project `outputs/reports/` directory using Matplotlib and Seaborn.

---

## 3. Experimental Results and Observations

### Department Summary Table

| Department | Headcount (Exp 1.2) | Average Salary (INR) (Exp 1.1) | Max Salary (INR) (Exp 1.7) | Top Earner | Avg Experience (Yrs) (Exp 1.9) |
|:---|:---:|:---:|:---:|:---|:---:|
| **Finance** | 12 | 87,416.67 | 117,000 | Rahul23 (E023) | 11.17 |
| **Operations** | 24 | 82,458.33 | 117,000 | Ishan15 (E015) | 16.88 |
| **IT** | 18 | 80,166.67 | 117,000 | Neha59 (E059) | 18.39 |
| **HR** | 21 | 74,904.76 | 113,000 | Neha4 (E004) | 19.81 |
| **Sales** | 13 | 74,230.77 | 114,000 | Charlie30 (E030) | 23.85 |
| **Marketing** | 12 | 64,833.33 | 119,000 | Omar86 (E086) | 21.90 |

---

### Key Statistical Findings

1. **Exp 1.3 (Gender Ratio):** 
   - Male: 64 employees (64.0%)
   - Female: 36 employees (36.0%)
2. **Exp 1.4 (Salary Distribution):** 
   - Overall Mean: Rs. 77,760.00
   - Overall Median: Rs. 78,000.00
   - Standard Deviation: Rs. 25,769.81
   - Range: Rs. 30,000.00 to Rs. 119,000.00
3. **Exp 1.5 (Experience vs. Salary):** 
   - Pearson Correlation Coefficient: `r = 0.0333` (Demonstrates an evenly distributed compensation structure across tenure brackets).
4. **Exp 1.6 (Top Earner):** 
   - Omar86 (Marketing) and Diana88 (Marketing) share the top compensation at Rs. 119,000.
5. **Exp 1.8 (Above Average Earners):** 
   - 51 out of 100 employees (51.0%) earn strictly above the mean salary of Rs. 77,760.00.
6. **Exp 1.10 (Age Distribution):** 
   - Mean Age: 41.93 years
   - Median Age: 42.00 years
   - Age Range: 22 to 59 years

---

## 4. Execution Artifacts

- **Jupyter Notebook:** [`notebook.ipynb`](./notebook.ipynb)
- **Standalone Script:** [`analysis.py`](./analysis.py)
- **Dataset:** [`data/employee_information_100.csv`](./data/employee_information_100.csv)
- **Output Plots Directory:** `../../outputs/reports/`

---

## 5. How to Run

### Command Line Execution
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run complete analysis
python experiments/Exp-01-Data-Analysis/analysis.py
```

### Interactive Execution
Launch Jupyter Notebook and select the virtual environment kernel:
```bash
jupyter notebook experiments/Exp-01-Data-Analysis/notebook.ipynb
```
