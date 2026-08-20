from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data_path = Path(__file__).parent / "data" / "employee_information_100.csv"
df = pd.read_csv(data_path)

# 1.1 Average salary of each department (bar chart)
df.groupby("Department")["Salary"].mean().plot(kind="bar", title="Average Salary by Department", ylabel="Salary")
plt.tight_layout()
plt.show()

# 1.2 Number of employees in each department
print("1.2 Number of employees in each department:")
print(df["Department"].value_counts())

# 1.3 Percentage of male and female employees (pie chart)
df["Gender"].value_counts().plot(kind="pie", autopct="%.1f%%", title="Gender Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 1.4 Salary distribution (histogram)
df["Salary"].plot(kind="hist", bins=10, title="Salary Distribution", edgecolor="black")
plt.xlabel("Salary")
plt.tight_layout()
plt.show()

# 1.5 Relationship between experience and salary (scatter plot)
df.plot(kind="scatter", x="Experience_Years", y="Salary", title="Experience vs Salary")
plt.tight_layout()
plt.show()

# 1.6 Top 10 employees with highest salary
print("\n1.6 Top 10 employees with highest salary:")
print(df.nlargest(10, "Salary")[["Employee_ID", "Name", "Department", "Salary"]])

# 1.7 Highest salary in every department
print("\n1.7 Highest salary in every department:")
print(df.groupby("Department")["Salary"].max())

# 1.8 Employees whose salary is greater than the overall average salary
avg_sal = df["Salary"].mean()
print(f"\n1.8 Employees with salary > overall average ({avg_sal:.2f}):")
print(df[df["Salary"] > avg_sal][["Employee_ID", "Name", "Department", "Salary"]])

# 1.9 Average years of experience in each department
print("\n1.9 Average years of experience in each department:")
print(df.groupby("Department")["Experience_Years"].mean())

# 1.10 Distribution of employee ages (histogram)
df["Age"].plot(kind="hist", bins=10, title="Age Distribution", edgecolor="black")
plt.xlabel("Age")
plt.tight_layout()
plt.show()
