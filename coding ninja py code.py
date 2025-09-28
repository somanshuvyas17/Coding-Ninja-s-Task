import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# 1. Data Loading and Exploration
# =================================================================
# Load the dataset
df = pd.read_csv("student_wellbeing_dataset.csv")

print("--- Initial Data Exploration ---")
print(f"Original shape: {df.shape}")
# print(df.info())
# print(df.head())

# Check for duplicates and missing values
duplicate_count = df.duplicated().sum()
missing_values = df.isnull().sum()
print(f"Number of duplicate rows: {duplicate_count}")
# print("Missing values:\n", missing_values[missing_values > 0])


# =================================================================
# 2. Data Preprocessing
# =================================================================

# Handle Duplicates
df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df.shape}")

# Handle Missing Values (Imputation with Mean)
# The missing values are clustered in Sleep_Hours, Screen_Time, and Attendance.
mean_sleep = df['Sleep_Hours'].mean()
mean_screen = df['Screen_Time'].mean()
mean_attendance = df['Attendance'].mean()

df['Sleep_Hours'].fillna(mean_sleep, inplace=True)
df['Screen_Time'].fillna(mean_screen, inplace=True)
df['Attendance'].fillna(mean_attendance, inplace=True)

# Convert Categorical Values to Numeric

# Extracurricular: Convert 'Yes'/'No' to 1/0
df['Extracurricular_Numeric'] = df['Extracurricular'].map({'Yes': 1, 'No': 0})

# Stress_Level: Convert to ordinal (Low=0, Medium=1, High=2)
stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Stress_Level_Numeric'] = df['Stress_Level'].map(stress_mapping)

# Drop original categorical and ID columns for core analysis
df_cleaned = df.drop(columns=['Student_ID', 'Extracurricular', 'Stress_Level'])

print("--- Preprocessing Complete ---")
# print(df_cleaned.head())


# =================================================================
# 3. Exploratory Data Analysis (EDA)
# =================================================================

# A. Find relationships between study/sleep/screen time and CGPA (Correlation)
print("\n--- EDA: Correlation Analysis ---")
correlation_matrix = df_cleaned[['Hours_Study', 'Sleep_Hours', 'Screen_Time', 'CGPA', 'Attendance']].corr()
cgpa_correlations = correlation_matrix['CGPA'].sort_values(ascending=False)
print("Correlation with CGPA:\n", cgpa_correlations)

# B. Visualization 1: Scatter plots for Hours_Study, Sleep_Hours, Screen_Time vs CGPA
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Relationship between Student Activities and CGPA', fontsize=16)

sns.regplot(ax=axes[0], x='Hours_Study', y='CGPA', data=df_cleaned, scatter_kws={'alpha':0.3, 's':10})
axes[0].set_title('Study Hours vs. CGPA')
axes[0].set_xlabel('Hours of Study')
axes[0].set_ylabel('CGPA')

sns.regplot(ax=axes[1], x='Sleep_Hours', y='CGPA', data=df_cleaned, scatter_kws={'alpha':0.3, 's':10})
axes[1].set_title('Sleep Hours vs. CGPA')
axes[1].set_xlabel('Sleep Hours')
axes[1].set_ylabel('CGPA')

sns.regplot(ax=axes[2], x='Screen_Time', y='CGPA', data=df_cleaned, scatter_kws={'alpha':0.3, 's':10})
axes[2].set_title('Screen Time vs. CGPA')
axes[2].set_xlabel('Screen Time (Hours)')
axes[2].set_ylabel('CGPA')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('activity_cgpa_relationships.png')
plt.close()

# C. Visualization 2: CGPA across different stress levels (Box Plot)
df_viz = df.copy() # Use the DataFrame with original categorical labels for clearer plotting
stress_level_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
df_viz['Stress_Level_Label'] = df_viz['Stress_Level_Numeric'].map(stress_level_labels)

plt.figure(figsize=(7, 6))
sns.boxplot(x='Stress_Level_Label', y='CGPA', data=df_viz.sort_values('Stress_Level_Numeric'), palette='Set2', order=['Low', 'Medium', 'High'])
plt.title('CGPA Distribution Across Stress Levels')
plt.xlabel('Stress Level')
plt.ylabel('CGPA')
# plt.savefig('cgpa_by_stress_level.png')
plt.close()

# D. Visualization 3: CGPA for students with/without extracurriculars (Box Plot)
extracurricular_labels = {0: 'No', 1: 'Yes'}
df_viz['Extracurricular_Label'] = df_viz['Extracurricular_Numeric'].map(extracurricular_labels)

plt.figure(figsize=(7, 6))
sns.boxplot(x='Extracurricular_Label', y='CGPA', data=df_viz, palette='Pastel1')
plt.title('CGPA Distribution: Extracurricular vs. No Extracurricular')
plt.xlabel('Participates in Extracurriculars')
plt.ylabel('CGPA')
# plt.savefig('cgpa_by_extracurricular.png')
plt.close()

# E. Summary Statistics for Insights
print("\n--- EDA: Summary Statistics ---")
cgpa_by_stress = df_viz.groupby('Stress_Level_Label')['CGPA'].agg(['mean', 'std']).sort_values('mean', ascending=False)
print("CGPA Summary by Stress Level:\n", cgpa_by_stress)

cgpa_by_extracurricular = df_viz.groupby('Extracurricular_Label')['CGPA'].agg(['mean', 'std']).sort_values('mean', ascending=False)
print("\nCGPA Summary by Extracurricular Participation:\n", cgpa_by_extracurricular)

# =================================================================
# 4. Export Cleaned Dataset (Re-exporting for confirmation/completeness)
# =================================================================
# Prepare final cleaned dataset
final_cleaned_df = df.drop(columns=['Extracurricular', 'Stress_Level'])
final_cleaned_df.to_csv('student_wellbeing_cleaned.csv', index=False)
print("\nCleaned dataset successfully exported to student_wellbeing_cleaned.csv.")