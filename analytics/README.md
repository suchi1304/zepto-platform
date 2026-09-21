# Module 2: Titanic Analytics & Predictive Modeling Pipeline

## 1. Missing-Value Handling Strategy & Percentages
* `embarked` / `embark_town`: **0.22% missing (< 5% threshold)** -> Affected rows dropped.
* `age`: **19.87% missing (5%–30% threshold)** -> Median imputation applied.
* `deck`: **77.10% missing (> 30% threshold)** -> Column dropped due to extreme sparsity.

## 2. Univariate Outliers & Skewness Analysis
* **IQR Outliers**:
  * `age`: IQR = 13.00, Lower = 2.50, Upper = 54.50. Outliers: 66
  * `fare`: IQR = 23.09, Lower = -26.72, Upper = 65.63. Outliers: 116
* **Fare Skewness**:
  * Mean: 32.10 | Median: 14.45 | Mode: 8.05
  * **Conclusion**: Fare distribution is **right-skewed** because `Mean > Median > Mode`.

## 3. Bivariate Survival Breakdowns & Correlation Analysis
* **Survival Rates**:
  * By Sex: Female: 74.20% | Male: 18.89%
  * By Pclass: 1st: 62.96% | 2nd: 47.28% | 3rd: 24.24%
  * By Sex + Pclass: 1st Class Female: 96.81%, 3rd Class Male: 13.54%
* **6x6 Correlation Matrix**: Evaluated on `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare` (`adult_male` and `alone` excluded as derived flags).
* **Two Strongest Off-Diagonal Correlations**:
  1. `pclass` & `fare` (r = -0.549): Strong inverse relationship reflecting higher ticket prices for first-class cabins.
  2. `sibsp` & `parch` (r = 0.415): Moderate positive correlation indicating passengers often traveled in family units.

## 4. Multivariate Data Story (4 Charts)
1. **Chart 1 (`chart1_sex_pclass_survival.png`)**: Shows survival prioritization of females across all passenger tiers, with 1st-class females exceeding 95% survival.
2. **Chart 2 (`chart2_age_fare_survival.png`)**: Demonstrates higher survival density in upper fare regions across all age groups.
3. **Chart 3 (`chart3_pclass_fare_survival.png`)**: Highlights extreme fare outliers concentrated in 1st class, directly mapping to improved evacuation priority.
4. **Chart 4 (`chart4_family_pclass_survival.png`)**: Demonstrates that moderate family sizes (2–4 members) had better survival rates than solo travelers or large families.

## 5. Model Comparison Table

| Model Group | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MAE | RMSE | R² | Adj-R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classifier** | Logistic Regression | 0.7978 | 0.7612 | 0.7391 | 0.7500 | 0.8465 | - | - | - | - |
| **Classifier** | Decision Tree | 0.8202 | 0.8333 | 0.6812 | 0.7495 | 0.8378 | - | - | - | - |
| **Classifier** | Random Forest (Tuned) | **0.8371** | **0.8485** | **0.7101** | **0.7731** | **0.8652** | - | - | - | - |
| **Regression** | Linear Regression (`fare`) | - | - | - | - | - | **19.85** | **33.42** | **0.3851** | **0.3640** |

* **Heteroscedasticity Conclusion**: The residual plot for the fare regression exhibits a classic outward fanning pattern as predicted values rise, indicating heteroscedasticity due to high ticket price variance in luxury berths.
* **Imbalance Handling Conclusion**: Applying SMOTE on the training fold improved recall for non-survivors without compromising precision compared to class-weighted loss.
* **Final Model Recommendation**: The **Tuned Random Forest Classifier** is selected for deployment due to superior generalization balance (ROC-AUC: 0.8652, F1: 0.7731, OOB Score: 0.8123), maintaining high precision while minimizing false survivorship positives.