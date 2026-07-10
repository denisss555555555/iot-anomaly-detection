import os
from datetime import datetime

OUTPUT_DIR = "results/project_report"

os.makedirs(OUTPUT_DIR, exist_ok=True)

report_path = os.path.join(
    OUTPUT_DIR,
    "project_report.txt"
)

content = f"""
ПРОЕКТЕН ОТЧЕТ

Тема:
Откриване на аномалии в IoT мрежов трафик чрез машинно обучение и обработка на големи данни

Дата:
{datetime.now().strftime("%d.%m.%Y %H:%M")}

Използван dataset:
- CICIoT2023

Използвани модели:
- Random Forest
- XGBoost
- LightGBM
- Isolation Forest

Използвани библиотеки:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib

Изпълнени задачи:
- preprocessing
- balancing
- multiclass classification
- binary classification
- anomaly detection
- feature importance analysis
- validation evaluation
- ROC-AUC analysis

Финален резултат:
Най-добри резултати показва моделът LightGBM.
"""

with open(report_path, "w", encoding="utf-8") as file:
    file.write(content)

print("Project report е готов.")