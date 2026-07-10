import os
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

# ============================================================
# FONT
# ============================================================

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"

arial_path = r"C:\Windows\Fonts\arial.ttf"
arial_bold_path = r"C:\Windows\Fonts\arialbd.ttf"

pdfmetrics.registerFont(TTFont(FONT_REGULAR, arial_path))
pdfmetrics.registerFont(TTFont(FONT_BOLD, arial_bold_path))

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "results/final_pdf_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PDF_PATH = os.path.join(
    OUTPUT_DIR,
    "iot_ids_experimental_report_readable.pdf"
)

# ============================================================
# STYLES
# ============================================================

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="TitleBG",
    parent=styles["Title"],
    fontName=FONT_BOLD,
    fontSize=18,
    leading=24,
    alignment=1,
    spaceAfter=20
))

styles.add(ParagraphStyle(
    name="HeadingBG",
    parent=styles["Heading1"],
    fontName=FONT_BOLD,
    fontSize=14,
    leading=18,
    spaceBefore=14,
    spaceAfter=8
))

styles.add(ParagraphStyle(
    name="TextBG",
    parent=styles["BodyText"],
    fontName=FONT_REGULAR,
    fontSize=10,
    leading=14,
    spaceAfter=8
))

styles.add(ParagraphStyle(
    name="CaptionBG",
    parent=styles["BodyText"],
    fontName=FONT_REGULAR,
    fontSize=9,
    leading=12,
    alignment=1,
    spaceAfter=8
))

# ============================================================
# HELPERS
# ============================================================

def add_table(elements, data, col_widths=None):
    table = Table(data, hAlign="LEFT", colWidths=col_widths)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))


def add_image(elements, path, caption, max_width=16 * cm, max_height=18 * cm):
    if not os.path.exists(path):
        return

    img = Image(path)

    img_width = img.imageWidth
    img_height = img.imageHeight

    scale = min(max_width / img_width, max_height / img_height)

    img.drawWidth = img_width * scale
    img.drawHeight = img_height * scale

    elements.append(Paragraph(caption, styles["CaptionBG"]))
    elements.append(img)
    elements.append(Spacer(1, 14))


def read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ============================================================
# DOCUMENT
# ============================================================

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    rightMargin=1.5 * cm,
    leftMargin=1.5 * cm,
    topMargin=1.5 * cm,
    bottomMargin=1.5 * cm
)

elements = []

# ============================================================
# TITLE
# ============================================================

elements.append(Paragraph(
    "Автоматично генериран експериментален отчет",
    styles["TitleBG"]
))

elements.append(Paragraph(
    "Тема: Откриване на аномалии в IoT мрежов трафик чрез машинно обучение и обработка на големи данни",
    styles["TextBG"]
))

elements.append(Paragraph(
    f"Дата на генериране: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
    styles["TextBG"]
))

elements.append(Spacer(1, 16))

# ============================================================
# SUMMARY
# ============================================================

elements.append(Paragraph("1. Обобщение", styles["HeadingBG"]))

elements.append(Paragraph(
    "Настоящият отчет представя резултатите от проведените експерименти върху CICIoT2023 dataset. "
    "Използвани са Random Forest, XGBoost, LightGBM, Isolation Forest и Ensemble Voting модели. "
    "Допълнително са включени SHAP анализ, PCA визуализация, cross-validation и симулация на поток от данни.",
    styles["TextBG"]
))

# ============================================================
# MODEL RESULTS
# ============================================================

elements.append(Paragraph("2. Основни резултати на моделите", styles["HeadingBG"]))

model_table = [
    ["Модел", "Accuracy", "Macro F1", "Weighted F1"],
    ["Random Forest", "97.35%", "0.96", "0.97"],
    ["XGBoost", "97.48%", "0.97", "0.97"],
    ["LightGBM", "97.63%", "0.97", "0.98"],
]

add_table(elements, model_table)

elements.append(Paragraph(
    "LightGBM показва най-добри резултати и е избран като основен модел за финална оценка.",
    styles["TextBG"]
))

# ============================================================
# MODEL COMPARISON
# ============================================================

elements.append(Paragraph("3. Сравнение на моделите", styles["HeadingBG"]))

add_image(
    elements,
    "results/model_comparison/model_comparison_plot.png",
    "Фигура 1. Сравнение между Random Forest, XGBoost и LightGBM.",
    max_width=15 * cm,
    max_height=10 * cm
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

elements.append(Paragraph("4. Validation Confusion Matrix", styles["HeadingBG"]))

add_image(
    elements,
    "results/validation_results/validation_confusion_matrix.png",
    "Фигура 2. Validation confusion matrix за LightGBM.",
    max_width=17 * cm,
    max_height=16 * cm
)

elements.append(PageBreak())

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elements.append(Paragraph("5. Feature Importance", styles["HeadingBG"]))

fi_df = read_csv_if_exists("results/feature_importance/lightgbm_feature_importance.csv")

if fi_df is not None:
    top10 = fi_df.head(10)

    fi_table = [["Feature", "Importance"]]

    for _, row in top10.iterrows():
        fi_table.append([
            str(row["Feature"]),
            str(row["Importance"])
        ])

    add_table(elements, fi_table)

add_image(
    elements,
    "results/feature_importance/lightgbm_top20_features.png",
    "Фигура 3. Top 20 най-важни характеристики за LightGBM.",
    max_width=16 * cm,
    max_height=12 * cm
)

# ============================================================
# SHAP
# ============================================================

elements.append(Paragraph("6. SHAP анализ", styles["HeadingBG"]))

elements.append(Paragraph(
    "SHAP анализът показва кои характеристики влияят най-силно върху решенията на LightGBM модела.",
    styles["TextBG"]
))

add_image(
    elements,
    "results/shap_analysis/lightgbm_shap_bar.png",
    "Фигура 4. SHAP bar plot.",
    max_width=16 * cm,
    max_height=10 * cm
)

add_image(
    elements,
    "results/shap_analysis/lightgbm_shap_summary.png",
    "Фигура 5. SHAP summary plot.",
    max_width=16 * cm,
    max_height=12 * cm
)

elements.append(PageBreak())

# ============================================================
# ROC-AUC
# ============================================================

elements.append(Paragraph("7. ROC-AUC анализ", styles["HeadingBG"]))

add_image(
    elements,
    "results/roc_auc/roc_auc_curve.png",
    "Фигура 6. ROC-AUC крива за бинарна класификация.",
    max_width=14 * cm,
    max_height=12 * cm
)

# ============================================================
# PCA
# ============================================================

elements.append(Paragraph("8. PCA визуализация", styles["HeadingBG"]))

add_image(
    elements,
    "results/pca_visualization/pca_2d_visualization.png",
    "Фигура 7. PCA визуализация на IoT мрежовия трафик.",
    max_width=16 * cm,
    max_height=12 * cm
)

# ============================================================
# STREAM SIMULATION
# ============================================================

elements.append(Paragraph("9. Stream simulation", styles["HeadingBG"]))

stream_df = read_csv_if_exists("results/stream_simulation/stream_simulation_summary.csv")

if stream_df is not None:
    row = stream_df.iloc[0]

    stream_table = [
        ["Показател", "Стойност"],
        ["Общо записи", int(row["total_records"])],
        ["Общо време", f"{row['total_time_seconds']:.4f} sec"],
        ["Средна latency", f"{row['average_latency_seconds']:.4f} sec"],
        ["Среден throughput", f"{row['average_throughput_records_per_second']:.2f} records/sec"],
    ]

    add_table(elements, stream_table)

elements.append(Paragraph(
    "Резултатите показват, че системата може да обработва голям брой записи за кратко време, "
    "което я прави подходяща за near-real-time IDS сценарии.",
    styles["TextBG"]
))

# ============================================================
# CROSS VALIDATION
# ============================================================

elements.append(Paragraph("10. Cross-validation", styles["HeadingBG"]))

cv_df = read_csv_if_exists("results/cross_validation/lightgbm_cross_validation.csv")

if cv_df is not None:
    cv_table = [["Fold", "Weighted F1"]]

    for _, row in cv_df.iterrows():
        cv_table.append([
            str(row["fold"]),
            str(row["weighted_f1"])
        ])

    add_table(elements, cv_table)

# ============================================================
# ALL CLASSES
# ============================================================

elements.append(Paragraph("11. Експеримент с всички класове", styles["HeadingBG"]))

elements.append(Paragraph(
    "Проведен е допълнителен експеримент с всички налични класове от CICIoT2023 dataset. "
    "Целта е да се оцени устойчивостта на модела при по-сложна и небалансирана многокласова класификация.",
    styles["TextBG"]
))

add_image(
    elements,
    "results/main_vs_all_classes/main_vs_all_classes_plot.png",
    "Фигура 8. Сравнение между основния и разширения експеримент.",
    max_width=15 * cm,
    max_height=10 * cm
)

# ============================================================
# CONCLUSION
# ============================================================

elements.append(Paragraph("12. Заключение", styles["HeadingBG"]))

elements.append(Paragraph(
    "Проведените експерименти показват, че LightGBM е най-подходящият модел за откриване "
    "на аномалии и атаки в IoT мрежов трафик. Моделът постига най-високи резултати при "
    "многокласова класификация и показва стабилност при validation и cross-validation. "
    "Допълнителните експерименти със SHAP, PCA, stream simulation и dashboard разширяват "
    "практическата стойност на проекта.",
    styles["TextBG"]
))

# ============================================================
# BUILD
# ============================================================

doc.build(elements)

print("=" * 70)
print("PDF отчетът е създаден успешно.")
print(f"Файл: {PDF_PATH}")
print("=" * 70)