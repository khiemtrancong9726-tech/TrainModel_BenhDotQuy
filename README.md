# Train Model Dự Đoán Bệnh Đột Quỵ (Stroke Prediction)

Dự án xây dựng mô hình Machine Learning để dự đoán nguy cơ đột quỵ (stroke) dựa trên các chỉ số sức khỏe và nhân khẩu học của bệnh nhân.

---

## Mục tiêu

Phân loại nhị phân: dự đoán khả năng một bệnh nhân bị đột quỵ (`stroke = 1`) hay không (`stroke = 0`) dựa trên các đặc trưng đầu vào.

Do tính chất y tế của bài toán, mô hình được tối ưu để **tối đa hóa Recall** (giảm thiểu bỏ sót ca bệnh), thay vì Accuracy đơn thuần.

---

## Dataset

| Thông tin | Chi tiết |
|---|---|
| File | `File dữ liệu thô.csv` |
| Số mẫu | ~5.110 bệnh nhân |
| Tỷ lệ mất cân bằng | ~4.8% ca đột quỵ (Imbalanced Classification) |

**Các đặc trưng đầu vào:**

| Feature | Mô tả |
|---|---|
| `age` | Tuổi bệnh nhân |
| `gender` | Giới tính (Male / Female) |
| `hypertension` | Cao huyết áp (0/1) |
| `heart_disease` | Bệnh tim (0/1) |
| `ever_married` | Đã từng kết hôn |
| `work_type` | Loại công việc |
| `Residence_type` | Khu vực sinh sống (Urban / Rural) |
| `avg_glucose_level` | Chỉ số glucose trung bình |
| `bmi` | Chỉ số khối cơ thể |
| `smoking_status` | Tình trạng hút thuốc |

**Target:** `stroke` — 0: Không đột quỵ, 1: Đột quỵ

---

## Quy trình thực hiện

### Phần 1 — Tải & Kiểm tra dữ liệu
- Import thư viện và cấu hình môi trường
- Load dataset, loại bỏ cột `id` (không có giá trị dự báo)
- Chuyển đổi kiểu dữ liệu sang `category` cho các biến phân loại

### Phần 2 — Khám phá dữ liệu (EDA)
- **Missing values:** Chỉ có `bmi` bị thiếu → dùng Median Imputation
- **Correlation:** `age` có tương quan cao nhất với nguy cơ đột quỵ
- **Class imbalance:** Tỷ lệ lớp 1 chỉ ~4.8% → cần chiến lược đặc biệt
- **Outliers:** `avg_glucose_level` và `bmi` có phân phối lệch → giữ lại vì liên quan bệnh lý
- **Data cleaning:** Loại bỏ `gender = 'Other'` (chỉ có 1 mẫu, gây nhiễu khi encoding)

### Phần 3 — Tiền xử lý dữ liệu
- **Pipeline số:** Median Imputation + StandardScaler
- **Pipeline phân loại:** Most-Frequent Imputation + OneHotEncoder
- **Chia dữ liệu:** Train (70%) / Validation (15%) / Test (15%)

### Phần 4 — Lựa chọn mô hình
So sánh 6 thuật toán bằng **Recall** (Stratified K-Fold, k=5) với `class_weight='balanced'`:

| Mô hình | Ghi chú |
|---|---|
| Logistic Regression | **Mô hình tốt nhất** |
| Decision Tree | |
| Random Forest | |
| SVM (SVC) | |
| KNN | |
| XGBoost | `scale_pos_weight=19` để xử lý imbalance |

### Phần 5 — Huấn luyện & Tinh chỉnh ngưỡng
- **Mô hình cuối:** Logistic Regression với `class_weight='balanced'`
- **Threshold Tuning:** Tìm ngưỡng tối ưu theo F1-Score thay vì dùng ngưỡng mặc định 0.5
- **Human-in-the-loop:** Trực quan hóa Precision-Recall theo ngưỡng để chuyên gia y tế tự chọn điểm cân bằng phù hợp

### Phần 6 — Đánh giá trên Test Set
- Áp dụng ngưỡng đã chọn lên tập Test (chỉ dùng 1 lần duy nhất)
- Báo cáo cuối: Recall, Precision, F1-Score, ROC-AUC

---

## Cách chạy

```bash
# 1. Cài đặt thư viện
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 2. Đặt file dữ liệu đúng đường dẫn
# Mặc định notebook đọc từ: ./Data_Stroke/stroke_data_raw.csv
# Hoặc chỉnh lại đường dẫn trong Cell 4 của notebook

# 3. Chạy toàn bộ notebook
jupyter notebook source.ipynb
```

---

## Công nghệ sử dụng

- **Python 3**
- `pandas`, `numpy` — Xử lý dữ liệu
- `matplotlib`, `seaborn` — Trực quan hóa
- `scikit-learn` — Pipeline, preprocessing, mô hình, đánh giá
- `xgboost` — XGBoost classifier

---

## Lưu ý quan trọng

> Mô hình này chỉ mang tính tham khảo và nghiên cứu học thuật. **Không dùng để thay thế chẩn đoán y tế chuyên nghiệp.**
