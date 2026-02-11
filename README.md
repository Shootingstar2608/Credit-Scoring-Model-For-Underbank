# 🏦 Home Credit Default Risk — Credit Scoring System

Hệ thống chấm điểm tín dụng dành cho **người chưa có lịch sử tín dụng** (unbanked), sử dụng dữ liệu từ cuộc thi [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) trên Kaggle.

Hệ thống kết hợp **LightGBM** + **SHAP Explainability** + **FICO Scoring** + **AI Advisor (OpenRouter)** trong một ứng dụng **Streamlit** hoàn chỉnh.

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Dữ Liệu](#-dữ-liệu)
- [Pipeline ML](#-pipeline-ml)
- [Webapp](#-webapp)
- [Cài Đặt & Chạy](#-cài-đặt--chạy)
- [AI Advisor (OpenRouter)](#-ai-advisor-openrouter)
- [Kết Quả](#-kết-quả)

---

## 🎯 Tổng Quan

| Thành phần | Chi tiết |
|-----------|---------|
| **Bài toán** | Binary classification — dự đoán khách hàng có vỡ nợ hay không |
| **Đối tượng** | Người chưa có lịch sử tín dụng (no credit bureau history) |
| **Model** | LightGBM (`n_estimators=1000, max_depth=6, learning_rate=0.03`) |
| **Features** | 48 features (34 raw + 14 engineered) |
| **Validation** | 5-fold Stratified K-Fold CV |
| **Baseline AUC** | **0.767** |
| **Enhanced AUC** | **0.781** (114 features, dùng thêm 3 bảng phụ) |
| **Scoring** | FICO 300–850 (log-odds transform) |
| **Explainability** | SHAP values — top 10 yếu tố ảnh hưởng |
| **AI Advisor** | Nhận xét tín dụng bằng tiếng Việt qua OpenRouter LLM |

---

## 🏗 Kiến Trúc Hệ Thống

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Streamlit   │────▶│  Scoring Engine   │────▶│  LightGBM Model  │
│  (app.py)    │     │  (engine.py)      │     │  (.pkl artifacts) │
└──────┬───────┘     └────────┬─────────┘     └──────────────────┘
       │                      │
       │                      ├──▶ FICO Score (300-850)
       │                      ├──▶ SHAP Explanation (top 10)
       │                      └──▶ Credit Limit Suggestion
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│  AI Advisor   │────▶│  OpenRouter API   │
│  (advisor.py) │     │  (100+ LLM models)│
└──────────────┘     └──────────────────┘
```

---

## 📂 Cấu Trúc Thư Mục

```
HomeCreditDefaultRisk/
├── README.md                          # File này
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── input/                             # Dữ liệu gốc (Kaggle) — git ignored
│   ├── application_train.csv          # 307,511 rows — bảng chính
│   ├── application_test.csv           # 48,744 rows — test set
│   ├── bureau.csv                     # Lịch sử tín dụng từ Bureau
│   ├── bureau_balance.csv             # Chi tiết số dư Bureau
│   ├── previous_application.csv       # Đơn vay trước đó tại Home Credit
│   ├── installments_payments.csv      # Lịch sử trả góp
│   ├── POS_CASH_balance.csv           # Số dư POS/Cash
│   ├── credit_card_balance.csv        # Số dư thẻ tín dụng
│   └── HomeCredit_columns_description.csv
│
├── notebooks/                         # Jupyter notebooks
│   ├── credit_scoring_no_history.ipynb      # ⭐ Pipeline chính (48 features, AUC 0.767)
│   ├── credit_scoring_enhanced.ipynb        # Pipeline mở rộng (114 features, AUC 0.781)
│   ├── start-here-a-gentle-introduction.ipynb
│   ├── home-credit-complete-eda-feature-importance.ipynb
│   ├── base-model-with-0-804-auc-on-home-credit.ipynb
│   └── mafs5440-jiang-song-wan-qiu.ipynb
│
├── model_artifacts/                   # Model đã train — git ignored
│   ├── lgbm_credit_scoring.pkl        # LightGBM model
│   ├── feature_names.json             # 48 feature names
│   ├── label_encoders.pkl             # Categorical encoders
│   ├── isotonic_calibrator.pkl        # Probability calibration
│   ├── shap_explainer.pkl             # SHAP TreeExplainer
│   └── feature_descriptions.json      # Feature metadata
│
├── preprocessed_dataset/              # Datasets đã xử lý
│   ├── credit_scoring_baseline_dataset.csv    # 307K × 55 cols (135 MB)
│   ├── credit_scoring_enhanced_dataset.csv    # 307K × 121 cols (266 MB)
│   └── feature_sets.json
│
├── analysis/                          # Kết quả phân tích
│   ├── feature_importance_comparison.png
│   ├── oof_distribution.png
│   └── oof_predictions.csv
│
├── image/                             # Ảnh minh họa
│
└── webapp/                            # 🌐 Streamlit Web Application
    ├── app.py                         # Main UI (form + results + gauge)
    ├── config.py                      # Business rules, FICO tiers, labels
    ├── Dockerfile                     # Docker deployment
    ├── README.md                      # Webapp-specific docs
    ├── report.md                      # Báo cáo kỹ thuật
    ├── .streamlit/config.toml         # Streamlit theme & server config
    └── scoring/
        ├── engine.py                  # CreditScoringEngine class
        └── advisor.py                 # AI Credit Advisor (OpenRouter)
```

---

## 📊 Dữ Liệu

Dữ liệu từ cuộc thi [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) trên Kaggle, gồm 8 bảng:

| Bảng | Rows | Mô tả |
|------|-----:|-------|
| `application_train` | 307,511 | Hồ sơ đăng ký vay (bảng chính) |
| `application_test` | 48,744 | Test set (không có TARGET) |
| `bureau` | 1,716,428 | Lịch sử tín dụng từ Credit Bureau |
| `bureau_balance` | 27,299,925 | Số dư hàng tháng từ Bureau |
| `previous_application` | 1,670,214 | Đơn vay trước đó tại Home Credit |
| `installments_payments` | 13,605,401 | Lịch sử trả góp |
| `POS_CASH_balance` | 10,001,358 | Số dư POS/cash hàng tháng |
| `credit_card_balance` | 3,840,312 | Số dư thẻ tín dụng hàng tháng |

### Đặc điểm đối tượng

Bài toán tập trung vào **người chưa có lịch sử tín dụng** — nhóm khách hàng mà các hệ thống truyền thống (credit bureau) không có dữ liệu để đánh giá. Thay vào đó, model sử dụng:

- **Thông tin nhân khẩu học**: tuổi, giới tính, học vấn, hôn nhân
- **Tài chính**: thu nhập, số tiền vay, kỳ hạn trả góp
- **Việc làm**: nghề nghiệp, thâm niên, loại tổ chức
- **Tài sản**: nhà ở, xe cộ, bất động sản
- **Điểm tín dụng thay thế** (EXT_SOURCE): viễn thông, tiện ích, TMĐT
- **Liên lạc**: số kênh liên lạc có thể xác minh
- **Mạng lưới xã hội**: số người quen có nợ xấu

---

## 🔬 Pipeline ML

### Baseline (48 features → AUC 0.767)

```
Raw Data (34 features)
    ↓ Cleaning (fillna, drop high-null)
    ↓ Feature Engineering (+14 features)
    │   ├── Financial ratios (credit/income, annuity/income, ...)
    │   ├── EXT_SOURCE aggregates (mean, prod, min, max)
    │   ├── Social default total
    │   ├── Age group binning
    │   └── Contact count
    ↓ Label Encoding (categoricals)
    ↓ LightGBM (5-fold StratifiedKFold)
    ↓ Isotonic Calibration
    → FICO Score (300-850)
```

### Enhanced (114 features → AUC 0.781)

Mở rộng thêm **66 features** từ 3 bảng phụ:
- `previous_application` → 22 features (số đơn, tỷ lệ duyệt, trung bình từ chối...)
- `installments_payments` → 24 features (tỷ lệ trả đúng hạn, trễ hạn TB...)
- `POS_CASH_balance` → 20 features (số tháng active, tỷ lệ hoàn thành...)

### FICO Scoring Formula

```
FICO = 600 − (40/ln2) × ln(odds) − penalties
```

Trong đó:
- `odds = p / (1-p)` với p = raw probability từ LightGBM
- Penalty: −20 điểm (no credit history), −15/source (neutral EXT_SOURCE), dampen high EXT_SOURCE

| Tier | FICO Range | Quyết định |
|------|:----------:|-----------|
| 🌟 Exceptional | 800–850 | Duyệt tự động, lãi suất tốt nhất |
| ✅ Very Good | 740–799 | Duyệt tự động, lãi suất ưu đãi |
| 👍 Good | 670–739 | Duyệt tự động, lãi suất tiêu chuẩn |
| ⚠️ Fair | 580–669 | Cần thẩm định viên xem xét |
| ❌ Poor | 300–579 | Từ chối |

---

## 🌐 Webapp

### Tính năng

- **Form nhập liệu** tiếng Việt — 7 nhóm thông tin với help text chi tiết
- **FICO Score Gauge** (300–850) — Plotly interactive
- **SHAP Explanation** — top 10 yếu tố ảnh hưởng (đã lọc bỏ feature "ẩn")
- **Đề xuất cải thiện** — tự động từ SHAP risk factors
- **Hạn mức tín dụng** — dựa trên FICO tier × thu nhập
- **AI Advisor** — nhận xét tín dụng chi tiết bằng tiếng Việt qua OpenRouter LLM
- **5 tier xếp hạng** — hiển thị trực quan vị trí người dùng

### Screenshots

Giao diện gồm:
1. **Sidebar**: Config AI + Form nhập liệu (7 nhóm)
2. **Main**: Gauge → KPI metrics → SHAP chart → Đề xuất → AI Advisor → Bảng xếp hạng

---

## 🚀 Cài Đặt & Chạy

### Yêu cầu

- Python ≥ 3.10
- Dữ liệu Kaggle trong `input/`
- Model artifacts trong `model_artifacts/` (train từ notebook)

### 1. Clone & cài dependencies

```bash
git clone <repo-url>
cd HomeCreditDefaultRisk
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Tải dữ liệu từ [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) và giải nén vào `input/`.

### 3. Train model (nếu chưa có artifacts)

Mở và chạy notebook:

```bash
jupyter notebook notebooks/credit_scoring_no_history.ipynb
```

Notebook sẽ tự động export model artifacts vào `model_artifacts/`.

### 4. Chạy webapp

```bash
cd webapp
streamlit run app.py
```

Truy cập http://localhost:8501

### 5. Docker (tùy chọn)

```bash
cd webapp
docker build -t credit-scoring .
docker run -p 8501:8501 credit-scoring
```

---

## 🤖 AI Advisor (OpenRouter)

Tích hợp [OpenRouter](https://openrouter.ai) — gateway truy cập hàng trăm model AI qua một API key duy nhất.

### Cách sử dụng

1. Đăng ký tại https://openrouter.ai → lấy API key miễn phí
2. Mở expander **"🤖 Nhận Xét AI"** trên sidebar
3. Nhập API key + chọn model
4. Nhấn **Đánh Giá** → nhận phân tích AI chi tiết

### Model gợi ý (miễn phí)

| Model | ID |
|-------|----|
| Gemini 2.0 Flash | `google/gemini-2.0-flash-exp:free` |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct:free` |
| Qwen3 235B | `qwen/qwen3-235b-a22b:free` |
| DeepSeek V3 | `deepseek/deepseek-chat-v3-0324:free` |
| Mistral Small 3.1 | `mistralai/mistral-small-3.1-24b-instruct:free` |

> Không có API key? Hệ thống tự động dùng **template-based assessment** — nhận xét tín dụng cấu trúc từ SHAP + rules, không cần internet.

### AI Advisor Output

Bản nhận xét AI bao gồm:
1. **Tổng quan** — đánh giá chung về hồ sơ
2. **Điểm mạnh** — yếu tố tích cực (SHAP safe)
3. **Điểm cần cải thiện** — yếu tố rủi ro (SHAP risk)
4. **Phân tích chi tiết** — giải thích từng yếu tố
5. **Khuyến nghị** — 3-5 hành động cụ thể
6. **Kết luận** — quyết định cuối cùng

---

## 📈 Kết Quả

### Model Performance

| Metric | Baseline (48 feat) | Enhanced (114 feat) |
|--------|:------------------:|:-------------------:|
| AUC ROC | 0.767 | 0.781 |
| Features | 48 | 114 |
| Data sources | 1 bảng | 4 bảng |
| CV Folds | 5 | 5 |

### Feature Importance (Top 10)

| # | Feature | Mô tả |
|---|---------|-------|
| 1 | EXT_SOURCE_2 | Điểm tiện ích (Điện/Nước) |
| 2 | EXT_SOURCE_3 | Điểm TMĐT |
| 3 | EXT_SOURCE_1 | Điểm viễn thông |
| 4 | AGE_YEARS | Tuổi |
| 5 | CREDIT_INCOME_RATIO | Tỷ lệ vay/thu nhập |
| 6 | AMT_GOODS_PRICE | Giá trị hàng hóa |
| 7 | EMPLOYMENT_YEARS | Thâm niên làm việc |
| 8 | AMT_CREDIT | Số tiền vay |
| 9 | ANNUITY_INCOME_RATIO | Trả góp/thu nhập |
| 10 | NAME_EDUCATION_TYPE | Trình độ học vấn |

---

## 🛠 Tech Stack

| Thành phần | Công nghệ |
|-----------|----------|
| ML Model | LightGBM 4.x |
| Explainability | SHAP (TreeExplainer) |
| Calibration | Isotonic Regression (sklearn) |
| Web Framework | Streamlit |
| Visualization | Plotly |
| AI Integration | OpenRouter API (OpenAI SDK) |
| Data Processing | Pandas, NumPy |
| Deployment | Docker |

---

## 📝 License

Dự án sử dụng dữ liệu từ cuộc thi Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk). Dữ liệu thuộc bản quyền của Home Credit Group.

---

## 👥 Credits

- **Home Credit Group** — Dữ liệu & bài toán
- **Kaggle Community** — Notebooks tham khảo (EDA, feature engineering)
- **OpenRouter** — LLM API gateway cho AI Advisor
