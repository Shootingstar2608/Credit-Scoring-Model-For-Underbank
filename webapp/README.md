# Credit Scoring Web Application

Ứng dụng đánh giá tín dụng cho người **chưa có lịch sử tín dụng**, sử dụng LightGBM + SHAP.

## Kiến trúc

```
webapp/
├── app.py                  # Streamlit UI chính
├── scoring/
│   ├── __init__.py
│   ├── engine.py           # Inference engine (feature engineering + predict)
│   └── explainer.py        # SHAP explanation generator
├── config.py               # Cấu hình business rules & thresholds
├── requirements.txt        # Dependencies
├── Dockerfile              # Container hóa
└── .streamlit/
    └── config.toml         # Streamlit theme config
```

## Cài đặt & Chạy

### Local
```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

### Docker
```bash
cd webapp
docker build -t credit-scoring .
docker run -p 8501:8501 credit-scoring
```

## Features
- Form nhập thông tin cá nhân (7 nhóm feature)
- Credit Score 0-1000 với gauge chart
- Tiered Approval System (4 tầng)
- SHAP explanation cho từng dự đoán
- Credit limit recommendation
- Responsive UI tiếng Việt
