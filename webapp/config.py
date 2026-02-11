"""
Business rules & application configuration.
"""

# ============================================================
# TIERED APPROVAL SYSTEM  (FICO 300 – 850)
# ============================================================
TIERS = [
    {
        "name": "Exceptional (Xuất sắc)",
        "score_min": 800,
        "score_max": 850,
        "color": "#00c853",
        "icon": "🌟",
        "interest_modifier": -0.03,  # Giảm 3% lãi suất
        "description": "Hồ sơ xuất sắc — Duyệt tự động, lãi suất tốt nhất",
        "action": "AUTO_APPROVE_PREFERRED",
    },
    {
        "name": "Very Good (Rất tốt)",
        "score_min": 740,
        "score_max": 799,
        "color": "#43a047",
        "icon": "✅",
        "interest_modifier": -0.01,
        "description": "Hồ sơ rất tốt — Duyệt tự động, lãi suất ưu đãi",
        "action": "AUTO_APPROVE_STANDARD",
    },
    {
        "name": "Good (Tốt)",
        "score_min": 670,
        "score_max": 739,
        "color": "#2196f3",
        "icon": "👍",
        "interest_modifier": 0.0,
        "description": "Hồ sơ tốt — Duyệt tự động, lãi suất tiêu chuẩn",
        "action": "AUTO_APPROVE_STANDARD",
    },
    {
        "name": "Fair (Trung bình)",
        "score_min": 580,
        "score_max": 669,
        "color": "#ff9800",
        "icon": "⚠️",
        "interest_modifier": 0.04,  # Tăng 4% lãi suất
        "description": "Hồ sơ trung bình — Cần thẩm định viên xem xét thủ công",
        "action": "MANUAL_REVIEW",
    },
    {
        "name": "Poor (Kém)",
        "score_min": 300,
        "score_max": 579,
        "color": "#f44336",
        "icon": "❌",
        "interest_modifier": None,
        "description": "Hồ sơ rủi ro cao — Từ chối, đề xuất cải thiện",
        "action": "REJECT",
    },
]

# ============================================================
# CREDIT LIMIT CALCULATION
# ============================================================
BASE_CREDIT_LIMIT = 50_000_000  # 50 triệu VND
INCOME_MULTIPLIER_MAX = 10  # Tối đa 10x thu nhập năm
MIN_CREDIT_LIMIT = 5_000_000  # 5 triệu VND
MAX_CREDIT_LIMIT = 500_000_000  # 500 triệu VND

# ============================================================
# RISK LEVEL MAPPING (cho hiển thị đơn giản)
# ============================================================
RISK_LEVELS = {
    "VERY_LOW": {"label": "Rất thấp", "color": "#00c853", "emoji": "🟢"},
    "LOW": {"label": "Thấp", "color": "#66bb6a", "emoji": "🟢"},
    "MEDIUM": {"label": "Trung bình", "color": "#ff9800", "emoji": "🟡"},
    "HIGH": {"label": "Cao", "color": "#f44336", "emoji": "🔴"},
    "VERY_HIGH": {"label": "Rất cao", "color": "#b71c1c", "emoji": "🔴"},
}

# ============================================================
# MODEL ARTIFACTS PATH
# ============================================================
MODEL_DIR = "../model_artifacts"

# ============================================================
# SHAP FEATURE LABELS (Vietnamese)
# ============================================================
FEATURE_LABELS_VI = {
    # --- Direct input features ---
    "CODE_GENDER": "Giới tính",
    "CNT_CHILDREN": "Số con",
    "CNT_FAM_MEMBERS": "Số thành viên GĐ",
    "NAME_EDUCATION_TYPE": "Học vấn",
    "NAME_FAMILY_STATUS": "Hôn nhân",
    "AMT_INCOME_TOTAL": "Thu nhập năm",
    "AMT_CREDIT": "Số tiền vay",
    "AMT_ANNUITY": "Trả góp/tháng",
    "AMT_GOODS_PRICE": "Giá trị hàng hóa",
    "NAME_CONTRACT_TYPE": "Loại hợp đồng",
    "NAME_INCOME_TYPE": "Loại thu nhập",
    "OCCUPATION_TYPE": "Nghề nghiệp",
    "ORGANIZATION_TYPE": "Loại tổ chức",
    "FLAG_OWN_CAR": "Có ô tô",
    "FLAG_OWN_REALTY": "Có BĐS",
    "OWN_CAR_AGE": "Tuổi xe",
    "NAME_HOUSING_TYPE": "Loại nhà ở",
    "EXT_SOURCE_1": "Viễn thông (Viettel/VNPT)",
    "EXT_SOURCE_2": "Tiện ích (Điện/Nước)",
    "EXT_SOURCE_3": "TMĐT (Shopee/Lazada)",
    "FLAG_EMP_PHONE": "SĐT cơ quan",
    "FLAG_WORK_PHONE": "SĐT nơi làm việc",
    "FLAG_PHONE": "SĐT nhà",
    "FLAG_EMAIL": "Có email",
    "AGE_YEARS": "Tuổi",
    "EMPLOYMENT_YEARS": "Thâm niên (năm)",
    "REGISTRATION_YEARS": "ĐK cư trú (năm)",
    "ID_PUBLISH_YEARS": "Đổi CCCD cách đây (năm)",
    "PHONE_CHANGE_DAYS": "Đổi SĐT cách đây (ngày)",
    "ORGANIZATION_TYPE": "Loại tổ chức",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Người quen vỡ nợ (30 ngày)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Người quen vỡ nợ (60 ngày)",
    # --- Engineered features (tính từ input) ---
    "CREDIT_INCOME_RATIO": "Tỷ lệ vay/thu nhập",
    "ANNUITY_INCOME_RATIO": "Trả góp/thu nhập",
    "CREDIT_TERM_MONTHS": "Kỳ hạn vay (tháng)",
    "PAYMENT_RATE": "Tốc độ trả nợ",
    "INCOME_PER_PERSON": "Thu nhập/người (= TN ÷ số người)",
    "GOODS_CREDIT_RATIO": "Giá hàng/khoản vay",
    "EMPLOYED_TO_AGE_RATIO": "Thâm niên/tuổi",
    "EXT_SOURCE_MEAN": "TB 3 nguồn điểm thay thế",
    "EXT_SOURCE_PROD": "Tích 3 nguồn điểm",
    "EXT_SOURCE_MIN": "Nguồn điểm thấp nhất",
    "EXT_SOURCE_MAX": "Nguồn điểm cao nhất",
    "SOCIAL_DEF_TOTAL": "Tổng vỡ nợ mạng XH",
    "AGE_GROUP": "Nhóm tuổi",
    "CONTACT_COUNT": "Số kênh liên lạc",
}

# ============================================================
# IMPROVEMENT SUGGESTIONS by category
# ============================================================
IMPROVEMENT_SUGGESTIONS = {
    "income": "Tăng thu nhập hoặc có thêm nguồn thu phụ để cải thiện tỷ lệ vay/thu nhập.",
    "employment": "Tiếp tục duy trì công việc ổn định — thâm niên làm việc lâu sẽ giúp cải thiện điểm.",
    "credit_amount": "Giảm số tiền vay hoặc chọn kỳ hạn dài hơn để giảm áp lực trả góp hàng tháng.",
    "assets": "Tích lũy tài sản (nhà, xe) để chứng minh năng lực tài chính.",
    "contact": "Cung cấp đầy đủ thông tin liên lạc (email, SĐT nơi làm việc) để tăng độ tin cậy.",
    "ext_source": "Cải thiện điểm tín dụng thay thế bằng cách thanh toán hóa đơn đúng hạn (điện, nước, viễn thông).",
    "social": "Hạn chế liên kết tài chính với người có lịch sử vỡ nợ.",
}
