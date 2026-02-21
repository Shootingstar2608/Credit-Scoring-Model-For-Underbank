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
    "CREDIT_GOODS_RATIO": "Khoản vay/giá hàng",
    "EMPLOYED_TO_AGE_RATIO": "Thâm niên/tuổi",
    "INCOME_CREDIT_PERC": "Thu nhập/khoản vay",
    "EXT_SOURCES_MEAN": "TB 3 nguồn điểm thay thế",
    "EXT_SOURCES_STD": "Độ lệch 3 nguồn điểm",
    "EXT_SOURCES_PROD": "Tích 3 nguồn điểm",
    "EXT_SOURCE_1x2": "Viễn thông × Tiện ích",
    "EXT_SOURCE_2x3": "Tiện ích × TMĐT",
    "EXT_SOURCE_1x3": "Viễn thông × TMĐT",
    "SOCIAL_CIRCLE_DEFAULT": "Tổng vỡ nợ mạng XH",
    "AGE_GROUP": "Nhóm tuổi",
    "CONTACT_COUNT": "Số kênh liên lạc",
    # --- Bureau features (lịch sử tín dụng tổ chức khác) ---
    "BUREAU_LOAN_COUNT": "Số khoản vay tổ chức khác",
    "BUREAU_ACTIVE_COUNT": "Số khoản vay đang nợ",
    "BUREAU_CLOSED_COUNT": "Số khoản vay đã tất toán",
    "BUREAU_CLOSED_RATIO": "Tỷ lệ tất toán",
    "BUREAU_HAD_OVERDUE": "Đã từng quá hạn",
    "BUREAU_MAX_OVERDUE": "Ngày quá hạn nhiều nhất",
    "BUREAU_AMT_CREDIT_SUM": "Tổng dư nợ tổ chức khác",
    "BUREAU_AMT_CREDIT_MEAN": "TB khoản vay tổ chức khác",
    "BUREAU_AMT_DEBT_SUM": "Tổng nợ hiện tại",
    "BUREAU_AMT_OVERDUE_SUM": "Tổng số tiền quá hạn",
    "BUREAU_CREDIT_DURATION_MEAN": "TB thời hạn khoản vay",
    "BUREAU_DEBT_RATIO": "Tỷ lệ nợ/tín dụng",
    "BUREAU_DPD_TOTAL": "Tổng lần trễ hạn (bureau)",
    "BUREAU_MONTHS_HISTORY": "Độ dài lịch sử tín dụng",
    # --- Previous application features ---
    "PREV_APP_COUNT": "Số lần nộp đơn vay trước",
    "PREV_APPROVED_RATIO": "Tỷ lệ được duyệt",
    "PREV_APPROVED_COUNT": "Số lần được duyệt",
    "PREV_REFUSED_COUNT": "Số lần bị từ chối",
    "PREV_AVG_APPLICATION": "TB số tiền đăng ký vay",
    "PREV_AVG_CREDIT": "TB số tiền được duyệt",
    "PREV_MAX_CREDIT": "Khoản vay lớn nhất",
    "PREV_AVG_ANNUITY": "TB trả góp cũ",
    "PREV_AVG_DOWN_PAYMENT": "TB đặt cọc",
    "PREV_DAYS_LAST_APP": "Đơn vay gần nhất",
    "PREV_DAYS_FIRST_APP": "Đơn vay đầu tiên",
    "PREV_CREDIT_VS_APPLICATION": "Tỷ lệ duyệt/đăng ký",
    "PREV_ANNUITY_VS_CURRENT": "Trả góp cũ/hiện tại",
    # --- Installments features ---
    "INSTALL_COUNT": "Tổng kỳ trả góp",
    "INSTALL_LATE_RATIO": "Tỷ lệ trả trễ",
    "INSTALL_LATE_COUNT": "Số lần trả trễ",
    "INSTALL_DAYS_DIFF_MEAN": "TB ngày trả sớm/trễ",
    "INSTALL_DAYS_DIFF_MAX": "Trễ hạn lâu nhất",
    "INSTALL_PAYMENT_RATIO_MEAN": "TB tỷ lệ trả đủ",
    "INSTALL_PAYMENT_RATIO_MIN": "Tỷ lệ trả thấp nhất",
    "INSTALL_AMT_PAYMENT_SUM": "Tổng đã trả",
    "INSTALL_AMT_INSTALMENT_SUM": "Tổng phải trả",
    "INSTALL_OVERALL_PAYMENT_RATIO": "Tỷ lệ trả tổng thể",
    # --- POS_CASH features ---
    "POS_CONTRACT_COUNT": "Số khoản trả góp POS",
    "POS_MONTHS_COUNT": "Số tháng POS",
    "POS_DPD_MAX": "Quá hạn POS (ngày)",
    "POS_DPD_MEAN": "TB quá hạn POS",
    "POS_DPD_DEF_MAX": "Quá hạn POS tối đa",
    "POS_COMPLETED_COUNT": "POS đã hoàn thành",
    "POS_ACTIVE_COUNT": "POS đang active",
    "POS_COMPLETED_RATIO": "Tỷ lệ hoàn thành POS",
    # --- Credit card features ---
    "CC_CARD_COUNT": "Số thẻ tín dụng",
    "CC_BALANCE_MEAN": "TB dư nợ thẻ TD",
    "CC_BALANCE_MAX": "Dư nợ thẻ TD cao nhất",
    "CC_LIMIT_MEAN": "TB hạn mức thẻ",
    "CC_UTILIZATION_MEAN": "TB sử dụng hạn mức thẻ",
    "CC_UTILIZATION_MAX": "Sử dụng hạn mức cao nhất",
    "CC_DRAWINGS_COUNT": "Số lần rút tiền thẻ",
    "CC_PAYMENT_TOTAL_MEAN": "TB thanh toán thẻ",
    "CC_MIN_INSTALLMENT_MEAN": "TB trả tối thiểu",
    "CC_DPD_MAX": "Quá hạn thẻ TD (ngày)",
    "CC_DPD_MEAN": "TB quá hạn thẻ TD",
    "CC_MONTHS_COUNT": "Số tháng dùng thẻ",
    "CC_PAYMENT_VS_BALANCE": "Thanh toán/dư nợ thẻ",
    # --- Cross features ---
    "BUREAU_DEBT_INCOME_RATIO": "Nợ tổ chức khác/thu nhập",
    "BUREAU_CREDIT_VS_CURRENT": "Tín dụng cũ/khoản vay mới",
    "TOTAL_LOAN_COUNT": "Tổng số khoản vay",
    "GOOD_PAYMENT_SCORE": "Điểm hành vi trả nợ tốt",
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
    "bureau": "Tất toán các khoản vay cũ tại tổ chức khác để giảm tỷ lệ nợ và chứng minh khả năng trả nợ.",
    "prev_app": "Tránh nộp quá nhiều đơn vay liên tục — mỗi lần bị từ chối sẽ ảnh hưởng tiêu cực đến hồ sơ.",
    "installment": "Trả đúng hạn các kỳ trả góp hiện có. Tỷ lệ trả đúng hạn cao giúp cải thiện điểm đáng kể.",
    "credit_card": "Giữ tỷ lệ sử dụng thẻ tín dụng dưới 30% hạn mức. Thanh toán đầy đủ mỗi tháng thay vì chỉ trả tối thiểu.",
}
