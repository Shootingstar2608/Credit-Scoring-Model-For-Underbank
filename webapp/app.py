"""
Credit Scoring Web Application — Streamlit UI
Đánh giá tín dụng cho người chưa có lịch sử tín dụng.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file (OPENROUTER_API_KEY, AI_MODEL)

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Allow imports from webapp root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FEATURE_LABELS_VI,
    IMPROVEMENT_SUGGESTIONS,
    MODEL_DIR,
    RISK_LEVELS,
    TIERS,
)
from scoring.advisor import CreditAdvisor
from scoring.engine import CreditScoringEngine

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Scoring — Đánh Giá Tín Dụng",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #bbdefb; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Score card */
    .score-card {
        text-align: center;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .score-number {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .score-label { font-size: 0.9rem; opacity: 0.8; }

    /* Tier card */
    .tier-card {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border-left: 6px solid;
        margin-bottom: 0.8rem;
    }

    /* SHAP bar */
    .shap-bar-positive { background: #ef5350; border-radius: 4px; height: 18px; }
    .shap-bar-negative { background: #66bb6a; border-radius: 4px; height: 18px; }

    /* Section divider */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a237e;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Suggestion card */
    .suggestion-card {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }

    /* Metric box */
    .metric-box {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-box .value { font-size: 1.4rem; font-weight: 700; }
    .metric-box .label { font-size: 0.75rem; color: #757575; }

    div[data-testid="stForm"] {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD ENGINE (cached)
# ============================================================
@st.cache_resource
def load_engine():
    return CreditScoringEngine(MODEL_DIR)


engine = load_engine()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="main-header">
    <h1>🏦 Credit Scoring — Đánh Giá Tín Dụng</h1>
    <p>Dành cho khách hàng chưa có lịch sử tín dụng • LightGBM + SHAP Explainability</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — Form nhập liệu
# ============================================================
with st.sidebar:
    # ---- AI Advisor config (top of sidebar) ----
    # with st.expander("🤖 Nhận Xét AI (OpenRouter)", expanded=False):
    #     st.caption(
    #         "Dùng [OpenRouter](https://openrouter.ai) để truy cập hàng trăm model AI "
    #         "(bao gồm model miễn phí). Đăng ký tại openrouter.ai → lấy API key."
    #     )
    #     ai_api_key = st.text_input(
    #         "🔑 OpenRouter API Key",
    #         type="password",
    #         help="Lấy key tại https://openrouter.ai/keys — Để trống để dùng template tự động (không cần AI).",
    #     )
    #     ai_model_preset = st.selectbox(
    #         "🧠 Model gợi ý",
    #         [""] + list(OPENROUTER_MODELS.keys()),
    #         format_func=lambda x: (
    #             "— Tự nhập bên dưới —" if x == "" else OPENROUTER_MODELS.get(x, x)
    #         ),
    #         index=1,  # default: first free model
    #         help="Chọn model gợi ý hoặc chọn '— Tự nhập —' để điền tên model tùy ý.",
    #     )
    #     ai_model_custom = st.text_input(
    #         "✏️ Hoặc nhập tên model",
    #         value="",
    #         placeholder="vd: google/gemini-2.0-flash-exp:free",
    #         help="Nhập model ID từ openrouter.ai/models. Nếu điền ở đây sẽ ưu tiên hơn dropdown.",
    #     )
    #     ai_model = (
    #         ai_model_custom.strip()
    #         if ai_model_custom.strip()
    #         else (ai_model_preset or DEFAULT_MODEL)
    #     )

    ai_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    ai_model = os.environ.get("AI_MODEL", "deepseek/deepseek-r1-0528:free")

    st.markdown("---")
    st.markdown("### 📝 Thông Tin Đăng Ký Vay")
    st.caption("Điền đầy đủ thông tin bên dưới rồi nhấn **Đánh giá**.")

    with st.form("credit_form"):
        # ---- Nhóm 1: Nhân khẩu học ----
        st.markdown(
            '<p class="section-title">👤 Thông tin cá nhân</p>', unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox(
                "Giới tính",
                ["M", "F"],
                format_func=lambda x: "Nam" if x == "M" else "Nữ",
                help="Chọn giới tính của bạn",
            )
            age = st.number_input(
                "Tuổi",
                min_value=18,
                max_value=80,
                value=30,
                step=1,
                help="Tuổi hiện tại. Người lớn tuổi hơn thường có điểm tốt hơn",
            )
        with col2:
            children = st.number_input(
                "Số con",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Số con đang nuôi dưỡng. Nhiều con = gánh nặng tài chính lớn hơn",
            )
            family = st.number_input(
                "Thành viên gia đình",
                min_value=1,
                max_value=30,
                value=2,
                step=1,
                help="Tổng số người cùng sống trong hộ (bao gồm bản thân)",
            )

        col_id1, col_id2 = st.columns(2)
        with col_id1:
            id_publish_years = st.number_input(
                "Đổi CCCD/CMND cách đây (năm)",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                help="Bạn đổi CCCD/CMND lần cuối cách đây bao lâu? CCCD mới = đáng tin cậy hơn",
            )
        with col_id2:
            registration_years = st.number_input(
                "Đăng ký cư trú cách đây (năm)",
                min_value=0,
                max_value=60,
                value=10,
                step=1,
                help="Thời gian đăng ký cư trú tại địa chỉ hiện tại. Ổn định lâu = điểm tốt hơn",
            )

        education = st.selectbox(
            "Trình độ học vấn",
            [
                "Lower secondary",
                "Secondary / secondary special",
                "Incomplete higher",
                "Higher education",
                "Academic degree",
            ],
            index=3,
            format_func={
                "Lower secondary": "THCS",
                "Secondary / secondary special": "THPT / Trung cấp",
                "Incomplete higher": "Cao đẳng / ĐH dở dang",
                "Higher education": "Đại học",
                "Academic degree": "Sau đại học",
            }.get,
            help="Cấp học cao nhất bạn đã hoàn thành",
        )
        family_status = st.selectbox(
            "Tình trạng hôn nhân",
            ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"],
            index=1,
            format_func={
                "Single / not married": "Độc thân",
                "Married": "Đã kết hôn",
                "Civil marriage": "Sống chung",
                "Separated": "Ly thân",
                "Widow": "Góa",
            }.get,
            help="Tình trạng gia đình hiện tại",
        )

        # ---- Nhóm 2: Tài chính ----
        st.markdown(
            '<p class="section-title">💰 Thông tin khoản vay</p>',
            unsafe_allow_html=True,
        )
        monthly_income = st.number_input(
            "Thu nhập hàng tháng (VNĐ)",
            min_value=0,
            value=12_000_000,
            step=500_000,
            help="Lương thực nhận mỗi tháng (trước thuế). VD: 12 triệu/tháng → nhập 12,000,000",
        )
        credit = st.number_input(
            "Số tiền muốn vay (VNĐ)",
            min_value=0,
            value=300_000_000,
            step=10_000_000,
            help="Tổng khoản tiền bạn muốn vay. Vay càng lớn so với thu nhập → rủi ro càng cao",
        )
        annuity = st.number_input(
            "Trả góp hàng tháng (VNĐ)",
            min_value=0,
            value=10_000_000,
            step=500_000,
            help="Số tiền gốc + lãi phải trả mỗi tháng. VD: Vay 300tr, kỳ hạn 36 tháng → ~10tr/tháng",
        )
        goods_price = st.number_input(
            "Giá trị hàng hóa / mục đích vay (VNĐ)",
            min_value=0,
            value=280_000_000,
            step=10_000_000,
            help="Giá thực tế của tài sản bạn mua (nhà, xe, hàng hóa). Nếu vay > giá trị = rủi ro",
        )
        contract_type = st.selectbox(
            "Loại hợp đồng",
            ["Cash loans", "Revolving loans"],
            format_func=lambda x: (
                "Vay tiền mặt (trả góp cố định)"
                if x == "Cash loans"
                else "Tín dụng tuần hoàn (như thẻ tín dụng)"
            ),
            help="Vay tiền mặt = vay 1 lần, trả góp đều. Tín dụng tuần hoàn = hạn mức quay vòng",
        )

        # ---- Nhóm 3: Việc làm ----
        st.markdown(
            '<p class="section-title">💼 Việc làm & Nghề nghiệp</p>',
            unsafe_allow_html=True,
        )
        emp_years = st.number_input(
            "Số năm đi làm",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5,
            help="Thâm niên ở công việc hiện tại. Làm lâu = ổn định = điểm tốt hơn",
        )
        _income_type_map = {
            "Working": "Đi làm (lương)",
            "Commercial associate": "Đối tác thương mại",
            "Pensioner": "Hưu trí",
            "State servant": "Công chức/Viên chức",
            "Businessman": "Kinh doanh tự do",
            "Student": "Sinh viên",
            "Unemployed": "Thất nghiệp",
            "Maternity leave": "Nghỉ thai sản",
        }
        income_type = st.selectbox(
            "Loại thu nhập",
            list(_income_type_map.keys()),
            format_func=_income_type_map.get,
            help="Nguồn thu nhập chính của bạn hiện tại",
        )
        _occupation_map = {
            None: "— Bỏ qua —",
            "Laborers": "Lao động phổ thông",
            "Sales staff": "Nhân viên kinh doanh",
            "Core staff": "Nhân viên chính",
            "Managers": "Quản lý",
            "Drivers": "Tài xế",
            "High skill tech staff": "Kỹ thuật cao",
            "Accountants": "Kế toán",
            "Medicine staff": "Y tế",
            "Cooking staff": "Đầu bếp",
            "Security staff": "Bảo vệ",
            "Cleaning staff": "Vệ sinh",
            "Private service staff": "Dịch vụ tư nhân",
            "Low-skill Laborers": "Lao động giản đơn",
            "Waiters/barmen staff": "Phục vụ",
            "Secretaries": "Thư ký",
            "Realty agents": "Môi giới BĐS",
            "IT staff": "Công nghệ thông tin",
            "HR staff": "Nhân sự",
        }
        occupation = st.selectbox(
            "Nghề nghiệp (không bắt buộc)",
            list(_occupation_map.keys()),
            format_func=_occupation_map.get,
            help="Chọn nghề gần nhất với công việc hiện tại. Bỏ qua nếu không muốn khai",
        )

        _org_type_map = {
            None: "— Không khai báo —",
            "Business Entity Type 3": "Doanh nghiệp tư nhân",
            "Self-employed": "Tự kinh doanh",
            "Business Entity Type 2": "Công ty TNHH",
            "Business Entity Type 1": "Công ty cổ phần",
            "Government": "Cơ quan nhà nước",
            "Medicine": "Y tế / Bệnh viện",
            "School": "Trường học (phổ thông)",
            "Kindergarten": "Mẫu giáo / Mầm non",
            "University": "Đại học / Cao đẳng",
            "Construction": "Xây dựng",
            "Military": "Quân đội",
            "Police": "Công an",
            "Bank": "Ngân hàng",
            "Insurance": "Bảo hiểm",
            "Agriculture": "Nông nghiệp",
            "Restaurant": "Nhà hàng / Ăn uống",
            "Hotel": "Khách sạn / Lưu trú",
            "Transport: type 4": "Vận tải / Logistics",
            "Security": "Bảo vệ / An ninh",
            "Telecom": "Viễn thông / CNTT",
            "Trade: type 7": "Thương mại / Bán lẻ",
            "Industry: type 9": "Công nghiệp / Sản xuất",
            "Other": "Khác",
        }
        organization = st.selectbox(
            "Loại tổ chức nơi làm việc",
            list(_org_type_map.keys()),
            format_func=_org_type_map.get,
            help="Loại công ty/tổ chức bạn đang làm việc. Ảnh hưởng đến đánh giá sự ổn định công việc",
        )

        # ---- Nhóm 4: Tài sản ----
        st.markdown(
            '<p class="section-title">🏠 Tài sản & Nhà ở</p>', unsafe_allow_html=True
        )
        col3, col4 = st.columns(2)
        with col3:
            own_car = st.selectbox(
                "Sở hữu ô tô?",
                ["N", "Y"],
                format_func=lambda x: "Có" if x == "Y" else "Không",
                help="Bạn có đang sở hữu ô tô đứng tên không?",
            )
        with col4:
            own_realty = st.selectbox(
                "Sở hữu BĐS?",
                ["N", "Y"],
                format_func=lambda x: "Có" if x == "Y" else "Không",
                index=1,
                help="Bạn có nhà/đất đứng tên không? Có BĐS giúp giảm rủi ro đáng kể",
            )

        car_age = None
        if own_car == "Y":
            car_age = st.number_input(
                "Tuổi xe (năm)",
                min_value=0,
                max_value=80,
                value=5,
                step=1,
                help="Xe đã sử dụng bao nhiêu năm? Xe mới = giá trị tài sản cao hơn",
            )

        _housing_map = {
            "House / apartment": "Nhà riêng / Căn hộ (sở hữu)",
            "With parents": "Ở cùng bố mẹ",
            "Municipal apartment": "Chung cư nhà nước",
            "Rented apartment": "Thuê nhà / phòng trọ",
            "Office apartment": "Căn hộ văn phòng",
            "Co-op apartment": "Chung cư hợp tác xã",
        }
        housing = st.selectbox(
            "Loại nhà ở",
            list(_housing_map.keys()),
            format_func=_housing_map.get,
            help="Bạn đang ở đâu? Nhà sở hữu được đánh giá tốt hơn nhà thuê",
        )

        # ---- Nhóm 5: Điểm thay thế (External) ----
        st.markdown(
            '<p class="section-title">📊 Điểm tín dụng thay thế</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Thang điểm: **0.0** (rất kém) → **1.0** (rất tốt). Để mặc định 0.5 nếu không rõ."
        )
        ext1 = st.number_input(
            "📱 Viễn thông (Viettel, VNPT, Mobi)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="ext1",
            help="Dựa trên: thời gian dùng SIM, tần suất nạp tiền, thanh toán cước đúng hạn. Dùng SIM lâu năm + trả cước đều = điểm cao",
        )
        ext2 = st.number_input(
            "💡 Tiện ích (Điện, Nước, Internet)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="ext2",
            help="Dựa trên: lịch sử thanh toán hóa đơn điện/nước/internet. Trả đúng hạn nhiều tháng liên tục = điểm cao",
        )
        ext3 = st.number_input(
            "🛒 TMĐT (Shopee, Lazada, Tiki)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="ext3",
            help="Dựa trên: tuổi tài khoản, tần suất mua, tỷ lệ hoàn thành đơn, lịch sử 'mua trước trả sau'",
        )

        # ---- Nhóm 6: Liên lạc ----
        st.markdown(
            '<p class="section-title">📱 Thông tin liên lạc</p>', unsafe_allow_html=True
        )
        st.caption(
            "Tick vào nếu bạn **có thể cung cấp** thông tin liên lạc này. Càng nhiều kênh → càng đáng tin cậy."
        )
        col8, col9 = st.columns(2)
        with col8:
            has_emp_phone = st.checkbox(
                "SĐT cơ quan",
                value=True,
                help="SĐT bàn tổng đài của công ty bạn đang làm",
            )
            has_work_phone = st.checkbox(
                "SĐT nơi làm việc",
                value=False,
                help="SĐT trực tiếp phòng ban / bộ phận của bạn",
            )
        with col9:
            has_phone = st.checkbox(
                "SĐT nhà", value=True, help="Số điện thoại cố định tại nhà riêng"
            )
            has_email = st.checkbox(
                "Có email", value=True, help="Bạn có email cá nhân đang sử dụng không?"
            )

        phone_change_days = st.number_input(
            "Đổi SĐT gần nhất cách đây (ngày)",
            min_value=0,
            max_value=10000,
            value=365,
            step=30,
            help="Bạn đổi số điện thoại lần cuối cách đây bao nhiêu ngày? Dùng SĐT căng lâu = đáng tin cậy hơn",
        )

        # ---- Nhóm 7: Mạng lưới xã hội ----
        st.markdown(
            '<p class="section-title">👥 Mạng lưới xã hội</p>', unsafe_allow_html=True
        )
        st.caption(
            "Trong số bạn bè/người thân của bạn, có ai **vỡ nợ/trễ hạn trả nợ** gần đây không?"
        )
        col_soc1, col_soc2 = st.columns(2)
        with col_soc1:
            def_30 = st.number_input(
                "Vỡ nợ trong 30 ngày",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Số người quen bị vỡ nợ/trễ hạn trong 30 ngày gần nhất. 0 = không có ai",
            )
        with col_soc2:
            def_60 = st.number_input(
                "Vỡ nợ trong 60 ngày",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Số người quen bị vỡ nợ/trễ hạn trong 60 ngày gần nhất. 0 = không có ai",
            )

        # ---- Submit ----
        st.markdown("---")
        submitted = st.form_submit_button(
            "🔍  Đánh Giá Tín Dụng", use_container_width=True, type="primary"
        )

# ============================================================
# BUILD USER INPUT DICT
# ============================================================
user_input = {
    "CODE_GENDER": gender,
    "AGE_YEARS": float(age),
    "CNT_CHILDREN": int(children),
    "CNT_FAM_MEMBERS": float(family),
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family_status,
    "AMT_INCOME_TOTAL": float(monthly_income) * 12,  # Model cần thu nhập năm
    "AMT_CREDIT": float(credit),
    "AMT_ANNUITY": float(annuity),
    "AMT_GOODS_PRICE": float(goods_price),
    "NAME_CONTRACT_TYPE": contract_type,
    "EMPLOYMENT_YEARS": float(emp_years),
    "NAME_INCOME_TYPE": income_type,
    "OCCUPATION_TYPE": occupation,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "OWN_CAR_AGE": float(car_age) if car_age is not None else np.nan,
    "NAME_HOUSING_TYPE": housing,
    "EXT_SOURCE_1": float(ext1),
    "EXT_SOURCE_2": float(ext2),
    "EXT_SOURCE_3": float(ext3),
    "FLAG_EMP_PHONE": 1 if has_emp_phone else 0,
    "FLAG_WORK_PHONE": 1 if has_work_phone else 0,
    "FLAG_PHONE": 1 if has_phone else 0,
    "FLAG_EMAIL": 1 if has_email else 0,
    "ID_PUBLISH_YEARS": float(id_publish_years),
    "REGISTRATION_YEARS": float(registration_years),
    "PHONE_CHANGE_DAYS": float(phone_change_days),
    "ORGANIZATION_TYPE": organization if organization else "XNA",
    "DEF_30_CNT_SOCIAL_CIRCLE": float(def_30),
    "DEF_60_CNT_SOCIAL_CIRCLE": float(def_60),
}


# ============================================================
# HELPER: Gauge chart
# ============================================================
def make_gauge(score: int, tier_color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 48, "color": tier_color}},
            gauge={
                "axis": {
                    "range": [300, 850],
                    "tickwidth": 1,
                    "tickcolor": "#ccc",
                    "tickvals": [300, 400, 500, 580, 670, 740, 800, 850],
                },
                "bar": {"color": tier_color, "thickness": 0.3},
                "bgcolor": "#f5f5f5",
                "steps": [
                    {"range": [300, 580], "color": "#ffcdd2"},  # Poor
                    {"range": [580, 670], "color": "#fff9c4"},  # Fair
                    {"range": [670, 740], "color": "#bbdefb"},  # Good
                    {"range": [740, 800], "color": "#c8e6c9"},  # Very Good
                    {"range": [800, 850], "color": "#a5d6a7"},  # Exceptional
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
            title={"text": "FICO Score (300 – 850)", "font": {"size": 16}},
        )
    )
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10))
    return fig


# ============================================================
# HELPER: SHAP waterfall chart
# ============================================================
def make_shap_chart(shap_top: list[dict]) -> go.Figure:
    if not shap_top:
        return None

    features = [
        FEATURE_LABELS_VI.get(s["feature"], s["feature"]) for s in reversed(shap_top)
    ]
    values = [s["shap_value"] for s in reversed(shap_top)]
    colors = ["#ef5350" if v > 0 else "#66bb6a" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            textfont={"size": 11},
        )
    )
    fig.update_layout(
        title={"text": "Giải thích kết quả (SHAP)", "font": {"size": 14}},
        xaxis_title="Mức ảnh hưởng",
        height=max(300, 36 * len(shap_top)),
        margin=dict(l=10, r=10, t=40, b=30),
        yaxis={"tickfont": {"size": 12}},
        plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_width=1, line_color="grey")
    return fig


# ============================================================
# MAIN CONTENT — Kết quả
# ============================================================
if submitted:
    with st.spinner("Đang phân tích hồ sơ..."):
        result = engine.predict(user_input)
    st.session_state.scoring_result = result
    st.session_state.scoring_input = dict(user_input)
    st.session_state.chat_messages = []  # Reset chat khi đánh giá mới

if "scoring_result" in st.session_state:
    result = st.session_state.scoring_result
    score = result["credit_score"]
    proba = result["probability"]
    risk = RISK_LEVELS[result["risk_key"]]

    # Determine tier
    tier = next(
        (t for t in TIERS if t["score_min"] <= score <= t["score_max"]), TIERS[-1]
    )

    # ---- Row 1: Score + Tier ----
    col_score, col_tier = st.columns([1, 1.3])

    with col_score:
        fig_gauge = make_gauge(score, tier["color"])
        st.plotly_chart(fig_gauge, width="stretch")

    with col_tier:
        st.markdown(
            f"""
<div class="tier-card" style="background: {tier['color']}15; border-color: {tier['color']};">
    <h2 style="margin:0; color: {tier['color']};">{tier['icon']} {tier['name']}</h2>
    <p style="margin: 0.5rem 0 0; font-size: 1rem;">{tier['description']}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        # KPI metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="value" style="color: {tier["color"]};">{score}</div><div class="label">FICO Score</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="value" style="color: {risk["color"]};">{proba:.1%}</div><div class="label">Xác suất vỡ nợ</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            cl = result["credit_limit"]
            cl_text = f"{cl:,.0f}" if cl > 0 else "N/A"
            st.markdown(
                f'<div class="metric-box"><div class="value">{cl_text}</div><div class="label">Hạn mức đề xuất</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---- Row 2: SHAP explanation + Suggestions ----
    col_shap, col_suggest = st.columns([1.4, 1])

    with col_shap:
        st.markdown(
            '<p class="section-title">🔬 Giải Thích Kết Quả (SHAP)</p>',
            unsafe_allow_html=True,
        )
        fig_shap = make_shap_chart(result["shap_top_features"])
        if fig_shap:
            st.plotly_chart(fig_shap, width="stretch")
            st.caption(
                "🔴 **Đỏ** = tăng rủi ro vỡ nợ &nbsp;|&nbsp; 🟢 **Xanh** = giảm rủi ro vỡ nợ. "
                "Thanh càng dài → ảnh hưởng càng lớn."
            )
        else:
            st.info("SHAP explainer chưa sẵn sàng.")

    with col_suggest:
        st.markdown(
            '<p class="section-title">💡 Đề Xuất Cải Thiện</p>', unsafe_allow_html=True
        )

        # Generate suggestions based on SHAP
        shown = set()
        for item in result.get("shap_top_features", []):
            if item["direction"] != "risk":
                continue
            feat = item["feature"]
            # Map feature → suggestion category
            if feat in (
                "CREDIT_INCOME_RATIO",
                "ANNUITY_INCOME_RATIO",
                "AMT_INCOME_TOTAL",
                "INCOME_PER_PERSON",
            ):
                key = "income"
            elif feat in ("EMPLOYMENT_YEARS", "EMPLOYED_TO_AGE_RATIO", "DAYS_EMPLOYED"):
                key = "employment"
            elif feat in (
                "AMT_CREDIT",
                "CREDIT_TERM_MONTHS",
                "AMT_ANNUITY",
                "PAYMENT_RATE",
                "GOODS_CREDIT_RATIO",
            ):
                key = "credit_amount"
            elif feat in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE"):
                key = "assets"
            elif feat in (
                "CONTACT_COUNT",
                "FLAG_EMP_PHONE",
                "FLAG_WORK_PHONE",
                "FLAG_PHONE",
                "FLAG_EMAIL",
            ):
                key = "contact"
            elif feat.startswith("EXT_SOURCE"):
                key = "ext_source"
            elif feat in (
                "SOCIAL_DEF_TOTAL",
                "DEF_30_CNT_SOCIAL_CIRCLE",
                "DEF_60_CNT_SOCIAL_CIRCLE",
            ):
                key = "social"
            else:
                continue

            if key not in shown:
                shown.add(key)
                st.markdown(
                    f'<div class="suggestion-card">💡 {IMPROVEMENT_SUGGESTIONS[key]}</div>',
                    unsafe_allow_html=True,
                )

        if not shown:
            st.success("Hồ sơ của bạn rất tốt! Không cần cải thiện thêm. 🎉")

        # Decision details
        st.markdown(
            '<p class="section-title">📋 Chi Tiết Quyết Định</p>',
            unsafe_allow_html=True,
        )
        decision_data = {
            "FICO Score": f"{score}/850",
            "Xác suất vỡ nợ": f"{proba:.2%}",
            "Mức rủi ro": f"{risk['emoji']} {risk['label']}",
            "Quyết định": f"{tier['icon']} {tier['name']}",
            "Hạn mức đề xuất": (
                f"{result['credit_limit']:,.0f}"
                if result["credit_limit"] > 0
                else "Không cấp"
            ),
        }
        if tier["interest_modifier"] is not None:
            base_rate = 12.0  # Base interest rate %
            adjusted = base_rate + tier["interest_modifier"] * 100
            decision_data["Lãi suất dự kiến"] = f"{adjusted:.1f}%/năm"

        for k, v in decision_data.items():
            st.markdown(f"**{k}:** {v}")

    # ---- Row 3: Template Assessment (ALWAYS shown) ----
    st.markdown("---")
    st.markdown(
        '<p class="section-title">📋 Nhận Xét Tín Dụng</p>',
        unsafe_allow_html=True,
    )
    template_advisor = CreditAdvisor()  # No API key → always template
    template_result = template_advisor.assess(
        result, st.session_state.scoring_input, FEATURE_LABELS_VI
    )
    st.markdown(template_result.text)

    # ---- Row 4: Tiered System Legend ----
    st.markdown("---")
    st.markdown(
        '<p class="section-title">📊 Hệ Thống Xếp Hạng Tín Dụng</p>',
        unsafe_allow_html=True,
    )
    tier_cols = st.columns(len(TIERS))
    for i, t in enumerate(TIERS):
        with tier_cols[i]:
            is_current = t["name"] == tier["name"]
            border = f"3px solid {t['color']}" if is_current else "1px solid #e0e0e0"
            bg = f"{t['color']}18" if is_current else "#fafafa"
            st.markdown(
                f"""
<div style="border: {border}; background: {bg}; border-radius: 10px; padding: 0.8rem; text-align: center; min-height: 130px;">
    <div style="font-size: 1.8rem;">{t['icon']}</div>
    <div style="font-weight: 700; color: {t['color']}; font-size: 0.85rem;">{t['name']}</div>
    <div style="font-size: 0.75rem; color: #757575; margin-top: 0.3rem;">{t['score_min']}–{t['score_max']} điểm</div>
    {'<div style="margin-top:0.3rem; font-size:0.7rem; font-weight:600; color:' + t["color"] + ';">◄ BẠN Ở ĐÂY</div>' if is_current else ''}
</div>
""",
                unsafe_allow_html=True,
            )

    # ---- Row 5: AI Chatbot ----
    st.markdown("---")
    st.markdown(
        '<p class="section-title">🤖 Hỏi Tư Vấn Viên AI</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Hỏi bất kỳ câu hỏi nào về hồ sơ tín dụng — "
        "AI sẽ tư vấn dựa trên kết quả vừa chấm điểm."
    )
    st.warning(
        "⚠️ **Lưu ý:** AI có thể mắc sai sót. Mọi câu trả lời chỉ mang tính "
        "tham khảo, không phải cam kết từ ngân hàng. Luôn kiểm chứng thông tin "
        "quan trọng với chuyên viên tư vấn.",
        icon="⚠️",
    )

    if not ai_api_key:
        st.info(
            "💡 Tính năng tư vấn AI chưa được kích hoạt. "
            "Vui lòng cấu hình OPENROUTER_API_KEY trong file .env."
        )
    else:
        # Display chat history
        for msg in st.session_state.get("chat_messages", []):
            avatar = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input(
                "Câu hỏi",
                placeholder="VD: Làm sao để tăng điểm tín dụng của tôi?",
                label_visibility="collapsed",
            )
            send_btn = st.form_submit_button(
                "Gửi câu hỏi 🚀", use_container_width=True
            )

        if send_btn and question.strip():
            st.session_state.chat_messages.append(
                {"role": "user", "content": question}
            )
            # Limit chat history to last 10 messages (prevent context drift)
            if len(st.session_state.chat_messages) > 10:
                st.session_state.chat_messages = st.session_state.chat_messages[-10:]
            with st.spinner("🤖 AI đang suy nghĩ..."):
                chat_advisor = CreditAdvisor(
                    api_key=ai_api_key, model=ai_model
                )
                response = chat_advisor.chat(
                    st.session_state.chat_messages,
                    result,
                    st.session_state.scoring_input,
                    FEATURE_LABELS_VI,
                )
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": response}
            )
            st.rerun()

else:
    # ---- Landing page ----
    st.markdown(
        """
### 👈 Điền thông tin ở thanh bên trái để bắt đầu

Ứng dụng sẽ đánh giá hồ sơ tín dụng của bạn dựa trên:

| Nhóm | Thông tin |
|------|----------|
| 👤 **Cá nhân** | Tuổi, giới tính, học vấn, tình trạng hôn nhân |
| 💰 **Tài chính** | Thu nhập, số tiền vay, kỳ hạn |
| 💼 **Việc làm** | Nghề nghiệp, thâm niên |
| 🏠 **Tài sản** | Nhà ở, xe cộ |
| 📊 **Điểm thay thế** | Viễn thông, tiện ích (điện/nước), thương mại điện tử |
| 📱 **Liên lạc** | Kênh liên lạc có sẵn |

**Kết quả bao gồm:**
- 📈 **FICO Score** (300-850)
- 🎯 **Xác suất vỡ nợ** (đã hiệu chỉnh)
- 🔬 **Giải thích SHAP** cho từng yếu tố
- 💡 **Đề xuất cải thiện** cụ thể
- 💳 **Hạn mức tín dụng đề xuất** (VNĐ)
"""
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #9e9e9e; font-size: 0.8rem; padding: 0.5rem;">
    Credit Scoring Demo • LightGBM + SHAP • Model AUC: 0.767 • 48 features<br>
    Dành cho người chưa có lịch sử tín dụng — Chỉ sử dụng dữ liệu từ hồ sơ đăng ký
</div>
""",
    unsafe_allow_html=True,
)
