"""
Credit Scoring Web App — Phiên bản tối ưu (Streamlit)
- Form sidebar theo expanders, dễ quét
- Kết quả theo tabs: Kết quả | SHAP | Nhận xét | Hỏi AI
- Nút Chỉnh sửa hồ sơ, format VNĐ, glossary, footer không hardcode
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import plotly.graph_objects as go
import streamlit as st

WEBAPP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBAPP_DIR))
MODEL_DIR_OVERRIDE = WEBAPP_DIR.parent / "model_artifacts"

from config import (
    FEATURE_LABELS_VI,
    IMPROVEMENT_SUGGESTIONS,
    RISK_LEVELS,
    TIERS,
)
from auth import authenticate, register
from pdf_report import build_credit_report_pdf
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
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.6rem; }
    .main-header p { color: #bbdefb; margin: 0.25rem 0 0 0; font-size: 0.9rem; }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a237e;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.35rem;
        margin: 0.8rem 0 0.5rem 0;
    }
    .tier-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 6px solid;
        margin-bottom: 0.6rem;
    }
    .metric-box {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
    }
    .metric-box .value { font-size: 1.3rem; font-weight: 700; }
    .metric-box .label { font-size: 0.75rem; color: #757575; }
    .suggestion-card {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 0.6rem 0.8rem;
        border-radius: 0 8px 8px 0;
        margin: 0.35rem 0;
        font-size: 0.9rem;
    }
    .glossary-item { margin: 0.25rem 0; font-size: 0.9rem; }
    div[data-testid="stForm"] { border: 1px solid #e0e0e0; border-radius: 12px; padding: 1rem !important; }
</style>
""", unsafe_allow_html=True)


def format_vnd(x: float) -> str:
    """Format số tiền VNĐ: 12_000_000 -> '12 triệu' hoặc '12.5 tr'."""
    if x <= 0 or np.isnan(x):
        return "0"
    if x >= 1e9:
        return f"{x/1e9:.1f} tỷ"
    if x >= 1e6:
        return f"{x/1e6:.0f} triệu" if x % 1e6 == 0 else f"{x/1e6:.1f} triệu"
    if x >= 1e3:
        return f"{x/1e3:.0f} nghìn"
    return f"{x:,.0f}"


@st.cache_resource
def load_engine():
    model_dir = str(MODEL_DIR_OVERRIDE) if MODEL_DIR_OVERRIDE.exists() else None
    if model_dir is None:
        from config import MODEL_DIR
        model_dir = str(Path(MODEL_DIR).resolve() if not Path(MODEL_DIR).is_absolute() else MODEL_DIR)
    return CreditScoringEngine(model_dir)


# ============================================================
# AUTH
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.auth_page = "login"
if "report_pdf_bytes" not in st.session_state:
    st.session_state.report_pdf_bytes = None
if "report_pdf_valid" not in st.session_state:
    st.session_state.report_pdf_valid = False
if "report_pdf_meta" not in st.session_state:
    st.session_state.report_pdf_meta = {}
if "report_pdf_signature" not in st.session_state:
    st.session_state.report_pdf_signature = ""


def do_logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.auth_page = "login"
    for key in [
        "scoring_result",
        "scoring_input",
        "chat_messages",
        "report_pdf_bytes",
        "report_pdf_valid",
        "report_pdf_meta",
        "report_pdf_signature",
    ]:
        st.session_state.pop(key, None)


if not st.session_state.authenticated:
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Credit Scoring — Đánh Giá Tín Dụng</h1>
        <p>Dành cho khách hàng chưa có lịch sử tín dụng • Mô hình ML + SHAP</p>
    </div>
    """, unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 1.5, 1])
    with col_center:
        if st.session_state.auth_page == "login":
            st.markdown("### 🔐 Đăng nhập")
            with st.form("login_form"):
                username = st.text_input("Tên đăng nhập", placeholder="admin")
                password = st.text_input("Mật khẩu", type="password", placeholder="••••••")
                if st.form_submit_button("Đăng nhập", width="stretch"):
                    if username and password:
                        user = authenticate(username.strip().lower(), password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.rerun()
                        else:
                            st.error("Sai tên đăng nhập hoặc mật khẩu.")
                    else:
                        st.warning("Vui lòng nhập đầy đủ.")
            if st.button("📝 Chưa có tài khoản? Đăng ký", width="stretch"):
                st.session_state.auth_page = "register"
                st.rerun()
        else:
            st.markdown("### 📝 Đăng ký")
            with st.form("register_form"):
                reg_name = st.text_input("Họ tên", placeholder="Nguyễn Văn A")
                reg_username = st.text_input("Tên đăng nhập", placeholder="nguyenvana", help="Chữ và số, ≥3 ký tự")
                reg_password = st.text_input("Mật khẩu", type="password", placeholder="≥6 ký tự")
                reg_password2 = st.text_input("Nhập lại mật khẩu", type="password")
                if st.form_submit_button("Đăng ký", width="stretch"):
                    if reg_password != reg_password2:
                        st.error("Mật khẩu nhập lại không khớp.")
                    else:
                        err = register(reg_username, reg_password, reg_name)
                        if err:
                            st.error(err)
                        else:
                            st.success("Đăng ký thành công. Hãy đăng nhập.")
                            st.session_state.auth_page = "login"
                            st.rerun()
            if st.button("🔐 Đã có tài khoản? Đăng nhập", width="stretch"):
                st.session_state.auth_page = "login"
                st.rerun()
    st.stop()


# ============================================================
# HEADER (sau khi đăng nhập)
# ============================================================
engine = load_engine()
ai_api_key = os.environ.get("OPENROUTER_API_KEY", "")
ai_model = os.environ.get("AI_MODEL", "deepseek/deepseek-r1-0528:free")

hdr_left, hdr_right = st.columns([4, 1])
with hdr_left:
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Credit Scoring — Đánh Giá Tín Dụng</h1>
        <p>Dành cho khách hàng chưa có lịch sử tín dụng • Mô hình ML + SHAP</p>
    </div>
    """, unsafe_allow_html=True)
with hdr_right:
    user = st.session_state.user
    role_badge = "🔑 Admin" if user["role"] == "admin" else "👤 User"
    st.markdown(
        f'<div style="text-align:right; padding:0.5rem 0;"><strong>{user["name"]}</strong><br><span style="color:#888;">{role_badge}</span></div>',
        unsafe_allow_html=True,
    )
    st.button("🚪 Đăng xuất", on_click=do_logout, width="stretch")


# ============================================================
# FORM: Main (khi chưa có kết quả) hoặc Sidebar (khi đã có kết quả)
# Chưa có kết quả → form nằm MAIN (rộng). Có kết quả → form nằm SIDEBAR (để chỉnh & đánh giá lại).
# ============================================================
_has_result = "scoring_result" in st.session_state
# st là module không dùng được với "with" → dùng st.container() cho main, st.sidebar cho sidebar
_form_place = st.sidebar if _has_result else st.container()

if not _has_result:
    with st.sidebar:
        st.markdown("### 🏦 Credit Scoring")
        st.caption("Form điền bên **phải** (vùng rộng). Điền xong nhấn **Đánh giá tín dụng**.")
        st.markdown(
            """
### Hướng dẫn điền hồ sơ

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
        with st.expander("📖 Giải thích thuật ngữ"):
            st.markdown("- **FICO (300–850):** Thang điểm tín dụng. Càng cao càng dễ duyệt vay.")
            st.markdown("- **SHAP:** Đo ảnh hưởng từng yếu tố. Đỏ = tăng rủi ro, xanh = giảm rủi ro.")
            st.markdown("- **Điểm thay thế:** Từ viễn thông, hóa đơn điện nước, TMĐT khi chưa có lịch sử vay.")

with _form_place:
    if _has_result:
        st.markdown("### ✏️ Chỉnh hồ sơ & đánh giá lại")
        st.caption("Sửa bên dưới rồi nhấn **Đánh giá tín dụng**.")
    else:
        st.markdown("### 📝 Hồ sơ đăng ký vay")
        st.caption("Điền đầy đủ các nhóm dưới đây rồi nhấn **Đánh giá tín dụng** ở cuối.")

    with st.form("credit_form"):
        # Khi form ở main: 2 cột (Cá nhân | Khoản vay). Khi form ở sidebar: 1 cột, xếp dọc.
        if not _has_result:
            row1_left, row1_right = st.columns(2)
        else:
            row1_left = st.container()
            row1_right = st.container()

        with row1_left:
            with st.expander("👤 Cá nhân", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    gender = st.selectbox(
                        "Giới tính",
                        ["M", "F"],
                        format_func=lambda x: "Nam" if x == "M" else "Nữ",
                        help="Chọn giới tính của bạn",
                    )
                    age = st.number_input("Tuổi", 18, 80, 30, 1, help="Tuổi hiện tại")
                with col2:
                    children = st.number_input(
                        "Số con",
                        0,
                        20,
                        0,
                        1,
                        help="Số con đang nuôi dưỡng. Nhiều con = gánh nặng tài chính lớn hơn",
                    )
                    family = st.number_input(
                        "Thành viên gia đình",
                        1,
                        30,
                        2,
                        1,
                        help="Tổng số người cùng sống trong hộ (bao gồm bản thân)",
                    )
                id_publish_years = st.number_input(
                    "Đổi CCCD/CMND cách đây (năm)",
                    0,
                    50,
                    5,
                    1,
                    help="Bạn đổi CCCD/CMND lần cuối cách đây bao lâu? CCCD mới = đáng tin cậy hơn",
                )
                registration_years = st.number_input(
                    "Đăng ký cư trú cách đây (năm)",
                    0,
                    60,
                    10,
                    1,
                    help="Thời gian đăng ký cư trú tại địa chỉ hiện tại. Ổn định lâu = điểm tốt hơn",
                )
                education = st.selectbox(
                    "Trình độ học vấn",
                    ["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"],
                    index=3,
                    format_func={"Lower secondary": "THCS", "Secondary / secondary special": "THPT/Trung cấp", "Incomplete higher": "CĐ/ĐH dở dang", "Higher education": "Đại học", "Academic degree": "Sau đại học"}.get,
                    help="Cấp học cao nhất bạn đã hoàn thành",
                )
                family_status = st.selectbox(
                    "Tình trạng hôn nhân",
                    ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"],
                    index=1,
                    format_func={"Single / not married": "Độc thân", "Married": "Đã kết hôn", "Civil marriage": "Sống chung", "Separated": "Ly thân", "Widow": "Góa"}.get,
                    help="Tình trạng gia đình hiện tại",
                )

        with row1_right:
            with st.expander("💰 Khoản vay", expanded=True):
                st.caption("VD: 12 triệu/tháng → nhập 12000000")
                monthly_income = st.number_input(
                    "Thu nhập hàng tháng (VNĐ)",
                    0,
                    value=12_000_000,
                    step=500_000,
                    help="Lương thực nhận mỗi tháng (trước thuế). VD: 12 triệu/tháng → nhập 12,000,000",
                )
                credit = st.number_input(
                    "Số tiền muốn vay (VNĐ)",
                    0,
                    value=300_000_000,
                    step=10_000_000,
                    help="Tổng khoản tiền bạn muốn vay. Vay càng lớn so với thu nhập → rủi ro càng cao",
                )
                annuity = st.number_input(
                    "Trả góp hàng tháng (VNĐ)",
                    0,
                    value=10_000_000,
                    step=500_000,
                    help="Số tiền gốc + lãi phải trả mỗi tháng. VD: Vay 300tr, kỳ hạn 36 tháng → ~10tr/tháng",
                )
                goods_price = st.number_input(
                    "Giá trị hàng hóa / mục đích vay (VNĐ)",
                    0,
                    value=280_000_000,
                    step=10_000_000,
                    help="Giá thực tế của tài sản bạn mua (nhà, xe, hàng hóa). Nếu vay > giá trị = rủi ro",
                )
                contract_type = st.selectbox(
                    "Loại hợp đồng",
                    ["Cash loans", "Revolving loans"],
                    format_func=lambda x: "Vay tiền mặt (trả góp)" if x == "Cash loans" else "Tín dụng tuần hoàn",
                    help="Vay tiền mặt = vay 1 lần, trả góp đều. Tín dụng tuần hoàn = hạn mức quay vòng",
                )

        # ---- Nhóm 3: Việc làm ----
        with st.expander("💼 Việc làm", expanded=False):
            emp_years = st.number_input(
                "Số năm đi làm",
                0.0,
                60.0,
                5.0,
                0.5,
                help="Thâm niên ở công việc hiện tại. Làm lâu = ổn định = điểm tốt hơn",
            )
            _income_type_map = {"Working": "Đi làm", "Commercial associate": "Đối tác TM", "Pensioner": "Hưu trí", "State servant": "Công chức", "Businessman": "Kinh doanh", "Student": "Sinh viên", "Unemployed": "Thất nghiệp", "Maternity leave": "Nghỉ thai sản"}
            income_type = st.selectbox(
                "Loại thu nhập",
                list(_income_type_map.keys()),
                format_func=_income_type_map.get,
                help="Nguồn thu nhập chính của bạn hiện tại",
            )
            _occupation_map = {
                None: "— Bỏ qua —", "Laborers": "Lao động phổ thông", "Sales staff": "NV kinh doanh", "Core staff": "Nhân viên chính", "Managers": "Quản lý", "Drivers": "Tài xế", "High skill tech staff": "Kỹ thuật cao", "Accountants": "Kế toán", "Medicine staff": "Y tế", "Cooking staff": "Đầu bếp", "Security staff": "Bảo vệ", "Cleaning staff": "Vệ sinh", "Private service staff": "Dịch vụ tư nhân", "Low-skill Laborers": "LĐ giản đơn", "Waiters/barmen staff": "Phục vụ", "Secretaries": "Thư ký", "Realty agents": "Môi giới BĐS", "IT staff": "CNTT", "HR staff": "Nhân sự",
            }
            occupation = st.selectbox(
                "Nghề nghiệp (không bắt buộc)",
                list(_occupation_map.keys()),
                format_func=_occupation_map.get,
                help="Chọn nghề gần nhất với công việc hiện tại. Bỏ qua nếu không muốn khai",
            )
            _org_type_map = {
                None: "— Không khai —", "Business Entity Type 3": "DN tư nhân", "Self-employed": "Tự kinh doanh", "Business Entity Type 2": "TNHH", "Business Entity Type 1": "Cổ phần", "Government": "Nhà nước", "Medicine": "Y tế", "School": "Trường học", "Kindergarten": "Mầm non", "University": "ĐH/CĐ", "Construction": "Xây dựng", "Military": "Quân đội", "Police": "Công an", "Bank": "Ngân hàng", "Insurance": "Bảo hiểm", "Agriculture": "Nông nghiệp", "Restaurant": "Nhà hàng", "Hotel": "Khách sạn", "Transport: type 4": "Vận tải", "Security": "Bảo vệ", "Telecom": "Viễn thông", "Trade: type 7": "Thương mại", "Industry: type 9": "Công nghiệp", "Other": "Khác",
            }
            organization = st.selectbox(
                "Loại tổ chức",
                list(_org_type_map.keys()),
                format_func=_org_type_map.get,
                help="Loại công ty/tổ chức bạn đang làm việc. Ảnh hưởng đến đánh giá sự ổn định công việc",
            )

        # ---- Nhóm 4: Tài sản ----
        with st.expander("🏠 Tài sản & Nhà ở", expanded=False):
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
                    0,
                    80,
                    5,
                    1,
                    help="Xe đã sử dụng bao nhiêu năm? Xe mới = giá trị tài sản cao hơn",
                )
            _housing_map = {"House / apartment": "Nhà riêng/Căn hộ", "With parents": "Ở cùng bố mẹ", "Municipal apartment": "Chung cư NN", "Rented apartment": "Thuê nhà", "Office apartment": "Căn hộ VP", "Co-op apartment": "Chung cư HTX"}
            housing = st.selectbox(
                "Loại nhà ở",
                list(_housing_map.keys()),
                format_func=_housing_map.get,
                help="Bạn đang ở đâu? Nhà sở hữu được đánh giá tốt hơn nhà thuê",
            )

        # ---- Nhóm 5: Điểm thay thế ----
        with st.expander("📊 Điểm tín dụng thay thế (0–1)", expanded=False):
            st.caption("Thang điểm: 0.0 (rất kém) → 1.0 (rất tốt). Để mặc định 0.5 nếu không rõ.")
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
        with st.expander("📱 Liên lạc", expanded=False):
            col8, col9 = st.columns(2)
            with col8:
                has_emp_phone = st.checkbox(
                    "SĐT cơ quan",
                    value=True,
                    help="SĐT bàn tổng đài của công ty bạn đang làm",
                )
                has_work_phone = st.checkbox(
                    "SĐT nơi làm",
                    value=False,
                    help="SĐT trực tiếp phòng ban / bộ phận của bạn",
                )
            with col9:
                has_phone = st.checkbox(
                    "SĐT nhà",
                    value=True,
                    help="Số điện thoại cố định tại nhà riêng",
                )
                has_email = st.checkbox(
                    "Email",
                    value=True,
                    help="Bạn có email cá nhân đang sử dụng không?",
                )
            phone_change_days = st.number_input(
                "Đổi SĐT gần nhất (ngày)",
                0,
                10000,
                365,
                30,
                help="Bạn đổi số điện thoại lần cuối cách đây bao nhiêu ngày? Dùng SĐT càng lâu = đáng tin cậy hơn",
            )

        # ---- Nhóm 7: Mạng lưới ----
        with st.expander("👥 Mạng lưới xã hội", expanded=False):
            def_30 = st.number_input(
                "Người quen vỡ nợ 30 ngày",
                0,
                10,
                0,
                1,
                help="Số người quen bị vỡ nợ/trễ hạn trong 30 ngày gần nhất. 0 = không có ai",
            )
            def_60 = st.number_input(
                "Người quen vỡ nợ 60 ngày",
                0,
                10,
                0,
                1,
                help="Số người quen bị vỡ nợ/trễ hạn trong 60 ngày gần nhất. 0 = không có ai",
            )

        st.markdown("---")
        submitted = st.form_submit_button("🔍 Đánh giá tín dụng", width="stretch", type="primary")


# Build user_input (giống app.py)
user_input = {
    "CODE_GENDER": gender,
    "AGE_YEARS": float(age),
    "CNT_CHILDREN": int(children),
    "CNT_FAM_MEMBERS": float(family),
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family_status,
    "AMT_INCOME_TOTAL": float(monthly_income) * 12,
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
# HELPERS: Gauge, SHAP chart
# ============================================================
def make_gauge(score: int, tier_color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 44, "color": tier_color}},
        gauge={
            "axis": {"range": [300, 850], "tickwidth": 1, "tickcolor": "#ccc", "tickvals": [300, 400, 500, 580, 670, 740, 800, 850]},
            "bar": {"color": tier_color, "thickness": 0.3},
            "bgcolor": "#f5f5f5",
            "steps": [
                {"range": [300, 580], "color": "#ffcdd2"},
                {"range": [580, 670], "color": "#fff9c4"},
                {"range": [670, 740], "color": "#bbdefb"},
                {"range": [740, 800], "color": "#c8e6c9"},
                {"range": [800, 850], "color": "#a5d6a7"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": score},
        },
        title={"text": "FICO Score (300–850)", "font": {"size": 14}},
    ))
    fig.update_layout(height=260, margin=dict(l=25, r=25, t=45, b=10))
    return fig


def make_shap_chart(shap_top: list) -> go.Figure | None:
    if not shap_top:
        return None
    features = [FEATURE_LABELS_VI.get(s["feature"], s["feature"]) for s in reversed(shap_top)]
    values = [s["shap_value"] for s in reversed(shap_top)]
    colors = ["#ef5350" if v > 0 else "#66bb6a" for v in values]
    fig = go.Figure(go.Bar(x=values, y=features, orientation="h", marker_color=colors, text=[f"{v:+.3f}" for v in values], textposition="outside", textfont={"size": 11}))
    fig.update_layout(title={"text": "Giải thích (SHAP)", "font": {"size": 14}}, xaxis_title="Mức ảnh hưởng", height=max(280, 32 * len(shap_top)), margin=dict(l=10, r=10, t=35, b=25), yaxis={"tickfont": {"size": 11}}, plot_bgcolor="white")
    fig.add_vline(x=0, line_width=1, line_color="grey")
    return fig


def collect_improvement_suggestions(shap_top_features: list[dict]) -> list[str]:
    suggestions = []
    seen_keys = set()
    for item in shap_top_features:
        if item.get("direction") != "risk":
            continue
        feat = item.get("feature")
        key = None
        if feat in ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "AMT_INCOME_TOTAL", "INCOME_PER_PERSON"):
            key = "income"
        elif feat in ("EMPLOYMENT_YEARS", "EMPLOYED_TO_AGE_RATIO", "DAYS_EMPLOYED"):
            key = "employment"
        elif feat in ("AMT_CREDIT", "CREDIT_TERM_MONTHS", "AMT_ANNUITY", "PAYMENT_RATE", "GOODS_CREDIT_RATIO"):
            key = "credit_amount"
        elif feat in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE"):
            key = "assets"
        elif feat in ("CONTACT_COUNT", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"):
            key = "contact"
        elif isinstance(feat, str) and feat.startswith("EXT_SOURCE"):
            key = "ext_source"
        elif feat in ("SOCIAL_DEF_TOTAL", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"):
            key = "social"
        if key and key not in seen_keys:
            seen_keys.add(key)
            suggestion = IMPROVEMENT_SUGGESTIONS.get(key)
            if suggestion:
                suggestions.append(suggestion)
    return suggestions


# ============================================================
# MAIN: Submit → lưu result
# ============================================================
if submitted:
    with st.spinner("Đang phân tích hồ sơ..."):
        result = engine.predict(user_input)
    st.session_state.scoring_result = result
    st.session_state.scoring_input = dict(user_input)
    st.session_state.chat_messages = []
    st.session_state.report_pdf_bytes = None
    st.session_state.report_pdf_valid = False
    st.session_state.report_pdf_meta = {}
    st.session_state.report_pdf_signature = ""
    st.rerun()


# ============================================================
# Chưa có kết quả → dừng (form đã nằm ở main phía trên)
# ============================================================
if "scoring_result" not in st.session_state:
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#9e9e9e; font-size:0.8rem;">Credit Scoring Demo • Dành cho người chưa có lịch sử tín dụng</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# CÓ KẾT QUẢ — Tabs
# ============================================================
result = st.session_state.scoring_result
score = result["credit_score"]
proba = result["probability"]
risk = RISK_LEVELS[result["risk_key"]]
tier = next((t for t in TIERS if t["score_min"] <= score <= t["score_max"]), TIERS[-1])

# Nút chỉnh sửa + Chuẩn bị/Tải PDF
current_pdf_signature = f"{result.get('credit_score')}|{result.get('probability')}|{result.get('risk_key')}|{repr(sorted(st.session_state.scoring_input.items()))}"
if st.session_state.get("report_pdf_signature") != current_pdf_signature:
    st.session_state.report_pdf_bytes = None
    st.session_state.report_pdf_valid = False
    st.session_state.report_pdf_meta = {}
    st.session_state.report_pdf_signature = ""

col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    if st.button("✏️ Chỉnh sửa hồ sơ / Đánh giá lại", width="stretch"):
        for key in [
            "scoring_result",
            "scoring_input",
            "chat_messages",
            "report_pdf_bytes",
            "report_pdf_valid",
            "report_pdf_meta",
            "report_pdf_signature",
        ]:
            st.session_state.pop(key, None)
        st.rerun()
with col_btn2:
    if st.button("🧾 Chuẩn bị báo cáo PDF", width="stretch"):
        with st.spinner("Đang tạo báo cáo PDF..."):
            _advisor = CreditAdvisor()
            _assessment_result = _advisor.assess(result, st.session_state.scoring_input, FEATURE_LABELS_VI)
            _suggestions = collect_improvement_suggestions(result.get("shap_top_features", []))
            _fig_g = make_gauge(score, tier["color"])
            _fig_s = make_shap_chart(result["shap_top_features"])
            _pdf_bytes, _pdf_meta = build_credit_report_pdf(
                result=result,
                user_input=st.session_state.scoring_input,
                feature_labels=FEATURE_LABELS_VI,
                assessment_text=_assessment_result.text,
                suggestions_list=_suggestions,
                fig_gauge=_fig_g,
                fig_shap=_fig_s,
                return_meta=True,
            )
        st.session_state.report_pdf_bytes = _pdf_bytes
        st.session_state.report_pdf_valid = True
        st.session_state.report_pdf_meta = _pdf_meta
        st.session_state.report_pdf_signature = current_pdf_signature

if st.session_state.get("report_pdf_valid") and st.session_state.get("report_pdf_bytes"):
    _download_kwargs = dict(
        label="📥 Tải báo cáo PDF",
        data=st.session_state.report_pdf_bytes,
        file_name="bao_cao_danh_gia_tin_dung.pdf",
        mime="application/pdf",
        width="stretch",
    )
    try:
        import inspect

        if "on_click" in inspect.signature(st.download_button).parameters:
            _download_kwargs["on_click"] = "ignore"
    except Exception:
        pass
    st.download_button(**_download_kwargs)
    _pdf_warnings = st.session_state.get("report_pdf_meta", {}).get("warnings", [])
    if _pdf_warnings:
        st.warning("\n".join([f"- {w}" for w in _pdf_warnings]))
else:
    st.caption("Nhấn **Chuẩn bị báo cáo PDF** trước, rồi bấm **Tải báo cáo PDF**.")

tab_overview, tab_shap, tab_assessment, tab_chat = st.tabs(["📈 Kết quả", "🔬 Giải thích SHAP", "📋 Nhận xét", "🤖 Hỏi AI"])

# ---- Tab 1: Kết quả ----
with tab_overview:
    col_score, col_tier = st.columns([1, 1.2])
    with col_score:
        fig_gauge = make_gauge(score, tier["color"])
        st.plotly_chart(fig_gauge, width="stretch")
        st.caption(f"Mức hiện tại: **{tier['name']}**")
    with col_tier:
        st.markdown(
            f'<div class="tier-card" style="background:{tier["color"]}15; border-color:{tier["color"]};"><h2 style="margin:0; color:{tier["color"]};">{tier["icon"]} {tier["name"]}</h2><p style="margin:0.4rem 0 0; font-size:0.95rem;">{tier["description"]}</p></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="value" style="color:{tier["color"]};">{score}</div><div class="label">FICO Score</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="value" style="color:{risk["color"]};">{proba:.1%}</div><div class="label">Xác suất vỡ nợ</div></div>', unsafe_allow_html=True)
        with m3:
            cl = result["credit_limit"]
            cl_text = format_vnd(cl) if cl > 0 else "N/A"
            st.markdown(f'<div class="metric-box"><div class="value">{cl_text}</div><div class="label">Hạn mức đề xuất</div></div>', unsafe_allow_html=True)
        st.markdown("**Chi tiết:**")
        st.markdown(f"- FICO: {score}/850 • Xác suất vỡ nợ: {proba:.2%} • Rủi ro: {risk['emoji']} {risk['label']}")
        if tier["interest_modifier"] is not None:
            adj = 12.0 + tier["interest_modifier"] * 100
            st.markdown(f"- Lãi suất dự kiến: **{adj:.1f}%/năm**")

    st.markdown("---")
    st.markdown("**💡 Đề xuất cải thiện**")
    _suggestions_overview = collect_improvement_suggestions(result.get("shap_top_features", []))
    for suggestion in _suggestions_overview:
        st.markdown(f'<div class="suggestion-card">💡 {suggestion}</div>', unsafe_allow_html=True)
    if not _suggestions_overview:
        st.success("Hồ sơ tốt, không cần cải thiện thêm.")

    st.markdown("---")
    st.markdown("**📊 Hệ thống xếp hạng**")
    tier_cols = st.columns(len(TIERS))
    for i, t in enumerate(TIERS):
        with tier_cols[i]:
            is_current = t["name"] == tier["name"]
            border = f"3px solid {t['color']}" if is_current else "1px solid #e0e0e0"
            bg = f"{t['color']}18" if is_current else "#fafafa"
            cur = " ◄ Bạn" if is_current else ""
            st.markdown(f'<div style="border:{border}; background:{bg}; border-radius:8px; padding:0.6rem; text-align:center; min-height:100px;"><div style="font-size:1.4rem;">{t["icon"]}</div><div style="font-weight:700; color:{t["color"]}; font-size:0.8rem;">{t["name"]}</div><div style="font-size:0.7rem; color:#757575;">{t["score_min"]}–{t["score_max"]}{cur}</div></div>', unsafe_allow_html=True)

# ---- Tab 2: SHAP ----
with tab_shap:
    st.markdown("🔴 **Đỏ** = tăng rủi ro vỡ nợ • 🟢 **Xanh** = giảm rủi ro. Thanh càng dài → ảnh hưởng càng lớn.")
    fig_shap = make_shap_chart(result["shap_top_features"])
    if fig_shap:
        st.plotly_chart(fig_shap, width="stretch")
    else:
        st.info("SHAP chưa sẵn sàng.")

# ---- Tab 3: Nhận xét ----
with tab_assessment:
    template_advisor = CreditAdvisor()
    template_result = template_advisor.assess(result, st.session_state.scoring_input, FEATURE_LABELS_VI)
    # Hiển thị tóm tắt + expander chi tiết
    lines = template_result.text.strip().split("\n")
    summary_lines = [l for l in lines if l.startswith("### 1.") or l.startswith("- **FICO") or l.startswith("- **Xác suất") or (l.startswith("### 5.") and "Kết luận" in "".join(lines[lines.index(l):lines.index(l)+2]))]
    intro = "\n".join(lines[: min(15, len(lines))])
    st.markdown(intro)
    if len(lines) > 15:
        with st.expander("📄 Xem đầy đủ nhận xét"):
            st.markdown(template_result.text)

# ---- Tab 4: Chat AI ----
with tab_chat:
    st.caption("Hỏi về hồ sơ tín dụng — AI tư vấn dựa trên kết quả vừa chấm.")
    st.warning("AI có thể sai. Chỉ tham khảo, không thay cam kết ngân hàng.", icon="⚠️")
    if not ai_api_key:
        st.info("Cấu hình **OPENROUTER_API_KEY** trong file `.env` để bật tư vấn AI. Lấy key tại [OpenRouter](https://openrouter.ai/keys).")
    else:
        for msg in st.session_state.get("chat_messages", []):
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input("Câu hỏi", placeholder="VD: Làm sao tăng điểm tín dụng?", label_visibility="collapsed")
            if st.form_submit_button("Gửi 🚀", width="stretch") and question.strip():
                st.session_state.chat_messages.append({"role": "user", "content": question})
                if len(st.session_state.chat_messages) > 10:
                    st.session_state.chat_messages = st.session_state.chat_messages[-10:]
                with st.spinner("AI đang trả lời..."):
                    chat_advisor = CreditAdvisor(api_key=ai_api_key, model=ai_model)
                    response = chat_advisor.chat(st.session_state.chat_messages, result, st.session_state.scoring_input, FEATURE_LABELS_VI)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()

# Footer (không hardcode AUC/features)
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#9e9e9e; font-size:0.8rem;">Credit Scoring Demo • Dành cho người chưa có lịch sử tín dụng • Chỉ dùng dữ liệu hồ sơ đăng ký</div>',
    unsafe_allow_html=True,
)
