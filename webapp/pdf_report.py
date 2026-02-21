"""
Tạo báo cáo PDF kết quả đánh giá tín dụng.
- Font tiếng Việt (DejaVu Sans / Arial)
- Nhúng biểu đồ: Gauge FICO, SHAP.
"""

import io
import os
import re
from pathlib import Path
from typing import Any

# reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Font tiếng Việt: ưu tiên font nhúng trong repo, sau đó fallback hệ thống.
# Nếu không có TTF khả dụng thì dùng Helvetica (kém với dấu tiếng Việt).
_PDF_FONT_NAME = "Helvetica"
_EMBEDDED_FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "DejaVuSans.ttf"
_SYSTEM_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
]


def _register_vietnamese_font():
    global _PDF_FONT_NAME
    if _PDF_FONT_NAME != "Helvetica":
        return
    font_candidates = []

    # 1) Font nhúng trong repo (ổn định nhất giữa môi trường dev/prod)
    if _EMBEDDED_FONT_PATH.exists():
        font_candidates.append(_EMBEDDED_FONT_PATH)

    # 2) Fallback theo hệ điều hành
    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    font_candidates.extend(
        [
            windir / "Fonts" / "arial.ttf",
            windir / "Fonts" / "Arial.ttf",
        ]
    )
    font_candidates.extend(_SYSTEM_FONT_CANDIDATES)

    seen = set()
    for font_path in font_candidates:
        font_path = Path(font_path)
        key = str(font_path).lower()
        if key in seen:
            continue
        seen.add(key)
        if not font_path.exists():
            continue
        if font_path.suffix.lower() not in {".ttf", ".otf"}:
            continue
        try:
            pdfmetrics.registerFont(TTFont("VNFont", str(font_path)))
            _PDF_FONT_NAME = "VNFont"
            return
        except Exception:
            continue


def _format_value(key: str, value) -> str:
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return "—"
    if "AMT_" in key or "INCOME" in key or key == "AMT_CREDIT":
        try:
            x = float(value)
            if x >= 1e9:
                return f"{x/1e9:.1f} tỷ VNĐ"
            if x >= 1e6:
                return f"{x/1e6:.0f} triệu VNĐ"
            return f"{x:,.0f}"
        except (TypeError, ValueError):
            return str(value)
    if key == "CODE_GENDER":
        return "Nam" if value == "M" else "Nữ"
    if "FLAG_OWN" in key:
        return "Có" if value == "Y" else "Không"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def plotly_fig_to_png_bytes(fig, width=500, height=280):
    """Chuyển Plotly figure sang PNG bytes (cần kaleido)."""
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", width=width, height=height)
    except Exception:
        return None


def build_credit_report_pdf(
    result: dict,
    user_input: dict,
    feature_labels: dict,
    assessment_text: str,
    suggestions_list: list,
    fig_gauge=None,
    fig_shap=None,
    return_meta: bool = False,
) -> bytes | tuple[bytes, dict[str, Any]]:
    """
    Tạo PDF báo cáo đánh giá tín dụng.
    Trả về bytes PDF. Font tiếng Việt, có nhúng biểu đồ.
    """
    _register_vietnamese_font()
    fn = _PDF_FONT_NAME
    warnings: list[str] = []
    gauge_embedded = False
    shap_embedded = False

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    title_style = ParagraphStyle(
        name="VNTitle",
        fontName=fn,
        fontSize=18,
        leading=22,
        spaceAfter=12,
        alignment=1,
    )
    heading_style = ParagraphStyle(
        name="VNHeading",
        fontName=fn,
        fontSize=12,
        leading=14,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        name="VNBody",
        fontName=fn,
        fontSize=10,
        leading=12,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        name="VNSmall",
        fontName=fn,
        fontSize=9,
        leading=11,
        spaceAfter=4,
    )

    def p(text: str, style=body_style):
        # Escape & cho XML; giữ <b>, </b> để in đậm
        text = (text or "").replace("&", "&amp;")
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
        return Paragraph(text, style)

    flowables = []

    # ---- Tiêu đề ----
    flowables.append(p("BÁO CÁO ĐÁNH GIÁ TÍN DỤNG", title_style))
    flowables.append(p("Dành cho khách hàng chưa có lịch sử tín dụng", small_style))
    flowables.append(Spacer(1, 0.5 * cm))

    # ---- 1. Thông tin hồ sơ ----
    flowables.append(p("1. Thông tin hồ sơ đăng ký", heading_style))
    data = [["Thông tin", "Giá trị"]]
    for key, raw_value in user_input.items():
        label = feature_labels.get(key, key)
        value = _format_value(key, raw_value)
        data.append([label, value])
    t = Table(data, colWidths=[5 * cm, 10 * cm])
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), fn),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ]
        )
    )
    flowables.append(t)
    flowables.append(Spacer(1, 0.5 * cm))

    # ---- 2. Kết quả ----
    score = result.get("credit_score", 0)
    proba = result.get("probability", 0)
    risk_key = result.get("risk_key", "")
    credit_limit = result.get("credit_limit", 0)
    cl_text = f"{credit_limit/1e6:.0f} triệu VNĐ" if credit_limit > 0 else "Không cấp"

    flowables.append(p("2. Kết quả đánh giá", heading_style))
    flowables.append(p(f"• FICO Score: <b>{score}/850</b>", body_style))
    flowables.append(p(f"• Xác suất vỡ nợ: <b>{proba:.2%}</b>", body_style))
    flowables.append(p(f"• Mức rủi ro: {risk_key}", body_style))
    flowables.append(p(f"• Hạn mức đề xuất: {cl_text}", body_style))

    # Biểu đồ Gauge
    gauge_bytes = plotly_fig_to_png_bytes(fig_gauge, width=480, height=260)
    if gauge_bytes:
        gauge_embedded = True
        img = Image(io.BytesIO(gauge_bytes), width=14 * cm, height=7 * cm)
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.append(img)
    elif fig_gauge is not None:
        warnings.append(
            "Không thể nhúng biểu đồ Gauge vào PDF (cần kaleido hoạt động đúng)."
        )
    flowables.append(Spacer(1, 0.5 * cm))

    # ---- 3. Giải thích (SHAP) ----
    flowables.append(p("3. Giải thích kết quả (SHAP)", heading_style))
    flowables.append(
        p("Đỏ = tăng rủi ro vỡ nợ; Xanh = giảm rủi ro. Thanh càng dài thì ảnh hưởng càng lớn.", small_style)
    )
    shap_bytes = plotly_fig_to_png_bytes(fig_shap, width=500, height=max(280, 32 * 10))
    if shap_bytes:
        shap_embedded = True
        img_shap = Image(io.BytesIO(shap_bytes), width=14 * cm, height=8 * cm)
        flowables.append(img_shap)
    elif fig_shap is not None:
        warnings.append(
            "Không thể nhúng biểu đồ SHAP vào PDF (cần kaleido hoạt động đúng)."
        )
    flowables.append(Spacer(1, 0.5 * cm))

    # ---- 4. Nhận xét ----
    flowables.append(p("4. Nhận xét tín dụng", heading_style))
    for line in (assessment_text or "").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("###"):
            flowables.append(p(line.replace("#", "").strip(), heading_style))
        else:
            flowables.append(p(line, body_style))
    flowables.append(Spacer(1, 0.5 * cm))

    # ---- 6. Đề xuất ----
    if suggestions_list:
        flowables.append(p("6. Đề xuất cải thiện", heading_style))
        for s in suggestions_list:
            flowables.append(p(f"• {s}", body_style))

    flowables.append(Spacer(1, 1 * cm))
    flowables.append(p("— Báo cáo được tạo tự động từ hệ thống Credit Scoring —", small_style))

    
    doc.build(flowables)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()

    if return_meta:
        return (
            pdf_bytes,
            {
                "font_name": fn,
                "gauge_embedded": gauge_embedded,
                "shap_embedded": shap_embedded,
                "warnings": warnings,
            },
        )
    return pdf_bytes

# --- Demo: Tạo file PDF khi chạy trực tiếp ---
if __name__ == "__main__":
    # Dữ liệu mẫu
    result = {"credit_score": 720, "probability": 0.12, "risk_key": "Trung bình", "credit_limit": 50000000}
    user_input = {"CODE_GENDER": "M", "AMT_INCOME_TOTAL": 12000000}
    feature_labels = {"CODE_GENDER": "Giới tính", "AMT_INCOME_TOTAL": "Thu nhập"}
    assessment_text = "Khách hàng có thu nhập ổn định."
    suggestions_list = ["Tăng thu nhập", "Thanh toán đúng hạn"]
    fig_gauge = None
    fig_shap = None
    pdf_bytes = build_credit_report_pdf(result, user_input, feature_labels, assessment_text, suggestions_list, fig_gauge, fig_shap)
    with open("report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Đã tạo file report.pdf. Hãy mở file này để xem báo cáo.")
