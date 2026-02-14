"""
AI Credit Advisor — V1
Generates natural-language credit assessments in Vietnamese using LLM APIs.

Uses OpenRouter (https://openrouter.ai) as the unified LLM gateway.
Supports hundreds of models including free tiers.
Falls back to template-based assessment when no API key is configured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# SYSTEM PROMPT (Vietnamese credit advisor persona)
# ============================================================
SYSTEM_PROMPT = """\
Bạn là chuyên gia tư vấn tín dụng cao cấp tại một ngân hàng lớn ở Việt Nam.
Nhiệm vụ: phân tích kết quả chấm điểm tín dụng và đưa ra nhận xét chi tiết bằng tiếng Việt.

Quy tắc:
- Luôn viết tiếng Việt, giọng văn chuyên nghiệp nhưng dễ hiểu.
- Sử dụng đơn vị tiền tệ VNĐ.
- Đưa ra đánh giá khách quan dựa trên dữ liệu, không đoán mò.
- Nêu rõ điểm mạnh và điểm yếu của hồ sơ.
- Đề xuất hành động cụ thể, khả thi để cải thiện.
- Trả lời có cấu trúc rõ ràng với các mục đánh giá.
- KHÔNG bịa thêm thông tin ngoài dữ liệu được cung cấp.
"""

# ============================================================
# CHAT SYSTEM PROMPT (interactive Q&A)
# ============================================================
CHAT_SYSTEM_PROMPT = """\
Bạn là tư vấn viên tín dụng AI tại ngân hàng Việt Nam.
Bạn đang tư vấn cho một khách hàng vừa được chấm điểm tín dụng.
Kết quả chấm điểm đã được cung cấp ở tin nhắn đầu tiên.

Quy tắc:
- Trả lời bằng tiếng Việt, thân thiện, chuyên nghiệp.
- Dựa trên dữ liệu thực tế của khách hàng để tư vấn.
- Đưa ra lời khuyên cụ thể, khả thi.
- Trả lời ngắn gọn, tập trung vào câu hỏi (tối đa 200 từ).
- KHÔNG bịa thêm thông tin ngoài dữ liệu được cung cấp.
- Sử dụng emoji phù hợp để thân thiện hơn.
"""

# ============================================================
# USER PROMPT TEMPLATE
# ============================================================
USER_PROMPT_TEMPLATE = """\
## Kết quả chấm điểm tín dụng

**FICO Score:** {fico_score}/850
**Xác suất vỡ nợ:** {probability:.2%}
**Mức rủi ro:** {risk_level}
**Hạn mức đề xuất:** {credit_limit}

## Thông tin hồ sơ
- Giới tính: {gender}
- Tuổi: {age} tuổi
- Trình độ: {education}
- Hôn nhân: {family_status}
- Thu nhập tháng: {monthly_income} VNĐ
- Số tiền vay: {credit_amount} VNĐ
- Trả góp/tháng: {annuity} VNĐ
- Thâm niên: {emp_years} năm
- Nghề nghiệp: {occupation}
- Sở hữu ô tô: {own_car}
- Sở hữu BĐS: {own_realty}
- Điểm viễn thông: {ext1}
- Điểm tiện ích: {ext2}
- Điểm TMĐT: {ext3}

## Top yếu tố ảnh hưởng (SHAP)
{shap_summary}

## Yêu cầu
Hãy viết một bản nhận xét tín dụng chi tiết bao gồm:
1. **Tổng quan**: Đánh giá chung về hồ sơ (2-3 câu).
2. **Điểm mạnh**: Liệt kê các yếu tố tích cực (dựa trên SHAP xanh/safe).
3. **Điểm cần cải thiện**: Liệt kê các yếu tố rủi ro (dựa trên SHAP đỏ/risk).
4. **Phân tích chi tiết**: Giải thích tại sao từng yếu tố lại ảnh hưởng đến điểm.
5. **Khuyến nghị**: 3-5 hành động cụ thể để cải thiện điểm tín dụng.
6. **Kết luận**: Quyết định cuối cùng và lời khuyên.
"""


# ============================================================
# Data classes
# ============================================================
@dataclass
class AdvisorResult:
    """Result from AI advisor."""

    text: str
    provider: str  # "openrouter" | "template"
    model: str | None = None
    success: bool = True
    error: str | None = None


# ============================================================
# OpenRouter API caller
# ============================================================
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _call_openrouter(api_key: str, system: str, user_msg: str, model: str) -> str:
    """
    Call OpenRouter via the OpenAI-compatible SDK.
    OpenRouter proxies hundreds of models (GPT, Gemini, Claude, Llama, etc.)
    through a single endpoint + API key.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://credit-scoring-demo.streamlit.app",
            "X-Title": "Credit Scoring AI Advisor",
        },
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=2000,
    )
    return resp.choices[0].message.content


def _call_openrouter_chat(api_key: str, messages: list[dict], model: str) -> str:
    """Call OpenRouter with a full message list (for multi-turn chat)."""
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://credit-scoring-demo.streamlit.app",
            "X-Title": "Credit Scoring AI Advisor",
        },
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=1500,
    )
    return resp.choices[0].message.content


# ============================================================
# Popular models on OpenRouter (curated list for UI)
# ============================================================
OPENROUTER_MODELS = {
    # --- Free models ---
    "deepseek/deepseek-r1-0528:free": "🆓 DeepSeek R1 0528 (free)",
    "nvidia/nemotron-3-nano-30b-a3b:free": "🆓 NVIDIA Nemotron 3 Nano (free)",
    "stepfun/step-3.5-flash:free": "🆓 StepFun 3.5 Flash (free)",
    "z-ai/glm-4.5-air:free": "🆓 GLM 4.5 Air (free)",
    "arcee-ai/trinity-large-preview:free": "🆓 Arcee Trinity Large (free)",
    # --- Cheap models ---
    "google/gemini-2.5-flash-preview": "💰 Gemini 2.5 Flash (rẻ)",
    "openai/gpt-4o-mini": "💰 GPT-4o Mini ($0.15/1M)",
    "anthropic/claude-3.5-haiku": "💰 Claude 3.5 Haiku ($0.80/1M)",
    "deepseek/deepseek-chat": "💰 DeepSeek V3 ($0.27/1M)",
}

DEFAULT_MODEL = "deepseek/deepseek-r1-0528:free"

# Fallback chain: if primary model fails (429/404), try next one
FALLBACK_MODELS = [
    "deepseek/deepseek-r1-0528:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "stepfun/step-3.5-flash:free",
    "z-ai/glm-4.5-air:free",
    "arcee-ai/trinity-large-preview:free",
]


# ============================================================
# Template fallback (no API key)
# ============================================================
_RISK_DESCRIPTIONS = {
    "VERY_LOW": "Hồ sơ được đánh giá ở mức rủi ro rất thấp — xếp hạng xuất sắc.",
    "LOW": "Hồ sơ có mức rủi ro thấp — xếp hạng rất tốt.",
    "MEDIUM": "Hồ sơ có mức rủi ro trung bình — xếp hạng khá.",
    "HIGH": "Hồ sơ có mức rủi ro cao — cần cải thiện đáng kể.",
    "VERY_HIGH": "Hồ sơ có mức rủi ro rất cao — khó được phê duyệt.",
}


def _template_assessment(
    result: dict[str, Any],
    user_input: dict[str, Any],
    feature_labels: dict[str, str],
) -> str:
    """Generate a structured assessment WITHOUT calling any LLM."""
    score = result["credit_score"]
    proba = result["probability"]
    risk_key = result["risk_key"]
    shap_top = result.get("shap_top_features", [])

    lines = []
    lines.append("##  Nhận Xét Tín Dụng Tự Động\n")

    # 1. Overview
    lines.append("### 1. Tổng quan")
    lines.append(f"- **FICO Score:** {score}/850")
    lines.append(f"- **Xác suất vỡ nợ:** {proba:.2%}")
    lines.append(f"- {_RISK_DESCRIPTIONS.get(risk_key, '')}\n")

    # 2. Strengths
    safe_factors = [s for s in shap_top if s["direction"] == "safe"]
    if safe_factors:
        lines.append("### 2. Điểm mạnh ")
        for s in safe_factors[:5]:
            name = feature_labels.get(s["feature"], s["feature"])
            lines.append(
                f"- **{name}**: Đóng góp tích cực (SHAP = {s['shap_value']:+.4f})"
            )
        lines.append("")

    # 3. Weaknesses
    risk_factors = [s for s in shap_top if s["direction"] == "risk"]
    if risk_factors:
        lines.append("### 3. Điểm cần cải thiện ")
        for s in risk_factors[:5]:
            name = feature_labels.get(s["feature"], s["feature"])
            lines.append(f"- **{name}**: Tăng rủi ro (SHAP = {s['shap_value']:+.4f})")
        lines.append("")

    # 4. Recommendations
    lines.append("### 4. Khuyến nghị")
    recs = _generate_template_recommendations(risk_factors, user_input)
    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    # 5. Conclusion
    lines.append("### 5. Kết luận")
    if score >= 740:
        lines.append(
            "Hồ sơ đủ điều kiện phê duyệt với điều kiện tốt. "
            "Nên duy trì thói quen tài chính hiện tại."
        )
    elif score >= 670:
        lines.append(
            "Hồ sơ có thể được phê duyệt nhưng lãi suất có thể cao hơn. "
            "Cải thiện các yếu tố rủi ro sẽ giúp giảm chi phí vay."
        )
    elif score >= 580:
        lines.append(
            "Hồ sơ ở mức cận biên. Cần cải thiện một số yếu tố "
            "trước khi nộp đơn để tăng khả năng phê duyệt."
        )
    else:
        lines.append(
            "Hồ sơ hiện tại có rủi ro cao. Khuyến nghị cải thiện "
            "các yếu tố nêu trên trong 3-6 tháng rồi nộp đơn lại."
        )

    lines.append("\n---")
    lines.append(
        "*Nhận xét được tạo tự động bằng hệ thống đánh giá nội bộ.*"
    )

    return "\n".join(lines)


def _generate_template_recommendations(
    risk_factors: list[dict], user_input: dict
) -> list[str]:
    """Generate recommendations based on risk factors."""
    recs: list[str] = []

    risk_features = {s["feature"] for s in risk_factors}

    if risk_features & {
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "AMT_INCOME_TOTAL",
        "INCOME_PER_PERSON",
    }:
        recs.append(
            "Tăng thu nhập hoặc giảm số tiền vay để cải thiện tỷ lệ nợ/thu nhập."
        )

    if risk_features & {"EMPLOYMENT_YEARS", "EMPLOYED_TO_AGE_RATIO", "DAYS_EMPLOYED"}:
        recs.append("Duy trì công việc hiện tại lâu hơn để chứng minh sự ổn định.")

    if risk_features & {
        "AMT_CREDIT",
        "CREDIT_TERM_MONTHS",
        "AMT_ANNUITY",
        "PAYMENT_RATE",
        "GOODS_CREDIT_RATIO",
    }:
        recs.append("Xem xét giảm số tiền vay hoặc kéo dài kỳ hạn trả góp.")

    if risk_features & {"FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE"}:
        recs.append(
            "Tích lũy tài sản (nhà, xe) làm tài sản đảm bảo sẽ tăng điểm đáng kể."
        )

    if risk_features & {
        "CONTACT_COUNT",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    }:
        recs.append("Cung cấp đầy đủ các kênh liên lạc (SĐT, email, SĐT cơ quan).")

    if any(f.startswith("EXT_SOURCE") for f in risk_features):
        recs.append(
            "Cải thiện điểm tín dụng thay thế bằng cách thanh toán hóa đơn đúng hạn."
        )

    if risk_features & {
        "SOCIAL_DEF_TOTAL",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    }:
        recs.append("Tránh liên kết tài chính với người đang có nợ xấu.")

    if not recs:
        recs.append("Tiếp tục duy trì thói quen tài chính tốt hiện tại.")

    return recs[:5]


# ============================================================
# MAIN CLASS
# ============================================================


class CreditAdvisor:
    """
    AI Credit Advisor — generates Vietnamese credit assessments.

    Uses OpenRouter as the unified LLM gateway (https://openrouter.ai).
    Falls back to template-based output when no API key is provided.

    Usage:
        advisor = CreditAdvisor(api_key="sk-or-...", model="google/gemini-2.0-flash-exp:free")
        result = advisor.assess(engine_result, user_input, feature_labels)
        print(result.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL

    # ----------------------------------------------------------
    def assess(
        self,
        result: dict[str, Any],
        user_input: dict[str, Any],
        feature_labels: dict[str, str],
    ) -> AdvisorResult:
        """
        Generate a credit assessment.

        Parameters
        ----------
        result : dict from CreditScoringEngine.predict()
        user_input : raw user input dict
        feature_labels : FEATURE_LABELS_VI from config

        Returns
        -------
        AdvisorResult with .text (Markdown), .provider, .success
        """
        # If no API key, use template fallback
        if not self.api_key:
            text = _template_assessment(result, user_input, feature_labels)
            return AdvisorResult(text=text, provider="template", success=True)

        # Build the prompt
        user_msg = self._build_prompt(result, user_input, feature_labels)

        # Build model chain: primary model first, then fallbacks
        models_to_try = [self.model]
        for fb in FALLBACK_MODELS:
            if fb != self.model:
                models_to_try.append(fb)

        # Try each model in chain
        last_error = None
        for model in models_to_try:
            try:
                logger.info("Trying model: %s", model)
                text = _call_openrouter(self.api_key, SYSTEM_PROMPT, user_msg, model)
                return AdvisorResult(
                    text=text,
                    provider="openrouter",
                    model=model,
                    success=True,
                )
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Only retry on rate-limit (429) or not-found (404)
                if "429" in error_str or "404" in error_str:
                    logger.warning("Model %s failed (%s), trying next...", model, error_str[:80])
                    continue
                else:
                    # Other errors (auth, network) — don't retry
                    break

        # All models failed — fallback to template
        logger.error("All OpenRouter models failed. Last error: %s", last_error)
        text = _template_assessment(result, user_input, feature_labels)
        text += (
            f"\n\n> *Lỗi khi gọi AI (đã thử {len(models_to_try)} model): {last_error}. "
            f"Đã sử dụng nhận xét tự động.*"
        )
        return AdvisorResult(
            text=text,
            provider="template",
            model=self.model,
            success=False,
            error=str(last_error),
        )

    # ----------------------------------------------------------
    def chat(
        self,
        chat_messages: list[dict],
        result: dict[str, Any],
        user_input: dict[str, Any],
        feature_labels: dict[str, str],
    ) -> str:
        """
        Interactive chat: answer user questions about their credit score.

        Parameters
        ----------
        chat_messages : list of {"role": "user"/"assistant", "content": "..."}
        result : dict from CreditScoringEngine.predict()
        user_input : raw user input dict
        feature_labels : FEATURE_LABELS_VI from config

        Returns
        -------
        str : AI response text
        """
        if not self.api_key:
            return "⚠️ Chức năng tư vấn AI chưa được cấu hình."

        # Build context from scoring result
        context = self._build_prompt(result, user_input, feature_labels)

        # Full message list: system + context + chat history
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Đây là kết quả chấm điểm tín dụng của tôi:\n\n{context}",
            },
            {
                "role": "assistant",
                "content": (
                    "Tôi đã xem kết quả chấm điểm tín dụng của bạn rồi. "
                    "Bạn muốn tôi tư vấn gì ạ? 😊"
                ),
            },
        ]
        messages.extend(chat_messages)

        # Try models with fallback
        models_to_try = [self.model] + [
            m for m in FALLBACK_MODELS if m != self.model
        ]
        last_error = None

        for model in models_to_try:
            try:
                logger.info("Chat: trying model %s", model)
                response = _call_openrouter_chat(self.api_key, messages, model)
                # Strip any <think>...</think> reasoning blocks (DeepSeek R1)
                import re
                response = re.sub(
                    r"<think>.*?</think>", "", response, flags=re.DOTALL
                ).strip()
                return response
            except Exception as e:
                last_error = e
                if "429" in str(e) or "404" in str(e):
                    logger.warning("Chat: model %s failed, trying next...", model)
                    continue
                break

        return (
            "⚠️ Không thể kết nối AI lúc này. Vui lòng thử lại sau.\n\n"
            f"*Lỗi: {last_error}*"
        )

    # ----------------------------------------------------------
    def _build_prompt(
        self,
        result: dict[str, Any],
        user_input: dict[str, Any],
        feature_labels: dict[str, str],
    ) -> str:
        """Build the user prompt from scoring results."""
        # SHAP summary
        shap_lines = []
        for s in result.get("shap_top_features", []):
            direction = (
                "🔴 Tăng rủi ro" if s["direction"] == "risk" else "🟢 Giảm rủi ro"
            )
            name = feature_labels.get(s["feature"], s["feature"])
            shap_lines.append(f"- {name}: SHAP = {s['shap_value']:+.4f} ({direction})")
        shap_summary = "\n".join(shap_lines) if shap_lines else "Không có dữ liệu SHAP."

        # Risk level mapping
        risk_map = {
            "VERY_LOW": "Rất thấp",
            "LOW": "Thấp",
            "MEDIUM": "Trung bình",
            "HIGH": "Cao",
            "VERY_HIGH": "Rất cao",
        }

        # Gender
        gender_map = {"M": "Nam", "F": "Nữ"}

        # Education
        edu_map = {
            "Lower secondary": "THCS",
            "Secondary / secondary special": "THPT / Trung cấp",
            "Incomplete higher": "Cao đẳng / ĐH dở dang",
            "Higher education": "Đại học",
            "Academic degree": "Sau đại học",
        }

        # Family status
        fam_map = {
            "Single / not married": "Độc thân",
            "Married": "Đã kết hôn",
            "Civil marriage": "Sống chung",
            "Separated": "Ly thân",
            "Widow": "Góa",
        }

        # Credit limit display
        cl = result.get("credit_limit", 0)
        cl_text = f"{cl:,.0f} VNĐ" if cl > 0 else "Không cấp"

        # Monthly income (model gets annual, convert back)
        annual_income = user_input.get("AMT_INCOME_TOTAL", 0)
        monthly = annual_income / 12

        # Occupation
        occ = user_input.get("OCCUPATION_TYPE")
        occ_text = occ if occ else "Không khai báo"

        return USER_PROMPT_TEMPLATE.format(
            fico_score=result["credit_score"],
            probability=result["probability"],
            risk_level=risk_map.get(result["risk_key"], result["risk_key"]),
            credit_limit=cl_text,
            gender=gender_map.get(user_input.get("CODE_GENDER", ""), "N/A"),
            age=int(user_input.get("AGE_YEARS", 0)),
            education=edu_map.get(user_input.get("NAME_EDUCATION_TYPE", ""), "N/A"),
            family_status=fam_map.get(user_input.get("NAME_FAMILY_STATUS", ""), "N/A"),
            monthly_income=f"{monthly:,.0f}",
            credit_amount=f"{user_input.get('AMT_CREDIT', 0):,.0f}",
            annuity=f"{user_input.get('AMT_ANNUITY', 0):,.0f}",
            emp_years=user_input.get("EMPLOYMENT_YEARS", 0),
            occupation=occ_text,
            own_car="Có" if user_input.get("FLAG_OWN_CAR") == "Y" else "Không",
            own_realty="Có" if user_input.get("FLAG_OWN_REALTY") == "Y" else "Không",
            ext1=user_input.get("EXT_SOURCE_1", 0.5),
            ext2=user_input.get("EXT_SOURCE_2", 0.5),
            ext3=user_input.get("EXT_SOURCE_3", 0.5),
            shap_summary=shap_summary,
        )
