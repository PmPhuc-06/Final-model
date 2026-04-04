from __future__ import annotations

import math
from typing import Iterable

from engine_common import (
    TU_KHOA_TIENG_ANH,
    TU_KHOA_TAI_CHINH_VN,
    chuan_hoa_khong_dau,
    chuan_hoa_text,
    co_cum_tu,
    phat_hien_co_do,
    tien_xu_ly_day_du,
    tim_doan_nghi_ngo,
    tinh_chat_luong_van_ban,
)


HYBRID_METADATA_FEATURE_NAMES = [
    # ─ Thống kê căn bản ─
    "log_char_len",
    "log_token_count",
    "avg_token_len",
    "ocr_quality",
    "digit_ratio",
    "uppercase_ratio",
    "newline_density",
    # ─ Tín hiệu gian lận rule-based ─
    "red_flag_count",
    "suspicious_segment_count",
    # ─ Tín hiệu ngôn ngữ ─
    "english_keyword_ratio",
    "has_related_party",
    "has_invoice_risk",
    "has_cashflow_risk",
    "has_off_balance_risk",
    # ─ Tín hiệu tài chính mở rộng (AuditBERT Feature Fusion) ─
    "financial_term_density_vn",   # Mật độ từ khóa tài chính tiếng Việt
    "financial_term_density_en",   # Mật độ từ khóa tài chính tiếng Anh (MFinBERT signal)
    "round_number_ratio",          # Tỷ lệ số tròn (Benford's Law signal) - Baseline signal
    "accounting_code_count",       # Số lượng mã khoản kế toán (VAS: 111, 131, 511...)
    "loss_keyword_count",          # Đếm từ "âm", "lỗ", "giảm", "thiếu"
    "abnormal_profit_signal",      # Lợi nhuận biến động cực lớn (% vs dạng)
]


def trich_xuat_metadata_features(text: str) -> dict[str, float]:
    tokens_sach, _ = tien_xu_ly_day_du(text)
    chat_luong = tinh_chat_luong_van_ban(text, tokens_sach)
    red_flags = phat_hien_co_do(text)
    suspicious_segments = tim_doan_nghi_ngo(text)

    normalized = chuan_hoa_text(text)
    normalized_ascii = chuan_hoa_khong_dau(text)
    variants = [normalized, normalized_ascii]

    char_len = len(text.strip())
    token_count = len(tokens_sach)
    letter_count = sum(1 for ch in text if ch.isalpha())
    digit_count = sum(1 for ch in text if ch.isdigit())
    uppercase_count = sum(1 for ch in text if ch.isupper())
    newline_count = text.count("\n")

    raw_tokens = text.lower().split()
    english_hits = sum(1 for token in raw_tokens if token.strip(".,:;!?") in TU_KHOA_TIENG_ANH)
    english_keyword_ratio = english_hits / max(len(raw_tokens), 1)

    avg_token_len = sum(len(token) for token in tokens_sach) / max(token_count, 1)

    has_related_party = 1.0 if co_cum_tu(variants, ["bên liên quan", "ben lien quan"]) else 0.0
    has_invoice_risk = 1.0 if co_cum_tu(variants, ["hóa đơn giả", "hoa don gia", "doanh thu khống"]) else 0.0
    has_cashflow_risk = 1.0 if co_cum_tu(variants, ["dòng tiền âm", "dong tien am"]) else 0.0
    has_off_balance_risk = 1.0 if co_cum_tu(
        variants,
        ["ngoài bảng cân đối", "ngoai bang can doi", "off-balance sheet"],
    ) else 0.0

    # ─ Feature Fusion mở rộng cho AuditBERT ─────────────────────────────
    text_lower = text.lower()

    # 1. Mật độ thuật ngữ tài chính tiếng Việt (Baseline TF-IDF signal)
    vn_hits = sum(1 for kw in TU_KHOA_TAI_CHINH_VN if kw in text_lower)
    financial_term_density_vn = vn_hits / max(len(raw_tokens), 1)

    # 2. Mật độ thuật ngữ tài chính tiếng Anh (MFinBERT signal proxy)
    fin_en_keywords = {
        "assets", "liabilities", "equity", "revenue", "profit", "loss",
        "cash", "receivable", "payable", "provision", "impairment",
        "related", "party", "disclosure", "going-concern", "restatement",
        "misstatement", "overstated", "fictitious", "embezzlement",
    }
    en_fin_hits = sum(1 for t in raw_tokens if t.strip(".,;") in fin_en_keywords)
    financial_term_density_en = en_fin_hits / max(len(raw_tokens), 1)

    # 3. Tỷ lệ số tròn (Benford's Law heuristic — Baseline signal)
    import re as _re
    all_numbers = _re.findall(r"\b\d+\b", text)
    round_count = sum(1 for n in all_numbers if n.endswith("000") and len(n) >= 6)
    round_number_ratio = round_count / max(len(all_numbers), 1)

    # 4. Số lượng mã khoản kế toán VAS (chữu ảnh hưởng từ BCTC thực tế)
    vas_codes = _re.findall(r"\b(1[0-9]{2}|2[0-9]{2}|3[0-9]{2}|4[0-9]{2}|5[0-9]{2}|6[0-9]{2}|7[0-9]{2})\b", text)
    accounting_code_count = float(len(set(vas_codes)))  # dùng set để khử trùng

    # 5. Đếm từ âm thế / lỗ / giảm / thiếu
    loss_kws = ["lỗ", "am", "âm", "giảm", "thiếu", "không đủ", "không đạt", "không đọc", "khó đòi"]
    loss_keyword_count = float(sum(text_lower.count(kw) for kw in loss_kws))

    # 6. Tín hiệu biến động lợi nhuận bất thường
    abnormal_profit_signal = 1.0 if co_cum_tu(
        variants,
        ["lợi nhuận đột biến", "tăng đột ngột", "biến động lớn", "không giải thích được",
         "unusual profit", "abnormal gain"],
    ) else 0.0

    return {
        "log_char_len": math.log1p(char_len),
        "log_token_count": math.log1p(token_count),
        "avg_token_len": avg_token_len,
        "ocr_quality": chat_luong,
        "digit_ratio": digit_count / max(char_len, 1),
        "uppercase_ratio": uppercase_count / max(letter_count, 1),
        "newline_density": newline_count / max(char_len, 1),
        "red_flag_count": float(len(red_flags)),
        "suspicious_segment_count": float(len(suspicious_segments)),
        "english_keyword_ratio": english_keyword_ratio,
        "has_related_party": has_related_party,
        "has_invoice_risk": has_invoice_risk,
        "has_cashflow_risk": has_cashflow_risk,
        "has_off_balance_risk": has_off_balance_risk,
        # Feature Fusion mở rộng
        "financial_term_density_vn": financial_term_density_vn,
        "financial_term_density_en": financial_term_density_en,
        "round_number_ratio": round_number_ratio,
        "accounting_code_count": accounting_code_count,
        "loss_keyword_count": loss_keyword_count,
        "abnormal_profit_signal": abnormal_profit_signal,
    }


def vector_hoa_metadata_features(
    feature_map: dict[str, float],
    feature_names: Iterable[str] = HYBRID_METADATA_FEATURE_NAMES,
) -> list[float]:
    return [float(feature_map.get(name, 0.0)) for name in feature_names]


def tao_hybrid_feature_vector(text: str, raw_model_prob: float) -> tuple[dict[str, float], list[float]]:
    feature_map = trich_xuat_metadata_features(text)
    vector = [float(raw_model_prob)] + vector_hoa_metadata_features(feature_map)
    return feature_map, vector


def lam_tron_metadata_features(feature_map: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: round(float(value), digits) for key, value in feature_map.items()}
