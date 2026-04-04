from __future__ import annotations

import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


MAU_TACH_TU = re.compile(r"\w+", re.UNICODE)
MAU_HTML = re.compile(r"<[^>]+>")
MAU_KHOANG_TRANG = re.compile(r"\s+")

try:
    from underthesea import word_tokenize as tach_tu_underthesea
except Exception:  # pragma: no cover - optional dependency
    tach_tu_underthesea = None

try:
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional dependency
    GoogleTranslator = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_SPLITTER_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGCHAIN_SPLITTER_AVAILABLE = False

from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Hằng số chung
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "và", "la", "là", "của", "cua", "các", "cac", "những", "nhung",
    "được", "duoc", "cho", "với", "voi", "một", "mot", "có", "co",
    "trong", "khi", "đã", "da", "đang", "dang", "từ", "tu",
    "the", "and", "or", "of", "to", "is", "are",
}

TU_KHOA_TIENG_ANH = {
    # Từ thông dụng
    "the", "and", "of", "to", "is", "are", "with", "from", "that", "this",
    # Thuật ngữ kế toán / kiểm toán chuẩn quốc tế
    "revenue", "invoice", "liabilities", "audit", "transactions", "profit",
    "assets", "equity", "expenses", "income", "loss", "earnings", "cash",
    "balance", "sheet", "statement", "financial", "accounting", "fiscal",
    "disclosure", "provision", "receivable", "payable", "inventory",
    "depreciation", "amortization", "impairment", "goodwill", "intangible",
    # Dấu hiệu rủi ro / gian lận tài chính
    "related", "party", "off-balance", "fictitious", "overstated",
    "understated", "manipulation", "misstatement", "restatement",
    "round-tripping", "kickback", "embezzlement", "bribery",
    "misappropriation", "concealment", "falsification",
    # Thuật ngữ thuyết minh BCTC
    "note", "notes", "contingent", "commitment", "subsequent",
    "materiality", "going-concern", "qualified", "disclaimer",
}

TU_KHOA_TAI_CHINH_VN = [
    # Từ khóa tiếng Việt về gian lận kế toán
    "doanh thu khống", "doanh thu ao", "hóa đơn giả", "hóa đơn khống",
    "chi phí ảo", "chi phí khống", "khai khống", "khai thiếu",
    "thổi phồng", "che giấu", "gian lận", "gian dối", "làm giả",
    "sai lệch", "sai phạm", "vi phạm", "không trung thực",
    "bên liên quan", "giao dịch nội bộ", "chuyển giá",
    "lỗ lũy kế", "âm vốn chủ", "mất khả năng thanh toán",
    "dự phòng không đủ", "không trích lập", "trích lập thiếu",
    "ngoài bảng cân đối", "tài sản ảo", "công nợ ẩn",
    "thu nhập bất thường", "lợi nhuận đột biến", "biến động lớn",
    "không có chứng từ", "thiếu chứng từ", "chứng từ không hợp lệ",
]

CUM_TU_GHEP = [
    "báo cáo tài chính", "bao cao tai chinh",
    "bên liên quan", "ben lien quan",
    "doanh thu khống", "doanh thu tăng",
    "hóa đơn giả", "hoa don gia",
    "dòng tiền", "dong tien",
    "kiểm toán viên", "kiem toan vien",
    "chính sách kế toán", "chinh sach ke toan",
    "ngoài bảng cân đối", "ngoai bang can doi",
    "lợi nhuận sau thuế", "loi nhuan sau thue",
    "vốn chủ sở hữu", "von chu so huu",
    "tổng tài sản", "tong tai san",
    "lưu chuyển tiền tệ", "luu chuyen tien te",
]
# Alias tương thích ngược
CU_M_TU_GHEP = CUM_TU_GHEP

# Hằng số cho baseline/transformer model
BASELINE_CHECKPOINT      = "baseline_fraud_checkpoint.pkl"

# PhoBERT — backbone tốt nhất cho tiếng Việt
PHOBERT_MODEL_NAME       = "vinai/phobert-base"
PHOBERT_CHECKPOINT       = "phobert_fraud_checkpoint.pt"
PHOBERT_MAX_LEN          = 256
PHOBERT_BATCH_SIZE       = 8
PHOBERT_EPOCHS           = 5
PHOBERT_LR               = 2e-5
PHOBERT_PATIENCE         = 2
PHOBERT_THRESHOLD_METRIC = "f2"

# MFinBERT — hiểu thuật ngữ tài chính tiếng Anh tốt
MFINBERT_MODEL_NAME       = "sonnv/MFinBERT"
MFINBERT_CHECKPOINT       = "mfinbert_fraud_checkpoint.pt"
MFINBERT_MAX_LEN          = 256
MFINBERT_BATCH_SIZE       = 8
MFINBERT_EPOCHS           = 5
MFINBERT_LR               = 2e-5
MFINBERT_PATIENCE         = 2
MFINBERT_THRESHOLD_METRIC = "f2"

# AuditBERT-VN — mô hình tổng hợp cuối cùng
# Backbone: PhoBERT (tiếng Việt tốt nhất) + Hybrid Metadata mở rộng
# (tích hợp tín hiệu tài chính từ Baseline + MFinBERT vào feature vector)
AUDITBERT_MODEL_NAME       = "vinai/phobert-base"
AUDITBERT_CHECKPOINT       = "auditbert_fraud_checkpoint.pt"
AUDITBERT_MAX_LEN          = 256
AUDITBERT_BATCH_SIZE       = 8
AUDITBERT_EPOCHS           = 6
AUDITBERT_LR               = 1e-5
AUDITBERT_PATIENCE         = 3
AUDITBERT_THRESHOLD_METRIC = "f2"

# Recursive chunking
CHUNK_MAX_CHARS = PHOBERT_MAX_LEN * 3   # ~768 ký tự
CHUNK_OVERLAP   = 120


# ===========================================================================
# PREPROCESSING PIPELINE
# ===========================================================================

def xu_ly_van_ban(text: str) -> list[str]:
    return MAU_TACH_TU.findall(text.lower())


def bo_dau(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def chuan_hoa_text(text: str) -> str:
    lowered = text.lower()
    lowered = MAU_KHOANG_TRANG.sub(" ", lowered).strip()
    return lowered


def chuan_hoa_khong_dau(text: str) -> str:
    return chuan_hoa_text(bo_dau(text))


def co_cum_tu(texts: list[str], phrases: list[str]) -> bool:
    for phrase in phrases:
        normalized_phrase       = chuan_hoa_text(phrase)
        normalized_ascii_phrase = chuan_hoa_khong_dau(phrase)
        if any(
            normalized_phrase in text_variant or normalized_ascii_phrase in text_variant
            for text_variant in texts
        ):
            return True
    return False


def lam_sach_van_ban(text: str) -> str:
    text = MAU_HTML.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹ_\s]", " ", text)
    text = chuan_hoa_text(text)
    return text


def tach_tu_tieng_viet(text: str) -> list[str]:
    if not text:
        return []
    if tach_tu_underthesea is not None:
        try:
            tokenized = tach_tu_underthesea(text, format="text")
            return tokenized.split()
        except Exception:
            pass
    text_thay_the = f" {text} "
    for cum_tu in CU_M_TU_GHEP:
        mau = re.escape(cum_tu)
        text_thay_the = re.sub(mau, cum_tu.replace(" ", "_"), text_thay_the)
    return xu_ly_van_ban(text_thay_the)


def bo_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]


def tien_xu_ly_day_du(text: str) -> tuple[list[str], str]:
    van_ban_sach = lam_sach_van_ban(text)
    tokens = tach_tu_tieng_viet(van_ban_sach)
    tokens = bo_stopwords(tokens)
    van_ban_san_sang = " ".join(tokens)
    return tokens, van_ban_san_sang


def tao_van_ban_hien_thi(text: str) -> str:
    van_ban = lam_sach_van_ban(text)
    return dich_hien_thi_sang_viet(van_ban)


def la_van_ban_tieng_anh(text: str) -> bool:
    tokens = xu_ly_van_ban(text)
    if not tokens:
        return False
    ascii_ratio  = sum(1 for ch in text if ord(ch) < 128) / max(len(text), 1)
    english_hits = sum(1 for token in tokens if token.lower() in TU_KHOA_TIENG_ANH)
    return ascii_ratio > 0.9 and english_hits >= 3


def dich_hien_thi_sang_viet(text: str) -> str:
    if not text:
        return text
    if not la_van_ban_tieng_anh(text):
        return text
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source="auto", target="vi").translate(text)
    except Exception:
        return text


# RED FLAGS — rule-based

def phat_hien_co_do(text: str) -> list[str]:
    lowered       = chuan_hoa_text(text)
    lowered_ascii = chuan_hoa_khong_dau(text)
    text_variants = [lowered, lowered_ascii]
    flags: list[str] = []

    revenue_growth_phrases = [
        "doanh thu tăng", "doanh thu tăng mạnh", "doanh thu tăng đột biến",
    ]
    weak_cashflow_phrases = [
        "dòng tiền âm", "dòng tiền từ hoạt động kinh doanh âm",
        "dòng tiền hoạt động âm", "tiền từ hoạt động kinh doanh âm",
    ]
    if co_cum_tu(text_variants, revenue_growth_phrases) and co_cum_tu(text_variants, weak_cashflow_phrases):
        flags.append("doanh_thu_khong_di_kem_dong_tien")

    checks = {
        "doanh_thu_tang_nhung_loi_nhuan_giam": [
            "doanh thu tăng nhưng lợi nhuận gộp giảm",
            "doanh thu tăng nhưng lợi nhuận giảm",
            "revenue increased but gross profit decreased",
        ],
        "hoa_don_gia_hoac_thoi_phong_doanh_thu": [
            "hóa đơn giả", "hoa don gia",
            "thổi phồng doanh thu", "ghi nhận doanh thu khống", "doanh thu khống",
        ],
        "che_giau_no_phai_tra": [
            "che giấu nợ phải trả",
            "nợ ngoài bảng cân đối kế toán",
            "thỏa thuận ngoài bảng cân đối",
            "off-balance sheet arrangement",
            "off-balance sheet liabilities",
            "off-balance sheet financing",
        ],
        "dau_hieu_quan_tri_rui_ro": [
            "bên liên quan", "ben lien quan",
            "thay đổi kiểm toán viên", "thay doi kiem toan vien",
            "thay đổi chính sách kế toán", "thay doi chinh sach ke toan",
        ],
        "window_dressing_cuoi_ky": [
            "tập trung vào ngày cuối",
            "tap trung vao ngay cuoi",
            "ngày cuối cùng của niên độ",
            "ngay cuoi cung cua nien do",
            "cuối niên độ kế toán",
            "cuoi nien do ke toan",
            "tập trung phát sinh vào ngày 31",
            "tap trung phat sinh vao ngay 31",
            "giải ngân tập trung vào cuối",
            "giai ngan tap trung vao cuoi",
        ],
    }
    for rule_id, phrases in checks.items():
        if co_cum_tu(text_variants, phrases):
            flags.append(rule_id)

    _off_balance_phrases = [
        "off-balance", "ngoài bảng cân đối",
    ]
    _ngu_canh_che_giau = [
        "che giấu", "che dấu", "giấu nợ", "không công bố",
        "không minh bạch", "hidden liabilities", "concealed",
        "undisclosed liabilities", "thỏa thuận ngầm",
    ]
    if (
        co_cum_tu(text_variants, _off_balance_phrases)
        and co_cum_tu(text_variants, _ngu_canh_che_giau)
        and "che_giau_no_phai_tra" not in flags
    ):
        flags.append("che_giau_no_phai_tra")

    return flags


def tach_doan(text: str) -> list[str]:
    parts = re.split(r"(?:\n{2,}|[\n\r]+|(?<=[\.\?\!])\s+)", text)
    return [part.strip() for part in parts if part and part.strip()]


def tim_doan_nghi_ngo(text: str) -> list[dict[str, str]]:
    ket_qua: list[dict[str, str]] = []
    for doan in tach_doan(text):
        danh_sach_co_do = phat_hien_co_do(doan)
        if danh_sach_co_do:
            ket_qua.append({"snippet": doan[:300], "flags": ", ".join(danh_sach_co_do)})
    return ket_qua


# ===========================================================================
# RECURSIVE CHUNKING
# ===========================================================================

def tach_chunks_recursive(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    if LANGCHAIN_SPLITTER_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        return [c for c in chunks if c.strip()]

    # Fallback: cắt cứng có overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


# ===========================================================================
# DATACLASS — dùng chung cho cả 2 model
# ===========================================================================

@dataclass
class PredictionResult:
    label: int
    muc_do_rui_ro: str
    model_probability_fraud: float
    model_probability_non_fraud: float
    probability_fraud: float
    probability_non_fraud: float
    threshold_used: float
    chat_luong_van_ban: float
    red_flags: list[str]
    explanation: list[str]
    top_terms: list[str]
    van_ban_sach: str
    van_ban_embedding: str
    doan_nghi_ngo: list[dict[str, str]]
    metadata_features: dict[str, float] = field(default_factory=dict)
    raw_text_probability_fraud: float | None = None
    raw_text_probability_non_fraud: float | None = None
    bias_flags: list[str] = field(default_factory=list)
    explainability: dict[str, object] = field(default_factory=dict)


def tach_text_va_label_tu_dataset(data: object) -> tuple[list[str], list[int]]:
    if isinstance(data, dict) and "records" in data:
        data = data["records"]

    if isinstance(data, list):
        texts = [str(item["text"]) for item in data]
        labels = [int(item["label"]) for item in data]
    elif isinstance(data, dict):
        if "texts" not in data or "labels" not in data:
            raise ValueError("JSON object phải chứa đủ 2 khóa 'texts' và 'labels'.")
        texts = [str(text) for text in data["texts"]]
        labels = [int(label) for label in data["labels"]]
    else:
        raise ValueError(f"Format JSON không hỗ trợ: {type(data)}")

    if len(texts) != len(labels):
        raise ValueError("Số lượng texts và labels không khớp nhau.")
    return texts, labels


def tai_du_lieu_json(json_path: str) -> tuple[list[str], list[int]]:
    raw = Path(json_path).read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("File dataset trống.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        try:
            data = [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Dataset phải là JSON array, JSON object hoặc JSONL hợp lệ."
            ) from exc

    return tach_text_va_label_tu_dataset(data)


# ===========================================================================
# DATA SPLIT UTILITIES
# ===========================================================================

@dataclass
class DataSplit:
    train_texts:  list[str]
    train_labels: list[int]
    val_texts:    list[str]
    val_labels:   list[int]
    test_texts:   list[str]
    test_labels:  list[int]

    def summary(self) -> dict:
        def _stats(labels: list[int]) -> dict:
            n = len(labels)
            f = sum(labels)
            return {"total": n, "fraud": f, "non_fraud": n - f,
                    "fraud_ratio": round(f / n, 4) if n else 0.0}
        return {
            "train": _stats(self.train_labels),
            "val":   _stats(self.val_labels),
            "test":  _stats(self.test_labels),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("=" * 55)
        print(f"{'Split':<8} {'Total':>7} {'Fraud':>7} {'Non-F':>7} {'%Fraud':>8}")
        print("-" * 55)
        for split_name in ("train", "val", "test"):
            st = s[split_name]
            print(f"{split_name:<8} {st['total']:>7} {st['fraud']:>7} "
                  f"{st['non_fraud']:>7} {st['fraud_ratio']:>7.2%}")
        print("=" * 55)


def _loc_bo_trung_lap(
    texts: list[str],
    labels: list[int],
) -> tuple[list[str], list[int]]:
    seen: set[str] = set()
    clean_texts, clean_labels = [], []
    for text, label in zip(texts, labels):
        if text not in seen:
            seen.add(text)
            clean_texts.append(text)
            clean_labels.append(label)
    removed = len(texts) - len(clean_texts)
    if removed:
        print(f"[DataSplit] Đã loại {removed} mẫu exact-duplicate.")
    return clean_texts, clean_labels


def tao_train_val_test_split(
    texts: list[str],
    labels: list[int],
    ty_le_train: float = 0.8,
    ty_le_val:   float = 0.1,
    seed: int = 42,
    loc_trung_lap: bool = True,
) -> DataSplit:
    from sklearn.model_selection import train_test_split

    texts_list  = list(texts)
    labels_list = list(labels)

    if loc_trung_lap:
        texts_list, labels_list = _loc_bo_trung_lap(texts_list, labels_list)

    ty_le_val_test = 1.0 - ty_le_train
    stratify_labels = labels_list
    if len(set(labels_list)) < 2 or min(Counter(labels_list).values()) < 2:
        stratify_labels = None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts_list, labels_list,
        test_size=ty_le_val_test,
        random_state=seed,
        stratify=stratify_labels,
    )

    ty_le_val_trong_tmp = ty_le_val / ty_le_val_test
    stratify_tmp = y_tmp
    if len(set(y_tmp)) < 2 or min(Counter(y_tmp).values()) < 2:
        stratify_tmp = None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1.0 - ty_le_val_trong_tmp,
        random_state=seed,
        stratify=stratify_tmp,
    )

    split = DataSplit(
        train_texts=X_train, train_labels=y_train,
        val_texts=X_val,     val_labels=y_val,
        test_texts=X_test,   test_labels=y_test,
    )
    split.print_summary()
    return split


def luu_split_ra_file(
    split: DataSplit,
    thu_muc: str = ".",
    ten_file_prefix: str = "split",
) -> dict[str, str]:
    out_dir = Path(thu_muc)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name, texts, labels in [
        ("train", split.train_texts, split.train_labels),
        ("val",   split.val_texts,   split.val_labels),
        ("test",  split.test_texts,  split.test_labels),
    ]:
        file_path = out_dir / f"{ten_file_prefix}_{split_name}.jsonl"
        with file_path.open("w", encoding="utf-8") as fh:
            for text, label in zip(texts, labels):
                fh.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
        paths[split_name] = str(file_path.resolve())
        print(f"[DataSplit] Đã lưu {split_name}: {file_path} ({len(texts)} mẫu)")
    return paths


def tai_split_tu_file(
    thu_muc: str = ".",
    ten_file_prefix: str = "split",
) -> DataSplit:
    in_dir = Path(thu_muc)
    result: dict[str, tuple[list[str], list[int]]] = {}
    for split_name in ("train", "val", "test"):
        file_path = in_dir / f"{ten_file_prefix}_{split_name}.jsonl"
        texts, labels = [], []
        with file_path.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                texts.append(obj["text"])
                labels.append(int(obj["label"]))
        result[split_name] = (texts, labels)
        print(f"[DataSplit] Đã tải {split_name}: {len(texts)} mẫu")
    return DataSplit(
        train_texts=result["train"][0], train_labels=result["train"][1],
        val_texts=result["val"][0],     val_labels=result["val"][1],
        test_texts=result["test"][0],   test_labels=result["test"][1],
    )


# ===========================================================================
# HELPER FUNCTIONS — metrics, utils
# ===========================================================================

def tinh_chat_luong_van_ban(text: str, tokens_sach: list[str]) -> float:
    raw_tokens = xu_ly_van_ban(text)
    if not raw_tokens:
        return 0.0
    clean_ratio      = len(tokens_sach) / len(raw_tokens)
    long_token_ratio = sum(1 for t in tokens_sach if len(t) >= 3) / max(len(tokens_sach), 1)
    ascii_noise      = sum(1 for t in raw_tokens if any(c.isdigit() for c in t) and len(t) > 6) / len(raw_tokens)
    score = 0.6 * clean_ratio + 0.5 * long_token_ratio - 0.3 * ascii_noise
    return max(0.0, min(1.0, score))


def xep_muc_do_rui_ro(score: float) -> str:
    if score >= 0.75:
        return "Cao"
    if score >= 0.45:
        return "Trung bình"
    return "Thấp"


def tim_nguong_toi_uu(
    y_true: list[int],
    y_scores: list[float],
    metric: str = "f2",
) -> tuple[float, dict[str, float]]:
    if not y_scores or sum(y_true) == 0:
        return 0.5, danh_gia_du_doan(y_true, y_scores, threshold=0.5)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    best_threshold = 0.5
    best_score     = -1.0
    best_metrics: dict[str, float] = {}

    for p, r, thr in zip(precisions[:-1], recalls[:-1], thresholds):
        if metric == "f2":
            score = fbeta_score_from_pr(p, r, beta=2.0)
        elif metric == "f1":
            score = fbeta_score_from_pr(p, r, beta=1.0)
        elif metric == "f0_5":
            score = fbeta_score_from_pr(p, r, beta=0.5)
        elif metric == "recall":
            score = r
        elif metric == "precision":
            score = p
        else:
            score = fbeta_score_from_pr(p, r, beta=2.0)

        if score > best_score:
            best_score     = score
            best_threshold = float(thr)
            best_metrics   = danh_gia_du_doan(y_true, y_scores, threshold=best_threshold)

    if not best_metrics:
        best_metrics = danh_gia_du_doan(y_true, y_scores, threshold=best_threshold)

    return best_threshold, best_metrics


def fbeta_score_from_pr(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denom   = beta_sq * precision + recall
    return (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0


def danh_gia_du_doan(
    y_true: list[int],
    y_scores: list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    zero_div = 0.0

    has_pos = sum(y_true) > 0
    has_both_classes = len(set(y_true)) > 1
    has_pred_variance = len(set(y_pred)) > 1

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_div)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=zero_div)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=zero_div)),
        "f0_5":      float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=zero_div)),
        "f2":        float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=zero_div)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auc_roc":   float(roc_auc_score(y_true, y_scores)) if has_both_classes else 0.0,
        "auprc":     float(average_precision_score(y_true, y_scores)) if has_pos else 0.0,
        "mcc":       float(matthews_corrcoef(y_true, y_pred)) if (has_both_classes and has_pred_variance) else 0.0,
    }


def doc_file_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


# ===========================================================================
# TRAINING DATA MẶC ĐỊNH
# ===========================================================================

TRAINING_TEXTS = [
    "Doanh thu tăng nhờ hợp đồng hợp lệ và đã được đối chiếu với sao kê ngân hàng.",
    "Chi phí được ghi nhận chính xác và kiểm soát nội bộ vận hành hiệu quả.",
    "Tất cả giao dịch đều có xác nhận bên thứ ba và chứng từ đầy đủ.",
    "Kiểm kê hàng tồn kho khớp với sổ kho và dòng tiền từ hoạt động kinh doanh dương.",
    "Thuyết minh minh bạch, biên lợi nhuận gộp ổn định và không có giao dịch bất thường.",
    "Doanh thu bị thổi phồng từ hóa đơn giả và xác nhận khách hàng đáng ngờ.",
    "Doanh nghiệp che giấu nợ phải trả trong các thỏa thuận ngoài bảng cân đối với bên liên quan.",
    "Lợi nhuận bị thao túng để đạt chỉ tiêu và thay đổi chính sách kế toán không có giải trình.",
    "Dòng tiền từ hoạt động kinh doanh âm dù doanh thu tăng mạnh vào cuối kỳ.",
    "Phát hiện giao dịch bất thường với bên liên quan và thiếu phụ lục hợp đồng.",
    "Doanh thu được ghi nhận sớm kèm thỏa thuận ngầm và dấu hiệu hóa đơn giả.",
    "Ban lãnh đạo thay đổi kiểm toán viên liên tục và áp lực đạt KPI lợi nhuận rất cao.",
]

TRAINING_LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

TEST_CASES = [
    ("Doanh thu tăng hợp lệ nhờ hợp đồng đã ký và tiền thu về tốt từ khách hàng.", 0),
    ("Doanh thu khống từ hóa đơn giả được dùng để thổi phồng lợi nhuận và che giấu nợ phải trả.", 1),
    ("Doanh nghiệp thay đổi chính sách kế toán và có nhiều giao dịch với bên liên quan cần lưu ý.", 1),
    ("Thuyết minh rõ ràng, chi phí bình thường và số dư các tài khoản đã được xác minh.", 0),
]
