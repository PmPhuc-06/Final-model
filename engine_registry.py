from __future__ import annotations

from dataclasses import dataclass

from engine_auditbert import MoHinhGianLanAuditBERT
from engine_baseline import MoHinhGianLan
from engine_common import (
    AUDITBERT_CHECKPOINT,
    BASELINE_CHECKPOINT,
    MFINBERT_CHECKPOINT,
    PHOBERT_CHECKPOINT,
)
from engine_mfinbert import MoHinhGianLanMFinBERT
from engine_phobert import MoHinhGianLanPhoBERT


MODEL_BASELINE  = "baseline"
MODEL_PHOBERT   = "phobert"
MODEL_MFINBERT  = "mfinbert"
MODEL_AUDITBERT = "auditbert"


@dataclass(frozen=True)
class ModelRegistryEntry:
    key: str
    label: str
    description: str
    model_class: type
    is_transformer: bool = False
    checkpoint_path: str | None = None


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    MODEL_BASELINE: ModelRegistryEntry(
        key=MODEL_BASELINE,
        label="Baseline",
        description="TF-IDF + Logistic Regression (nhanh, giải thích được)",
        model_class=MoHinhGianLan,
        is_transformer=False,
        checkpoint_path=BASELINE_CHECKPOINT,
    ),
    MODEL_PHOBERT: ModelRegistryEntry(
        key=MODEL_PHOBERT,
        label="PhoBERT",
        description="PhoBERT fine-tune + hybrid metadata (tiếng Việt tốt)",
        model_class=MoHinhGianLanPhoBERT,
        is_transformer=True,
        checkpoint_path=PHOBERT_CHECKPOINT,
    ),
    MODEL_MFINBERT: ModelRegistryEntry(
        key=MODEL_MFINBERT,
        label="MFinBERT",
        description="MFinBERT fine-tune + hybrid metadata (tài chính tiếng Anh)",
        model_class=MoHinhGianLanMFinBERT,
        is_transformer=True,
        checkpoint_path=MFINBERT_CHECKPOINT,
    ),
    MODEL_AUDITBERT: ModelRegistryEntry(
        key=MODEL_AUDITBERT,
        label="AuditBERT-VN",
        description=(
            "Mô hình tổng hợp cuối cùng: PhoBERT backbone + "
            "Feature Fusion (Baseline signal + MFinBERT signal). "
            "Chỉ load 1 checkpoint, không cần chạy song song."
        ),
        model_class=MoHinhGianLanAuditBERT,
        is_transformer=True,
        checkpoint_path=AUDITBERT_CHECKPOINT,
    ),
}

MODEL_CHOICES = tuple(MODEL_REGISTRY.keys())
TRANSFORMER_MODEL_CHOICES = tuple(
    key for key, entry in MODEL_REGISTRY.items() if entry.is_transformer
)
MODEL_QUERY_DESCRIPTION = "Một trong: " + ", ".join(MODEL_CHOICES)


def lay_muc_registry(loai: str) -> ModelRegistryEntry:
    if loai not in MODEL_REGISTRY:
        raise KeyError(f"Model không được hỗ trợ: {loai}")
    return MODEL_REGISTRY[loai]


def lay_lop_mo_hinh(loai: str) -> type:
    return lay_muc_registry(loai).model_class


def tao_mo_hinh_theo_loai(loai: str):
    return lay_lop_mo_hinh(loai)()


def la_model_transformer(loai: str) -> bool:
    return lay_muc_registry(loai).is_transformer


def danh_sach_model_ho_tro() -> list[str]:
    return list(MODEL_CHOICES)

def tao_ket_qua_du_doan(
    loai: str,
    prediction,
    model_source: str = "unknown",
    top_terms_method: str = "unknown",
) -> dict:
    entry = lay_muc_registry(loai)
    return {
        "model": loai,
        "model_label": entry.label,
        "model_source": model_source,
        "top_terms_method": top_terms_method,
        "label": "Gian lận" if prediction.label == 1 else "Bình thường",
        "label_int": prediction.label,
        "muc_do_rui_ro": prediction.muc_do_rui_ro,
        "fraud_probability": round(prediction.probability_fraud, 4),
        "non_fraud_probability": round(prediction.probability_non_fraud, 4),
        "model_fraud_probability": round(prediction.model_probability_fraud, 4),
        "threshold_used": round(prediction.threshold_used, 4),
        "chat_luong_van_ban": round(prediction.chat_luong_van_ban, 4),
        "red_flags": prediction.red_flags,
        "explanation": prediction.explanation,
        "top_terms": prediction.top_terms,
        "van_ban_sach": prediction.van_ban_sach,
        "doan_nghi_ngo": prediction.doan_nghi_ngo,
    }