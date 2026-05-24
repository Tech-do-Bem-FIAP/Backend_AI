"""
Validação manual dos inputs da API.

Não usamos pydantic para manter o projeto enxuto (3.9-compatível). As
funções abaixo retornam ``(ok, payload_ou_erro)``.
"""
from __future__ import annotations

from typing import Any

FEATURE_KEYS: tuple[str, ...] = (
    "densidade_pop",
    "idh",
    "dist_ubs_km",
    "historico_atendimentos",
    "pct_exames_pendentes",
    "populacao_infantil_pct",
    "dia_semana_cadastro",
)

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "densidade_pop": (0.0, 50_000.0),
    "idh": (0.0, 1.0),
    "dist_ubs_km": (0.0, 50.0),
    "historico_atendimentos": (0.0, 1_000.0),
    "pct_exames_pendentes": (0.0, 1.0),
    "populacao_infantil_pct": (0.0, 1.0),
    "dia_semana_cadastro": (0.0, 6.0),
}


def _erro(msg: str) -> tuple[bool, str]:
    return False, msg


def detectar_modo(data: Any) -> tuple[str, str]:
    """Identifica se o input é via 'bairro' ou via 'features'.

    Retorna ``(modo, erro)`` onde ``modo`` é 'bairro' | 'features' | ''.
    """
    if not isinstance(data, dict):
        return "", "payload deve ser um objeto JSON"
    if "bairro" in data:
        return "bairro", ""
    # se vier ao menos uma das features esperadas, tratamos como modo features
    if any(k in data for k in FEATURE_KEYS):
        return "features", ""
    return "", "payload deve conter 'bairro' ou as features numericas"


def validate_bairro_input(data: dict) -> tuple[bool, Any]:
    bairro = data.get("bairro")
    if not isinstance(bairro, str) or not bairro.strip():
        return _erro("campo 'bairro' deve ser uma string nao vazia")
    return True, bairro.strip()


def validate_features_input(data: dict) -> tuple[bool, Any]:
    """Valida e normaliza o payload do modo 'features'.

    ``dia_semana_cadastro`` é opcional; se ausente, usa 0.
    """
    if not isinstance(data, dict):
        return _erro("payload deve ser um objeto JSON")

    obrigatorias = [k for k in FEATURE_KEYS if k != "dia_semana_cadastro"]
    faltando = [k for k in obrigatorias if k not in data]
    if faltando:
        return _erro(f"features faltando: {faltando}")

    normalizado: dict[str, float] = {}
    for k in FEATURE_KEYS:
        v = data.get(k, 0 if k == "dia_semana_cadastro" else None)
        if v is None:
            return _erro(f"feature '{k}' eh obrigatoria")
        try:
            f = float(v)
        except (TypeError, ValueError):
            return _erro(f"feature '{k}' deve ser numerica")
        lo, hi = FEATURE_RANGES[k]
        if not (lo <= f <= hi):
            return _erro(f"feature '{k}'={f} fora do range esperado [{lo}, {hi}]")
        normalizado[k] = f

    return True, normalizado
