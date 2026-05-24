"""
API Flask do Backend_AI.

Endpoints
---------
- ``GET  /health``         -> health check
- ``POST /predict``        -> aceita modo 'features' (7 numericas) ou modo
                              'bairro' (lookup interno no CSV)
- ``POST /predict/batch``  -> array de inputs nos mesmos formatos
- ``GET  /ranking``        -> lista todos os bairros conhecidos com a classe
                              predita pelo modelo, ordenados por prioridade

CORS liberado para o frontend local e os deploys publicos.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from api.schemas import (
    FEATURE_KEYS,
    detectar_modo,
    validate_bairro_input,
    validate_features_input,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backend_ai")

ROOT = Path(__file__).resolve().parent.parent
MODELO_PATH = ROOT / "modelo" / "modelo_demanda.joblib"
CSV_PATH = ROOT / "data" / "demanda_bairros.csv"

CORS_ORIGINS = [
    "http://localhost:5173",
    "https://techdobem.labs-lcs.com",
    "https://techdobem.vercel.app",
]

# -------------------------------------------------------------------- carga ---


def _carregar_modelo() -> dict[str, Any]:
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {MODELO_PATH}. "
            "Rode antes: python modelo/train.py"
        )
    artefato = joblib.load(MODELO_PATH)
    log.info("modelo carregado: %s", artefato.get("versao", "?"))
    return artefato


def _carregar_lookup() -> dict[str, dict[str, float]]:
    """Agrega features medias por bairro a partir do CSV."""
    if not CSV_PATH.exists():
        log.warning("CSV de bairros nao encontrado em %s", CSV_PATH)
        return {}
    df = pd.read_csv(CSV_PATH)
    agrupado = df.groupby("bairro")[list(FEATURE_KEYS)].mean()
    tabela = {nome: row.to_dict() for nome, row in agrupado.iterrows()}
    log.info("lookup carregado: %d bairros", len(tabela))
    return tabela


def _carregar_coords_bairros() -> dict[str, dict[str, float]]:
    """Coords medias (lat/lng) por bairro, lidas do CSV."""
    if not CSV_PATH.exists():
        return {}
    df = pd.read_csv(CSV_PATH)
    agrupado = df.groupby("bairro")[["latitude", "longitude"]].mean()
    return {nome: {"lat": float(row["latitude"]), "lng": float(row["longitude"])}
            for nome, row in agrupado.iterrows()}


MODELO = _carregar_modelo()
PIPELINE = MODELO["pipeline"]
LABEL_ENCODER = MODELO["label_encoder"]
VERSAO = MODELO["versao"]
LOOKUP_BAIRROS = _carregar_lookup()
COORDS_BAIRROS = _carregar_coords_bairros()

# -------------------------------------------------------------------- app -----

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})


def _predizer(features: dict[str, float]) -> dict[str, Any]:
    """Roda o pipeline carregado e retorna classe + probabilidades."""
    X = pd.DataFrame([[features[k] for k in FEATURE_KEYS]], columns=list(FEATURE_KEYS))
    pred = PIPELINE.predict(X)[0]
    classe = str(LABEL_ENCODER.inverse_transform([pred])[0])

    proba: dict[str, float] = {}
    if hasattr(PIPELINE, "predict_proba"):
        probs = PIPELINE.predict_proba(X)[0]
        for cls, p in zip(LABEL_ENCODER.classes_, probs):
            proba[str(cls)] = float(round(float(p), 4))

    return {
        "classe": classe,
        "probabilidades": proba,
        "modelo_versao": VERSAO,
    }


def _resolver_input(payload: Any) -> tuple[bool, Any, str | None]:
    """Resolve um payload individual para um dict de features.

    Retorna ``(ok, features_ou_erro, bairro_resolvido)``.
    """
    modo, erro_modo = detectar_modo(payload)
    if not modo:
        return False, erro_modo, None

    if modo == "bairro":
        ok, valor = validate_bairro_input(payload)
        if not ok:
            return False, valor, None
        nome = valor
        if nome not in LOOKUP_BAIRROS:
            return False, f"bairro '{nome}' nao encontrado", None
        feats = {k: float(LOOKUP_BAIRROS[nome][k]) for k in FEATURE_KEYS}
        return True, feats, nome

    # modo features
    ok, valor = validate_features_input(payload)
    if not ok:
        return False, valor, None
    return True, valor, None


# -------------------------------------------------------------------- routes --


@app.get("/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "modelo_carregado": True,
            "versao": VERSAO,
            "bairros_disponiveis": len(LOOKUP_BAIRROS),
        }
    )


@app.post("/predict")
def predict() -> Any:
    try:
        payload = request.get_json(silent=True)
    except Exception:
        return jsonify({"erro": "JSON malformado"}), 400
    if payload is None:
        return jsonify({"erro": "JSON malformado ou ausente"}), 400

    ok, valor, bairro = _resolver_input(payload)
    if not ok:
        # 404 quando o erro for de bairro inexistente
        if isinstance(valor, str) and valor.startswith("bairro '") and "nao encontrado" in valor:
            return jsonify({"erro": valor}), 404
        return jsonify({"erro": valor}), 400

    try:
        resp = _predizer(valor)
    except Exception as exc:  # pragma: no cover
        log.exception("erro inferindo")
        return jsonify({"erro": f"falha na predicao: {exc}"}), 500

    if bairro is not None:
        resp["bairro"] = bairro
    return jsonify(resp)


@app.post("/predict/batch")
def predict_batch() -> Any:
    try:
        payload = request.get_json(silent=True)
    except Exception:
        return jsonify({"erro": "JSON malformado"}), 400
    if not isinstance(payload, list):
        return jsonify({"erro": "payload deve ser um array JSON"}), 400

    saidas: list[dict[str, Any]] = []
    for i, item in enumerate(payload):
        ok, valor, bairro = _resolver_input(item)
        if not ok:
            saidas.append({"index": i, "erro": valor})
            continue
        try:
            resp = _predizer(valor)
            if bairro is not None:
                resp["bairro"] = bairro
            resp["index"] = i
            saidas.append(resp)
        except Exception as exc:  # pragma: no cover
            saidas.append({"index": i, "erro": f"falha na predicao: {exc}"})

    return jsonify(saidas)


@app.get("/ranking")
def ranking() -> Any:
    """Lista todos os bairros conhecidos com a predicao do modelo.

    Ordenado por prioridade decrescente: bairros classificados como "Alta"
    aparecem primeiro, depois "Media", depois "Baixa". Dentro de cada classe,
    ordena pela probabilidade da classe predita (maior confianca primeiro).
    """
    ordem_classe = {"Alta": 0, "Media": 1, "Baixa": 2}
    itens: list[dict[str, Any]] = []
    for nome, features in LOOKUP_BAIRROS.items():
        try:
            resp = _predizer({k: float(features[k]) for k in FEATURE_KEYS})
        except Exception:
            log.exception("falha no ranking pro bairro %s", nome)
            continue
        coords = COORDS_BAIRROS.get(nome, {"lat": 0.0, "lng": 0.0})
        itens.append({
            "bairro": nome,
            "classe": resp["classe"],
            "probabilidades": resp["probabilidades"],
            "coords": coords,
        })

    itens.sort(key=lambda x: (
        ordem_classe.get(x["classe"], 99),
        -x["probabilidades"].get(x["classe"], 0.0),
    ))

    return jsonify({
        "modelo_versao": VERSAO,
        "total": len(itens),
        "itens": itens,
    })


@app.errorhandler(404)
def _not_found(_e: Any) -> Any:
    return jsonify({"erro": "rota nao encontrada"}), 404


@app.errorhandler(405)
def _bad_method(_e: Any) -> Any:
    return jsonify({"erro": "metodo nao permitido"}), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
