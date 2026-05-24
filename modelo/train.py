"""
Treina 3 modelos diferentes para classificação multi-classe da demanda
odontológica (Baixa / Media / Alta), seleciona o melhor pelo f1_macro
em validação cruzada estratificada (5 folds), e serializa o pipeline
vencedor em ``modelo/modelo_demanda.joblib``.

Também grava ``modelo/metrics.json`` com as métricas no conjunto de teste,
o nome do modelo escolhido, top-5 feature importances e a data do treino.

Uso:
    python modelo/train.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore

    XGB_OK = True
except Exception:  # pragma: no cover - fallback gracioso
    XGB_OK = False
    from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "demanda_bairros.csv"
MODELO_PATH = ROOT / "modelo" / "modelo_demanda.joblib"
METRICS_PATH = ROOT / "modelo" / "metrics.json"

FEATURES = [
    "densidade_pop",
    "idh",
    "dist_ubs_km",
    "historico_atendimentos",
    "pct_exames_pendentes",
    "populacao_infantil_pct",
    "dia_semana_cadastro",  # decoy: o modelo deve aprender a ignorar
]
TARGET = "demanda"

SEED = 42


def carregar_dados() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV nao encontrado em {CSV_PATH}. "
            "Rode antes: python data/gerar_dataset.py"
        )
    return pd.read_csv(CSV_PATH)


def construir_modelos() -> dict[str, Pipeline]:
    """Constrói os 3 pipelines candidatos."""
    rf = Pipeline(
        steps=[
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=SEED,
                n_jobs=-1,
            )),
        ]
    )

    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                random_state=SEED,
            )),
        ]
    )

    if XGB_OK:
        boost = Pipeline(
            steps=[
                ("clf", XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.08,
                    max_depth=5,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=SEED,
                    n_jobs=-1,
                )),
            ]
        )
        return {"random_forest": rf, "xgboost": boost, "logistic_regression": logreg}

    gboost = Pipeline(
        steps=[
            ("clf", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=4,
                random_state=SEED,
            )),
        ]
    )
    return {"random_forest": rf, "gradient_boosting": gboost, "logistic_regression": logreg}


def main() -> None:
    print(f"[train] lendo {CSV_PATH.relative_to(ROOT)}")
    df = carregar_dados()
    print(f"[train] dataset: {len(df)} linhas, {df['bairro'].nunique()} bairros")

    X = df[FEATURES].copy()
    y_raw = df[TARGET].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"[train] classes: {list(le.classes_)}")

    # split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    modelos = construir_modelos()

    resultados: dict[str, dict] = {}
    melhor_nome = None
    melhor_f1 = -1.0

    for nome, pipe in modelos.items():
        print(f"\n[cv] {nome}: rodando 5-fold stratified...")
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        media = float(np.mean(scores))
        desvio = float(np.std(scores))
        resultados[nome] = {"f1_macro_cv_mean": media, "f1_macro_cv_std": desvio}
        print(f"     f1_macro CV = {media:.4f} (+/- {desvio:.4f})")
        if media > melhor_f1:
            melhor_f1 = media
            melhor_nome = nome

    assert melhor_nome is not None
    print(f"\n[selecao] vencedor (f1_macro CV): {melhor_nome} = {melhor_f1:.4f}")

    pipe_vencedor: Pipeline = modelos[melhor_nome]
    pipe_vencedor.fit(X_train, y_train)

    y_pred = pipe_vencedor.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred).tolist()
    print(f"\n[teste] accuracy  = {acc:.4f}")
    print(f"[teste] f1_macro  = {f1m:.4f}")
    print("[teste] classification_report:")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

    # feature importances (top 5) quando disponível
    feat_importances: list[dict] = []
    try:
        clf = pipe_vencedor.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importancias = clf.feature_importances_
            pares = sorted(
                zip(FEATURES, importancias),
                key=lambda kv: float(kv[1]),
                reverse=True,
            )
            feat_importances = [
                {"feature": f, "importance": float(v)} for f, v in pares[:5]
            ]
        elif hasattr(clf, "coef_"):
            coefs = np.mean(np.abs(clf.coef_), axis=0)
            pares = sorted(
                zip(FEATURES, coefs),
                key=lambda kv: float(kv[1]),
                reverse=True,
            )
            feat_importances = [
                {"feature": f, "importance": float(v)} for f, v in pares[:5]
            ]
    except Exception as exc:  # pragma: no cover
        print(f"[warn] nao foi possivel extrair feature importances: {exc}")

    # serializa pipeline + label encoder juntos
    artefato = {
        "pipeline": pipe_vencedor,
        "label_encoder": le,
        "features": FEATURES,
        "modelo_nome": melhor_nome,
        "versao": f"{melhor_nome}-v1",
    }
    MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, MODELO_PATH)
    print(f"\n[save] modelo salvo em {MODELO_PATH.relative_to(ROOT)}")

    metrics = {
        "modelo_escolhido": melhor_nome,
        "versao": f"{melhor_nome}-v1",
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_macro_cv": resultados,
        "confusion_matrix": cm,
        "classes": list(le.classes_),
        "feature_importances_top5": feat_importances,
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
        "features": FEATURES,
        "xgboost_disponivel": XGB_OK,
        "data_treino": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[save] metricas salvas em {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
