"""
Gerador de dataset sintético de demanda odontológica por bairro de São Paulo.

Cria ~2000 linhas em data/demanda_bairros.csv para servir de base ao
treinamento do classificador multi-classe (Baixa/Media/Alta).

Reprodutível: usa numpy seed fixa = 42.

Uso:
    python data/gerar_dataset.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
LINHAS_POR_BAIRRO = 25

# --- 80 bairros reais de São Paulo, com coordenadas aproximadas e perfil socioeconômico ---
# tier: 1 = mais nobre (IDH alto), 3 = mais vulnerável (IDH baixo)
BAIRROS = [
    # nome, lat, lon, tier
    ("Brás",                 -23.5419, -46.6155, 3),
    ("Pinheiros",            -23.5670, -46.6920, 1),
    ("Itaim Bibi",           -23.5847, -46.6770, 1),
    ("Bela Vista",           -23.5616, -46.6489, 2),
    ("Capão Redondo",        -23.6657, -46.7805, 3),
    ("Cidade Tiradentes",    -23.5872, -46.4022, 3),
    ("Vila Mariana",         -23.5896, -46.6347, 1),
    ("Mooca",                -23.5519, -46.5970, 2),
    ("Tatuapé",              -23.5380, -46.5765, 2),
    ("Lapa",                 -23.5266, -46.7050, 2),
    ("Santana",              -23.5018, -46.6285, 2),
    ("Vila Madalena",        -23.5544, -46.6920, 1),
    ("Perdizes",             -23.5378, -46.6750, 1),
    ("Higienópolis",         -23.5470, -46.6580, 1),
    ("Liberdade",            -23.5587, -46.6360, 2),
    ("Aclimação",            -23.5710, -46.6280, 2),
    ("Saúde",                -23.6166, -46.6386, 2),
    ("Vila Olímpia",         -23.5950, -46.6870, 1),
    ("Morumbi",              -23.6020, -46.7220, 1),
    ("Butantã",              -23.5710, -46.7200, 2),
    ("Vila Sônia",           -23.5970, -46.7320, 2),
    ("Cambuci",              -23.5680, -46.6230, 2),
    ("Bom Retiro",           -23.5260, -46.6360, 3),
    ("Jardim Paulista",      -23.5680, -46.6700, 1),
    ("Jabaquara",            -23.6470, -46.6440, 2),
    ("Sacomã",               -23.6125, -46.6055, 3),
    ("Ipiranga",             -23.5946, -46.5980, 2),
    ("Vila Prudente",        -23.5810, -46.5805, 2),
    ("Itaquera",             -23.5400, -46.4570, 3),
    ("São Mateus",           -23.6010, -46.4810, 3),
    ("Guaianases",           -23.5430, -46.4170, 3),
    ("Ermelino Matarazzo",   -23.5000, -46.4830, 3),
    ("São Miguel Paulista",  -23.4980, -46.4430, 3),
    ("Vila Curuçá",          -23.5130, -46.4170, 3),
    ("Lajeado",              -23.5430, -46.3990, 3),
    ("Iguatemi",             -23.5840, -46.4380, 3),
    ("Cidade Líder",         -23.5570, -46.4630, 3),
    ("Penha",                -23.5260, -46.5430, 2),
    ("Cangaíba",             -23.5170, -46.5170, 3),
    ("Aricanduva",           -23.5540, -46.5320, 2),
    ("Vila Formosa",         -23.5680, -46.5460, 2),
    ("Belém",                -23.5400, -46.6020, 2),
    ("Pari",                 -23.5260, -46.6150, 3),
    ("Brasilândia",          -23.4660, -46.6890, 3),
    ("Cachoeirinha",         -23.4750, -46.6580, 3),
    ("Casa Verde",           -23.5060, -46.6660, 2),
    ("Vila Maria",           -23.5030, -46.5860, 2),
    ("Vila Guilherme",       -23.5170, -46.5970, 2),
    ("Tucuruvi",             -23.4790, -46.6020, 2),
    ("Mandaqui",             -23.4830, -46.6310, 2),
    ("Tremembé",             -23.4640, -46.6090, 2),
    ("Jaçanã",               -23.4640, -46.5740, 3),
    ("Jardim Helena",        -23.4830, -46.4310, 3),
    ("Vila Medeiros",        -23.4940, -46.5780, 3),
    ("Vila Andrade",         -23.6320, -46.7270, 2),
    ("Campo Limpo",          -23.6500, -46.7570, 3),
    ("Jardim Ângela",        -23.6940, -46.7670, 3),
    ("Jardim São Luís",      -23.6700, -46.7400, 3),
    ("Cidade Ademar",        -23.6620, -46.6580, 3),
    ("Pedreira",             -23.6940, -46.6680, 3),
    ("Cidade Dutra",         -23.7150, -46.7030, 3),
    ("Socorro",              -23.6970, -46.7150, 2),
    ("Marsilac",             -23.9020, -46.7080, 3),
    ("Parelheiros",          -23.8270, -46.7370, 3),
    ("Grajaú",               -23.7770, -46.7060, 3),
    ("Anhanguera",           -23.4400, -46.7900, 3),
    ("Perus",                -23.4040, -46.7470, 3),
    ("Pirituba",             -23.4790, -46.7220, 2),
    ("Jaraguá",              -23.4570, -46.7480, 3),
    ("Freguesia do Ó",       -23.4980, -46.7000, 2),
    ("Limão",                -23.5010, -46.6650, 2),
    ("Raposo Tavares",       -23.5870, -46.7570, 2),
    ("Rio Pequeno",          -23.5750, -46.7470, 2),
    ("Jardim Bonfiglioli",   -23.5800, -46.7240, 2),
    ("Sé",                   -23.5500, -46.6340, 2),
    ("República",            -23.5430, -46.6420, 2),
    ("Consolação",           -23.5530, -46.6610, 1),
    ("Vila Buarque",         -23.5430, -46.6480, 2),
    ("Campo Belo",           -23.6210, -46.6700, 1),
    ("Moema",                -23.6010, -46.6650, 1),
    ("Chácara Klabin",       -23.6040, -46.6360, 2),
    ("Vila Leopoldina",      -23.5350, -46.7340, 1),
]


def _idh_base_por_tier(tier: int, rng: np.random.Generator) -> float:
    """Sorteia IDH base do bairro de acordo com tier socioeconômico."""
    if tier == 1:
        return float(rng.uniform(0.85, 0.95))
    if tier == 2:
        return float(rng.uniform(0.75, 0.87))
    return float(rng.uniform(0.65, 0.78))


def _densidade_base_por_tier(tier: int, rng: np.random.Generator) -> float:
    if tier == 1:
        return float(rng.uniform(8_000, 14_000))
    if tier == 2:
        return float(rng.uniform(10_000, 18_000))
    return float(rng.uniform(12_000, 25_000))


def _dist_ubs_base_por_tier(tier: int, rng: np.random.Generator) -> float:
    # bairros mais vulneráveis tendem a ter UBSs um pouco mais distantes
    if tier == 1:
        return float(rng.exponential(0.8))
    if tier == 2:
        return float(rng.exponential(1.4))
    return float(rng.exponential(2.4))


def gerar() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    registros: list[dict] = []
    for nome, lat, lon, tier in BAIRROS:
        idh_base = _idh_base_por_tier(tier, rng)
        densidade_base = _densidade_base_por_tier(tier, rng)
        dist_base = _dist_ubs_base_por_tier(tier, rng)

        for _ in range(LINHAS_POR_BAIRRO):
            idh = float(np.clip(idh_base + rng.normal(0, 0.02), 0.60, 0.98))
            densidade_pop = float(np.clip(densidade_base + rng.normal(0, 1_500), 2_000, 30_000))
            dist_ubs_km = float(np.clip(dist_base + abs(rng.normal(0, 0.3)), 0.05, 12.0))
            historico_atendimentos = int(rng.poisson(10))
            pct_exames_pendentes = float(np.clip(rng.beta(2, 5), 0.0, 1.0))
            populacao_infantil_pct = float(np.clip(rng.normal(0.22, 0.05), 0.05, 0.45))
            dia_semana_cadastro = int(rng.integers(0, 7))  # decoy
            jitter_lat = float(rng.normal(0, 0.005))
            jitter_lon = float(rng.normal(0, 0.005))

            registros.append(
                {
                    "bairro": nome,
                    "densidade_pop": round(densidade_pop, 2),
                    "idh": round(idh, 4),
                    "dist_ubs_km": round(dist_ubs_km, 3),
                    "historico_atendimentos": historico_atendimentos,
                    "pct_exames_pendentes": round(pct_exames_pendentes, 4),
                    "populacao_infantil_pct": round(populacao_infantil_pct, 4),
                    "dia_semana_cadastro": dia_semana_cadastro,
                    "latitude": round(lat + jitter_lat, 6),
                    "longitude": round(lon + jitter_lon, 6),
                }
            )

    df = pd.DataFrame(registros)

    # --- Score latente para o label (com ruído real, não determinístico) ---
    def _norm(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    idh_norm = _norm(df["idh"])
    dist_norm = _norm(df["dist_ubs_km"])
    pct_norm = _norm(df["pct_exames_pendentes"])
    dens_norm = _norm(df["densidade_pop"])

    ruido = rng.normal(0, 0.12, size=len(df))

    score = (
        0.40 * (1 - idh_norm)
        + 0.25 * dist_norm
        + 0.20 * pct_norm
        + 0.15 * dens_norm
        + ruido
    )

    df["demanda"] = pd.qcut(score, q=3, labels=["Baixa", "Media", "Alta"]).astype(str)

    return df


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "demanda_bairros.csv"

    df = gerar()
    df.to_csv(out, index=False)

    print(f"OK -> {out}")
    print(f"linhas: {len(df)} | bairros distintos: {df['bairro'].nunique()}")
    print("distribuicao do label:")
    print(df["demanda"].value_counts().to_string())


if __name__ == "__main__":
    main()
