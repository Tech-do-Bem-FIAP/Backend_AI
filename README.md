# Tech do Bem — Backend AI (Sprint 4)

Modelo de **classificação multi-classe** que prediz a demanda odontológica (`Baixa` / `Media` / `Alta`) por bairro de São Paulo, com **API Flask** servindo o modelo serializado em `.joblib`. Entrega da disciplina **Artificial Intelligence e Chatbot** — 1TDSPR / FIAP.

## Integrantes

| RM       | Nome                          |
|----------|-------------------------------|
| RM568542 | Hugo Souza de Jesus           |
| RM566815 | Lucas Campanhã dos Santos     |
| RM567010 | Lucas Marcelino Pompeu        |

## Conteúdo do repositório

| Caminho | Descrição |
|---|---|
| `data/gerar_dataset.py` | Gerador reproduzível (seed=42) do dataset sintético |
| `data/demanda_bairros.csv` | ~2050 linhas, 82 bairros reais de SP, label `Baixa/Media/Alta` |
| `notebooks/01_pipeline_demanda.ipynb` | Pipeline completa + EDA com 5+ gráficos e matriz de confusão |
| `modelo/train.py` | Treina os 3 modelos, seleciona vencedor por `f1_macro` 5-fold CV |
| `modelo/modelo_demanda.joblib` | Pipeline vencedor + LabelEncoder serializados (joblib) |
| `modelo/metrics.json` | Métricas no teste, comparação CV, top-5 feature importances |
| `api/app.py` | API Flask: `/health`, `/predict`, `/predict/batch` |
| `api/schemas.py` | Validação manual dos payloads |
| `Dockerfile` | Container com gunicorn (2 workers) na porta 5001 |
| `requirements.txt` | scikit-learn, xgboost, flask, flask-cors, joblib, pandas, numpy |
| `DOCUMENTACAO.md` | Documentação atendendo à rubrica da Sprint 4 |
| `README.md` | Este arquivo |

## Sobre o projeto

A **Tech do Bem** é uma ONG que oferece atendimento odontológico gratuito a populações vulneráveis. Este módulo prevê a **demanda odontológica** por bairro a partir de indicadores socioeconômicos, e é consumido pelo frontend para apresentar um mapa de calor priorizado e uma página dinâmica por bairro (`/colaborador/area/:bairro`).

O sistema completo é composto por quatro repositórios:

- [`Tech-do-Bem-FIAP/Backend_AI`](https://github.com/Tech-do-Bem-FIAP/Backend_AI) — este repositório (API Python + modelo)
- [`Tech-do-Bem-FIAP/Backend_Java`](https://github.com/Tech-do-Bem-FIAP/Backend_Java) — API Java Quarkus (CRUD do domínio)
- [`Tech-do-Bem-FIAP/Frontend`](https://github.com/Tech-do-Bem-FIAP/Frontend) — SPA React + Vite + TypeScript
- [`Tech-do-Bem-FIAP/Database`](https://github.com/Tech-do-Bem-FIAP/Database) — schema e seed Oracle

## Como executar

### Local (Python 3.12 recomendado)

```bash
git clone https://github.com/Tech-do-Bem-FIAP/Backend_AI.git
cd Backend_AI

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (opcional) regerar dataset — já vem commitado
python data/gerar_dataset.py

# treinar e gerar .joblib + metrics.json
python modelo/train.py

# subir a API em http://localhost:5001
python -m flask --app api.app run --port 5001
```

> **macOS:** a porta `5000` é usada pelo AirPlay Receiver — por isso usamos `5001`.

### Docker

```bash
docker build -t backend-ai .
docker run -p 5001:5001 backend-ai
curl http://localhost:5001/health
```

### Teste rápido

```bash
curl http://localhost:5001/health
curl -X POST http://localhost:5001/predict \
     -H 'Content-Type: application/json' \
     -d '{"bairro":"Brás"}'
```

## Atendimento à rubrica Sprint 4 (Artificial Intelligence e Chatbot)

| Critério                                            | Pontuação | Localização                                |
|-----------------------------------------------------|-----------|---------------------------------------------|
| Pipeline e seleção de modelo (.ipynb)               | 40 pts    | `notebooks/01_pipeline_demanda.ipynb` + `modelo/train.py` |
| Serialização (.joblib) e deploy (.py)               | 40 pts    | `modelo/modelo_demanda.joblib` + `api/app.py` |
| Documento com demonstrações (.pdf)                  | 20 pts    | `DOCUMENTACAO.md` (exportável via Print → PDF) |

Detalhes de cada item em [`DOCUMENTACAO.md`](DOCUMENTACAO.md).
