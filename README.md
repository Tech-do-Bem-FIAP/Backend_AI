# Backend_AI — Tech do Bem (FIAP, Sprint 4 IA)

Modelo de **classificação multi-classe** que prediz a demanda
odontológica (`Baixa` / `Media` / `Alta`) por bairro de São Paulo, com
**API Flask** servindo o modelo serializado em `.joblib`.

Este é o módulo de IA do ecossistema **Tech do Bem** — plataforma
acadêmica de gestão odontológica desenvolvida na disciplina de IA da
FIAP (turma **1TDSPR**).

- Backend Java (regra de negócio): https://github.com/Tech-do-Bem-FIAP/Backend_Java
- Frontend React: https://github.com/Tech-do-Bem-FIAP/Frontend

---

## Autores

| Nome | RM | Turma |
| --- | --- | --- |
| Lucas Campanhã dos Santos | 566815 | 1TDSPR |

---

## Estrutura

```
Backend_AI/
├── api/
│   ├── __init__.py
│   ├── app.py                # Flask: /health, /predict, /predict/batch
│   └── schemas.py            # validacao manual dos payloads
├── data/
│   ├── gerar_dataset.py      # gerador reproduzivel (seed=42)
│   └── demanda_bairros.csv   # ~2050 linhas, 82 bairros reais de SP
├── docs/
│   └── demonstracao_sprint4.md
├── modelo/
│   ├── train.py              # treina os 3 modelos, escolhe vencedor
│   ├── modelo_demanda.joblib # pipeline + LabelEncoder serializados
│   └── metrics.json          # accuracy, f1_macro, top-5 importances
├── notebooks/
│   └── 01_pipeline_demanda.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## O que o modelo faz

Dado um conjunto de indicadores socioeconômicos de um bairro, classifica
a demanda odontológica em **Baixa**, **Media** ou **Alta**.

### Features (7 numéricas)

| Feature | Descrição |
| --- | --- |
| `densidade_pop` | densidade populacional (hab/km²) |
| `idh` | IDH do bairro (0–1) |
| `dist_ubs_km` | distância média até a UBS mais próxima (km) |
| `historico_atendimentos` | nº de atendimentos no histórico |
| `pct_exames_pendentes` | % de exames pendentes (0–1) |
| `populacao_infantil_pct` | % da população com idade ≤ 14 anos (0–1) |
| `dia_semana_cadastro` | 0–6 — **decoy** (irrelevante, para validar feature importance) |

### Como o label é gerado (dataset sintético)

Score latente com ruído gaussiano:

```
s = 0.40 * (1 - idh_norm)
  + 0.25 * dist_norm
  + 0.20 * pct_exames_norm
  + 0.15 * densidade_norm
  + N(0, 0.12)
```

depois discretizado por **quantis (terços)** → `Baixa / Media / Alta`.
O ruído impede acurácia perfeita, fornecendo uma baseline realista.

---

## Modelos comparados

Treinamento + comparação por **validação cruzada estratificada 5-fold**
no `f1_macro`:

1. `RandomForestClassifier` (sklearn) — baseline robusto, sem normalização
2. `XGBClassifier` (xgboost) — referência tabular forte
3. `LogisticRegression` em `Pipeline` com `StandardScaler` — controle linear

> Se a instalação do `xgboost` falhar no ambiente, o `train.py` usa
> automaticamente `GradientBoostingClassifier` do sklearn no lugar.

A escolha do vencedor e o salvamento do pipeline completo (incluindo
qualquer pré-processador) acontece no final do treino.

---

## Métricas alcançadas no último treino

(arquivo fonte: `modelo/metrics.json`)

| Métrica | Valor |
| --- | --- |
| Modelo escolhido | `logistic_regression` |
| Accuracy (teste) | **0.639** |
| F1-macro (teste) | **0.631** |
| F1-macro (CV 5-fold) | 0.587 |
| Linhas de treino / teste | 1640 / 410 |

**Top-5 feature importances**

1. `idh`
2. `pct_exames_pendentes`
3. `dist_ubs_km`
4. `densidade_pop`
5. `historico_atendimentos`

A feature `dia_semana_cadastro` (decoy) ficou fora do top 5 — o modelo
aprendeu o sinal correto.

---

## Execução LOCAL

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) (re)gerar o dataset (opcional, ja vem commitado)
python data/gerar_dataset.py

# 2) treinar e salvar .joblib + metrics.json
python modelo/train.py

# 3) subir a API em http://localhost:5001
python -m flask --app api.app run --port 5001 --debug
```

> **Atenção macOS:** a porta `5000` é usada pelo AirPlay Receiver. Por
> isso fixamos `5001`.

> **Compatibilidade Python:** o `requirements.txt` está pinado para
> **Python 3.12** (ambiente alvo do Docker). Em ambientes mais antigos
> (ex.: Python 3.9 do CommandLineTools do macOS) instale versões mais
> conservadoras: `numpy<2 pandas<2.2 scikit-learn<1.5 xgboost<2.1
> flask>=3.0 flask-cors>=5 joblib>=1.3`.

---

## Execução via DOCKER

```bash
docker build -t backend-ai .
docker run -p 5001:5001 backend-ai

curl http://localhost:5001/health
```

A imagem usa `gunicorn -w 2` para servir `api.app:app`.

---

## Endpoints

### `GET /health`

```bash
curl http://localhost:5001/health
```

```json
{
  "status": "ok",
  "modelo_carregado": true,
  "versao": "logistic_regression-v1",
  "bairros_disponiveis": 82
}
```

### `POST /predict` — modo "features completas"

```bash
curl -X POST http://localhost:5001/predict \
     -H 'Content-Type: application/json' \
     -d '{
           "densidade_pop": 12000,
           "idh": 0.78,
           "dist_ubs_km": 1.5,
           "historico_atendimentos": 45,
           "pct_exames_pendentes": 0.3,
           "populacao_infantil_pct": 0.22,
           "dia_semana_cadastro": 3
         }'
```

```json
{
  "classe": "Baixa",
  "modelo_versao": "logistic_regression-v1",
  "probabilidades": {"Alta": 0.0686, "Baixa": 0.708, "Media": 0.2235}
}
```

### `POST /predict` — modo "lookup por bairro"

A API agrega as features médias do bairro a partir do CSV e roda a
predição. Retorna **404** se o bairro não existir.

```bash
curl -X POST http://localhost:5001/predict \
     -H 'Content-Type: application/json' \
     -d '{"bairro":"Brás"}'
```

```json
{
  "bairro": "Brás",
  "classe": "Alta",
  "modelo_versao": "logistic_regression-v1",
  "probabilidades": {"Alta": 0.6667, "Baixa": 0.0553, "Media": 0.278}
}
```

### `POST /predict/batch`

Recebe um array misto dos dois formatos acima e devolve um array de
respostas (cada item recebe um `index` correspondente ao input):

```bash
curl -X POST http://localhost:5001/predict/batch \
     -H 'Content-Type: application/json' \
     -d '[{"bairro":"Pinheiros"},{"bairro":"Capão Redondo"}]'
```

### Códigos de erro

| Status | Quando |
| --- | --- |
| 400 | JSON ausente / malformado / features fora do range |
| 404 | bairro não encontrado em `data/demanda_bairros.csv` |
| 405 | método HTTP errado |
| 500 | falha inesperada na predição |

### CORS

A whitelist de origens é:

- `http://localhost:5173` (frontend Vite local)
- `https://techdobem.labs-lcs.com` (deploy próprio)
- `https://techdobem.vercel.app` (deploy Vercel)

---

## Reproduzir o dataset

```bash
python data/gerar_dataset.py
```

Saída: `data/demanda_bairros.csv` (~2050 linhas, 82 bairros, label
balanceado).

## Reproduzir o treino

```bash
python modelo/train.py
```

Saída: `modelo/modelo_demanda.joblib` + `modelo/metrics.json`.

---

## Notebook

`notebooks/01_pipeline_demanda.ipynb` contém o mesmo pipeline do
`train.py`, com EDA + visualizações (histogramas, heatmap de correlação,
matriz de confusão, barras de feature importances).

---

## Licença / contexto acadêmico

Projeto entregue como parte da disciplina de **Inteligência Artificial
— FIAP — 1TDSPR**, Sprint 4 (100 pontos: 40 pipeline + 40 deploy + 20
demonstração).
