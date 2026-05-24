# Demonstração — Sprint 4 IA — Tech do Bem

**Disciplina:** Inteligência Artificial — FIAP
**Turma:** 1TDSPR
**Sprint:** 4 (entrega de 100 pontos)

## Autores

| Nome | RM |
| --- | --- |
| Lucas Campanhã dos Santos | 566815 |

> Este documento serve como _template_ do PDF de entrega. Os screenshots
> abaixo devem ser substituídos pelos prints reais antes da exportação.

---

## 1. O que foi construído

Modelo de classificação multi-classe que prediz a **demanda odontológica**
(`Baixa` / `Media` / `Alta`) de um bairro de São Paulo a partir de
indicadores socioeconômicos, exposto como API HTTP (`/predict`,
`/predict/batch`, `/health`).

- Dataset sintético reproduzível (`data/gerar_dataset.py`) com **2050
  linhas** distribuídas em **82 bairros reais** de SP.
- 3 modelos comparados: `RandomForestClassifier`, `XGBClassifier`,
  `LogisticRegression` (com `StandardScaler`).
- Seleção pelo `f1_macro` em validação cruzada **5-fold estratificada**.
- Serialização do pipeline vencedor em `modelo/modelo_demanda.joblib`.
- API Flask com `flask-cors` configurado para o frontend e o gunicorn em
  produção (Docker).

---

## 2. Como rodar localmente

```bash
git clone https://github.com/Tech-do-Bem-FIAP/Backend_AI.git
cd Backend_AI

# 1) ambiente
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) (re)gerar dataset
python data/gerar_dataset.py

# 3) treinar modelo
python modelo/train.py

# 4) subir a API
python -m flask --app api.app run --port 5001
```

A API fica em `http://localhost:5001`.

### Testes manuais

```bash
curl http://localhost:5001/health
curl -X POST http://localhost:5001/predict \
     -H 'Content-Type: application/json' \
     -d '{"bairro":"Brás"}'
curl -X POST http://localhost:5001/predict \
     -H 'Content-Type: application/json' \
     -d '{"densidade_pop":12000,"idh":0.78,"dist_ubs_km":1.5,"historico_atendimentos":45,"pct_exames_pendentes":0.3,"populacao_infantil_pct":0.22,"dia_semana_cadastro":3}'
```

---

## 3. Como rodar via Docker

```bash
docker build -t backend-ai .
docker run -p 5001:5001 backend-ai
curl http://localhost:5001/health
```

O container usa `gunicorn -w 2` na porta 5001 e copia somente os
artefatos necessários (`api/`, `modelo/modelo_demanda.joblib`,
`data/demanda_bairros.csv`).

---

## 4. Screenshots (preencher antes do PDF)

> _TODO: substituir pelos prints reais._

- **Print 1 — `GET /health` no navegador / curl** → `[imagem aqui]`
- **Print 2 — `POST /predict` com payload `{"bairro":"Brás"}`** → `[imagem aqui]`
- **Print 3 — Página web do frontend consumindo a API** → `[imagem aqui]`
- **Print 4 — Notebook com matriz de confusão renderizada** → `[imagem aqui]`

---

## 5. Matriz de confusão

> _TODO: exportar a imagem da matriz de confusão renderizada no notebook
> (`notebooks/01_pipeline_demanda.ipynb`, célula 6) e inserir aqui._

`[imagem: matriz_confusao.png]`

---

## 6. Métricas alcançadas

Valores atualizados a partir de `modelo/metrics.json` após o treino:

| Métrica | Valor |
| --- | --- |
| Modelo escolhido | `logistic_regression` |
| `accuracy` (teste) | ~0.64 |
| `f1_macro` (teste) | ~0.63 |
| `f1_macro` (CV 5-fold) | ~0.59 |
| Linhas de treino / teste | 1640 / 410 |

> O nível de acurácia é compatível com o ruído gaussiano (σ=0.12) que
> foi propositalmente injetado no _score_ latente do gerador — não é um
> problema, é prova de que o pipeline não está “decorando” o rótulo.

### Top-5 feature importances

A feature `dia_semana_cadastro` foi introduzida como **decoy** (ruído
puro, sem relação com o label) e ficou fora do top-5 — evidência de que
o modelo aprendeu o sinal correto:

1. `idh`
2. `pct_exames_pendentes`
3. `dist_ubs_km`
4. `densidade_pop`
5. `historico_atendimentos`

---

## 7. Repositórios do projeto Tech do Bem

- Backend_AI (este repo): https://github.com/Tech-do-Bem-FIAP/Backend_AI
- Backend Java: https://github.com/Tech-do-Bem-FIAP/Backend_Java
- Frontend React: https://github.com/Tech-do-Bem-FIAP/Frontend
