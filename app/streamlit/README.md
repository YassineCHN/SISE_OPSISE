# 🔥 SISE – Analyse des logs Firewall

Application web Streamlit d'analyse des logs Iptables avec Machine Learning et LLM Mistral.

## 📁 Structure du projet

```
src/
├── app.py                        ← Accueil + auto-chargement des données
├── config.py                     ← Paramètres globaux
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── components/
│   └── ui.py                     ← CSS, composants UI réutilisables
├── modules/
│   ├── preprocessing.py          ← Chargement + parsing + enrichissement
│   ├── stats.py                  ← Fonctions statistiques
│   ├── ml.py                     ← Isolation Forest, DBSCAN, Random Forest, Z-score
│   └── llm.py                    ← Interface Mistral
├── pages/
│   ├── 1_Analyse_Descriptive.py  ← Actions, protocoles, ports, timeline, Sankey
│   ├── 2_DataTable.py            ← Table interactive + filtres + export CSV
│   ├── 3_Visualisation_IP.py     ← Top IPs, relations src→dst, profils
│   ├── 4_Statistiques.py         ← Crosstab, heatmaps horaires/hebdo, règles
│   ├── 5_Detection_Anomalies.py  ← Scénario 1 : Isolation Forest + DBSCAN
│   ├── 6_Random_Forest.py        ← Scénario 2 : RF + ROC + prédiction temps réel
│   ├── 7_Analyse_Temporelle.py   ← Scénario 3 : heatmaps + Z-score + profils horaires
│   ├── 8_Comportement_Attaques.py← IP src/dst, radar, corrélations, TOP stats
│   └── 9_Assistant_LLM.py        ← Chat Mistral cybersécurité
├── data/processed/
│   └── log_clean.parquet         ← ← PLACEZ VOTRE FICHIER ICI
└── utils/
```

## 🚀 Lancement local

```bash
# 1. Placer les données
cp /chemin/vers/log_clean.parquet data/processed/log_clean.parquet

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer (les données se chargent automatiquement)
streamlit run app.py
```

→ Accessible sur **http://localhost:8501**

## 🐳 Docker

```bash
# Placer d'abord le parquet dans data/processed/
cp /chemin/vers/log_clean.parquet data/processed/log_clean.parquet

# Build + lancement
docker compose up --build

# Avec clé Mistral
MISTRAL_API_KEY=votre_cle docker compose up --build
```

## 📋 Format des données

L'application accepte deux formats de parquet :

**Format brut** (colonne `raw_log` semi-structurée séparée par `;`) :
```
datetime;ip_src;ip_dst;protocol;port_src;port_dst;rule_id;action;interface_in;interface_out;fw_id
```

**Format structuré** (`log_clean.parquet`) : colonnes déjà séparées.

## 🤖 LLM Mistral

Deux façons de configurer la clé :
1. Dans la sidebar de l'application
2. Variable d'environnement : `export MISTRAL_API_KEY=votre_cle`

## 📑 Pages disponibles

| # | Page | Description |
|---|------|-------------|
| 🏠 | Accueil | KPIs globaux + vue rapide + synthèse LLM |
| 📊 | Analyse Descriptive | Actions, protocoles, ports, timeline, Sankey |
| 📋 | Table de données | Filtres dynamiques + export CSV |
| 🌐 | Visualisation IP | Top IPs, relations src→dst, profils comportementaux |
| 📈 | Statistiques | Crosstab, heatmaps horaires/hebdo, règles |
| 🔴 | Détection Anomalies | **Scénario 1** : Isolation Forest + DBSCAN |
| 🤖 | Random Forest | **Scénario 2** : Classification + ROC + prédiction |
| ⏱️ | Analyse Temporelle | **Scénario 3** : Heatmaps + Z-score |
| 🌍 | Comportement Attaques | Radar chart + corrélations + TOP stats projet |
| 💬 | Assistant LLM | Chat Mistral cybersécurité (ISO 27001 / ANSSI) |
