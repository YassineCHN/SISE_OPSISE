# 🛡️ SISE — Analyse des logs Firewall

> **Projet OPSISE · Master SISE 2025–2026**
> Plateforme Streamlit d'analyse de logs firewall : visualisation, cartographie géographique et détection d'anomalies par ML avec analyse IA.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![MotherDuck](https://img.shields.io/badge/Data-MotherDuck-yellow)](https://motherduck.com/)
[![Mistral](https://img.shields.io/badge/LLM-Mistral%20AI-orange)](https://mistral.ai/)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker)](https://www.docker.com/)
[![App](https://img.shields.io/badge/Live-opsise.streamlit.app-FF4B4B?logo=streamlit)](https://opsise.streamlit.app)

---

## 📖 Présentation

**SISE Firewall Analyser** est une application d'analyse de logs firewall développée dans le cadre du projet OPSISE. Elle permet d'explorer, visualiser et analyser des flux réseau à partir de données réelles (cloud) ou synthétiques (lab), avec des capacités de détection d'anomalies par machine learning et d'analyse IA des menaces.

L'application comprend :
- **Un dashboard de visualisation** pour explorer les distributions et tendances des flux réseau
- **Une carte géographique** pour visualiser les origines et destinations des IPs publiques
- **Une plateforme Sentinel** de détection d'anomalies et de classification comportementale
- **Un analyse IA** (Mistral) pour générer des rapports de menaces automatisés

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🏠 **Accueil** | KPIs globaux, timeline pédagogique, explorateur de données brutes avec export CSV |
| 📊 **Visualisation** | Distribution actions/protocoles/ports, top sources/destinations, heatmap temporelle |
| 🗺️ **Carte** | Géolocalisation des IPs publiques, visualisation pydeck (arcs, points), 7 KPIs réseau |
| 🔍 **Détection d'anomalies** | Isolation Forest sur les features comportementales par IP |
| 🤖 **Classification ML** | Random Forest — 4 profils : Scanner, Brute-Force, Flood, Normal |
| 📈 **Analyse temporelle** | Heatmaps, séries temporelles, corrélations inter-variables |
| 💬 **Analyse IA** | Rapports de menaces automatisés via Mistral AI (fallback intégré) |
| 📡 **Flux en direct** | Feed live des connexions réseau simulées |

---

## 🗄️ Sources de données

| Table | Description | Volume |
|---|---|---|
| `original_data` | Logs firewall cloud réels | ~4,5 M lignes |
| `generated_data` | Logs synthétiques OPSIE (IPs privées uniquement) | ~15 K lignes |

Colonnes : `datetime`, `ip_src`, `ip_dst`, `port_dst`, `protocol`, `action` (`PERMIT`/`DENY`), `rule_id`, `interface_in`, `interface_out`

---

## 🚀 Accès à l'application

### Streamlit Cloud *(recommandé)*

L'application est accessible en ligne sans installation :
**→ [opsise.streamlit.app](https://opsise.streamlit.app)**

### Docker

```bash
git clone https://github.com/YassineCHN/SISE_OPSISE.git
cd SISE_OPSISE
# Configurer src/.env (voir ci-dessous)
docker compose up --build
```

Ouvrir [http://localhost:8501](http://localhost:8501).

### Depuis une image `.tar`

```bash
docker load -i sise_opsise.tar
docker compose up
```

> `src/.env` doit être présent avant le lancement.

---

## 🎯 Utilisation

### Workflow recommandé

**1. Sélectionner la source de données** (sidebar)
- **MotherDuck** : données cloud en temps réel
- **Fichier local** : parquet ou CSV uploadé directement

**2. Explorer les données**
- Page **Visualisation** pour les distributions globales
- Page **Carte** pour l'analyse géographique des IPs publiques

**3. Détecter les anomalies** (page Sentinel)
- Lancer la détection **Isolation Forest** pour identifier les IPs suspectes
- Lancer la **classification Random Forest** pour profiler les comportements
- Consulter les courbes ROC et matrices de confusion

**4. Générer un rapport IA**
- Renseigner une clé API Mistral dans la sidebar (optionnel)
- Sans clé : des rapports de secours pré-rédigés sont utilisés automatiquement

---

## ⚙️ Configuration

Créer `src/.env` :

```env
DATA_SOURCE=motherduck

MOTHERDUCK_TOKEN=<votre_token>
MOTHERDUCK_DATABASE=my_db
MOTHERDUCK_TABLE_OPTIONS=original_data,generated_data
MOTHERDUCK_FALLBACK_TO_PARQUET=true

MISTRAL_API_KEY=<votre_clé>       # optionnel
MISTRAL_MODEL=mistral-small-latest
```

| Variable | Où la trouver |
|---|---|
| `MOTHERDUCK_TOKEN` | [app.motherduck.com](https://app.motherduck.com) → Settings → Tokens |
| `MISTRAL_API_KEY` | [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) |

---

## 🏗️ Architecture

```
SISE_OPSISE/
├── src/
│   ├── app.py                      # Page d'accueil
│   ├── pages/
│   │   ├── 1_Visualisation.py      # Dashboard analytique
│   │   ├── 2_Carte.py              # Carte géographique
│   │   └── 3_Sentinel_Avance.py   # Détection d'anomalies + ML + IA
│   ├── modules/
│   │   ├── preprocessing.py        # Chargement et nettoyage des données
│   │   ├── charts.py               # Graphiques Plotly réutilisables
│   │   ├── stats.py                # Agrégations statistiques
│   │   └── filters.py              # Filtres sidebar
│   ├── components/
│   │   ├── data_source_selector.py
│   │   ├── sentinel_theme.py
│   │   └── top_nav.py
│   ├── utils/
│   │   ├── network_utils.py        # Géolocalisation IP, labels ports
│   │   └── sentinel_llm_analyst.py
│   └── requirements.txt
├── exploration/                    # Notebooks d'exploration ML
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Dashboard | [Streamlit](https://streamlit.io/) |
| Données | [DuckDB](https://duckdb.org/) / [MotherDuck](https://motherduck.com/), Pandas, PyArrow |
| Visualisation | [Plotly](https://plotly.com/), [Pydeck](https://deckgl.readthedocs.io/), Matplotlib, Seaborn |
| ML | [Scikit-learn](https://scikit-learn.org/) (Isolation Forest, Random Forest), SciPy |
| LLM | [Mistral AI](https://mistral.ai/) |
| Déploiement | Docker, Streamlit Cloud |

---

## 👥 Réalisation du projet

Projet réalisé dans le cadre du **ChallengeSISE-OPSIE — Master SISE 2025–2026**, Université Lyon 2.

🔗 **Dépôt GitHub :** [YassineCHN/SISE_OPSISE](https://github.com/YassineCHN/SISE_OPSISE)

---

## 📝 Notes

- 🌍 `generated_data` ne contient que des IPs privées — la géolocalisation n'y est pas disponible
- ⚠️ DuckDB doit être en version ≤ 1.4.4 pour la compatibilité MotherDuck
- 🔑 La clé Mistral peut être saisie dans la sidebar même si une clé est déjà configurée
- ☁️ Sur Streamlit Cloud, `original_data` (~4,5 M lignes) est limité à 1 000 000 lignes pour respecter la contrainte mémoire de 1 Go. Configurer `MOTHERDUCK_ROW_LIMIT=1000000` dans les secrets. Les tendances restent représentatives. En local, aucune limite n'est appliquée.
