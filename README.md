# SISE — Analyse des logs Firewall 🛡️

Application Streamlit d'analyse de logs firewall développée dans le cadre du projet SISE/OPSISE 2026. Elle permet de visualiser, cartographier et détecter des anomalies dans des flux réseau à partir de données réelles (cloud) ou synthétiques (lab).

---

## ✨ Fonctionnalités

### 🏠 Accueil
- KPIs globaux (flux total, DENY/PERMIT, sources et destinations uniques)
- Timeline pédagogique des 6 étapes d'analyse
- Explorateur de données brutes avec filtres, recherche et export CSV

### 📊 1 — Visualisation
- Distribution des actions, protocoles et ports
- Top sources/destinations
- Analyse temporelle (heatmap horaire/journalière)
- Tableau de données filtrable

### 🗺️ 2 — Carte
- Géolocalisation des IPs publiques (API ip-api.com)
- Visualisation pydeck : arcs, points, labels
- 7 KPIs réseau (flux, trafic bloqué %, sources/destinations publiques)

### 🤖 3 — Sentinel Avancé
- Détection d'anomalies par Isolation Forest
- Classification comportementale par Random Forest (Scanner, Brute-Force, Flood, Normal)
- Analyse temporelle et corrélations
- Analyse IA des menaces via Mistral AI (avec rapports de secours intégrés)
- Flux réseau en direct

---

## 🗄️ Sources de données

| Table | Description | Lignes |
|-------|-------------|--------|
| `original_data` | Logs firewall cloud réels | ~4,5 M |
| `generated_data` | Logs synthétiques OPSIE (IPs privées) | ~15 K |

Colonnes communes : `datetime`, `ip_src`, `ip_dst`, `port_dst`, `protocol`, `action` (`PERMIT`/`DENY`), `rule_id`, `interface_in`, `interface_out`

---

## 🧰 Stack technique

- **Framework** : Streamlit
- **Données** : DuckDB / MotherDuck, Pandas, PyArrow
- **Visualisation** : Plotly, Pydeck, Matplotlib, Seaborn
- **ML** : Scikit-learn, SciPy
- **LLM** : Mistral AI
- **Déploiement** : Docker, Streamlit Cloud

---

## 🚀 Installation locale

### Prérequis
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommandé) ou pip

### Avec uv _(recommandé)_

```bash
git clone https://github.com/YassineCHN/SISE_OPSISE.git
cd SISE_OPSISE
uv sync                  # crée le venv et installe les dépendances automatiquement
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Avec venv + pip

```bash
git clone https://github.com/YassineCHN/SISE_OPSISE.git
cd SISE_OPSISE
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r src/requirements.txt
```

### ⚙️ Configuration

Créer `src/.env` à partir de l'exemple :

```env
# Source de données : "parquet" (local) ou "motherduck"
DATA_SOURCE=motherduck

# MotherDuck
MOTHERDUCK_TOKEN=<votre_token>
MOTHERDUCK_DATABASE=my_db
MOTHERDUCK_TABLE_OPTIONS=original_data,generated_data

# Mistral AI (optionnel — rapports de secours activés si absent)
MISTRAL_API_KEY=<votre_clé>
MISTRAL_MODEL=mistral-small-latest
```

### ▶️ Lancement

```bash
streamlit run src/app.py
```

L'application est disponible sur [http://localhost:8501](http://localhost:8501).

---

## 🐳 Déploiement Docker

```bash
# Build et démarrage
docker compose up --build

# Arrêt
docker compose down
```

> Le fichier `src/.env` est chargé automatiquement par docker-compose.

### 📦 Export de l'image (livraison sans build)

```bash
docker save sise_opsise-streamlit -o sise_opsise.tar
```

### 📥 Import de l'image

```bash
docker load -i sise_opsise.tar
docker compose up
```

> `src/.env` doit être présent avant de lancer `docker compose up`.

---

## ☁️ Déploiement Streamlit Cloud

1. Pusher le dépôt sur GitHub
2. Créer une app sur [share.streamlit.io](https://share.streamlit.io) avec `src/app.py` comme point d'entrée
3. Ajouter les secrets dans **Settings > Secrets** :

```toml
DATA_SOURCE = "motherduck"
MOTHERDUCK_TOKEN = "..."
MOTHERDUCK_DATABASE = "my_db"
MOTHERDUCK_TABLE_OPTIONS = "original_data,generated_data"
MISTRAL_API_KEY = "..."
MISTRAL_MODEL = "mistral-small-latest"
```

---

## 📁 Structure du projet

```
SISE_OPSISE/
├── src/
│   ├── app.py                  # Page d'accueil
│   ├── pages/
│   │   ├── 1_Visualisation.py
│   │   ├── 2_Carte.py
│   │   └── 3_Sentinel_Avance.py
│   ├── modules/
│   │   ├── preprocessing.py    # Chargement et nettoyage des données
│   │   ├── charts.py           # Graphiques Plotly réutilisables
│   │   ├── stats.py            # Agrégations statistiques
│   │   └── filters.py          # Filtres sidebar
│   ├── components/
│   │   ├── data_source_selector.py
│   │   ├── sentinel_theme.py
│   │   └── top_nav.py
│   ├── utils/
│   │   ├── network_utils.py    # Géolocalisation IP, labels ports
│   │   └── sentinel_llm_analyst.py
│   └── requirements.txt
├── exploration/                # Notebooks d'exploration (non conteneurisé)
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 📝 Notes

- 🔑 La clé Mistral peut être saisie directement dans la sidebar si elle est absente ou expirée
- 🛟 Sans clé Mistral, des rapports de secours pré-rédigés sont utilisés automatiquement
- 🌍 Les données `generated_data` ne contiennent que des IPs privées — la géolocalisation n'y est pas disponible
- ⚠️ DuckDB doit être en version ≤ 1.4.4 pour la compatibilité MotherDuck
