# CC Underwriting API — Complete Step-by-Step Guide

> LightGBM credit card underwriting API · FastAPI · PostgreSQL · Docker · GitHub Actions → Azure

**Model:** AUC 0.997 · Gini 0.994 · KS 0.944 · 171 features · 4,480 applicants

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Setup](#2-project-setup)
3. [Train the ML Model](#3-train-the-ml-model)
4. [Run Without Docker](#4-run-without-docker-pure-python)
5. [Run With Docker Compose](#5-run-with-docker-compose-recommended)
6. [Use the API](#6-use-the-api)
7. [Database and Persistence](#7-database-and-persistence)
8. [Run the Tests](#8-run-the-tests)
9. [Push to GitHub](#9-push-to-github)
10. [Set Up Azure Resources](#10-set-up-azure-resources)
11. [Configure GitHub Actions Secrets](#11-configure-github-actions-secrets)
12. [Deploy to Production](#12-deploy-to-production)
13. [Monitor and Maintain](#13-monitor-and-maintain)
14. [Troubleshooting](#14-troubleshooting)
15. [Project File Reference](#15-project-file-reference)

---

## 1. Prerequisites

Install all of these before starting. Commands are for macOS/Linux/WSL2.

### 1.1 Python 3.11

```bash
python3 --version          # must be 3.11 or higher

# macOS
brew install python@3.11

# Ubuntu / WSL2
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev -y
```

### 1.2 Docker Desktop

Download from https://www.docker.com/products/docker-desktop/

```bash
docker --version           # Docker version 24.x or higher
docker-compose --version   # Docker Compose version v2.x
```

> Windows users: install Docker Desktop and enable WSL2 integration
> in Docker Desktop → Settings → Resources → WSL Integration.

### 1.3 Azure CLI

```bash
# macOS
brew install azure-cli

# Ubuntu / WSL2
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az --version               # azure-cli 2.x
```

### 1.4 Git

```bash
git --version              # git version 2.x

# macOS: brew install git
# Ubuntu: sudo apt install git -y
```

---

## 2. Project Setup

### 2.1 Extract the project

```bash
tar -xzf cc_underwriting_api.tar.gz
cd cc_underwriting_api
ls -la
# You should see: app/  ml/  models/  migrations/  tests/  .github/
#                 Dockerfile  docker-compose.yml  requirements.txt  README.md
```

### 2.2 Create a Python virtual environment

A virtual environment keeps this project's dependencies isolated.

```bash
python3.11 -m venv .venv

# Activate — do this every time you open a new terminal
source .venv/bin/activate           # macOS / Linux / WSL2
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# You will see (.venv) in your terminal prompt when active
```

### 2.3 Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Takes 2-5 minutes on first install
```

### 2.4 Create your .env file

```bash
cp .env.example .env

# Generate a secure secret key and copy the output
python3 -c "import secrets; print(secrets.token_hex(32))"

# Edit .env — minimum required changes:
nano .env
```

Change these three values in `.env`:

```ini
SECRET_KEY=paste-the-64-char-hex-you-just-generated-here
POSTGRES_PASSWORD=choose-a-strong-password-min-16-chars
ADMIN_PASSWORD=choose-a-strong-admin-password
```

> The `.env` file is in `.gitignore` and will NEVER be committed to git.

---

## 3. Train the ML Model

The model must exist before starting the API.
Training reads the CSV, runs the full pipeline, and saves a `.joblib` artifact.

### 3.1 Place the data file

```bash
# Copy the CSV into the project directory
cp /path/to/cc_underwriting_5k_stratified11.csv .
ls *.csv    # confirm it is there
```

### 3.2 Run training

```bash
source .venv/bin/activate

python ml/train.py \
  --data cc_underwriting_5k_stratified11.csv \
  --out  models/cc_model_v1.joblib \
  --version v1.0.0
```

Expected output (takes 1-3 minutes):

```
09:12:01  INFO  ── Stage 1: Load ──
09:12:01  INFO    Raw shape: (4480, 200)
09:12:02  INFO  ── Stage 2: Drop columns ──
09:12:02  INFO    Dropping 2 high-missingness cols: ['military_status', 'promo_code_used']
09:12:03  INFO  ── Stage 5: Feature engineering ──
09:12:03  INFO    Engineered: 8  Log: 7  Encoded: 30
09:12:04  INFO  ── Stage 6: Feature matrix + correlation filter ──
09:12:04  INFO    Final: 171 features
09:12:08  INFO    AUC   : 0.9970
09:12:08  INFO    Gini  : 0.9939
09:12:08  INFO    KS    : 0.9443
09:12:09  INFO    Saved → models/cc_model_v1.joblib  (28.3 MB)
09:12:09  INFO  ✅ Training complete.
```

### 3.3 Verify the model file was created

```bash
ls -lh models/cc_model_v1.joblib
# Should show a file around 25-35 MB
```

---

## 4. Run Without Docker (Pure Python)

Use this for quick development and debugging.
You still need PostgreSQL — start just the database container:

### 4.1 Start only PostgreSQL

```bash
docker run -d \
  --name cc_postgres_dev \
  -e POSTGRES_DB=cc_underwriting \
  -e POSTGRES_USER=ccapi \
  -e POSTGRES_PASSWORD=ccapi_pass \
  -p 5432:5432 \
  postgres:16-alpine

# Wait 5 seconds, then verify
docker exec cc_postgres_dev pg_isready -U ccapi -d cc_underwriting
# Should print: /var/run/postgresql:5432 - accepting connections
```

### 4.2 Set DATABASE_URL in .env

```ini
DATABASE_URL=postgresql://ccapi:ccapi_pass@localhost:5432/cc_underwriting
```

### 4.3 Start FastAPI

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You will see:

```
INFO  Loading model from 'models/cc_model_v1.joblib' ...
INFO  Model 'v1.0.0' loaded — 171 features. AUC=0.997 Gini=0.994
INFO  Database tables verified / created.
INFO  Default admin user 'admin' created.
INFO  Startup complete — API is ready.
INFO  Uvicorn running on http://0.0.0.0:8000
```

Open http://localhost:8000/docs to see the interactive Swagger UI.

---

## 5. Run With Docker Compose (Recommended)

This starts the API and PostgreSQL together with one command.
No local Python or database setup needed.

### 5.1 Verify Docker is running

```bash
docker info    # should show system info without errors
```

### 5.2 Build and start all services

```bash
# From the project root directory (where docker-compose.yml lives)
docker-compose up -d

# First run takes 3-8 minutes to build the image and pull PostgreSQL
# Subsequent runs take 10-20 seconds
```

### 5.3 Check both services are healthy

```bash
docker-compose ps
```

Expected output:

```
NAME           STATUS                    PORTS
cc_postgres    Up 30s (healthy)          0.0.0.0:5432->5432/tcp
cc_api         Up 25s (health: starting) 0.0.0.0:8000->8000/tcp
```

Wait until `cc_api` shows `(healthy)` — takes about 30-60 seconds:

```bash
# Keep running this until you see (healthy)
docker-compose ps

# Or watch the logs while waiting
docker-compose logs -f
# Press Ctrl+C to stop watching
```

### 5.4 Verify the API is up

```bash
curl http://localhost:8000/health
```

Expected:

```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true,
  "model_version": "v1.0.0"
}
```

Open http://localhost:8000/docs for the Swagger UI.

### 5.5 Essential Docker Compose commands

```bash
docker-compose logs -f api          # stream API logs
docker-compose logs -f db           # stream PostgreSQL logs
docker-compose restart api          # restart API without touching database
docker-compose build api            # rebuild image after code changes
docker-compose build api && docker-compose up -d  # rebuild and restart

docker-compose down                 # stop everything — DATA IS PRESERVED
docker-compose down -v              # stop AND wipe database — DATA IS LOST
```

---

## 6. Use the API

### 6.1 Register a user (optional — admin exists already)

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "analyst1",
    "email": "analyst1@yourcompany.com",
    "password": "MyPassword@123"
  }'
```

### 6.2 Log in and get a JWT token

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=analyst1&password=MyPassword@123"
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

Copy the `access_token`. You will use it in every protected request.
Tokens expire after 60 minutes — just log in again to get a new one.

### 6.3 Run a prediction

```bash
# Replace YOUR_TOKEN with the token from the login step
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_ref": "APP-2024-001",
    "age": 35,
    "annual_income": 95000,
    "monthly_income": 7917,
    "fico_score": 740,
    "equifax_score": 745,
    "experian_score": 738,
    "transunion_score": 742,
    "avg_bureau_score": 741,
    "credit_utilization_ratio": 0.18,
    "debt_to_income_ratio": 0.25,
    "requested_credit_limit": 10000,
    "net_worth": 60000,
    "total_assets": 120000,
    "total_liabilities": 45000,
    "savings_account_balance": 15000,
    "total_monthly_expenses": 3200,
    "late_payments_last_24mo": 0,
    "bankruptcy_count": 0,
    "fraud_risk_score": 15
  }'
```

Response:

```json
{
  "applicant_ref": "APP-2024-001",
  "decision": "Approved",
  "approval_prob": 0.9312,
  "credit_score": 782,
  "risk_band": "Low",
  "risk_band_detail": {
    "band": "Low",
    "score_range": "740-819",
    "description": "Strong profile — standard approval"
  },
  "model_version": "v1.0.0",
  "prediction_id": 1,
  "timestamp": "2024-01-15T09:35:22"
}
```

### 6.4 View paginated prediction history

```bash
# Page 1, 10 records per page
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/predictions?page=1&size=10"

# Approved decisions only
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/predictions?page=1&size=50&decision=Approved"

# Single prediction by ID
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/predictions/1"
```

Pagination response always includes:

```json
{
  "total_count": 248,
  "page": 1,
  "size": 10,
  "has_next": true,
  "has_prev": false,
  "data": [...]
}
```

### 6.5 Using Swagger UI (easier for manual testing)

1. Open http://localhost:8000/docs
2. Click **POST /auth/token** → **Try it out**
3. Enter username and password → **Execute**
4. Copy the `access_token` from the response
5. Click the 🔒 **Authorize** button at the top of the page
6. Paste the token → **Authorize** → **Close**
7. All protected endpoints now work automatically

### 6.6 Admin operations

```bash
# Get admin token first (credentials from your .env)
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=Admin@12345" \
  -H "Content-Type: application/x-www-form-urlencoded"

# List all users
curl -H "Authorization: Bearer ADMIN_TOKEN" \
  http://localhost:8000/admin/users

# Deactivate a user
curl -X POST -H "Authorization: Bearer ADMIN_TOKEN" \
  http://localhost:8000/admin/users/2/deactivate
```

---

## 7. Database and Persistence

### 7.1 What is stored

**`users` table** — one row per API user:
- `username`, `email`, `hashed_password` (bcrypt — never plain text)
- `is_active`, `is_admin`, `last_login`

**`prediction_log` table** — every prediction ever made:
- Who called it (`user_id`), when (`created_at`), from where (`request_ip`)
- Full input JSON (`input_features`)
- Model output: `approval_prob`, `credit_score`, `decision`, `risk_band`
- `model_version` — which model version was used

### 7.2 How persistence works

Docker Compose creates a **named volume** `cc_underwriting_pgdata`.
Docker manages where this lives on your machine.

```bash
docker-compose down       # Stops containers. Volume SAFE. All data preserved.
docker-compose up -d      # Starts containers. PostgreSQL reads from volume. Data restored.

docker-compose down -v    # Stops AND deletes volume. ALL DATA LOST. Cannot undo.
```

Check where Docker stores your data:

```bash
docker volume inspect cc_underwriting_pgdata
# Shows the Mountpoint path on your machine
```

### 7.3 Connect to the database directly

```bash
# Open psql inside the running container
docker exec -it cc_postgres psql -U ccapi -d cc_underwriting

# Useful queries inside psql:
\dt                                          -- list all tables
SELECT * FROM users;                         -- see all users
SELECT COUNT(*) FROM prediction_log;         -- total predictions made
SELECT decision, COUNT(*) FROM prediction_log GROUP BY decision;
\q                                           -- quit psql
```

### 7.4 Backup and restore

```bash
# Backup
docker exec cc_postgres pg_dump -U ccapi cc_underwriting > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i cc_postgres psql -U ccapi cc_underwriting < backup_20240115.sql
```

---

## 8. Run the Tests

### 8.1 Install test dependencies

```bash
source .venv/bin/activate
pip install pytest pytest-asyncio httpx
```

### 8.2 Make sure the database is running

```bash
docker-compose up -d db
docker-compose ps    # wait until db shows (healthy)
```

### 8.3 Run all tests

```bash
export DATABASE_URL="postgresql://ccapi:ccapi_pass@localhost:5432/cc_underwriting"
export SECRET_KEY="test-secret-not-for-production"
export MODEL_PATH="models/cc_model_v1.joblib"
export ADMIN_PASSWORD="TestAdmin@123"
export ADMIN_EMAIL="admin@test.local"

pytest tests/ -v
```

Expected output:

```
tests/test_api.py::TestHealth::test_root PASSED
tests/test_api.py::TestHealth::test_health_returns_200 PASSED
tests/test_api.py::TestAuth::test_register_success PASSED
tests/test_api.py::TestAuth::test_login_success PASSED
tests/test_api.py::TestAuth::test_login_wrong_password PASSED
tests/test_api.py::TestPredict::test_predict_good_applicant PASSED
tests/test_api.py::TestPredict::test_predict_risky_applicant PASSED
tests/test_api.py::TestPagination::test_predictions_pagination_page1 PASSED
tests/test_api.py::TestModelInfo::test_model_metrics_acceptable PASSED
...
25 passed in 12.34s
```

### 8.4 Run with coverage report

```bash
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=html
open htmlcov/index.html    # macOS
```

---

## 9. Push to GitHub

### 9.1 Create a new GitHub repository

1. Go to https://github.com/new
2. Name: `cc-underwriting-api`
3. Visibility: **Private** (recommended)
4. Do NOT check any initialisation options (readme, gitignore, license)
5. Click **Create repository**

### 9.2 Initialise git and push

```bash
# From the project root
git init
git add .

# Verify: .env and *.joblib must NOT appear in this list
git status

git commit -m "Initial commit: CC Underwriting API"

git remote add origin https://github.com/YOUR_USERNAME/cc-underwriting-api.git
git branch -M main
git push -u origin main
```

### 9.3 Create the staging branch

```bash
git checkout -b staging
git push -u origin staging
git checkout main
```

### 9.4 Store the model in Azure Blob Storage

The `.joblib` file is excluded from git (too large, not appropriate for version control).
Store it in Azure Blob Storage and download it in CI/CD.

```bash
# Create a storage account (globally unique name)
az storage account create \
  --name ccunderwritingmodels \
  --resource-group rg-cc-underwriting \
  --sku Standard_LRS \
  --location eastus

# Create a container (like a folder)
az storage container create \
  --name models \
  --account-name ccunderwritingmodels

# Upload the model
az storage blob upload \
  --account-name ccunderwritingmodels \
  --container-name models \
  --name cc_model_v1.joblib \
  --file models/cc_model_v1.joblib

# Get the connection string — save this for the GitHub Secret
az storage account show-connection-string \
  --name ccunderwritingmodels \
  --resource-group rg-cc-underwriting \
  --query connectionString -o tsv
```

---

## 10. Set Up Azure Resources

Run these once to create all Azure infrastructure.

### 10.1 Log in to Azure

```bash
az login
# A browser opens — log in with your Azure account

# If multiple subscriptions, set the right one
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_NAME"
az account show    # verify active subscription
```

### 10.2 Create a Resource Group

```bash
az group create \
  --name rg-cc-underwriting \
  --location eastus
```

### 10.3 Create Azure Container Registry

```bash
# Name must be globally unique — only lowercase letters/numbers
az acr create \
  --resource-group rg-cc-underwriting \
  --name ccunderwritingacr \
  --sku Basic \
  --admin-enabled true

# Get credentials — save username and password for GitHub Secrets
az acr credential show --name ccunderwritingacr
```

### 10.4 Create managed PostgreSQL (Azure Database for PostgreSQL)

```bash
# Takes 5-10 minutes
az postgres flexible-server create \
  --resource-group rg-cc-underwriting \
  --name cc-underwriting-db \
  --location eastus \
  --admin-user ccapi \
  --admin-password "YourStrongPassword123!" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32 \
  --version 16

# Create the application database
az postgres flexible-server db create \
  --resource-group rg-cc-underwriting \
  --server-name cc-underwriting-db \
  --database-name cc_underwriting

# Get the server hostname — ends in .postgres.database.azure.com
az postgres flexible-server show \
  --resource-group rg-cc-underwriting \
  --name cc-underwriting-db \
  --query fullyQualifiedDomainName -o tsv

# Allow Azure services to reach the database
az postgres flexible-server firewall-rule create \
  --resource-group rg-cc-underwriting \
  --name cc-underwriting-db \
  --rule-name allow-azure-services \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

Your DATABASE_URL for production will be:
`postgresql://ccapi:YourStrongPassword123!@cc-underwriting-db.postgres.database.azure.com:5432/cc_underwriting`

### 10.5 Create App Service Plan

```bash
az appservice plan create \
  --name cc-underwriting-plan \
  --resource-group rg-cc-underwriting \
  --is-linux \
  --sku B2
# B2 = 2 cores, 3.5 GB RAM (~$75/month)
# B1 = 1 core, 1.75 GB RAM (~$14/month) — OK for low traffic
```

### 10.6 Create Web App for Containers

```bash
az webapp create \
  --resource-group rg-cc-underwriting \
  --plan cc-underwriting-plan \
  --name cc-underwriting-api \
  --deployment-container-image-name ccunderwritingacr.azurecr.io/cc-underwriting-api:latest

# Set the container port
az webapp config appsettings set \
  --resource-group rg-cc-underwriting \
  --name cc-underwriting-api \
  --settings WEBSITES_PORT=8000

# Get the production URL
az webapp show \
  --resource-group rg-cc-underwriting \
  --name cc-underwriting-api \
  --query defaultHostName -o tsv
# Prints: cc-underwriting-api.azurewebsites.net
```

### 10.7 Create Service Principal for GitHub Actions

```bash
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

az ad sp create-for-rbac \
  --name "cc-api-github-actions" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-cc-underwriting \
  --sdk-auth
```

This prints a JSON block. **Copy the entire JSON** — you will paste it as a GitHub Secret.

---

## 11. Configure GitHub Actions Secrets

### 11.1 Open secrets settings

GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

### 11.2 Add each secret

| Secret Name | Value | How to get it |
|---|---|---|
| `AZURE_CREDENTIALS` | Full JSON from Step 10.7 | Output of `az ad sp create-for-rbac` |
| `ACR_LOGIN_SERVER` | `ccunderwritingacr.azurecr.io` | Your ACR name + `.azurecr.io` |
| `ACR_USERNAME` | ACR admin username | `az acr credential show --name ccunderwritingacr --query username -o tsv` |
| `ACR_PASSWORD` | ACR admin password | `az acr credential show --name ccunderwritingacr --query passwords[0].value -o tsv` |
| `AZURE_WEBAPP_NAME` | `cc-underwriting-api` | Name from Step 10.6 |
| `AZURE_RESOURCE_GROUP` | `rg-cc-underwriting` | From Step 10.2 |
| `SECRET_KEY` | 64-char hex | `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `POSTGRES_PASSWORD` | Your DB password | Password from Step 10.4 |
| `POSTGRES_USER` | `ccapi` | As set in Step 10.4 |
| `DB_HOST` | `cc-underwriting-db.postgres.database.azure.com` | From Step 10.4 output |
| `ADMIN_PASSWORD` | A strong password | Choose one |
| `ADMIN_EMAIL` | `admin@yourcompany.com` | Your admin email |
| `AZURE_STORAGE_CONNECTION_STRING` | Full connection string | From Step 9.4 |

### 11.3 Add model download step to workflow

Open `.github/workflows/deploy.yml`. In the `test` job, after the
"Install dependencies" step, add:

```yaml
      - name: "Download model from Azure Blob Storage"
        run: |
          az storage blob download \
            --container-name models \
            --name cc_model_v1.joblib \
            --file models/cc_model_v1.joblib \
            --connection-string "${{ secrets.AZURE_STORAGE_CONNECTION_STRING }}"
```

Push this change:

```bash
git add .github/workflows/deploy.yml
git commit -m "Add model download step to CI workflow"
git push origin main
```

---

## 12. Deploy to Production

### 12.1 First deployment

```bash
# Any push to main triggers the full pipeline
git push origin main
```

### 12.2 Watch the pipeline

1. GitHub repository → **Actions** tab
2. Click the running workflow "CI/CD — Build, Test, Deploy"
3. Watch each job in real time

Pipeline sequence:

```
Job 1: Lint & Test        (2-5 min)
       ↓ passes
Job 2: Build & Push       (3-8 min) — builds Docker image, pushes to ACR
       ↓ (only on main branch)
Job 3: Deploy Web App     (2-3 min) — deploys to Azure App Service
       ↓
       Smoke test: GET /health → must return HTTP 200
```

### 12.3 Verify production

```bash
curl https://cc-underwriting-api.azurewebsites.net/health

# Expected:
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true
}
```

Open https://cc-underwriting-api.azurewebsites.net/docs for production Swagger UI.

### 12.4 Deploy to staging first (recommended workflow)

```bash
# Deploy to staging (Azure Container Instance — cheaper)
git checkout staging
git merge feature/my-change
git push origin staging
# Triggers Job 4: deploy-aci → staging container

# After testing staging, merge to main for production
git checkout main
git merge staging
git push origin main
# Triggers Jobs 1-3 → production
```

### 12.5 All future deployments

```bash
# Change code → commit → push → done
git add .
git commit -m "Describe change"
git push origin main    # GitHub Actions handles everything automatically
```

---

## 13. Monitor and Maintain

### 13.1 View live logs

```bash
# Production logs from Azure App Service
az webapp log tail \
  --name cc-underwriting-api \
  --resource-group rg-cc-underwriting

# Local Docker logs
docker-compose logs -f api
```

### 13.2 Check App Service status

```bash
az webapp show \
  --name cc-underwriting-api \
  --resource-group rg-cc-underwriting \
  --query state -o tsv    # Should print: Running
```

### 13.3 Retrain and deploy a new model version

```bash
# 1. Train with new data
python ml/train.py \
  --data new_data_q2_2024.csv \
  --out models/cc_model_v2.joblib \
  --version v2.0.0

# 2. Upload to Azure Blob Storage
az storage blob upload \
  --account-name ccunderwritingmodels \
  --container-name models \
  --name cc_model_v2.joblib \
  --file models/cc_model_v2.joblib

# 3. Update .env (and push config change)
# MODEL_PATH=models/cc_model_v2.joblib

# 4. Push to trigger deployment
git add .
git commit -m "Upgrade to model v2.0.0"
git push origin main
```

---

## 14. Troubleshooting

### Port 5432 already in use

```bash
lsof -i :5432                    # see what is using it
sudo service postgresql stop     # Ubuntu — stop local PostgreSQL
brew services stop postgresql    # macOS
# Or change the host port in docker-compose.yml: "5433:5432"
```

### API starts but model_loaded is false

```bash
# Check model file is inside the container
docker exec cc_api ls -lh /app/models/

# If empty, the volume mount is wrong — ensure docker-compose.yml has:
# volumes:
#   - ./models:/app/models:ro
# AND models/cc_model_v1.joblib exists on your host
```

### 401 Unauthorized after valid login

```bash
# Tokens expire after 60 minutes — log in again
# Also check: SECRET_KEY in .env must not have changed since login
# Changing SECRET_KEY invalidates all existing tokens
```

### GitHub Actions fails on deploy step

```bash
# Most common: AZURE_CREDENTIALS secret expired or missing permissions
# Recreate:
az ad sp delete --id "cc-api-github-actions"
az ad sp create-for-rbac \
  --name "cc-api-github-actions" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUB_ID/resourceGroups/rg-cc-underwriting \
  --sdk-auth
# Update AZURE_CREDENTIALS secret in GitHub with new JSON
```

### App Service returns 503 after deploy

```bash
# Normal during startup — model loading takes 20-30 seconds
# Wait 60 seconds and retry

# Check logs for actual error
az webapp log tail \
  --name cc-underwriting-api \
  --resource-group rg-cc-underwriting
```

### Training crashes with MemoryError

```bash
# Need at least 4 GB RAM
free -h    # Linux — check available memory

# Reduce memory usage in ml/train.py:
# n_estimators=200   (instead of 500)
# num_leaves=31      (instead of 63)
```

### docker-compose build fails with pip errors

```bash
# Try rebuilding without cache
docker-compose build --no-cache api

# Check internet connectivity inside Docker
docker run --rm python:3.11-slim pip install requests
```

---

## 15. Project File Reference

```
cc_underwriting_api/
│
├── app/                    FastAPI application
│   ├── main.py             All 12 routes + startup logic
│   │                       Routes: /, /health, /auth/*, /predict,
│   │                               /predictions, /model/info, /admin/*
│   ├── config.py           Settings from environment variables / .env
│   ├── database.py         SQLAlchemy engine, session factory, get_db()
│   ├── models.py           ORM: users table + prediction_log table
│   ├── schemas.py          Pydantic schemas (validation + Swagger docs)
│   ├── auth.py             bcrypt hashing, JWT sign/verify, Depends() guards
│   └── predict.py          ModelStore singleton, feature engineering mirror,
│                           prob_to_score(), get_risk_band()
│
├── ml/
│   └── train.py            12-stage training pipeline
│                           Stages: load → drop → encode → impute →
│                           engineer → correlate → split → SMOTE →
│                           scale → train → evaluate → save
│
├── models/
│   ├── .gitkeep            Keeps directory in git (model files excluded)
│   └── cc_model_v1.joblib  Generated by train.py (not in git)
│                           Contains: LightGBM, StandardScaler,
│                           171 feature names, 30 LabelEncoders, metrics
│
├── migrations/
│   └── init.sql            Runs once on first PostgreSQL container boot
│
├── tests/
│   └── test_api.py         25 integration tests
│                           Covers: health, auth, predict, pagination,
│                           model info, input validation, admin routes
│
├── .github/workflows/
│   └── deploy.yml          4-job CI/CD pipeline
│                           Job 1: pytest + lint
│                           Job 2: docker build + push to ACR
│                           Job 3: deploy to Azure App Service (main)
│                           Job 4: deploy to Azure Container Instance (staging)
│
├── Dockerfile              Multi-stage build (builder + runtime)
│                           Non-root user (uid 1001)
│                           Docker HEALTHCHECK on /health
│
├── docker-compose.yml      Local dev: API + PostgreSQL with named volume
│                           pgdata volume survives docker-compose down
│
├── docker-compose.prod.yml Production: pulls from ACR, bind-mount for DB,
│                           no exposed DB port, resource limits
│
├── requirements.txt        All deps with pinned versions
├── .env.example            Template — copy to .env
├── .gitignore              Excludes .env, *.joblib, __pycache__
└── README.md               This file
```

---

## Quick Reference Card

```bash
# ── Daily development ────────────────────────────────────────────
source .venv/bin/activate              # activate Python env
docker-compose up -d                   # start API + database
docker-compose logs -f api             # watch logs
docker-compose down                    # stop (data safe)

# ── Rebuild after code change ─────────────────────────────────────
docker-compose build api && docker-compose up -d

# ── Train model ───────────────────────────────────────────────────
python ml/train.py --data data.csv --out models/cc_model_v1.joblib

# ── Test ──────────────────────────────────────────────────────────
pytest tests/ -v

# ── Get JWT token ─────────────────────────────────────────────────
curl -X POST localhost:8000/auth/token \
  -d "username=admin&password=Admin@12345"

# ── Run prediction ─────────────────────────────────────────────────
curl -X POST localhost:8000/predict \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"fico_score": 740, "annual_income": 95000}'

# ── Deploy (automatic via git push) ──────────────────────────────
git add . && git commit -m "change" && git push origin main

# ── Production health check ───────────────────────────────────────
curl https://cc-underwriting-api.azurewebsites.net/health

# ── Production logs ───────────────────────────────────────────────
az webapp log tail --name cc-underwriting-api --resource-group rg-cc-underwriting

# ── Database backup ───────────────────────────────────────────────
docker exec cc_postgres pg_dump -U ccapi cc_underwriting > backup.sql

# ── Wipe and restart database (development only) ─────────────────
docker-compose down -v && docker-compose up -d
```
