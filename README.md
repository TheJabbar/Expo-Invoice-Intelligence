# Invoice Intelligence: Human-In-The-Loop OCR System

Production-ready invoice extraction system with human-in-the-loop learning that extracts 6 key fields from invoices (vendor, invoice no, date, tax, total, debit account) with confidence-based automation gating.

## Architecture

```mermaid
graph LR
    A[User Uploads Invoice] --> B[FastAPI Backend]
    B --> C[PP-OCRv4 OCR]
    C --> D[Rule-Based Extractor]
    D --> E[Confidence Calculator]
    E --> F{Confidence ≥ 0.75?}
    F -->|No| G[Gradio UI<br>Manual Correction]
    F -->|Yes| H[Auto-Post to Accounting]
    G --> I[Feedback Stored in PostgreSQL]
    I --> J[MLflow Retraining Pipeline]
    J --> K[Fine-tuned PP-OCRv4<br>v1.1]
    K --> C
    L[MinIO] -.->|Stores invoices| B
    M[Airflow] -.->|Triggers daily retrain|
```

## Features

- **OCR Processing**: PP-OCRv4 engine for accurate text recognition
- **Field Extraction**: Rule-based extraction of 6 key invoice fields
- **Confidence Calculation**: Multi-factor confidence scoring system
- **Confidence Gating**: Automation threshold at 0.75 confidence
- **Human-in-the-Loop**: Manual correction interface for low-confidence predictions
- **Safety Quarantine**: 72-hour validation period before retraining
- **Continuous Learning**: Daily retraining pipeline with MLflow
- **Audit Trail**: Complete history of predictions and corrections
- **Scalable Architecture**: Containerized services with Docker Compose

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Package Manager | UV | 0.2.0+ |
| Backend | FastAPI + Uvicorn | 0.109+ / 0.27+ |
| OCR Engine | PP-OCRv4 (PaddleOCR) | 2.7.0.3+ |
| Frontend | Gradio | 4.24+ |
| Storage | MinIO | RELEASE.2024-01-16T15-27-37Z |
| Database | PostgreSQL | 16 |
| ML Pipeline | MLflow | 2.11+ |
| Orchestration | Airflow | 2.8.1 (LocalExecutor) |
| Testing | pytest | 8.0+ |

## Quick Start

### Prerequisites

- Docker and Docker Compose (v2.23+)
- UV package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd invoice-intelligence
```

2. Install dependencies:
```bash
uv sync
```

3. Start all services:
```bash
docker-compose up --build -d
```

4. Wait for services to be healthy:
```bash
docker-compose ps
```

### Access Services

- **API**: http://localhost:8000/docs
- **UI**: http://localhost:7860
- **MinIO Console**: http://localhost:9001 (admin/adminadmin)
- **PostgreSQL**: localhost:5432 (admin/admin)

## Usage

1. Upload an invoice (PDF/JPG/PNG) to the Gradio UI
2. The system will automatically extract fields using OCR
3. If confidence is below 0.75, manual corrections are required
4. Submit corrections to improve the model
5. Validated corrections will be used for daily retraining after 72 hours

## Confidence Scoring System

The system uses a multi-factor confidence calculation:

1. **OCR Confidence**: Average confidence of OCR words matching the field
2. **Pattern Strength**: Confidence based on regex pattern matching
3. **Completeness**: Confidence based on field completeness and format
4. **Consistency**: Cross-field consistency validation

Each field receives a confidence score, and an overall confidence is calculated using harmonic mean to penalize low-confidence fields.

## Data Flow

1. **Ingestion**: Invoices uploaded via API stored in MinIO
2. **Processing**: OCR and field extraction with confidence scoring
3. **Gating**: High-confidence results auto-approved, low-confidence sent for review
4. **Correction**: Manual corrections collected via UI
5. **Validation**: Corrections quarantined for 72 hours before training eligibility
6. **Retraining**: Daily Airflow job trains new model with validated corrections
7. **Deployment**: New model version deployed for next inference cycle

## Safety Measures

- **72-Hour Quarantine**: Corrections must wait 72 hours before training eligibility
- **Format Validation**: Basic format checks on extracted values
- **Outlier Detection**: Statistical validation against historical data
- **Confidence Gap Check**: Prevents learning from user errors on high-confidence predictions

## Testing

Run the test suite:

```bash
uv run pytest tests/ -v
```

## Project Structure

```
invoice-intelligence/
├── pyproject.toml               # UV project configuration
├── docker-compose.yml           # Modern Compose (no version key)
├── Dockerfile.api               # UV-based FastAPI image
├── Dockerfile.ui                # UV-based Gradio image
├── init_db.sql                  # PostgreSQL schema
├── api/
│   ├── main.py                  # FastAPI endpoints
│   ├── db.py                    # DB connection
│   └── services/
│       ├── ocr.py               # PP-OCRv4 wrapper
│       ├── extractor.py         # Rule-based field extraction
│       └── confidence.py        # Confidence calculator
├── ui/
│   └── app.py                   # Gradio correction interface
├── ml/
│   ├── train.py                 # PP-OCRv4 fine-tuning (simulated)
│   └── safety.py                # Correction validation
├── airflow/
│   └── dags/
│       └── retrain.py           # Daily retraining DAG
├── tests/
│   ├── test_extractor.py        # Field extraction tests
│   └── test_safety.py           # Correction validation tests
├── sample_invoices/             # Sample invoice files
└── README.md                    # Setup instructions
```

## Monitoring

- **API Health**: `/health` endpoint
- **MLflow Tracking**: Monitor model performance and training metrics
- **PostgreSQL Logs**: Query performance and data integrity
- **Airflow UI**: Monitor retraining pipeline status

## Production Considerations

- Scale API workers based on load
- Configure external MLflow server for model registry
- Set up monitoring and alerting for critical services
- Implement backup strategies for PostgreSQL and MinIO
- Secure service communications with TLS
- Fine-tune confidence thresholds based on business requirements