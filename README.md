# Smart Nutrition Recommendation System

AI-powered analysis of lab reports to generate personalized nutrition plans, risk assessments, and actionable health insights.

## Architecture

| Phase | Component | Description |
|-------|-----------|-------------|
| **Phase 1** | Data Ingestion | Upload PDF/Image → OCR (pytesseract + PyMuPDF) → Regex extraction → Validation |
| **Phase 2** | AI/ML Core | RandomForest health classification + LightGBM nutrient prediction + SHAP explainability |
| **Phase 3** | Safety Engine | Clinical rule engine → Critical value detection → Medication interaction checks → Structured summary |
| **LLM** | Gemini | Receives only validated JSON → Generates diet plan, meal plan, lifestyle advice |

## Project Structure

```
project/
├── app.py                      # Flask application (routes & pipeline orchestration)
├── models/
│   ├── health_model.py         # RandomForest health classifier + SHAP
│   └── nutrient_model.py       # LightGBM / GBR nutrient target predictor
├── services/
│   ├── ocr_service.py          # PDF parsing + image OCR
│   ├── extraction_service.py   # Regex-based biomarker extraction
│   ├── validation_service.py   # Value validation + normalization
│   ├── rule_engine.py          # Clinical rules + risk scoring
│   └── llm_service.py          # Google Gemini integration
├── templates/
│   ├── index.html              # Home page
│   ├── upload.html             # Upload form + patient info
│   └── dashboard.html          # Results dashboard
├── static/css/style.css        # Custom styles
├── utils/config.py             # Configuration & constants
├── .env                        # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** installed on your system:
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`
3. **Google Gemini API Key**: Get one at https://makersuite.google.com/app/apikey

## Step-by-Step Setup

### 1. Clone / navigate to the project

```bash
cd smart-nutrition-system
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Edit the `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
SECRET_KEY=a-random-secret-string
```

### 5. Run the application

```bash
python app.py
```

The server starts at **http://localhost:5000**

### 6. Use the application

1. Open http://localhost:5000 in your browser
2. Click **Upload Lab Report**
3. Upload a PDF or image of a lab report
4. Fill in patient info (age, gender, activity level)
5. Click **Analyze Report**
6. View the dashboard with:
   - Biomarker table with status badges
   - Risk score and ML classification
   - SHAP feature importance chart
   - Personalized nutrient targets
   - AI-generated diet plan and weekly meal plan
   - Medication–nutrient interaction warnings
   - Emergency alert banner (if critical values detected)

## API Endpoint

For programmatic access, POST to `/api/analyze`:

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "biomarkers": {
      "hemoglobin": 13.5,
      "glucose": 145,
      "hba1c": 7.2,
      "ldl": 165,
      "hdl": 38,
      "total_cholesterol": 240,
      "triglycerides": 210,
      "vitamin_d": 18,
      "vitamin_b12": 350
    },
    "patient_info": {
      "age": 52,
      "gender": "male",
      "activity_level": "sedentary"
    },
    "medications": ["metformin", "statins"]
  }'
```

## Biomarkers Tracked

| Biomarker | Unit | Normal Range |
|-----------|------|-------------|
| Hemoglobin | g/dL | 12.0 – 17.5 |
| Glucose | mg/dL | 70 – 100 |
| HbA1c | % | 4.0 – 5.6 |
| LDL Cholesterol | mg/dL | 0 – 100 |
| HDL Cholesterol | mg/dL | 40 – 60 |
| Total Cholesterol | mg/dL | 0 – 200 |
| Triglycerides | mg/dL | 0 – 150 |
| Vitamin D | ng/mL | 30 – 100 |
| Vitamin B12 | pg/mL | 200 – 900 |

## Security

- API keys stored in `.env` (excluded from git)
- Uploaded files deleted immediately after processing
- Raw OCR text **never** sent to the LLM — only validated structured JSON
- Input validation on all form fields
- Medical disclaimer displayed on every results page

