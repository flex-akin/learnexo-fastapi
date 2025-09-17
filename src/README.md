
# LearNEXO Notebook → FastAPI

This folder contains files generated from your notebook to help you serve it as a FastAPI app.

## Files
- `notebook_code.py` — code auto-extracted from **LearNEXO_Recommndeir.ipynb** (all Python cells concatenated).
- `main.py` — a FastAPI server that imports `notebook_code` and exposes:
  - `GET /health` — health check
  - `GET /introspect` — lists exported attributes and shows the auto-detected function
  - `POST /predict` — calls your detected function with the request payload

## How detection works
- We try to find a primary function among: `predict`, `recommend`, `get_recommendations`, `infer`, `serve`, `main`.
- If none is found, the first defined function is used, or you can edit `main.py` and set `fn_name` manually.

## Running locally

```bash
# (Optional) create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Start the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open: http://127.0.0.1:8000/docs

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"sample": "value"}}'
```

If your notebook expects a list of records, send:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"x": 1}, {"x": 2}]}'
```

## Custom wiring
If `/introspect` shows the wrong function, open `main.py` and replace `fn_name` with the correct callable from `notebook_code`.

---

> Tip: Keep your heavy model loads at module import time in `notebook_code.py` (outside the function) so they load once when the server starts.
