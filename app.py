import __main__
import json
import os
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(
    r"C:\Users\Anant\Downloads\AI\project database\pima_best_pipeline.joblib"
)
HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8000"))


def zeros_to_nan(frame):
    """Convert biologically impossible zero values to NaN for imputation."""
    if not isinstance(frame, pd.DataFrame):
        return frame

    cleaned = frame.copy()
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for column in zero_as_missing:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def add_interactions(frame):
    """Recreate the same engineered features used when the model was trained."""
    if not isinstance(frame, pd.DataFrame):
        return frame

    featured = frame.copy()
    if {"Age", "BMI"}.issubset(featured.columns):
        featured["Age_BMI"] = featured["Age"] * featured["BMI"]
    if {"Glucose", "BMI"}.issubset(featured.columns):
        featured["Glucose_BMI"] = featured["Glucose"] * featured["BMI"]
    if {"Pregnancies", "Age"}.issubset(featured.columns):
        age_denominator = featured["Age"].replace(0, 1)
        featured["Preg_Age_Ratio"] = featured["Pregnancies"] / age_denominator
    return featured


__main__.zeros_to_nan = zeros_to_nan
__main__.add_interactions = add_interactions


def patch_legacy_model(model_bundle):
    """Patch attributes that changed in newer scikit-learn versions."""
    pipeline = model_bundle["pipeline"]

    if not hasattr(pipeline, "transform_input"):
        pipeline.transform_input = None

    for calibrated in getattr(pipeline, "calibrated_classifiers_", []):
        estimator = calibrated.estimator
        if not hasattr(estimator, "transform_input"):
            estimator.transform_input = None

        for _, step in getattr(estimator, "steps", []):
            if step.__class__.__name__ == "SimpleImputer" and not hasattr(step, "_fill_dtype"):
                step._fill_dtype = getattr(step, "_fit_dtype", np.float64)
            if hasattr(step, "n_jobs"):
                step.n_jobs = 1

    return pipeline


class DiabetesPredictor:
    def __init__(self, model_path):
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="sklearn",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="sklearn",
        )

        self.model_bundle = joblib.load(model_path)
        self.model = patch_legacy_model(self.model_bundle)
        self.features = self.model_bundle["features"]
        self.threshold = float(self.model_bundle.get("operating_threshold", 0.5))
        self.model_name = self.model_bundle.get("best_model_name", "Saved Pipeline")
        self.notes = self.model_bundle.get("notes", "")

    def predict(self, payload):
        frame = pd.DataFrame([payload])[self.features]
        probabilities = self.model.predict_proba(frame)[0]
        positive_probability = float(probabilities[1])
        predicted_positive = positive_probability >= self.threshold
        predicted_class = 1 if predicted_positive else 0

        return {
            "prediction": predicted_class,
            "label": "Higher diabetes risk" if predicted_positive else "Lower diabetes risk",
            "positive_probability": positive_probability,
            "negative_probability": float(probabilities[0]),
            "threshold": self.threshold,
            "model_name": self.model_name,
            "notes": self.notes,
        }


PREDICTOR = DiabetesPredictor(MODEL_PATH)


FORM_FIELDS = [
    {
        "name": "Pregnancies",
        "label": "Pregnancies",
        "type": "number",
        "step": "1",
        "min": "0",
        "value": "2",
        "help": "Number of pregnancies.",
    },
    {
        "name": "Glucose",
        "label": "Glucose",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "value": "138",
        "help": "Plasma glucose concentration.",
    },
    {
        "name": "BloodPressure",
        "label": "Blood Pressure",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "value": "72",
        "help": "Diastolic blood pressure (mm Hg).",
    },
    {
        "name": "SkinThickness",
        "label": "Skin Thickness",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "value": "35",
        "help": "Triceps skin fold thickness (mm).",
    },
    {
        "name": "Insulin",
        "label": "Insulin",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "value": "0",
        "help": "2-hour serum insulin (mu U/ml). Use 0 if unavailable.",
    },
    {
        "name": "BMI",
        "label": "BMI",
        "type": "number",
        "step": "0.1",
        "min": "0",
        "value": "33.6",
        "help": "Body mass index.",
    },
    {
        "name": "DiabetesPedigreeFunction",
        "label": "Diabetes Pedigree Function",
        "type": "number",
        "step": "0.001",
        "min": "0",
        "value": "0.627",
        "help": "Family-history based diabetes score.",
    },
    {
        "name": "Age",
        "label": "Age",
        "type": "number",
        "step": "1",
        "min": "1",
        "value": "47",
        "help": "Age in years.",
    },
]


def render_field(field):
    return f"""
    <label class="field">
      <span>{field["label"]}</span>
      <input
        name="{field["name"]}"
        type="{field["type"]}"
        step="{field["step"]}"
        min="{field["min"]}"
        value="{field["value"]}"
        required
      />
      <small>{field["help"]}</small>
    </label>
    """


HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Type 2 Diabetes Test Page</title>
  <link rel="stylesheet" href="/styles.css" />
</head>
<body>
  <main class="page">
    <section class="hero">
      <p class="eyebrow">AI/ML Project Demo</p>
      <h1>Type 2 Diabetes Detection</h1>
      <p class="intro">
        Test your saved model by entering the patient values below. The result is
        based on your trained pipeline, not a mock prediction.
      </p>
      <div class="meta">
        <div>
          <strong>Model</strong>
          <span>{PREDICTOR.model_name}</span>
        </div>
        <div>
          <strong>Decision Threshold</strong>
          <span>{PREDICTOR.threshold:.2f}</span>
        </div>
      </div>
    </section>

    <section class="panel">
      <form id="prediction-form" class="form-grid">
        {''.join(render_field(field) for field in FORM_FIELDS)}
        <div class="actions">
          <button type="submit">Run Prediction</button>
          <button type="button" id="reset-btn" class="secondary">Reset Sample Values</button>
        </div>
      </form>

      <article id="result" class="result hidden" aria-live="polite">
        <p class="status">Waiting for input...</p>
      </article>
    </section>

    <section class="footer-note">
      <p>
        This page is for project demonstration only and should not be used as
        medical advice or diagnosis.
      </p>
    </section>
  </main>

  <script>
    const form = document.getElementById("prediction-form");
    const result = document.getElementById("result");
    const resetBtn = document.getElementById("reset-btn");

    const sampleValues = {json.dumps({field["name"]: field["value"] for field in FORM_FIELDS})};

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      result.classList.remove("hidden");
      result.innerHTML = '<p class="status">Running your model...</p>';

      const formData = new FormData(form);
      const payload = Object.fromEntries(
        Array.from(formData.entries()).map(([key, value]) => [key, Number(value)])
      );

      try {{
        const response = await fetch("/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});

        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || "Prediction failed.");
        }}

        const riskPercent = (data.positive_probability * 100).toFixed(2);
        const safePercent = (data.negative_probability * 100).toFixed(2);
        const cardClass = data.prediction === 1 ? "result danger" : "result safe";

        result.className = cardClass;
        result.innerHTML = `
          <p class="badge">${{data.label}}</p>
          <h2>${{riskPercent}}% predicted probability</h2>
          <p class="summary">
            Lower-risk probability: <strong>${{safePercent}}%</strong><br />
            Model threshold used: <strong>${{Number(data.threshold).toFixed(2)}}</strong>
          </p>
          <p class="note">${{data.notes || "Prediction generated from the saved diabetes pipeline."}}</p>
        `;
      }} catch (error) {{
        result.className = "result error";
        result.innerHTML = `<p class="badge">Error</p><p>${{error.message}}</p>`;
      }}
    }});

    resetBtn.addEventListener("click", () => {{
      Object.entries(sampleValues).forEach(([name, value]) => {{
        const input = form.elements.namedItem(name);
        if (input) {{
          input.value = value;
        }}
      }});
      result.className = "result hidden";
      result.innerHTML = '<p class="status">Waiting for input...</p>';
    }});
  </script>
</body>
</html>
"""


CSS = """
:root {
  --bg: #f6efe6;
  --panel: rgba(255, 253, 249, 0.9);
  --ink: #1e2430;
  --muted: #5a6475;
  --accent: #bf5a36;
  --accent-2: #2f7d6d;
  --danger: #b64242;
  --border: rgba(30, 36, 48, 0.1);
  --shadow: 0 24px 60px rgba(73, 52, 37, 0.16);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(191, 90, 54, 0.18), transparent 30%),
    radial-gradient(circle at top right, rgba(47, 125, 109, 0.18), transparent 28%),
    linear-gradient(180deg, #f9f1e7 0%, #f3e9df 100%);
  min-height: 100vh;
}

.page {
  width: min(1100px, calc(100% - 32px));
  margin: 0 auto;
  padding: 40px 0 56px;
}

.hero {
  padding: 24px 0 18px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.78rem;
  color: var(--accent);
  margin-bottom: 10px;
  font-weight: 700;
}

h1 {
  margin: 0;
  font-size: clamp(2.2rem, 4vw, 4rem);
  line-height: 1;
}

.intro {
  max-width: 700px;
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.7;
}

.meta {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 24px;
}

.meta div,
.panel,
.result {
  background: var(--panel);
  border: 1px solid var(--border);
  backdrop-filter: blur(12px);
  box-shadow: var(--shadow);
  border-radius: 22px;
}

.meta div {
  padding: 14px 18px;
  min-width: 200px;
}

.meta strong,
.field span {
  display: block;
  margin-bottom: 6px;
}

.meta span {
  color: var(--muted);
}

.panel {
  padding: 24px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 18px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.field small {
  color: var(--muted);
  min-height: 36px;
}

input {
  border: 1px solid rgba(30, 36, 48, 0.14);
  border-radius: 14px;
  padding: 14px 16px;
  font-size: 1rem;
  background: #fffdf9;
}

input:focus {
  outline: 2px solid rgba(191, 90, 54, 0.26);
  border-color: var(--accent);
}

.actions {
  grid-column: 1 / -1;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 4px;
}

button {
  border: 0;
  border-radius: 999px;
  padding: 14px 22px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), #da8454);
  color: white;
}

button.secondary {
  background: white;
  color: var(--ink);
  border: 1px solid var(--border);
}

.result {
  margin-top: 22px;
  padding: 24px;
}

.result.hidden {
  display: none;
}

.result.safe {
  border-color: rgba(47, 125, 109, 0.2);
}

.result.danger {
  border-color: rgba(182, 66, 66, 0.24);
}

.result.error {
  border-color: rgba(182, 66, 66, 0.24);
}

.badge {
  display: inline-block;
  margin: 0 0 10px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(30, 36, 48, 0.08);
  font-size: 0.86rem;
  font-weight: 700;
}

.result.safe .badge {
  color: var(--accent-2);
  background: rgba(47, 125, 109, 0.1);
}

.result.danger .badge,
.result.error .badge {
  color: var(--danger);
  background: rgba(182, 66, 66, 0.1);
}

.result h2 {
  margin: 0 0 12px;
  font-size: clamp(1.8rem, 3vw, 2.6rem);
}

.summary,
.note,
.status,
.footer-note p {
  color: var(--muted);
  line-height: 1.7;
}

.footer-note {
  padding-top: 18px;
}

@media (max-width: 640px) {
  .page {
    width: min(100% - 18px, 1100px);
    padding-top: 24px;
  }

  .panel {
    padding: 18px;
  }

  .actions {
    flex-direction: column;
  }

  button {
    width: 100%;
  }
}
"""


class AppHandler(BaseHTTPRequestHandler):
    def _send(self, status_code, body, content_type):
        encoded = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        if self.path == "/styles.css":
            self._send(200, CSS, "text/css; charset=utf-8")
            return
        self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, json.dumps({"error": "Not found"}), "application/json")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            sanitized = {}
            for feature in PREDICTOR.features:
                if feature not in payload:
                    raise ValueError(f"Missing field: {feature}")
                sanitized[feature] = float(payload[feature])

            result = PREDICTOR.predict(sanitized)
            self._send(200, json.dumps(result), "application/json")
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}), "application/json")

    def log_message(self, format, *args):
        return


def main():
    print(f"Diabetes demo running at http://{HOST}:{PORT}")
    print(f"Using model: {MODEL_PATH}")
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
