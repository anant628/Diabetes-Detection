# Type 2 Diabetes Detection Demo

This project gives you a simple page to test your saved diabetes prediction model with user input.

## How to run

1. Open PowerShell in this folder.
2. Run:

```powershell
python app.py
```

3. Open `http://127.0.0.1:8000` in your browser.

You can also double-click `run.bat`.

## What it uses

- Your saved model file:
  `C:\Users\Anant\Downloads\AI\project database\pima_best_pipeline.joblib`
- A lightweight Python web server
- A browser form for the 8 model input fields

## Notes

- The page uses your real `.joblib` pipeline for prediction.
- Your model was saved with older scikit-learn objects, so the app includes a small compatibility patch to load it on this machine.
- This is for project/demo use only, not medical advice.
