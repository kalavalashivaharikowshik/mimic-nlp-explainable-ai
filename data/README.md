# Sample requests

Example bodies for `POST /predict` (synthetic, not real patients).

`features` order: `[age, gender (0 = male, 1 = female), heart_rate (bpm), sbp (mmHg), dbp (mmHg), mbp (mmHg), spo2 (%), creatinine (mg/dL), glucose (mg/dL), hemoglobin (g/dL)]`

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d @data/sample_input.json
```

Never put real MIMIC records in this folder: the PhysioNet data use agreement forbids sharing them.
