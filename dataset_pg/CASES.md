# PhysioCircuit Synthetic cases（L0）

120 cases total（20 dev / 100 blind test）+ 30 counterfactual pairs。

Cases per stratum and future time points：

| stratum | dev | blind | Total | items (blind test) |
|---|---:|---:|---:|---:|
| simple | 6 | 30 | 36 | 540 |
| conflict | 8 | 40 | 48 | 720 |
| turning | 4 | 20 | 24 | 360 |
| nonmono | 2 | 10 | 12 | 180 |

Total blind-test direction prediction items = 1800（= 100 cases x 6 steps x 3 organs）

## Sample (first blind-test case)

```json
{
  "case_id": "PG-D-021",
  "split": "blind_test",
  "field": "digestive",
  "stratum": "simple",
  "horizon": 6,
  "patient": {
    "age": 85,
    "sex": "M",
    "frailty": "mild",
    "setting": "surgical ward",
    "comorbidities": [
      "rheumatoid arthritis"
    ]
  },
  "presentation": {
    "chief_complaint": "post-prandial epigastric fullness, nausea after fatty meals",
    "duration_days": 11,
    "vital_signs": {
      "hr": 83,
      "map": 83,
      "temp": 38.4,
      "spo2": 97
    }
  },
  "observed_window": [
    {
      "t": 0,
      "note": "day -7",
      "labs": {
        "lipase": 2.43,
        "total bilirubin": 1.62,
        "ALT": 3.14,
        "ALP": 1.13
      }
    },
    {
      "t": 1,
      "note": "day -6",
      "labs": {
        "CRP": 3.0,
        "lipase": 1.97,
        "creatinine": 2.04,
        "ALP": 1.29
      }
    },
    {
      "t": 2,
      "note": "day -5",
      "labs": {
        "total bilirubin": 3.04,
        "creatinine": 1.44,
        "WBC": 0.51,
        "lipase": 2.97
      }
    }
  ],
  "mechanisms": [
    {
      "id": "MC-04",
      "name": "Acid load to the small bowel",
      "source": "stomach",
      "target": "intestine",
      "sign": "+",
      "lag": 1,
      "onset_step": 1,
      "strength": 0.87,
      "evidence": [
        "duodenal erosions",
        "gastric pH < 2"
      ],
      "confounder": "H2 blocker timing"
    }
  ],
  "gold": {
    "steps": [
      {
        "t": 1,
        "direction": {
          "stomach": "stable",
          "intestine": "worsen",
          "gallbladder": "stable"
        },
        "severity": {
          "stomach": 0.144,
          "intestine": 0.36,
          "gallbladder": 0.216
        },
        "phase": "baseline"
      },
      {
        "t": 2,
        "direction": {
          "stomach": "stable",
          "intestine": "worsen",
          "gallbladder": "stable"
        },
        "severity": {
          "stomach": 0.144,
          "intestine": 0.448,
          "gallbladder": 0.216
        },
        "phase": "load_accumulation"
      },
      {
        "t": 3,
        "direction": {
          "stomach": "stable",
          "intestine": "worsen",
          "gallbladder": "stable"
        },
        "severity": {
          "stomach": 0.144,
          "intestine": 0.537,
          "gallbladder": 0.216
        },
        "phase": "load_accumulation"
      },
      {
        "t": 4,
        "direction": {
          "stomach": "stable",
          "intestine": "worsen",
          "gallbladder": "stable"
        },
        "severity": {
          "stomach": 0.144,
          "intestine": 0.625,
          "gallbladder": 0.216
        },
        "phase": "load_accumulation"
      },
      {
        "t": 5,
        "direction": {
          "stomach": "stable",
          "intestine": "worsen",
          "gallbladder": "stable"
        },
        "severity": {
          "stomach": 0.144,
          "intestine": 0.714,
          "gallbladder": 0.21
```