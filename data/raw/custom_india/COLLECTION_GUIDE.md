# Custom Indian Road Image Collection Protocol

To test **RoadVision** on genuine local Indian road conditions without fabricating synthetic images, place real photographs captured on a smartphone or dashcam inside `data/raw/custom_india/` and record their metadata in `manifest_template.csv`.

## 1. Recommended Coverage Matrix (Target: 50–100 Photos)

| Environment / Condition | Suggested Count | Why It Matters |
| :--- | :--- | :--- |
| **Urban asphalt streets** | 15–20 | Shadows, lane paint, traffic clutter, patched tarmac. |
| **Rural / village damaged roads** | 15–20 | Unpaved shoulders, severe alligator cracking (`D20`), deep potholes (`D40`). |
| **State / national highways** | 10–15 | High-speed motion blur, longitudinal (`D00`) and transverse (`D10`) cracks. |
| **Concrete / paver roads** | 5–10 | Expansion joints (must not be confused with `D10` cracks). |
| **Wet / monsoon puddles & damp glare** | 10–15 | Water reflections inside potholes and low surface contrast. |
| **Dusk / shaded / low-light scenes** | 10 | Evaluates CLAHE/gamma enhancement robustness. |
| **Undamaged clean roads (negatives)** | 10–15 | Measures false-positive rate (FP) on healthy pavement. |

## 2. Capture Guidelines

1. **Angles:** Capture both oblique driver-perspective shots (1.0–1.5 m height) and top-down close-ups (60–90 cm).
2. **GPS Metadata:** Enable smartphone camera **Location Tags** when capturing outdoors so RoadVision's GPS module can read real EXIF coordinates automatically.
3. **Privacy & Safety:** Never photograph from moving traffic without a passenger or safe mount; avoid recognizable faces and vehicle license plates when possible.
4. **Leakage Prevention:** Keep custom photos strictly as an **unseen generalization test set** unless split by distinct road corridor/city.
5. **Known Scale Reference (Optional):** For calibrated physical dimension tests in Phase 8, place a ruler or standard reference object beside the defect in a duplicate shot and record the `px_per_cm` in `manifest.csv`.
