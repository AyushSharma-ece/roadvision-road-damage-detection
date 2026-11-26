# Contributing to RoadVision

Thank you for your interest in improving **RoadVision**!

## Development Standards

1. **Modular Architecture:** Keep preprocessing (`src/matlab/preprocessing/`), detection (`src/matlab/detection/`), severity (`src/matlab/severity/`), segmentation (`src/matlab/segmentation/`), video tracking (`src/matlab/video/`), GPS (`src/matlab/gps/`), database (`src/matlab/database/`), reporting (`src/matlab/reporting/`), and App Designer UI (`src/matlab/app/`) cleanly separated.
2. **Configuration Over Hardcoding:** Store tunable parameters, paths, and thresholds in `config/roadvision_config.json` and `config/classes.json`. Never commit machine-specific absolute paths or credentials.
3. **Dataset & License Hygiene:** Never commit raw datasets or large weights to Git. Register new dataset sources and licenses in `docs/DATASET_REGISTER.md`.
4. **Testing Requirement:** All PRs must pass:
   - MATLAB verification suites (`tests/matlab/TestCoreAndPreprocessing.m` and `tests/matlab/TestEngineModules.m`)
   - Python tests (`pytest tests/python/`)
5. **Scientific Integrity:** Never fabricate detection scores, evaluation metrics, or physical dimensions/pothole depth from uncalibrated 2D images.
