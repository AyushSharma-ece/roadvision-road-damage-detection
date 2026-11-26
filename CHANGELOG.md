# Changelog

All notable milestones in **RoadVision** are documented here.

## [1.0.0] - 2026-11-26
### Added
- **Phase 1:** Verified MATLAB R2024a, RTX 4050 GPU, Smart App Control workarounds, dataset licenses, and architecture report.
- **Phase 2:** Reproducible downloader (`download_datasets.py`), integrity/MD5/dHash deduplication & leakage-free splitter (`audit_datasets.py`), and `docs/DATASET_REGISTER.md`.
- **Phase 3:** Centralized JSON config (`roadvision_config.json`), extensible `ClassRegistry`, structured `RoadVisionLogger`, and `.gitignore`.
- **Phase 4:** Pure-MATLAB vectorized 5-stage DIP engine (`applyDenoising`, `applyContrastEnhancement`, `extractEdgesAndMorphology`, `letterboxResize`, `augmentRoadImage`, `comparePreprocessingStages`) and ablation evaluation.
- **Phase 5:** Pascal VOC XML <-> YOLO TXT <-> COCO JSON <-> MATLAB table converters (`convert_annotations.py`, `vocXmlToMatlabTable.m`).
- **Phase 6 & 9:** Trained and evaluated `YOLOv8s` (`18` ep, held-out `test mAP@0.5 = 0.4155`), `YOLOv8n` (`10` ep, `mAP@0.5 = 0.3242`), native MATLAB `SqueezeNet` grid detector (`10` ep), and native MATLAB `Light-UNet` crack segmenter (`dl_unet`, `DeepCrack-test-50` `0.7927` Dice).
- **Phase 7–14:** Dual-engine detector (`RoadVisionDetector.m`), pure-MATLAB bitmap annotation renderer (`annotateDetections.m`), severity engine (`estimateDefectSeverity.m`), modular segmentation (`segmentRoadDamage.m`, `rvBinaryOps.m`, `max_work_dim=800` speedup), Kalman + Hungarian video tracker (`RoadDefectTracker.m`, `processRoadVideo.m`, `13.46 FPS` E2E), OpenCV camera snapshot bridge, EXIF/manual GPS mapping, SQLite DB (`RoadVisionDatabase.m`), ReportLab PDF reports, and 7-tab `RoadVisionApp.m` UI with batch upload (`inspectImageBatch`) and corrected panel styling.
- **Phase 15–17:** Complete `32/32` automated MATLAB + Python test verification, multi-country & stress scenario evaluation (`evaluate_scenarios.py`), `15` visual showcase assets (`docs/assets/`), technical report (`docs/PROJECT_REPORT.md`), and root `README.md`.
