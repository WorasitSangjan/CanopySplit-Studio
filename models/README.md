# Pot detector

`pot_detector_yolo26s.pt` is a fine-tuned [Ultralytics YOLO26s model](https://docs.ultralytics.com/models/yolo26/)
for detecting pots and the colour card in top-down photos. It is distributed
under [AGPL-3.0](../LICENSE). The training software is available from
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

| | |
|---|---|
| Classes | `color_pad`, `pot` |
| Training | 88 epochs, best epoch 58 |
| Settings recorded in the checkpoint | Maximum 150 epochs, patience 30, batch size 4, image size 1280, seed 0 |
| Validation | precision 0.93, recall 0.98, mAP50 0.99, mAP50-95 0.92 |
| Dataset | Roboflow `plantphenomics/corn_experiment`, version 8 (CC BY 4.0) |
| Used with | confidence threshold 0.05 (set in `pipeline/detect_pots.py`) |

To use other weights: `python run_pipeline.py ... --weights path/to/best.pt`.
