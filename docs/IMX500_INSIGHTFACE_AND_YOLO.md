# IMX500 Face Pipeline Tutorial: InsightFace on Host, YOLO on Sensor

## Overview

This tutorial explains how to build an offline face pipeline with Sony IMX500, Ultralytics YOLO, and InsightFace. The recommended design is to use IMX500 only for **face detection** and to run InsightFace on the host device for alignment, embeddings, and identity matching, because the documented [Ultralytics IMX500 export workflow](https://docs.ultralytics.com/integrations/sony-imx500), Sony's [IMX500 deployment guidance](https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter), and the [InsightFace Python package documentation](https://github.com/deepinsight/insightface/blob/master/python-package/README.md) all point toward a detector-on-sensor and recognition-on-host architecture rather than a full face-recognition pipeline running directly on the sensor.[web:88][web:502]

The practical result is a split architecture:
- IMX500 detects faces on the camera.
- The host crops those faces and runs InsightFace recognition locally.
- All required weights can be preloaded so the whole system works offline.

## Architecture

The cleanest deployment model is:

| Component | Role | Why |
|---|---|---|
| IMX500 | Face detection | The public export path is built around compact edge object detection models, as described in the [Ultralytics IMX500 export guide](https://docs.ultralytics.com/integrations/sony-imx500). [web:88] |
| Host machine, such as a Mac or Raspberry Pi | Face alignment, embeddings, recognition | [InsightFace](https://github.com/deepinsight/insightface/blob/master/python-package/README.md) provides the recognition stack and can run fully offline once model packs are downloaded. [web:502] |
| YOLO detector | One-class `face` detector | This matches the IMX500 object detection workflow well. [web:88] |
| InsightFace `buffalo_l` | Recognition pipeline | It bundles detection, landmarks, recognition, and auxiliary models in one pack. [web:502] |

This division is easier to reproduce and more robust than trying to run a full ArcFace-style identity model directly on the sensor, because Sony IMX500 deployment focuses on compact exported models while InsightFace is designed to provide the host-side face analysis stack.[web:88][web:502]

## InsightFace setup

### Why `buffalo_l`

`buffalo_l` is the best default host-side pack when recognition quality matters more than model size. The [InsightFace Python package README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md) documents it as a packaged bundle that includes SCRFD-10GF for face detection, a ResNet50@WebFace600K recognition model, landmark models, and gender/age heads.[web:502]

That matters because it gives a complete local recognition stack: detect a face, align it, generate an embedding, and compare that embedding against enrolled identities. In other words, it reduces the amount of model assembly you have to do yourself.[web:502]

### Offline preload

InsightFace stores model packs under `~/.insightface/models/`, and the package otherwise downloads them automatically on first use if they are missing, as described in the [InsightFace README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md). For offline deployment, download the pack while online from the [InsightFace releases page](https://github.com/deepinsight/insightface/releases), extract it under `~/.insightface/models/buffalo_l/`, verify the ONNX files are present, and then initialize the model locally.[web:502][web:503]

```text
~/.insightface/models/buffalo_l/
```

```python
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
```

## YOLO detector choice

The IMX500 export flow is much more naturally aligned with object detection than face-recognition embeddings, so the practical approach is to train YOLO as a one-class detector for `face`. Ultralytics documents IMX export for YOLO with `format="imx"`, and community examples show packaging the export output into `network.rpk` for Raspberry Pi AI Camera deployment in the [Ultralytics IMX500 export guide](https://docs.ultralytics.com/integrations/sony-imx500) and related [community discussion](https://github.com/orgs/ultralytics/discussions/17496).[web:88][web:109]

A nano model such as `yolo11n.pt` is the safest starting point because smaller models are easier to train, easier to export, and a better fit for edge deployment constraints. The tutorial below uses `yolo11n.pt` for compatibility with the documented IMX500 flow.[web:88]

## Training data

### Recommended dataset

The best main dataset for face detection training is **WIDER FACE**. The benchmark paper, [WIDER FACE: A Face Detection Benchmark](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf), describes it as 32,203 images with 393,703 annotated faces and strong variation in scale, pose, and occlusion, which makes it a strong base dataset for robust face detection.[web:512]

That variety is useful for IMX500 because edge cameras often see small, partially occluded, or off-angle faces rather than clean portrait images. A detector trained only on easy frontal data will usually transfer poorly to real deployment conditions.[web:512]

### YOLO format

For Ultralytics training, labels should be converted to standard YOLO text format, where each line is:

```text
class_id center_x center_y width height
```

and the coordinates are normalized by image width and height, following the [Ultralytics object detection dataset documentation](https://docs.ultralytics.com/datasets/detect). The dataset should define one class only.[web:474]

```yaml
path: /absolute/path/to/datasets/faces
train: images/train
val: images/val

names:
  0: face
```

### Practical data strategy

A good staged approach is:
- Start with a smaller YOLO-ready face dataset if you want to validate the pipeline quickly.
- Move to WIDER FACE for the real detector training set.
- Add images from the target deployment environment later to improve robustness for your own camera, lighting, and background conditions.[web:512]

## Apple Silicon training setup

Ultralytics documents PyTorch MPS support on Apple Silicon, which makes an M4 Pro a practical local training machine for iteration and export in the broader [Ultralytics export documentation](https://docs.ultralytics.com/modes/export). It is not a replacement for a high-end CUDA GPU, but it is very usable for compact YOLO training and experimentation.[web:118]

A safe installation flow is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U ultralytics
python -c "import torch; print(torch.backends.mps.is_available())"
```

That final command confirms that PyTorch can see the Metal backend.

## Training and export workflow

The overall workflow is:

1. Prepare `face.yaml` and the YOLO-format training labels using the [Ultralytics dataset format guide](https://docs.ultralytics.com/datasets/detect). [web:474]
2. Train a compact YOLO detector on Apple Silicon using `device="mps"`. [web:118]
3. Resume from `last.pt` if training stops before completion, as shown in the tutorial script derived from the uploaded summary file. [file:543]
4. Export the best checkpoint with `format="imx"` using the [Ultralytics IMX500 export guide](https://docs.ultralytics.com/integrations/sony-imx500). [web:88]
5. Package `packerOut.zip` into `network.rpk` on the Raspberry Pi side using Sony's [IMX500 converter and packaging workflow](https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter). [web:88]

### Training and export script

The following script combines safe Apple Silicon defaults, automatic resume, and IMX export in one place:

```python
#!/usr/bin/env python3
from pathlib import Path
import json
import platform
import sys
import torch
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"
DATA_YAML = "face.yaml"

TRAIN_IMGSZ = 640
EXPORT_IMGSZ = 320

EPOCHS = 80
BATCH = 8
PATIENCE = 15
WORKERS = 0

PROJECT = "runs_face"
RUN_NAME = "m4pro_face"

AMP = False
CACHE = False
DETERMINISTIC = True
CLOSE_MOSAIC = 10
COS_LR = False


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sanity_check():
    print(json.dumps({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }, indent=2))

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")


def get_run_dir() -> Path:
    return Path(PROJECT) / RUN_NAME


def get_last_checkpoint() -> Path:
    return get_run_dir() / "weights" / "last.pt"


def get_best_checkpoint() -> Path:
    return get_run_dir() / "weights" / "best.pt"


def train_model(device: str):
    last_ckpt = get_last_checkpoint()

    if last_ckpt.exists():
        print(f"Resuming from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
        return results

    print(f"Starting new training run from pretrained weights: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=DATA_YAML,
        imgsz=TRAIN_IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        workers=WORKERS,
        device=device,
        project=PROJECT,
        name=RUN_NAME,
        pretrained=True,
        amp=AMP,
        cache=CACHE,
        deterministic=DETERMINISTIC,
        cos_lr=COS_LR,
        close_mosaic=CLOSE_MOSAIC,
        verbose=True,
    )
    return results


def export_imx():
    best_path = get_best_checkpoint()
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

    model = YOLO(str(best_path))
    exported = model.export(
        format="imx",
        data=DATA_YAML,
        imgsz=EXPORT_IMGSZ,
    )
    return best_path, exported


def main():
    sanity_check()
    device = choose_device()
    print(f"Using device: {device}")

    print("\n=== TRAIN OR RESUME ===")
    train_model(device)

    print("\n=== EXPORT IMX ===")
    best_path, exported = export_imx()

    print("\n=== DONE ===")
    print(f"Run directory: {get_run_dir()}")
    print(f"Last checkpoint: {get_last_checkpoint()}")
    print(f"Best checkpoint: {best_path}")
    print(f"Export result: {exported}")
    print("\nNext step on Raspberry Pi:")
    print("  imx500-package -i packerOut.zip -o rpk_output")


if __name__ == "__main__":
    main()
```

These defaults are conservative for Apple Silicon because `batch=8`, `workers=0`, and `amp=False` usually trade some speed for better stability on MPS, and they match the practical structure of the uploaded tutorial file.[file:543]

## IMX500 YOLO training data and subset extraction

### Why the subset matters

IMX500 export relies on quantization, and quantization needs a representative validation dataset to collect activation statistics before the final deployable model is generated. In practice, using the full validation set can cause the export process to be killed because memory usage can spike during statistics collection or immediately after quantization parameter calculation, so a smaller but representative subset is often the most reliable way to make the export complete, as reflected in the [Ultralytics IMX500 export guide](https://docs.ultralytics.com/integrations/sony-imx500), Sony's [IMX500 converter manual](https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter), and the related [Ultralytics discussion thread](https://github.com/orgs/ultralytics/discussions/17496).[web:88][web:109]

For a YOLO dataset, the subset must preserve the expected structure so the exporter can still resolve labels correctly. That means the reduced dataset should still contain matching `val/images/...` and `val/labels/...` entries, with one label file for each selected image, following the [Ultralytics dataset documentation](https://docs.ultralytics.com/datasets/detect) and [YOLODataset API reference](https://docs.ultralytics.com/reference/data/dataset).[web:474][web:473]

The `labels.cache` file should normally not be copied from the full dataset. It is safer to let Ultralytics rebuild the cache for the reduced dataset, because the cache depends on the actual contents and paths of the subset, which is consistent with how the [YOLODataset API](https://docs.ultralytics.com/reference/data/dataset) handles dataset loading and caching.[web:473]

### Reproducible subset workflow

Use this process:

1. Train the YOLO detector on the full training dataset.
2. Keep the full validation set for evaluation.
3. Create a smaller validation subset specifically for IMX500 export calibration.
4. Preserve `val/images` and matching `val/labels` in the reduced dataset.
5. Let Ultralytics rebuild the cache automatically.
6. Start export with 64 or 128 samples first.
7. Increase the subset size only if export succeeds and accuracy is still not sufficient, in line with Sony's [representative data guidance](https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter).[web:88]

### Subset extraction script

The following script creates a valid YOLO validation subset by randomly selecting image and label pairs from `val/images` and `val/labels`, copying both sides of the pair, and deleting stale cache files.

```python
#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_image_label_pairs(images_dir: Path, labels_dir: Path):
    pairs = []
    for img in images_dir.rglob('*'):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(images_dir)
        label = (labels_dir / rel).with_suffix('.txt')
        if label.exists():
            pairs.append((img, label, rel))
    return pairs


def remove_cache_files(root: Path):
    for cache_file in root.rglob('*.cache'):
        cache_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Create a smaller YOLO validation subset for IMX500 export, copying matching images and labels.'
    )
    parser.add_argument('source_root', help='Source dataset root containing val/images and val/labels')
    parser.add_argument('output_root', help='Output dataset root where val/images and val/labels will be created')
    parser.add_argument('--count', type=int, default=300, help='Number of image/label pairs to copy (default: 300)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--flat', action='store_true', help='Copy selected files into flat val/images and val/labels folders')
    args = parser.parse_args()

    src_root = Path(args.source_root)
    dst_root = Path(args.output_root)
    src_images = src_root / 'val' / 'images'
    src_labels = src_root / 'val' / 'labels'
    dst_images = dst_root / 'val' / 'images'
    dst_labels = dst_root / 'val' / 'labels'

    if not src_images.is_dir():
        raise SystemExit(f'Missing source images directory: {src_images}')
    if not src_labels.is_dir():
        raise SystemExit(f'Missing source labels directory: {src_labels}')

    pairs = collect_image_label_pairs(src_images, src_labels)
    if not pairs:
        raise SystemExit('No matching image/label pairs found in val/images and val/labels')

    rng = random.Random(args.seed)
    sample_size = min(args.count, len(pairs))
    selected = rng.sample(pairs, sample_size)

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for img, label, rel in selected:
        if args.flat:
            img_target = dst_images / img.name
            label_target = dst_labels / (img.stem + '.txt')
            i = 1
            while img_target.exists() or label_target.exists():
                img_target = dst_images / f'{img.stem}_{i}{img.suffix}'
                label_target = dst_labels / f'{img.stem}_{i}.txt'
                i += 1
        else:
            img_target = dst_images / rel
            label_target = (dst_labels / rel).with_suffix('.txt')
            img_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, img_target)
        shutil.copy2(label, label_target)

    remove_cache_files(dst_root / 'val')

    print(f'Selected {sample_size} matching image/label pairs from {len(pairs)} available pairs.')
    print(f'Validation subset written to: {dst_root.resolve()}')
    print('Copied: val/images and val/labels')
    print('Note: labels.cache is not copied; Ultralytics should regenerate it for the new subset.')


if __name__ == '__main__':
    main()
```

### Example command

```bash
python make_imx_val_subset_v2.py \
  /workspace/datasets/widerface \
  /workspace/datasets/widerface_imx_subset \
  --count 128
```

That command creates a reduced but valid YOLO validation set with matching images and labels under `val/images` and `val/labels`, ready for calibration-oriented export experiments.

## Export and packaging

After training, export the best checkpoint with:

```python
model.export(format="imx", data=DATA_YAML, imgsz=320)
```

Ultralytics documents IMX export with `format="imx"`, and the packaging step commonly shown for Raspberry Pi AI Camera deployment is part of the [Sony IMX500 converter workflow](https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter). That produces the `network.rpk` package used by the Raspberry Pi AI Camera workflow.[web:88]

## Offline deployment checklist

Use this checklist before final deployment:
- Download and preload `buffalo_l` into `~/.insightface/models/buffalo_l/` while online using the [InsightFace releases page](https://github.com/deepinsight/insightface/releases). [web:503]
- Verify the ONNX files are present locally using the [InsightFace Python package documentation](https://github.com/deepinsight/insightface/blob/master/python-package/README.md). [web:502]
- Download the starting YOLO weights before switching to offline mode. [web:88]
- Train and export the detector on the M4 Pro. [web:118][web:88]
- Package `packerOut.zip` into `network.rpk` on the Raspberry Pi side using the [Sony converter workflow](https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter). [web:88]
- Run detection on IMX500 and recognition on the host, fully offline once models are cached locally. [web:88][web:502]

## Final recommendation

The most reproducible solution is to let IMX500 do fast face detection and let InsightFace `buffalo_l` handle offline recognition on the host. Training a one-class YOLO detector on WIDER FACE, then exporting it with a carefully selected validation subset, gives a workflow that is both practical to reproduce and aligned with the documented IMX500 deployment path.[web:88][web:502][web:512]

## References

- Ultralytics, [Sony IMX500 Export for Ultralytics YOLO11](https://docs.ultralytics.com/integrations/sony-imx500)
- Ultralytics, [Model Export with Ultralytics YOLO](https://docs.ultralytics.com/modes/export)
- Ultralytics, [Object Detection Datasets Overview](https://docs.ultralytics.com/datasets/detect)
- Ultralytics, [YOLODataset API Reference](https://docs.ultralytics.com/reference/data/dataset)
- Sony AITRIOS, [IMX500 Converter User Manual](https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter)
- InsightFace, [Python Package README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md)
- InsightFace, [Releases](https://github.com/deepinsight/insightface/releases)
- Yang et al., [WIDER FACE: A Face Detection Benchmark](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf)
- Ultralytics Community, [IMX500 export discussion](https://github.com/orgs/ultralytics/discussions/17496)
- Source file used for rewrite context: attached file `imx500_insightface_yolo_summary_rewritten.md`
