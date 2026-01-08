# Dataset Formatting

The dataset is expected to be formatted in the COCO dataset format.

## Dataset File Structure

The dataset root is organized in a COCO-like hierarchy:

```
dataset_root/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/
│   ├── seq_001/
│   │   ├── img1/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── gt/
│   │       └── gt.txt
│   └── seq_002/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## JSON Data Structure
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "train/seq_001/img1/000001.jpg",
      "height": 720,
      "width": 1280,
      "frame_id": 1,
      "seq_id": "seq_001"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "track_id": 3,
      "bbox": [412.5, 233.1, 87.2, 190.4],
      "area": 16615.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person"}
  ]
}
```

## Tracker Inference
For the outputs in `outputs/<name>/*.txt`, each row is in `MOT-style` format
1. frame_id
2. track_id
3. x (top‑left)
4. y (top‑left)
5. w
6. h
7. score
8. class_id (your code always writes 1)
9. visibility (set to -1 in your code)
