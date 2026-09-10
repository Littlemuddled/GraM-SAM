# Data layout

The Phase-2 and Phase-3 entry points use this common, read-only layout:

```text
data/
  spine|Promise12|ISIC2016|BUSD/
    labeled_names_5_select.csv
    labeled_names_10_select.csv
    labeled_names_20_select.csv
    train/
      images/
      masks/
      mask_points.json
    val/
      images/
      masks/
      mask_points.json
```

Active-selection CSV files contain an `image_name` column. Prompt manifests
contain one record per image. Each `prompt_groups` entry stores a dataset class
ID, `(x, y)` coordinates, and binary point labels. The paper configuration
emits one positive point for each predicted non-background class.

Phase 1 consumes a separate CSV manifest with the columns `image_name`,
`uncertainty_path`, and `feature_path`. The two paths may be relative to the
manifest. Uncertainty maps and encoder features are NumPy `.npy` files. TTA
must use 10 invertible augmented views; predictions are inverse-transformed
before computing `U = mean(P) * (1 - mean(P))`.

Do not commit datasets, patient information, checkpoints, or generated caches.
