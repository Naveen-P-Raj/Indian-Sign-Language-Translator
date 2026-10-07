# Dataset

The training image dataset is not included in this repository.

Expected structure:

```text
data/ISL_Static_Data/
├── 0/
├── 1/
├── ...
├── A/
├── B/
└── Z/
```

Each class directory should contain JPG, JPEG, or PNG images.

After placing the dataset here, run:

```bash
python src/extract_features.py
```

The generated landmark features are stored in `data/ISL_Static_2Hand_Features/`.

Do not commit large/private datasets unless you have permission to redistribute them.
