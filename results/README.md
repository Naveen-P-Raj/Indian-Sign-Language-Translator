# Results

The original training/evaluation notebook reports a test accuracy of approximately **99%** on 4,571 samples from an 80/20 stratified split of 22,854 landmark-feature samples across 36 classes.

The notebook also generated a 36-class confusion matrix. The image is not included because the supplied project archive did not contain the generated plot file.

To regenerate it locally after preparing the feature dataset and model:

```bash
python src/evaluate_model.py
```

The command writes `results/confusion_matrix.png`.
