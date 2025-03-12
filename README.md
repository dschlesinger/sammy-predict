# Breast Mammography Laterality Prediction

This script predicts the view and laterality of unlabeled mammography images.

## Requirements

Install dependencies using:

```sh
pip install -r requirements.txt
```

## Usage

Run the script with:

```sh
python main.py
```

## Configuration

Variables to configure in [main.py](https://github.com/dschlesinger/sammy-predict/blob/master/main.py#L60).

- `IMG_FOLDER`: Path to the **folder** or **list** containing images (Dicom, JPEG, PNG, WebP).
- `MODEL_PATH`: Path to the trained model file (`SAMMY.pt`).
- `DATA_CSV`: Path to the CSV file containing image **labels**, `CC` or `MLO`.
- `IMAGE_PATH_COL`: Column name in `DATA_CSV` that associates images with entries.
- `IMAGE_COL_FIND`: Function to match image file names to `IMAGE_PATH_COL` entries.
- `VIEW_COL`: Column name in `DATA_CSV` that contains the view labels.
- `MODE`: Set to `predict` to generate predictions or `evaluate` to print statistics on selected dataset.
- `EVAL_METRICS`: List of metrics to be evaluated, (default Accuracy, AUC, Crossentropy Loss).

## Image Processing

Images are resized to 224x224, converted to grayscale, and normalized [0, 1] before being fed into the model.

## Dataset Handling

Two dataset loading strategies are implemented:

1. **EagerLoader**: Loads all images into memory at once (faster inference but high memory usage).
2. **LazyLoader**: Loads images on demand (lower memory usage but slower inference).

## Model (Torch)

A convolutional neural network (CNN) is used to classify images.

## Output

2 classes, [1,0] for `CC` and [0,1] for `MLO`

- When running in `evaluate` mode, classification accuracy, AUC, and cross-entropy loss are displayed. You can add metrics
- When running in `predict` mode, predictions are saved to `predictions.csv` containing:
  - `File`: Image file name
  - `View`: Predicted view (`CC` or `MLO`)
  - `Laterality`: Predicted laterality (`L` or `R`)

## Datasets Used

Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)

Moreira, I. C., Amaral, I., Domingues, I., Cardoso, A., Cardoso, M. J., & Cardoso, J. S. (2012). INbreast: Toward a full-field digital mammographic database [Data set]. Academic Radiology, 19(2), 236–248. [https://doi.org/10.1016/j.acra.2011.09.014](https://doi.org/10.1016/j.acra.2011.09.014)

## Name (Acronym)

### SAMMY

**S** - Sequential  
**A** - Automated  
**M** - Mammography  
**M** - Metadata  
**Y** - Yielder  

## Contact

- Denali Schlesinger, dsch28@bu.edu
- Yuanheng Mao, yuanhengm@gmail.com
