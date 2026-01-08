# Data Directory

This directory is for storing datasets used with the HEDPriv framework.

## UCI Heart Disease Dataset

The framework is designed for the UCI Heart Disease Dataset.

### Download Instructions

1. **Official Source**: 
   - Visit: https://archive.ics.uci.edu/ml/datasets/heart+disease
   - Download the processed Cleveland dataset

2. **Kaggle Alternative**:
   - Visit: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
   - Download `heart.csv`

3. **Place the file here**:
   ```
   data/heart.csv
   ```

### Expected Format

The CSV should contain the following columns:

| Column          | Type    | Description                    | Range      |
|-----------------|---------|--------------------------------|------------|
| Age             | Integer | Age in years                   | 29-77      |
| Sex             | Binary  | Gender (1=male, 0=female)      | 0-1        |
| ChestPainType   | Integer | Type of chest pain             | 0-3        |
| RestingBP       | Integer | Resting blood pressure (mm Hg) | 94-200     |
| Cholesterol     | Integer | Serum cholesterol (mg/dl)      | 126-564    |
| FastingBS       | Binary  | Fasting blood sugar > 120 mg/dl| 0-1        |
| RestingECG      | Integer | Resting ECG results            | 0-2        |
| MaxHR           | Integer | Maximum heart rate achieved    | 71-202     |
| ExerciseAngina  | Binary  | Exercise induced angina        | 0-1        |
| Oldpeak         | Float   | ST depression                  | 0.0-6.2    |
| ST_Slope        | Integer | Slope of peak exercise ST      | 0-2        |
| HeartDisease    | Binary  | Target variable                | 0-1        |

### Synthetic Data Generation

If you don't have the real dataset, the framework will automatically generate synthetic data for testing:

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
# This generates synthetic data automatically
data = preprocessor.load_heart_disease_data()
```

### Using Your Own Dataset

To use a different dataset:

1. Ensure it's in CSV format
2. Contains numerical features
3. Has no missing values (or use preprocessing to handle them)
4. Pass the filepath to the preprocessor:

```python
data = preprocessor.load_heart_disease_data('data/your_dataset.csv')
```

## Data Privacy Notice

**IMPORTANT**: This directory is in `.gitignore` to prevent accidental upload of sensitive data.

- Never commit actual patient/medical data to version control
- Use synthetic data for demonstrations
- Follow GDPR/HIPAA compliance guidelines
- Obtain proper consent and ethics approval for real data

## Sample Data Statistics

For the Heart Disease dataset:
- **Samples**: ~1000 patients
- **Features**: 11 features (4 used for encryption)
- **Selected Features**: Age, RestingBP, Cholesterol, MaxHR
- **Target**: Binary classification (heart disease presence)

## File Structure

```
data/
├── README.md          # This file
├── heart.csv          # (Download separately - not in repo)
├── raw/               # (Optional) Raw unprocessed data
└── processed/         # (Optional) Preprocessed data cache
```

## References

1. Detrano, R., et al. (1989). "International application of a new probability algorithm for the diagnosis of coronary artery disease." *American Journal of Cardiology*.
2. UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
3. Dataset Citation: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

---

**Note**: The `.csv` files in this directory are ignored by Git for privacy protection.
