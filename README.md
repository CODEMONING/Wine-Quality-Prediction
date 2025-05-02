# 🍷 Wine Quality Prediction

## 📊 Project Overview
This machine learning project predicts wine quality based on physicochemical properties. Using advanced classification models including Random Forest and Gradient Boosting, this analysis identifies key factors affecting wine quality and builds a robust predictive model to classify wines on a quality scale.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-yellow?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red?style=flat-square)

## 🔍 Dataset
The dataset contains chemical properties of wines, including:

| Feature | Description |
|---------|-------------|
| 🧪 Fixed acidity | The non-volatile acids in wine |
| 🧪 Volatile acidity | The amount of acetic acid in wine |
| 🧪 Citric acid | Adds freshness and flavor to wines |
| 🍯 Residual sugar | The amount of sugar remaining after fermentation |
| 🧂 Chlorides | The amount of salt in the wine |
| 💨 Free sulfur dioxide | Prevents microbial growth and oxidation |
| 💨 Total sulfur dioxide | Sum of free and bound forms of SO2 |
| 🌡️ Density | The density of the wine |
| 📏 pH | Describes how acidic or basic the wine is |
| 🧪 Sulphates | Additive that contributes to SO2 levels |
| 🍸 Alcohol | Percent alcohol content of the wine |
| ⭐ Quality | Target variable (score between 0 and 10) |

## 📁 Project Structure
```
📂 Wine_Quality_Prediction/
┣ 📂 data/
┃ ┗ 📄 winequalityn.csv
┣ 📂 notebooks/
┃ ┗ 📔 wine_quality_prediction.ipynb
┣ 📂 models/
┃ ┗ 📄 final_model.pkl
┣ 📂 images/
┃ ┣ 📊 quality_distribution.png
┃ ┣ 📊 correlation_heatmap.png
┃ ┣ 📊 feature_importance.png
┃ ┣ 📊 alcohol_vs_quality.png
┃ ┣ 📊 volatile_acidity_vs_quality.png
┃ ┣ 📊 confusion_matrix.png
┃ ┗ 📊 model_comparison.png
┣ 📄 requirements.txt
┣ 📄 README.md
┗ 📄 LICENSE
```

## 🔬 Methodology

### 1️⃣ Data Exploration & Preprocessing
- 📊 Analyzed distribution of features and target variable
- 🧹 Handled missing values and outliers
- ⚖️ Scaled features using StandardScaler
- 📈 Explored correlations between different wine properties

### 2️⃣ Feature Engineering & Selection
- 🔍 Identified correlation between features and wine quality
- 📊 Found that alcohol content has the strongest positive correlation with quality
- 🧪 Discovered that volatile acidity negatively impacts wine quality
- 🔬 Analyzed how sulphates and citric acid contribute to better quality scores

### 3️⃣ Model Development
- 🤖 Implemented multiple classification models:
  - 🌲 Random Forest Classifier
  - 🚀 Gradient Boosting Classifier
- 🎛️ Performed hyperparameter tuning using cross-validation
- 🔄 Used 5-fold cross-validation to ensure model robustness

### 4️⃣ Evaluation
- 📏 Compared model performance using accuracy, classification report, and confusion matrix
- 🔑 Analyzed feature importance to understand key quality determinants
- 📊 Evaluated model performance across different quality categories

## 📈 Results
- ✅ Achieved 77.1% accuracy with the best-performing model (Random Forest Classifier)
- 🔍 Discovered that alcohol content, volatile acidity, and sulphates are the most influential factors
- 📊 The model performs best at predicting medium-quality wines (scores 5-6)
- 🔬 Identified that high alcohol content (>11%) combined with low volatile acidity (<0.4) typically indicates higher quality wines

## 💡 Key Insights
- 🍸 Higher alcohol content generally correlates with higher wine quality
- 🧪 Lower volatile acidity is associated with better quality wines
- 🧪 Sulphates content shows positive correlation with wine quality
- 🔬 Total sulfur dioxide tends to be lower in higher quality wines
- 📊 Quality score distribution is imbalanced, with most wines falling in the medium quality range (5-6)

## 🛠️ Technologies Used
- ![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python) Python programming language
- ![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-yellow?style=flat-square&logo=pandas) For data manipulation and analysis
- ![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML_Models-orange?style=flat-square&logo=scikit-learn) For machine learning algorithms and evaluation
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red?style=flat-square) & ![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Viz-teal?style=flat-square) For data visualization
- ![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-blue?style=flat-square&logo=numpy) For numerical operations

## ⚙️ Installation
```bash
# Clone this repository
git clone https://github.com/shashikathi/Wine_Quality_Prediction.git

# Navigate to the project directory
cd Wine_Quality_Prediction

# Install required packages
pip install -r requirements.txt
```

## 🚀 Usage
Open the Jupyter notebook to see the full analysis:
```bash
jupyter notebook notebooks/wine_quality_prediction.ipynb
```

To load the trained model and make predictions:
```python
import pickle

# Load the model
with open('models/final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions (features must be scaled the same way as during training)
predictions = model.predict(scaled_features)
```

## 📊 Analysis Summary

The analysis revealed several important findings about wine quality predictors:

1. **Alcohol Content**: The single strongest predictor of wine quality, with higher alcohol content generally indicating higher quality wines.

2. **Volatile Acidity**: Shows a strong negative correlation with wine quality. Lower volatile acidity (less acetic acid) typically results in better wine quality.

3. **Sulphates**: Higher sulphate levels are associated with higher quality wines, likely due to their antioxidant properties.

4. **Citric Acid**: Moderate positive correlation with quality, contributing to wine freshness.

5. **Prediction Challenges**: The model has more difficulty accurately predicting very high quality (8-9) and very low quality (3-4) wines, likely due to fewer examples of these in the training data.

6. **Model Comparison**: Random Forest Classifier (77.1% accuracy) outperformed Gradient Boosting (76.3% accuracy), possibly due to its better handling of the feature space.

### 📈 Key Visualizations

#### Feature Importance
![Feature Importance](images/feature_importance.png)
*This chart shows the relative importance of each feature in predicting wine quality, with alcohol content, volatile acidity, and sulphates being the most significant predictors.*

#### Correlation Matrix
![Correlation Heatmap](images/correlation_heatmap.png)
*The correlation matrix reveals relationships between different wine properties, highlighting how each property correlates with quality and with other properties.*

#### Alcohol vs. Quality
![Alcohol vs. Quality](images/alcohol_vs_quality.png)
*This visualization demonstrates the positive relationship between alcohol content and wine quality, showing how higher alcohol percentage generally corresponds to higher quality ratings.*

#### Volatile Acidity vs. Quality
![Volatile Acidity vs. Quality](images/volatile_acidity_vs_quality.png)
*This plot illustrates the negative relationship between volatile acidity and wine quality, where lower acidity levels typically result in higher quality scores.*

## 🔮 Future Improvements
- 🔄 Implement more advanced models like XGBoost or neural networks
- 🌐 Deploy the model as a web application for real-time wine quality prediction
- 🌍 Expand the dataset with more wine varieties and regions
- ⚖️ Address class imbalance using techniques like SMOTE or class weights
- 🔬 Explore non-linear relationships between features using polynomial features

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact
- 👨‍💻 GitHub: [shashikathi](https://github.com/shashikathi)
- 🔗 LinkedIn: [shashikathi](http://linkedin.com/in/shashikathi/)
- 📧 Email: shashikathi56@gmail.com

---

### ⭐ If you find this project useful, please consider giving it a star!
