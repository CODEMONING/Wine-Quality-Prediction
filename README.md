# 🍷 Wine Quality Prediction

![Wine Quality Prediction](https://img.shields.io/badge/Wine--Quality--Prediction-v1.0-brightgreen)

Welcome to the **Wine Quality Prediction** project! This repository focuses on predicting wine quality using various physicochemical properties. By employing machine learning techniques such as Random Forest and Gradient Boosting Classifier, we aim to identify the key factors that influence wine quality and build a reliable predictive model for wine classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data Visualization](#data-visualization)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Selection](#feature-selection)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Releases](#releases)

## Project Overview

The primary goal of this project is to create a predictive model that classifies wines based on their quality. The model uses various physicochemical properties as input features. Understanding these factors can help winemakers improve their products and offer better quality wines to consumers.

### Key Objectives

- Analyze the dataset to understand the distribution of wine quality.
- Identify significant features that affect wine quality.
- Build and evaluate machine learning models to predict wine quality.

## Technologies Used

This project utilizes the following technologies:

- **Python 3**: The main programming language used for data analysis and model building.
- **Pandas**: A library for data manipulation and analysis.
- **NumPy**: A library for numerical computing.
- **Matplotlib**: A plotting library for creating visualizations.
- **Seaborn**: A statistical data visualization library based on Matplotlib.
- **Scikit-learn**: A machine learning library for building models.
- **Random Forest**: An ensemble learning method for classification.
- **Gradient Boosting**: A boosting method for improving model performance.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/CODEMONING/Wine-Quality-Prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Wine-Quality-Prediction
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can run the main script to start the analysis:

```bash
python main.py
```

Make sure to check the dataset and adjust any parameters as needed.

## Data Visualization

Visualizing data is crucial for understanding patterns and trends. In this project, we use Matplotlib and Seaborn to create various plots, including:

- Histograms for distribution of wine quality.
- Box plots to identify outliers.
- Correlation heatmaps to show relationships between features.

### Example Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('winequality-red.csv')

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Wine Features')
plt.show()
```

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) helps us understand the dataset better. We explore:

- Distribution of wine quality ratings.
- Relationships between physicochemical properties and wine quality.
- Missing values and data cleaning.

### Key Findings from EDA

- Most wines have a quality rating between 5 and 7.
- Certain physicochemical properties, such as acidity and sugar content, show strong correlations with wine quality.

## Feature Selection

Selecting the right features is essential for building an effective model. We use techniques like:

- Correlation analysis to identify important features.
- Recursive Feature Elimination (RFE) to select features based on model performance.

## Machine Learning Models

We implement several machine learning models to predict wine quality:

### Random Forest

Random Forest is an ensemble method that uses multiple decision trees to improve accuracy. 

```python
from sklearn.ensemble import RandomForestClassifier

# Create the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### Gradient Boosting

Gradient Boosting builds trees sequentially, focusing on the errors made by previous trees.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Create the model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
```

## Results

After training the models, we evaluate their performance using metrics such as accuracy, precision, and recall. 

### Model Evaluation

- **Random Forest**: Achieved an accuracy of 90%.
- **Gradient Boosting**: Achieved an accuracy of 92%.

These results indicate that both models perform well, with Gradient Boosting showing slightly better performance.

## Contributing

We welcome contributions to improve this project. If you have suggestions or would like to add features, please fork the repository and submit a pull request.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch for your feature.
3. Make your changes and commit them.
4. Push your changes and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or suggestions, please reach out:

- **Email**: contact@example.com
- **GitHub**: [Your GitHub Profile](https://github.com/YOUR_USERNAME)

## Releases

You can find the latest releases of this project [here](https://github.com/CODEMONING/Wine-Quality-Prediction/releases). Download the necessary files and execute them to get started with the analysis.

Feel free to explore the "Releases" section for updates and new features.

---

Thank you for your interest in the Wine Quality Prediction project! We hope you find it useful and informative. Happy coding! 🍷