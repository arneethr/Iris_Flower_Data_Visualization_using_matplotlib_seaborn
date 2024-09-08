# Iris Dataset Visualization 

## Overview
This Python code provides a series of visualizations and exploratory data analysis (EDA) on the famous **Iris dataset**, using `Pandas`, `NumPy`, `Matplotlib`, and `Seaborn`. The visualizations include joint plots, pair plots, box plots, violin plots, KDE plots, heatmaps, histograms, and more. The primary focus is to analyze the relationships between different features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) across the three species of the Iris flower (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`).

### Prerequisites
- **Python** (3.x)
- **Packages**: Install the following packages using `pip` if not already installed:
  ```bash
  pip install numpy pandas matplotlib seaborn
  ```

### Dataset
The Iris dataset used in the code is expected to be in a CSV file named `Iris.csv`. Make sure that the dataset contains the following columns:
- `Id`: Unique identifier for each record (dropped during analysis)
- `SepalLengthCm`: Sepal length in centimeters
- `SepalWidthCm`: Sepal width in centimeters
- `PetalLengthCm`: Petal length in centimeters
- `PetalWidthCm`: Petal width in centimeters
- `Species`: The species of the iris flower

### Code Structure and Description

#### 1. **Loading and Inspecting the Dataset**
```python
iris = pd.read_csv('Iris.csv')  # Load the Iris dataset
iris.drop('Id', axis=1, inplace=True)  # Drop the 'Id' column
iris.info()  # Check for null values and data types
iris['Species'].value_counts()  # Count unique values in the 'Species' column
```

#### 2. **Visualizing Species Count**
```python
sns.countplot(data=iris, x='Species', hue='Species')
plt.show()
```
- Displays a count plot of each species in the dataset.

#### 3. **Joint Plots for Sepal Dimensions**
```python
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=iris, kind='reg')  # With regression line
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=iris, kind='hex')  # Hexbin plot
```
- Joint plots showing relationships between `SepalLengthCm` and `SepalWidthCm`.

#### 4. **Scatter Plot with FacetGrid by Species**
```python
sns.FacetGrid(iris, hue='Species', height=5).map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm').add_legend()
```
- Visualizes the scatter plot of sepal dimensions with color-coded species.

#### 5. **Box Plots and Strip Plots**
```python
sns.boxplot(x='Species', y='PetalLengthCm', data=iris, hue='Species')
sns.stripplot(x='Species', y='SepalLengthCm', data=iris, jitter=True, edgecolor='gray', size=8)
```
- Box and strip plots showing the distribution of sepal and petal lengths across species.

#### 6. **Violin Plots**
```python
sns.violinplot(x='Species', y='SepalLengthCm', data=iris, hue='Species')
```
- A violin plot showing the distribution and density of sepal lengths for each species.

#### 7. **Pair Plots**
```python
sns.pairplot(data=iris, hue='Species')
```
- Pair plots visualize the pairwise relationships between all features in the dataset, separated by species.

#### 8. **Heatmap of Correlations**
```python
iris_numeric = iris.select_dtypes(include=[float, int])
sns.heatmap(iris_numeric.corr(), annot=True, cmap='cubehelix', linewidths=1, linecolor='k')
```
- Heatmap showing the correlation matrix of the numeric columns.

#### 9. **KDE Plot for Iris-setosa**
```python
sub = iris[iris['Species'] == 'Iris-setosa']
sns.kdeplot(x=sub['SepalLengthCm'], y=sub['SepalWidthCm'], fill=True, cmap="plasma")
```
- KDE plot showing the relationship between Sepal Length and Sepal Width for `Iris-setosa`.

#### 10. **Various Other Plots**
- **Histograms**: Distribution of features with histograms.
- **Swarm Plots**: Swarm plots showing individual points of petal lengths.
- **Area Plot**: Area plot showing different dimensions of the Iris flowers.
- **Dist Plot**: Distribution plot showing the Sepal Length distribution.
  
#### 11. **Final Plot**
```python
sns.set_style('darkgrid')
f, axes = plt.subplots(2, 2, figsize=(10, 10))
```
- A 2x2 subplot showing various visualizations like box plots, violin plots, strip plots, and histograms.

### Conclusion
The code provides a comprehensive exploration of the Iris dataset with a variety of visualizations, helping to understand the relationships between different features and how they vary across different species.

### Acknowledgment
The Iris dataset is a well-known dataset in machine learning and I got it from kaggle.com


