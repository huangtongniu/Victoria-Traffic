## Accident Severity Prediction and Risk Scenario Identification Using Machine Learning

### Research Questions

1. **Can we predict accident severity using machine learning models?**

2. **Can clustering techniques identify high-risk accident scenarios?**

### Data Sources

The dataset used in this project is from the [Victoria road crash data - Dataset - Victorian Government Data Directory](https://discover.data.vic.gov.au/dataset/victoria-road-crash-data).

This data has been consolidated from **Victoria Police reports** and **hospital injury information**, then validated and enriched to provide a comprehensive and detailed view of road crashes and injuries across Victoria. It includes information such as:

- Time and location of crashes
- Environmental and road conditions at the time of the crash
- Crash types (e.g., collisions, rollovers)
- Road user types (e.g., drivers, pedestrians, cyclists)
- Other relevant crash attributes

This dataset is valuable for road safety analysis, risk modeling, and policy development.

### Project Overview

Our project consists of four main stages:

#### 1. Data Preprocessing

We begin by loading three primary datasets:

```python
vehicle_df = pd.read_csv('filtered_vehicle.csv')
accident_df = pd.read_csv('accident.csv')
person_df = pd.read_csv('person.csv')
```

1. Missing Value Handling and Encoding

Using the notebook `fillandcode.ipynb`, we perform:

- Missing value imputation: All missing values are filled appropriately to ensure data completeness.
- Categorical encoding: Selected categorical variables are encoded for model compatibility.
- Feature creation: Additional features are derived where necessary to enhance the dataset.

After this step, the processed datasets are saved in the format:

`output_vehicle.csv` , `output_accident.csv` , `output_person.csv`

---

2. Feature Selection and Aggregation

Using the notebook `combine.ipynb`:

- Select important features from the cleaned datasets.
- Aggregate relevant fields, output as `output_vehicle_agg.csv` , `output_accident_agg.csv` , `output_person_agg.csv`
- Merge the datasets using `ACCIDENT_NO` as the primary key to form a consolidated accident-level table. This results in the following final merged dataset: `final_aggregated_data.csv`

#### 2. Correlation Analysis
1.Using mutual information to the evaluate the amount of sharing information through features and target features SEVERITY. 

2.Choosing features- takes causal reasoning and spurious correlations problems into account.

3.Got the Top20 features which have high MI scores and excluded the impact of confounding variables.

**How to run the code:**
1.Open 'Correlationanalysis.ipynb'
2.Change the path of data to `final_aggregated_data.csv` in the folder into your computer path
3.Run the code 


#### 3. Supervised Learning

1. Using 5/10/20 features ranked by correlation of severity to train KNN, RandomFroest, XGBoost models. And select the optimal feature number which is 20 through a tradeoff between accuracy and recall.

2. Use the 20 features to fine tune models' hyperparameters by grid search using macro recall as the metrics.

3. Finally, with the optimal hyperparameters found in the last step comparing through overall accuracy and recall and Class 0 (fatal accident) recall which giving the optimal model is Random Forest.

**How to run the code:**

Install the libraries used in `severity_prediction.ipynb`

Run it in jupyter lab with the `final_aggregated_data.csv`in the same folder

#### 4. Clustering

1. Standardization: All features were standardized using StandardScaler to ensure equal contribution.

2. Elbow Method: We used the Elbow Method to determine the optimal number of clusters. An elbow was observed at K = 3, so we performed KMeans clustering with 3 clusters.

3. PCA Visualization: To visualize the cluster structure, we used Principal Component Analysis (PCA) to reduce the high-dimensional feature space to 2 components.

4. Crash Frequency: Each PCA coordinate was rounded and used to compute crash frequencies (CRASH_COUNTS). These were visualized in scatter plots and box plots.

5. Find Cluster Characteristics
**How to run the code:**
Run Clustering.ipynb with `final_aggregated_data.csv` in the same directory.



This framework allows us to address both predictive and descriptive insights into traffic accidents, helping in both **severity forecasting** and **risk scenario identification**.