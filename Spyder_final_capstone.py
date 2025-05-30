# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:59:09 2025

@author: nhutd
"""

#pip install cfgrib
#import os
#print(os.path.exists("C:Users/nhutd/OneDrive/Desktop/BCIT/Spring 2025/BABI 9050/Capstone project/Data for project/NASA/fire_archive_M-C61_602405.csv"))

# Probability of fire apporach
## MODIS DATA 
import pandas as pd
import xarray as xr
import numpy as np
#import dask


# Show all columns
pd.set_option('display.max_columns', None)
#pd.set_option('display.expand_frame_repr', False)

# Load data
df_fire = pd.read_csv(r"C:\Users\nhutd\OneDrive\Desktop\BCIT\Spring 2025\BABI 9050\Capstone project\Data for project\NASA\fire_archive_M-C61_602405.csv")

#  Parse date, extract year and month
df_fire['acq_date'] = pd.to_datetime(df_fire['acq_date'])
df_fire['year'] = df_fire['acq_date'].dt.year
df_fire['month'] = df_fire['acq_date'].dt.month


df_fire = df_fire[
    (df_fire['latitude'] >= 48) & (df_fire['latitude'] <= 60) &
    (df_fire['longitude'] >= -139) & (df_fire['longitude'] <= -120)
]

# Create fire_occurred column
df_fire['fire_occurred'] = 1



# Aggregate to monthly, per grid cell (any fire in month = 1)
monthly_fire = df_fire.groupby(['year', 'month', 'latitude', 'longitude'])['fire_occurred'].max().reset_index()

print(monthly_fire.head())
print(monthly_fire.tail())
print(monthly_fire.info())


# Path to your GRIB file
nc_file1 = r"C:\Users\nhutd\OneDrive\Desktop\BCIT\Spring 2025\BABI 9050\Capstone project\Data for project\FWI data\data_0.nc"
nc_file2 = r"C:\Users\nhutd\OneDrive\Desktop\BCIT\Spring 2025\BABI 9050\Capstone project\Data for project\FWI data\data_1.nc"

# Open the datasets
ds0 = xr.open_dataset(nc_file1)
ds1 = xr.open_dataset(nc_file2)

# Normalize time in both datasets
ds0['valid_time'] = ds0.indexes['valid_time'].normalize()
ds1['valid_time'] = ds1.indexes['valid_time'].normalize()

# Merge again
ds_combined = xr.merge([ds0, ds1])

ds_small = ds_combined.sel(
    latitude=slice(60, 48),                 # North to South
    longitude=slice(220, 238)              # ERA5 uses 0–360 longitudes
)

df_climate = ds_small.to_dataframe().reset_index()


# Add year/month columns
df_climate['year'] = pd.to_datetime(df_climate['valid_time']).dt.year
df_climate['month'] = pd.to_datetime(df_climate['valid_time']).dt.month

print(df_climate['d2m'].value_counts())

# Convert longitude from 0–360 to –180–180 if needed, then snap
df_climate['longitude'] = df_climate['longitude'].apply(lambda x: x if x <= 180 else x - 360)

# Set grid size based on climate data
grid_size = 0.25

# Snap function
def snap_to_grid(val, grid_size):
    return round(val / grid_size) * grid_size

# Apply to fire data
monthly_fire['latitude'] = monthly_fire['latitude'].apply(lambda x: snap_to_grid(x, grid_size))
monthly_fire['longitude'] = monthly_fire['longitude'].apply(lambda x: snap_to_grid(x, grid_size))

# View result
print(monthly_fire.head())
print(monthly_fire.tail())
print(df_climate.head())
print(df_climate.tail())


print(monthly_fire.info())
print(df_climate.info())


# Select only the desired columns from the climate data
df_climate_filtered = df_climate[[
    'u10', 'v10', 'd2m', 't2m', 'lai_hv', 'year', 'month', 'latitude', 'longitude'
]]

# Merge 
merged = pd.merge(
    df_climate_filtered,
    monthly_fire[['year', 'month', 'latitude', 'longitude', 'fire_occurred']],
    on=['year', 'month', 'latitude', 'longitude'],
    how='left'
)

# Preview: number of matches after merging
merged_check = pd.merge(
    df_climate_filtered,
    monthly_fire,
    on=["year", "month", "latitude", "longitude"],
    how="inner"
) 

# Count matched records
print(monthly_fire.info())
print(df_climate_filtered.info())
print(monthly_fire.info())
print("Number of matched records:", len(merged_check))


# Fill missing fire_occurred with 0
merged['fire_occurred'] = merged['fire_occurred'].fillna(0).astype(int)



# Checking the merge data
print(merged.tail())
print(merged.info())

print(merged['year'].value_counts())

# data is inconsistence between the fire and non fire

# Option 1 - Downsizing data so the model wont bias nonfire prediction
fire = merged[merged['fire_occurred'] == 1]
non_fire = merged[merged['fire_occurred'] == 0]
non_fire_downsampled = non_fire.sample(n=len(fire)*3, random_state=42)  # e.g. 3:1
balanced = pd.concat([fire, non_fire_downsampled], ignore_index=True).sample(frac=1, random_state=42)
print(balanced['fire_occurred'].value_counts())
print(balanced.info())


# Option 2 - upsizing fire sample using SMOTE (Synthetic Minority Oversampling Technique

'''
How Does SMOTE Work?
The process:
Finds the minority class samples (in your case, where fire_occurred == 1).

For each minority sample, SMOTE picks one or more of its k-nearest neighbors (by default, 5 neighbors) among other minority samples.

For each pair, it creates a new synthetic data point (not just a copy!) by “interpolating” between the two real points.

For example, if you have two fire events with different weather conditions, SMOTE will create a new, fake “fire” event whose values are somewhere between the real ones.

The result:

You get new, plausible-looking fire events with slightly varied features, increasing the number of fire events in your dataset.

This helps your model see more “fire” examples and prevents it from being overwhelmed by the majority (non-fire) class.

'''

from imblearn.over_sampling import SMOTE

# Suppose your merged DataFrame is named 'merged'
# and your target is 'fire_occurred'

# 1. Separate features and target
feature_cols = ['u10', 'v10', 'd2m', 't2m', 'lai_hv', 'year', 'month', 'latitude', 'longitude']
X = merged[feature_cols]
y = merged['fire_occurred']

# 2. Apply SMOTE to upsample minority class
# sampling_strategy=0.3 means: after SMOTE, fires will be 30% of total (adjust as you like)
smote = SMOTE(random_state=42, sampling_strategy=0.4)
X_res, y_res = smote.fit_resample(X, y)

# 3. Combine back into a DataFrame if you want
balanced_smote = pd.DataFrame(X_res, columns=feature_cols)
balanced_smote['fire_occurred'] = y_res

# 4. Check new class balance

print(balanced_smote.info())
print(balanced_smote.tail(70))
print(balanced_smote['fire_occurred'].value_counts())

#print(balanced_smote[balanced_smote['fire_occurred'] == 1])
#print(balanced_smote[balanced_smote['fire_occurred'] == 0])


import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial import cKDTree


# Path to the biomass raster file
tif_path = r"C:\Users\nhutd\OneDrive\Desktop\BCIT\Spring 2025\BABI 9050\Capstone project\Data for project\Biomass\BC_fuel_type_epsg3347.tif"

# === Step 1: Load the TIFF ===

with rasterio.open(tif_path) as src:
    data = src.read(1)
    transform = src.transform
    nodata = src.nodata

# === Step 2: Extract valid pixels ===
rows, cols = np.where(data != nodata)
values = data[rows, cols]
xs, ys = rasterio.transform.xy(transform, rows, cols)


# === Step 1: Create DataFrame from raster coordinates ===
fuel_df = pd.DataFrame({
    "x": xs,
    "y": ys,
    "fuel_type": values
})

# === Step 2: Convert CRS to EPSG:4326 ===
transformer = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
fuel_df["longitude"], fuel_df["latitude"] = transformer.transform(
    fuel_df["x"].values, fuel_df["y"].values
)

# Drop x/y columns
fuel_df.drop(columns=["x", "y"], inplace=True)

# === Step 3: Round lat/lon to match fire data ===
fuel_df["latitude"] = fuel_df["latitude"].round(2)
fuel_df["longitude"] = fuel_df["longitude"].round(2)

# Drop duplicates after rounding
fuel_df = fuel_df.drop_duplicates(subset=["latitude", "longitude"])

# === Step 4: Round fire data to same precision ===
balanced_smote["latitude"] = balanced_smote["latitude"].round(2)
balanced_smote["longitude"] = balanced_smote["longitude"].round(2)

# === Step 5: Merge with fire data ===
final_df = balanced_smote.merge(
    fuel_df,
    on=["latitude", "longitude"],
    how="left"
)

# === Step 6: Fill missing fuel_type using spatial fallback ===
fuel_valid = fuel_df.dropna(subset=["fuel_type"]).copy()
tree = cKDTree(fuel_valid[["latitude", "longitude"]].values)

na_mask = final_df["fuel_type"].isna()
missing_coords = final_df.loc[na_mask, ["latitude", "longitude"]].values

radii = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,9.0]  # degrees

for radius in radii:
    print(f"Trying radius: {radius}°")
    distances, indices = tree.query(missing_coords, distance_upper_bound=radius)

    matched = distances != np.inf
    matched_idx = final_df.loc[na_mask].index[matched]
    matched_fuel_type = fuel_valid.iloc[indices[matched]]["fuel_type"].values

    final_df.loc[matched_idx, "fuel_type"] = matched_fuel_type

    na_mask = final_df["fuel_type"].isna()
    missing_coords = final_df.loc[na_mask, ["latitude", "longitude"]].values

    print(f"Remaining NaNs: {na_mask.sum()}")
    if na_mask.sum() == 0:
        break

# Create windspeed column 
final_df["windspeed"] = (final_df["u10"]**2 + final_df["v10"]**2)**0.5
final_df.drop(columns=["u10", "v10"], inplace=True) 

#Final check the data         
final_df.head()
final_df.info()
#final_df["fuel_type"].isna().sum()
print(final_df['fire_occurred'].value_counts()) 


# Defind X,Y 
X = final_df[['windspeed', 'd2m','t2m', 'lai_hv', 'year', 'month', 'latitude','longitude','fuel_type']]
y = final_df['fire_occurred']


import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier


# Split 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

"""
 Logistic Regression
"""

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict probabilities
logreg_probs = logreg.predict_proba(X_test)[:, 1]
logreg_probs

print("Logistic Regression AUC:", roc_auc_score(y_test, logreg_probs))


# Showing result 

## Define feature names used to create X_test
feature_columns = X.columns.tolist()
## Inverse transform X_test to get original values
X_test_original = scaler.inverse_transform(X_test)
X_test_df = pd.DataFrame(X_test_original, columns=feature_columns)

## 3. Convert y_test to Series for alignment
y_test_series = pd.Series(y_test, name="Actual_Label")

## 4. Predict probabilities
logreg_probs = logreg.predict_proba(X_test)[:, 1]

## 5. Predict binary outcomes using a threshold (e.g., 0.5)
logreg_preds = (logreg_probs >= 0.5).astype(int)

## 6. Build comparison DataFrame using original values
comparison_df = X_test_df.copy()
comparison_df["Predicted_Probability"] = logreg_probs
comparison_df["Predicted_Label"] = logreg_preds
comparison_df["Actual_Label"] = y_test_series.values

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
# === Accuracy, Recall, Precision ===
accuracy = accuracy_score(y_test, logreg_preds)
recall = recall_score(y_test, logreg_preds)
precision = precision_score(y_test, logreg_preds)
auc = roc_auc_score(y_test, logreg_probs)

# === Print the results ===
print(f"Logistic Regression AUC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")

## 7. Show result
print(comparison_df.head(10))


## RMSE
from sklearn.metrics import mean_squared_error
# Calculate RMSE on predicted probabilities vs. actual class labels
rmse = np.sqrt(mean_squared_error(y_test, logreg_probs))
print("Logistic Regression RMSE:", rmse)

"""
 Random Forest
"""

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("Random Forest AUC:", roc_auc_score(y_test, rf_probs))

# SHowing result
# Define feature names that were used to create X_test
feature_columns = X.columns.tolist()  # Ensure X is your original full feature DataFrame

# Convert X_test (NumPy array) to DataFrame
## Inverse transform X_test to get original values
X_test_original = scaler.inverse_transform(X_test)
X_test_df = pd.DataFrame(X_test_original, columns=feature_columns)

# Convert y_test to Series for alignment
y_test_series = pd.Series(y_test, name="Actual_Label")

# Predict binary outcomes using a threshold (e.g., 0.5)
rf_preds = (rf_probs >= 0.5).astype(int)

# Build comparison DataFrame
rf_comparison_df = X_test_df.copy()
rf_comparison_df["Predicted_Probability"] = rf_probs
rf_comparison_df["Predicted_Label"] = rf_preds
rf_comparison_df["Actual_Label"] = y_test_series.values


rf_comparison_df.to_excel("rf_fire_predictions_smote.xlsx", index=False)

# Show result
print(rf_comparison_df.head(10))

# Evaluation 
## RMSE
from sklearn.metrics import mean_squared_error
# Calculate RMSE on predicted probabilities vs. actual class labels
rmse = np.sqrt(mean_squared_error(y_test, rf_probs))
print("Random forest regression RMSE:", rmse)


# Feature_importance
feature_importance = rf.feature_importances_
features = pd.DataFrame(X).columns

# DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (RF)')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

'''
def plot_variable_effect(rf_model, X_ref_df, X_array, feature_name, num_points=100):
    """
    Plots the effect of a single feature on the average predicted fire probability.

    Args:
    - rf_model: Trained RandomForestClassifier
    - X_ref_df: Original unscaled feature DataFrame (for column reference)
    - X_array: The scaled or final input NumPy array used for prediction
    - feature_name: Name of the feature to vary
    - num_points: Number of test points across the feature's range
    """

    # Identify the feature index
    if feature_name not in X_ref_df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame")

    feature_index = list(X_ref_df.columns).index(feature_name)

    # Copy input array for modification
    X_temp = X_array.copy()

    # Get feature value range from unscaled data
    x_min = X_ref_df[feature_name].min()
    x_max = X_ref_df[feature_name].max()
    x_values = np.linspace(x_min, x_max, num_points)

    # Storage for average predictions
    y_probs = []

    # Loop over values to test effect
    for val in x_values:
        X_modified = X_temp.copy()
        X_modified[:, feature_index] = val  # set all rows for this feature

        probs = rf_model.predict_proba(X_modified)[:, 1]
        y_probs.append(np.mean(probs))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_probs, color='darkred')
    plt.xlabel(f'{feature_name}')
    plt.ylabel('Average Predicted Fire Probability')
    plt.title(f'Effect of {feature_name} on Fire Probability')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_variable_effect(
    rf_model=rf, 
    X_ref_df=X,         # your original DataFrame before scaling
    X_array=X_train,    # the scaled NumPy array used in training
    feature_name='t2m', # change this to the variable you want to test
    num_points=100
)
'''


# Confusion mattrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Predict class labels
rf_preds = rf.predict(X_test)

# Get the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()
total = tn + fp + fn + tp


# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()

# 3. Print counts
print(f"\n--- Confusion Matrix ---")
print(f"True Positives (TP): {tp} ({tp/total:.2%})")
print(f"True Negatives (TN): {tn} ({tn/total:.2%})")
print(f"False Positives (FP): {fp} ({fp/total:.2%})")
print(f"False Negatives (FN): {fn} ({fn/total:.2%})")

# Print performance metrics
print("\n--- Performance Metrics ---")
print(f"Accuracy : {accuracy_score(y_test, rf_preds):.2%}")
print(f"Precision: {precision_score(y_test, rf_preds):.2%}")
print(f"Recall   : {recall_score(y_test, rf_preds):.2%}")
print(f"F1 Score : {f1_score(y_test, rf_preds):.2%}")

# Random observation testing

# Define the columns used in your model
feature_columns = X.columns.tolist()  # same features used to train the model

# Generate a random observation based on existing feature ranges
random_obs = {
    col: np.random.uniform(X[col].min(), X[col].max())
    for col in feature_columns
}

# Convert to DataFrame with one row
random_df = pd.DataFrame([random_obs])

# Predict using the trained Random Forest model
pred_prob = rf.predict_proba(random_df)[:, 1][0]
pred_label = rf.predict(random_df)[0]

# Show result
print("🔍 Random Observation:")
print(random_df)
print("\n🔥 Predicted Probability of Fire:", round(pred_prob, 4))
print("🔥 Predicted Label:", pred_label)




# Cross validation
rf_model = RandomForestClassifier(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc_scores = cross_val_score(rf_model, X, y, cv=skf, scoring='roc_auc')

print(f"Random Forest Stratified CV AUC Scores: {cv_auc_scores}")
print(f"Random Forest Average CV AUC: {np.mean(cv_auc_scores)}")

"""
Every fold had a very high AUC (≈ 0.9999).

The average AUC is 0.99992, or 99.99%.

The x-axis is labeled “AUC Score” and shows values like 1.4e-5 + 9.999e-1, which is a scientific notation way of saying:

AUCs are all around 0.9999

There's very little variation between folds

The box represents the interquartile range (IQR) of the 5 AUC scores.

The vertical orange line in the middle is the median AUC.

The whiskers show the min and max AUC values (which are extremely close together here).

What this confirms:
Random Forest model consistently scores near-perfect AUCs (≈ 0.9999) on every fold.
The model is very stable — there's almost no variability between the folds.

"""

plt.figure(figsize=(6, 5))
plt.boxplot(cv_auc_scores, vert=False)
plt.title('Random Forest Stratified 5-Fold CV AUC')
plt.xlabel('AUC Score')
plt.grid(True)
plt.show()


"""
XGBOOST
"""
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
print("XGBoost AUC:", roc_auc_score(y_test, xgb_probs))

feature_columns = X.columns.tolist()  # assuming X is your full feature DataFrame
## Inverse transform X_test to get original values
X_test_original = scaler.inverse_transform(X_test)
X_test_df = pd.DataFrame(X_test_original, columns=feature_columns)

# Convert y_test to Series
y_test_series = pd.Series(y_test, name="Actual_Label")

# Predict labels with threshold
xgb_preds = (xgb_probs >= 0.5).astype(int)

# Build comparison DataFrame
xgb_comparison_df = X_test_df.copy()
xgb_comparison_df["Predicted_Probability"] = xgb_probs
xgb_comparison_df["Predicted_Label"] = xgb_preds
xgb_comparison_df["Actual_Label"] = y_test_series.values

xgb_comparison_df.head()

# Feature importance
feature_importance = xgb.feature_importances_
features = pd.DataFrame(X).columns

# DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.show()

# Evaluation

# Calculate RMSE on predicted probabilities vs. actual class labels
rmse = np.sqrt(mean_squared_error(y_test, xgb_probs))
print("XGBOOST regression RMSE:", rmse)

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
# === Accuracy, Recall, Precision ===
accuracy = accuracy_score(y_test, xgb_preds)
recall = recall_score(y_test, xgb_preds)
precision = precision_score(y_test, xgb_preds)
auc = roc_auc_score(y_test, xgb_preds)

# === Print the results ===
print(f"Logistic Regression AUC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")


# Cross validation 

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring='roc_auc')

print(f"XGBoost Stratified CV AUC Scores: {cv_auc_scores}")
print(f"XGBoost Average CV AUC: {np.mean(cv_auc_scores)}")

plt.figure(figsize=(6, 5))
plt.boxplot(cv_auc_scores, vert=False)
plt.title('XGBoost Stratified 5-Fold CV AUC')
plt.xlabel('AUC Score')

plt.grid(True)
plt.show()


# Impact 
## Temperature data 
bc_temp_path= r"C:\Users\nhutd\OneDrive\Desktop\BCIT\Spring 2025\BABI 9050\Capstone project\Data for project\BC_monthly_temp.csv"

try:
  bc_temp = pd.read_csv(bc_temp_path)
  print("File read successfully.")
except FileNotFoundError:
  print(f"File not found at: {'/content/BC_monthly_temp.csv'}")
except Exception as e:
  print(f"An error occurred while reading the file: {e}")
print(bc_temp.head())

# Drop irrelevant columns with all nulls or flags
columns_to_drop = [col for col in bc_temp.columns if 'Flag' in col or bc_temp[col].isna().all()]
bc_temp = bc_temp.drop(columns=columns_to_drop)

# Drop rows with missing mean temperature (can modify to fillna if preferred)
bc_temp = bc_temp.dropna(subset=['Mean Temp (°C)'])

# Create a datetime column for easier analysis
bc_temp['Date'] = pd.to_datetime(bc_temp[['Year', 'Month']].assign(DAY=1))

# Aggregate temperature by Year or Month

bc_temp = bc_temp.groupby(['Year'])['Mean Temp (°C)'].mean().reset_index()


# Preview
print(bc_temp.tail(50))


# Area burned by month

area_path = r"C:\Users\nhutd\OneDrive\Desktop\BCIT\Spring 2025\BABI 9050\Capstone project\Data for project\Area_burned_by_month.xlsx"
try:
  area_burned = pd.read_excel(area_path)
  print("File read successfully.")
except FileNotFoundError:
  print(f"File not found at: {'/content/Area_burned_by_month.xlsx'}")
except Exception as e:
  print(f"An error occurred while reading the file: {e}")
print(area_burned.head())

# Use the first row as the header
new_header = area_burned.iloc[0]
area_burned = area_burned[1:]
area_burned.columns = new_header

# Forward-fill the 'Jurisdiction' column
area_burned['Jurisdiction'] = area_burned['Jurisdiction'].ffill()

# Remove rows where 'Month' is NaN (they are just spacers)
area_burned = area_burned[area_burned['Month'].notna()]

# Identify year columns (they are float or int)
id_vars = ['Jurisdiction', 'Month', 'Data Qualifier']
value_vars = [col for col in area_burned.columns if isinstance(col, (int, float))]

# Melt the wide format into long format
area_burned = pd.melt(
    area_burned,
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='Year',
    value_name='Total area burned (ha)'
)

# Group to aggregate duplicates, just in case
final_area_burned = area_burned.groupby(
    ['Jurisdiction', 'Year', 'Month']
)['Total area burned (ha)'].sum().reset_index()

final_area_burned = final_area_burned[final_area_burned['Jurisdiction'].isin(['British Columbia'])]


# Drop the 'Jurisdiction' column
final_area_burned = final_area_burned.drop(columns=["Jurisdiction"])
# Drop rows with 'Unspecified' in the Month column
final_area_burned = final_area_burned[final_area_burned["Month"] != "Unspecified"]

# Convert 'Month' from text (e.g., "April") to numeric (e.g., 4)
final_area_burned["Month"] = pd.to_datetime(final_area_burned["Month"], format='%B').dt.month

final_area_burned = final_area_burned.sort_values(by=["Year", "Month"]).reset_index(drop=True)
final_area_burned = final_area_burned.groupby(['Year'])['Total area burned (ha)'].mean().reset_index()

# Preview the cleaned data
print(final_area_burned.head(50))

# Property damage

property_path = r"C:\Users\nhutd\OneDrive\Desktop\BCIT\Spring 2025\BABI 9050\Capstone project\Data for project\Property_losses_from_fires.xlsx"

try:
  property_df = pd.read_excel(property_path)
  print("File read successfully.")
except FileNotFoundError:
  print(f"File not found at: {'/content/Property_losses_from_fires.xlsx'}")
except Exception as e:
  print(f"An error occurred while reading the file: {e}")
print(property_df.head())

# Use the first row as the header
new_header = property_df.iloc[0]
property_df = property_df[1:]
property_df.columns = new_header

# Forward-fill the 'Jurisdiction' column
property_df['Jurisdiction'] = property_df['Jurisdiction'].ffill()

# Remove rows where 'Month' is NaN (they are just spacers)
property_df = property_df[property_df['Protection zone'].notna()]

# Identify year columns (they are float or int)
# Removing 'Data Qualifier' from id_vars as it's causing the KeyError
id_vars = ['Jurisdiction', 'Protection zone']
value_vars = [col for col in property_df.columns if isinstance(col, (int, float))]

# Melt the wide format into long format
property_df = pd.melt(
    property_df,
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='Year',
    value_name='Property loss'
)

# Group to aggregate duplicates, just in case
# Since 'Data Qualifier' is removed, adjust the groupby accordingly
property_df = property_df.groupby(
    ['Jurisdiction', 'Year']  
)['Property loss'].sum().reset_index()

final_property_df = property_df[property_df['Jurisdiction'].isin(['British Columbia'])]


# Drop Jurisdiction
final_property_df = final_property_df.drop(columns=["Jurisdiction"])
# Preview
final_property_df.head(50)



# Merge area burned and property loss on 'Year'
impact_df = pd.merge(final_area_burned, final_property_df, on="Year", how="inner")

# Merge the result with temperature data on 'Year'
impact_df = pd.merge(impact_df, bc_temp, on="Year", how="inner")

# View result
print(impact_df.head(60))

# 1. Drop 'Year' column
corr_df = impact_df.drop(columns=["Year"])

# 2. Compute correlation matrix
corr_matrix = corr_df.corr()

corr_matrix

import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold

# Prepare the DataFrame
df1 = impact_df.copy()  # Replace with your actual DataFrame if needed

# Define y1 (target) and X1 (predictors)
y1 = df1['Property loss']
X1 = df1[['Total area burned (ha)', 'Mean Temp (°C)']]

# Clean the data
X1 = X1.apply(pd.to_numeric, errors='coerce')
y1 = pd.to_numeric(y1, errors='coerce')
X1 = X1.fillna(X1.mean())
y1 = y1.fillna(y1.mean())

# Add constant (intercept) to X1
X1 = sm.add_constant(X1)
X1, y1 = X1.align(y1, join='inner', axis=0)

# Train-test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lowest_mse = float("inf")
best_model = None

for train_index, val_index in kf.split(X1_train):
    X1_fold_train, X1_fold_val = X1_train.iloc[train_index], X1_train.iloc[val_index]
    y1_fold_train, y1_fold_val = y1_train.iloc[train_index], y1_train.iloc[val_index]

    # Fit full model with all predictors
    model = sm.OLS(y1_fold_train, X1_fold_train).fit()
    y1_pred = model.predict(X1_fold_val)
    mse = mean_squared_error(y1_fold_val, y1_pred)

    if mse < lowest_mse:
        lowest_mse = mse
        best_model = model

# Output best model summary
print(best_model.summary())


def plot_variable_effect_ols(model, X_reference, feature_name, num_points=100):
    """
    Plots the effect of changing a feature on the predicted target using an OLS model.

    Args:
        model: Trained statsmodels OLS model.
        X_reference: A DataFrame of the reference inputs (e.g., X1_train).
        feature_name: The name of the feature to vary.
        num_points: Number of simulated points to test across the feature range.
    """
    X_temp = X_reference.copy()
    
    # Generate a range of values for the selected feature
    x_min, x_max = X_temp[feature_name].min(), X_temp[feature_name].max()
    x_values = np.linspace(x_min, x_max, num_points)
    
    y_preds = []
    for val in x_values:
        X_temp[feature_name] = val  # Set all rows of this column to the same value
        preds = model.predict(X_temp)
        y_preds.append(np.mean(preds))  # Average predicted outcome

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_preds, color='navy')
    plt.xlabel(f'{feature_name}')
    plt.ylabel('Average Predicted Property Loss')
    plt.title(f'Effect of {feature_name} on Predicted Property Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_variable_effect_ols(best_model, X1_train, feature_name='Mean Temp (°C)')
plot_variable_effect_ols(best_model, X1_train, feature_name='Total area burned (ha)')
"""
Test 2
"""

final_property_df = final_property_df.rename(columns={"Year": "year"})

# Step 1: Aggregate fire-climate data by year (excluding unused columns)
df_2 = final_df.copy()
df_2 = df_2.drop(columns=["month", "latitude", "longitude", "fire_occurred", "fuel_type"])
df_2 = df_2.groupby("year").mean().reset_index()

# Step 2: Merge with property loss data
impact_df_2 = pd.merge(df_2, final_property_df, on="year", how="inner")

# Step 3: Correlation matrix
corr_df = impact_df_2.drop(columns=["year"])
corr_matrix = corr_df.corr()
print(corr_matrix)

# Step 4: Regression setup
impact_df_2 = impact_df_2.iloc[:-2]  # drop last 2 rows if needed

# Define y2 (target) and X2 (predictors)
y2 = impact_df_2['Property loss']
X2 = impact_df_2.drop(columns=["Property loss"])

# Clean data
X2 = X2.apply(pd.to_numeric, errors='coerce')
y2 = pd.to_numeric(y2, errors='coerce')
X2 = X2.fillna(X2.mean())
y2 = y2.fillna(y2.mean())

# Add constant (intercept)
X2 = sm.add_constant(X2)
X2, y2 = X2.align(y2, join='inner', axis=0)

# Train-test split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lowest_mse = float("inf")
best_model2 = None

for train_index, val_index in kf.split(X2_train):
    X2_fold_train, X2_fold_val = X2_train.iloc[train_index], X2_train.iloc[val_index]
    y2_fold_train, y2_fold_val = y2_train.iloc[train_index], y2_train.iloc[val_index]

    model = sm.OLS(y2_fold_train, X2_fold_train).fit()
    y2_pred = model.predict(X2_fold_val)
    mse = mean_squared_error(y2_fold_val, y2_pred)

    if mse < lowest_mse:
        lowest_mse = mse
        best_model2 = model

# Output best model summary
print(best_model2.summary())


def plot_variable_effect_ols(model, X_reference, feature_name, num_points=100, title=None):
    """
    Plots the effect of changing a feature on the predicted target using an OLS model.

    Args:
        model: Trained statsmodels OLS model.
        X_reference: A DataFrame of the reference inputs (e.g., X1_train).
        feature_name: The name of the feature to vary.
        num_points: Number of simulated points to test across the feature range.
        title: Optional custom title for the plot.
    """
    X_temp = X_reference.copy()
    
    # Generate a range of values for the selected feature
    x_min, x_max = X_temp[feature_name].min(), X_temp[feature_name].max()
    x_values = np.linspace(x_min, x_max, num_points)
    
    y_preds = []
    for val in x_values:
        X_temp[feature_name] = val  # Set all rows of this column to the same value
        preds = model.predict(X_temp)
        y_preds.append(np.mean(preds))  # Average predicted outcome

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_preds, color='navy')
    plt.xlabel('Leaf Area Index')
    plt.ylabel('Average Predicted Property Loss (in Millions)')
    if title:
        plt.title(title)
    else:
        plt.title(f'Effect of {feature_name} on Predicted Property Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_variable_effect_ols(best_model2, X2_train, feature_name='t2m',
                         title='Impact of 2m Air Temperature on Fire-Related Property Loss')

plot_variable_effect_ols(best_model2, X2_train, feature_name='d2m',
                         title='Impact of Dewpoint Temperature on Predicted Loss')

plot_variable_effect_ols(best_model2, X2_train, feature_name='lai_hv',
                         title='Effect of Leaf Area Index on Fire Risk')

plot_variable_effect_ols(best_model2, X2_train, feature_name='windspeed',
                         title='Effect of Wind Speed on Fire-Related Property Loss')


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

# Apply model back to original data 
#Merge 
merged1= merged.copy()

final_merge = merged1.merge(
    fuel_df,
    on=["latitude", "longitude"],
    how="left"
)


# Fill missing fuel_type using spatial fallback 
fuel_valid = fuel_df.dropna(subset=["fuel_type"]).copy()
tree = cKDTree(fuel_valid[["latitude", "longitude"]].values)

na_mask = final_merge["fuel_type"].isna()
missing_coords = final_merge.loc[na_mask, ["latitude", "longitude"]].values

radii = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,9.0]  # degrees

for radius in radii:
    print(f"Trying radius: {radius}°")
    distances, indices = tree.query(missing_coords, distance_upper_bound=radius)

    matched = distances != np.inf
    matched_idx = final_merge.loc[na_mask].index[matched]
    matched_fuel_type = fuel_valid.iloc[indices[matched]]["fuel_type"].values

    final_merge.loc[matched_idx, "fuel_type"] = matched_fuel_type

    na_mask = final_merge["fuel_type"].isna()
    missing_coords = final_merge.loc[na_mask, ["latitude", "longitude"]].values

    print(f"Remaining NaNs: {na_mask.sum()}")
    if na_mask.sum() == 0:
        break

# === 1. Create Features and Target ===
final_merge["windspeed"] = (final_merge["u10"]**2 + final_merge["v10"]**2)**0.5
final_merge.drop(columns=["u10", "v10"], inplace=True)

X5 = final_merge[['windspeed', 'd2m', 't2m', 'lai_hv', 'fuel_type', 'latitude', 'longitude','month','year']]
y5 = final_merge['fire_occurred']

numerical_features = ['windspeed', 'd2m', 't2m', 'lai_hv']
categorical_features = ['fuel_type']

# === 2. Define Preprocessing ===
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# === 3. Transform Features ===
X_processed5 = preprocessor.fit_transform(X5)

# === 4. Train-Test Split (and keep original unprocessed features for display) ===
X_train5, X_test5, y_train5, y_test5 = train_test_split(
    X_processed5, y5, test_size=0.2, random_state=42, stratify=y5
)
X5_train_orig, X5_test_orig, _, _ = train_test_split(
    X5, y5, test_size=0.2, random_state=42, stratify=y5
)

# === 5. Train Model ===
rf5 = RandomForestClassifier(n_estimators=100, random_state=42)
rf5.fit(X_train5, y_train5)

# === 6. Predict ===
rf_probs5 = rf5.predict_proba(X_test5)[:, 1]
rf_preds5 = (rf_probs5 >= 0.5).astype(int)

# === 7. Evaluation Metrics ===
print("Random Forest AUC:", roc_auc_score(y_test5, rf_probs5))
rmse5 = np.sqrt(mean_squared_error(y_test5, rf_probs5))
print("Random Forest RMSE:", rmse5)

# === 8. Feature Importance (using separate transformer to avoid side effects) ===
preprocessor_for_importance = clone(preprocessor)
X_importance = preprocessor_for_importance.fit_transform(X5)
encoded_cat_names5 = preprocessor_for_importance.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names5 = numerical_features + list(encoded_cat_names5)

feature_importance5 = rf5.feature_importances_
importance_df5 = pd.DataFrame({
    'Feature': all_feature_names5,
    'Importance': feature_importance5
}).sort_values(by='Importance', ascending=False)


#importance_df5.to_excel("Random_Forest_Feature_Importance.xlsx", index=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df5)
plt.title('Feature Importance (Random Forest - Final Merge)')
plt.tight_layout()
plt.show()


def plot_variable_effect(rf_model, X_ref_df, X_array, feature_name, num_points=100, title=None,xlabel=None):
    """
    Plot the effect of one feature on the model’s average predicted probability.

    Args:
    - rf_model: trained RandomForestClassifier
    - X_ref_df: original unscaled feature DataFrame
    - X_array: scaled NumPy array version (used to train the model)
    - feature_name: the column to vary
    - num_points: how many points to simulate in the range
    """
    if feature_name not in X_ref_df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")

    # Index of the column to vary
    idx = X_ref_df.columns.get_loc(feature_name)
    X_temp = X_array.copy()

    # Value range (unscaled)
    feature_vals = np.linspace(
        X_ref_df[feature_name].min(),
        X_ref_df[feature_name].max(),
        num_points
    )

    y_probs = []
    for val in feature_vals:
        X_mod = X_temp.copy()
        val_scaled = (val - X_ref_df[feature_name].mean()) / X_ref_df[feature_name].std()
        X_mod[:, idx] = val_scaled
        probs = rf_model.predict_proba(X_mod)[:, 1]
        y_probs.append(np.mean(probs))

    # === Plot ===
    plt.figure(figsize=(5, 5))  # Square plot
    plt.plot(feature_vals, y_probs, color='navy', linewidth=2)  # Dark blue line

    # Clean axis labels and title
    plt.xlabel(xlabel if xlabel else f"{feature_name}", fontsize=10)
    plt.ylabel("Average Predicted Probabilities", fontsize=10)
    plt.title(title if title else f"Effect of {feature_name} on Fire Prediction")

    # Grid and ticks
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # Tight layout
    plt.tight_layout()
    plt.show()
    
plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='t2m',
    title="Temperature Influence on Predicted Fire Risk",
    xlabel="2-metre Air Temperature (K)"
)

plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='lai_hv',
    title="Leaf area Index Influence on Predicted Fire Risk",
    xlabel="Leaf Area Index"
)

plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='d2m',
    title="Dewpoint Influence on Predicted Fire Risk",
    xlabel="Dewpoint Temperature (K)"
)

plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='windspeed',
    title="Windspeed Influence on Predicted Fire Risk",
    xlabel="Windspeed (m/s)"
)


# === 9. Confusion Matrix ===
tn, fp, fn, tp = confusion_matrix(y_test5, rf_preds5).ravel()
total = tn + fp + fn + tp

print(f"\n--- Confusion Matrix ---")
print(f"True Positives (TP): {tp} ({tp / total:.2%})")
print(f"True Negatives (TN): {tn} ({tn / total:.2%})")
print(f"False Positives (FP): {fp} ({fp / total:.2%})")
print(f"False Negatives (FN): {fn} ({fn / total:.2%})")

# === 10. Other Metrics ===
print("\n--- Performance Metrics ---")
print(f"Accuracy : {accuracy_score(y_test5, rf_preds5):.2%}")
print(f"Precision: {precision_score(y_test5, rf_preds5):.2%}")
print(f"Recall   : {recall_score(y_test5, rf_preds5):.2%}")
print(f"F1 Score : {f1_score(y_test5, rf_preds5):.2%}")

# === 11. Prepare Comparison DataFrame (use original features, including raw fuel_type) ===
rf_comparison_df5 = X5_test_orig.copy()
rf_comparison_df5["Predicted_Probability"] = rf_probs5
rf_comparison_df5["Predicted_Label"] = rf_preds5
rf_comparison_df5["Actual_Label"] = y_test5.values

# === 12. View top results ===
print("\nTop 10 Prediction Samples (Original Feature View):")
print(final_merge.info())
print(rf_comparison_df5.info())

# Apply to full data (not just test set)
X_full_processed = preprocessor.transform(X5)
rf_probs_full = rf5.predict_proba(X_full_processed)[:, 1]
rf_preds_full = (rf_probs_full >= 0.5).astype(int)

# Combine with original features
rf_comparison_full = X5.copy()
rf_comparison_full["Predicted_Probability"] = rf_probs_full
rf_comparison_full["Predicted_Label"] = rf_preds_full
rf_comparison_full["Actual_Label"] = y5.values
print(rf_comparison_full.info())
# === 11. Export to Excel ===

rf_comparison_full.iloc[:1000000].to_excel("rf_comparison_sample.xlsx", index=False)
#monthly_fire.to_excel("fire_data.xlsx", index=False)


# === Confusion Matrix for Full Data ===
tn, fp, fn, tp = confusion_matrix(y5, rf_preds_full).ravel()
total_full = tn + fp + fn + tp

print(f"\n--- Confusion Matrix (Full Data) ---")
print(f"True Positives (TP): {tp} ({tp / total_full:.2%})")
print(f"True Negatives (TN): {tn} ({tn / total_full:.2%})")
print(f"False Positives (FP): {fp} ({fp / total_full:.2%})")
print(f"False Negatives (FN): {fn} ({fn / total_full:.2%})")

# === Performance Metrics for Full Data ===
print("\n--- Performance Metrics (Full Data) ---")
print(f"Accuracy : {accuracy_score(y5, rf_preds_full):.2%}")
print(f"Precision: {precision_score(y5, rf_preds_full):.2%}")
print(f"Recall   : {recall_score(y5, rf_preds_full):.2%}")
print(f"F1 Score : {f1_score(y5, rf_preds_full):.2%}")




"""
Stimulation
"""
# === 1. Define simulation parameters ===
n_samples = 10000  # Adjust as needed
future_year = 2025
future_months = [6, 7, 8, 9]  # Simulate for summer season

# Extract ranges from real data
feature_ranges = {
    'windspeed': (final_merge['windspeed'].min(), final_merge['windspeed'].max()),
    'd2m': (final_merge['d2m'].min(), final_merge['d2m'].max()),
    't2m': (final_merge['t2m'].min(), final_merge['t2m'].max()),
    'lai_hv': (final_merge['lai_hv'].min(), final_merge['lai_hv'].max()),
    'latitude': (final_merge['latitude'].min(), final_merge['latitude'].max()),
    'longitude': (final_merge['longitude'].min(), final_merge['longitude'].max())
}

# === 2. Generate simulated input data ===
np.random.seed(42)
simulated_data = pd.DataFrame({
    'windspeed': np.random.uniform(*feature_ranges['windspeed'], n_samples),
    'd2m': np.random.uniform(*feature_ranges['d2m'], n_samples),
    't2m': np.random.uniform(*feature_ranges['t2m'], n_samples),
    'lai_hv': np.random.uniform(*feature_ranges['lai_hv'], n_samples),
    'fuel_type': np.random.choice(final_merge['fuel_type'].dropna().unique(), n_samples),
    'latitude': np.random.uniform(*feature_ranges['latitude'], n_samples),
    'longitude': np.random.uniform(*feature_ranges['longitude'], n_samples),
    'month': np.random.choice(future_months, n_samples),
    'year': future_year
})

# === 3. Apply preprocessing ===
X_sim = preprocessor.transform(simulated_data)

# === 4. Make predictions ===
simulated_data['fire_probability'] = rf5.predict_proba(X_sim)[:, 1]
simulated_data['fire_prediction'] = (simulated_data['fire_probability'] >= 0.5).astype(int)

# === 5. Output sample results ===
print(simulated_data[['month', 'year', 'latitude', 'longitude', 'fire_probability', 'fire_prediction']].head())

# Optionally export
# simulated_data.to_csv("simulated_fire_forecast.csv", index=False)

# === 6. Optional visualization ===
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(simulated_data['fire_probability'], bins=30, kde=True)
plt.title("Fire Probability Distribution (Simulated Future)")
plt.xlabel("Fire Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# === Step 1: Copy and Round Coordinates ===
monthly_fire_rounded = monthly_fire.copy()
rf_comparison_rounded = rf_comparison_df5.copy()


# Round coordinates for consistent merging
monthly_fire_rounded["latitude"] = monthly_fire_rounded["latitude"].round(2)
monthly_fire_rounded["longitude"] = monthly_fire_rounded["longitude"].round(2)

rf_comparison_rounded["latitude"] = rf_comparison_rounded["latitude"].round(2)
rf_comparison_rounded["longitude"] = rf_comparison_rounded["longitude"].round(2)

# === Step 2: Ensure same dtype for merge keys ===
join_cols = ["year", "month", "latitude", "longitude"]
for col in join_cols:
    monthly_fire_rounded[col] = monthly_fire_rounded[col].astype(np.float64)
    rf_comparison_rounded[col] = rf_comparison_rounded[col].astype(np.float64)

# === Step 3: Aggregate duplicates to avoid merge explosion ===
monthly_summary = monthly_fire_rounded.groupby(join_cols, as_index=False)["fire_occurred"].max()
rf_summary = rf_comparison_rounded.groupby(join_cols, as_index=False)[["Predicted_Label", "Predicted_Probability"]].agg({
    "Predicted_Label": "max",
    "Predicted_Probability": "mean"
})

# === Step 4: Merge with right join to retain actual fire locations ===
comparison = pd.merge(
    rf_summary,
    monthly_summary,
    on=join_cols,
    how="right"
)

# Fill missing predictions as 0 (no prediction made by model at those points)
comparison["Predicted_Label"] = comparison["Predicted_Label"].fillna(0).astype(int)
comparison["fire_occurred"] = comparison["fire_occurred"].fillna(0).astype(int)

# === Step 5: Confusion Matrix ===
TP = ((comparison['Predicted_Label'] == 1) & (comparison['fire_occurred'] == 1)).sum()
FN = ((comparison['Predicted_Label'] == 0) & (comparison['fire_occurred'] == 1)).sum()
FP = ((comparison['Predicted_Label'] == 1) & (comparison['fire_occurred'] == 0)).sum()
TN = ((comparison['Predicted_Label'] == 0) & (comparison['fire_occurred'] == 0)).sum()
total = TP + FN + FP + TN

# === Step 6: Output ===
print(f"\n--- Confusion Matrix ---")
print(f" True Positives (TP): {TP} ({TP/total:.2%})")
print(f" True Negatives (TN): {TN} ({TN/total:.2%})")
print(f" False Negatives (FN): {FN} ({FN/total:.2%})")
print(f" False Positives (FP): {FP} ({FP/total:.2%})")


# Optional sanity checks
print("\nMissing values in merged comparison dataframe:", comparison.isna().sum().sum())
print("Number of comparison rows:", comparison.shape[0])



monthly_fire.to_excel("actual_fire.xlsx", index=False)
'''
def plot_variable_effect(rf_model, X_ref_df, X_array, feature_name, num_points=100, title=None,xlabel=None):
    """
    Plot the effect of one feature on the model’s average predicted probability.

    Args:
    - rf_model: trained RandomForestClassifier
    - X_ref_df: original unscaled feature DataFrame
    - X_array: scaled NumPy array version (used to train the model)
    - feature_name: the column to vary
    - num_points: how many points to simulate in the range
    """
    if feature_name not in X_ref_df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")

    # Index of the column to vary
    idx = X_ref_df.columns.get_loc(feature_name)
    X_temp = X_array.copy()

    # Value range (unscaled)
    feature_vals = np.linspace(
        X_ref_df[feature_name].min(),
        X_ref_df[feature_name].max(),
        num_points
    )

    y_probs = []
    for val in feature_vals:
        X_mod = X_temp.copy()
        val_scaled = (val - X_ref_df[feature_name].mean()) / X_ref_df[feature_name].std()
        X_mod[:, idx] = val_scaled
        probs = rf_model.predict_proba(X_mod)[:, 1]
        y_probs.append(np.mean(probs))

    # === Plot ===
    plt.figure(figsize=(5, 5))  # Square plot
    plt.plot(feature_vals, y_probs, color='navy', linewidth=2)  # Dark blue line

    # Clean axis labels and title
    plt.xlabel(xlabel if xlabel else f"{feature_name}", fontsize=10)
    plt.ylabel("Average Predicted Probabilities", fontsize=10)
    plt.title(title if title else f"Effect of {feature_name} on Fire Prediction")

    # Grid and ticks
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # Tight layout
    plt.tight_layout()
    plt.show()
    
plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='t2m',
    title="Temperature Influence on Predicted Fire Risk",
    xlabel="2-metre Air Temperature (K)"
)

plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='lai_hv',
    title="Leaf area Index Influence on Predicted Fire Risk",
    xlabel="Leaf Area Index"
)

plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='d2m',
    title="Dewpoint Influence on Predicted Fire Risk",
    xlabel="Dewpoint Temperature (K)"
)

plot_variable_effect(
    rf_model=rf5,
    X_ref_df=X5,
    X_array=X_train5,
    feature_name='windspeed',
    title="Windspeed Influence on Predicted Fire Risk",
    xlabel="Windspeed (m/s)"
)
'''