import pandas as pd
import numpy as np
from PIL import Image

import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import seaborn as sns

def read_data():
    """
    Reads grayscale image data from a specified directory, flattens each image into a 1D array,
    and stores the pixel data along with corresponding filenames (without extension) in a pandas DataFrame.
    The DataFrame is saved as 'prePCA_data.csv' for further processing (e.g., PCA).
    Prints progress every 1000 images processed.
    Returns:
        pd.DataFrame: DataFrame containing flattened pixel data with filenames as the first column ('objid').
    """
    pixel_data = []
    filenames = []
    i = 0

    directory = 'HIPS2FITS_petro'  # set directory path

    for root, _, files in os.walk(directory): 
        for filename in files:  # loop through files in the current directory
            i = i + 1
            full_filename = os.path.join(root, filename)
            filenames.append(filename[:-4])

            img_gray = Image.open(full_filename).convert("L")
            img_arr = np.array(img_gray)

            flat_pixels = img_arr.flatten()
            pixel_data.append(flat_pixels)

            if i % 1000 == 0:
                print(f"Processed {i} images")


    df = pd.DataFrame(pixel_data)
    df.insert(0, 'objid', filenames)
    df.to_csv('prePCA_data.csv', index=False)
    return df

def PCA_analysis():
    """
    Performs Principal Component Analysis (PCA) on a dataset loaded from 'prePCA_data.csv', retaining enough components to explain 95% of the variance.
    The function:
    - Loads the dataset and separates the 'objid' column (identifiers) from the feature data.
    - Standardizes the feature data.
    - Applies PCA to reduce dimensionality while retaining 95% of the variance.
    - Prints the number of principal components required to reach 95% variance.
    - Creates a new DataFrame containing the principal components and the original identifiers.
    - Saves the resulting DataFrame to 'postPCA_data.csv'.
    Returns:
        None
    """

    # Load CSV
    df = pd.read_csv("prePCA_data.csv", encoding="utf-8", dtype={'objid': str})

    # Separate filenames and pixel data
    filenames = df['objid']
    X = df.drop(columns=['objid'])

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA (retain 95% variance)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Number of components used
    #cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = X_pca.shape[1]
    print(f"Number of components to retain 95% variance: {n_components_95}")

    # Create DataFrame with PCs + filenames
    pca_columns = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df.insert(0, 'objid', filenames)

    # Optional: save to CSV
    pca_df.to_csv("postPCA_data.csv", index=False)



def concatenate_data():
    """
    Reads two CSV files, selects relevant columns, ensures 'objid' is of string type,
    and merges the dataframes on the 'objid' column using an inner join. The merged
    dataframe is saved to 'merged_data.csv' and returned.
    Returns:
        pandas.DataFrame: The merged dataframe containing columns from both input files.
    """
    df1 = pd.read_csv("postPCA_data.csv", encoding="utf-8", dtype={'objid': str})
    df2 = pd.read_csv("ZooSpecPhotoDR19_torradeflot.csv", encoding="utf-8", dtype={'objid': str})

    cols_to_keep = ['objid', 'spiral', 'elliptical']
    df2 = df2[cols_to_keep]

    df1['objid'] = df1['objid'].astype(str)
    df2['objid'] = df2['objid'].astype(str)

    merged_df = pd.merge(df1, df2, on="objid", how="inner")
    merged_df.to_csv("merged_data.csv", index=False)
    return merged_df


def custom_LinearSVC():
    """
    Loads galaxy classification data, preprocesses it, and trains a Linear Support Vector Classifier (LinearSVC)
    to distinguish between spiral and elliptical galaxies. The function performs the following steps:
    1. Loads data from 'merged_data.csv' and filters rows to include only those labeled as either spiral or elliptical.
    2. Splits the data into training and testing sets with stratification to preserve class distribution.
    3. Scales the features using StandardScaler.
    4. Trains a LinearSVC model with balanced class weights.
    5. Evaluates the model on the test set and prints the accuracy score.
    6. Computes and displays the confusion matrix as a heatmap.
    Returns:
        None
    """
    # Final cleaning of the data
    df = pd.read_csv("merged_data.csv")
    filtered_df = df[(df['elliptical'] + df['spiral']) == 1]

    X = filtered_df.drop(columns=['dr7objid', 'elliptical', 'spiral'], errors='ignore')
    y = filtered_df['spiral']

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    svm_model = LinearSVC(max_iter=10000, dual=False, class_weight='balanced', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    y_prediction = svm_model.predict(X_test_scaled)

    # Get the accuracy
    svm_model.score(X_test_scaled, y_test)
    print(svm_model.score(X_test_scaled, y_test))

    cm = confusion_matrix(y_test, y_prediction)

    sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='RdPu',
)

    
    

def pca_plot():
    """
    Performs Principal Component Analysis (PCA) on a preprocessed dataset and visualizes the cumulative explained variance.
    This function:
        - Loads a CSV file named 'prePCA_data.csv' with an 'objid' column.
        - Removes the 'objid' column from the dataset.
        - Standardizes the feature data.
        - Fits a PCA model to the standardized data.
        - Calculates and prints the number of principal components required to retain at least 95% of the variance.
        - Plots the cumulative explained variance as a function of the number of PCA components, highlighting the 95% threshold.
    Returns:
        None
    """
    # Load your CSV
    df = pd.read_csv("prePCA_data.csv", encoding="utf-8", dtype={'objid': str})

    # Remove filename column (and label if present)
    X = df.drop(columns=['objid'], errors='ignore')

    # Standardize data (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA (without specifying components yet)
    pca = PCA()
    pca.fit(X_scaled)

    # Compute cumulative explained variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Find number of components for 95% variance
    n_components_95 = np.argmax(cum_var >= 0.95) + 1
    print(f"Number of components to retain 95% variance: {n_components_95}")

    # (Optional) Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(cum_var, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axvline(x=n_components_95, color='g', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()



