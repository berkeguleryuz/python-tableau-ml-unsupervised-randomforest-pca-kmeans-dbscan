# Berke || Unsupervised
# 1. Random Forest proximity analysis
# 2. PCA reconstruction error
# 3. K-means clustering distances
# 4. DBSCAN

# Author: Berke Guleryuz
# Date: January 2025

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from tabpy.tabpy_tools.client import Client
import matplotlib.pyplot as plt
import seaborn as sns

class EnsembleFraudDetector:
    """An ensemble unsupervised fraud detection model."""
    
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_samples=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.pca_model = PCA(
            n_components=0.95,
            random_state=random_state
        )
        
        self.kmeans_model = KMeans(
            n_clusters=5,
            random_state=random_state
        )
        
        self.dbscan_model = DBSCAN(
            eps=0.5,
            min_samples=5,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """Preprocess and scale the data."""
        print("Starting data preprocessing...")
        
        if isinstance(df, pd.DataFrame):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(f"\nNumeric features: {', '.join(numeric_cols)}")
            
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            print(f"\nCategorical features (will be excluded): {', '.join(categorical_cols)}")

            X = df[numeric_cols].copy()
        else:
            X = df
        
        if len(X) == 0 or X.size == 0:
            raise ValueError("No numeric features found in input data")
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nPreprocessed {len(X)} transactions with {X.shape[1]} features")
        return X_scaled
    
    def train_and_predict(self, X):
        """Train the ensemble model and return anomaly scores and cluster labels."""
        print("Training ensemble model...")

        print("Computing Random Forest proximity scores...")
        rf_scores = self._get_rf_scores(X)

        print("Computing PCA reconstruction scores...")
        pca_scores = self._get_pca_scores(X)
        
        print("Computing K-means distance scores...")
        kmeans_scores, kmeans_labels = self._get_kmeans_scores(X)
        
        print("Computing DBSCAN scores...")
        dbscan_scores = self._get_dbscan_scores(X)
        
        ensemble_scores = (rf_scores + pca_scores + kmeans_scores + dbscan_scores) / 4
        
        silhouette_avg = silhouette_score(X, kmeans_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        unique_labels, counts = np.unique(kmeans_labels, return_counts=True)
        print("\nCluster Sizes:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(kmeans_labels)) * 100
            print(f"Cluster {label}: {count} ({percentage:.2f}%)")
        
        return ensemble_scores, kmeans_labels
    
    def _get_rf_scores(self, X):
        """Get anomaly scores using Random Forest proximity."""

        labels = np.zeros(X.shape[0])
        labels[:int(X.shape[0] * self.contamination)] = 1
        np.random.shuffle(labels)

        self.rf_model.fit(X, labels)
        importances = self.rf_model.feature_importances_

        weighted_distances = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            distances = np.sqrt(np.sum((X - X[i]) ** 2 * importances, axis=1))
            weighted_distances[i] = np.mean(np.sort(distances)[1:6])
        
        return (weighted_distances - np.min(weighted_distances)) / (np.max(weighted_distances) - np.min(weighted_distances))
    
    def _get_pca_scores(self, X):
        """Get anomaly scores using PCA reconstruction error."""
        self.pca_model.fit(X)
        X_transformed = self.pca_model.transform(X)
        X_reconstructed = self.pca_model.inverse_transform(X_transformed)
        reconstruction_errors = np.sqrt(np.sum((X - X_reconstructed) ** 2, axis=1))
        return (reconstruction_errors - np.min(reconstruction_errors)) / (np.max(reconstruction_errors) - np.min(reconstruction_errors))
    
    def _get_kmeans_scores(self, X):
        """Get anomaly scores using K-means clustering."""
        self.kmeans_model.fit(X)
        distances = self.kmeans_model.transform(X)
        min_distances = np.min(distances, axis=1)
        
        print("\nK-means Clustering Results:")
        labels = self.kmeans_model.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(labels)) * 100
            print(f"Cluster {label}: {count} samples ({percentage:.1f}%)")
        
        return (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances)), labels
    
    def _get_dbscan_scores(self, X):
        """Get anomaly scores using DBSCAN."""
        labels = self.dbscan_model.fit_predict(X)
        scores = np.zeros(X.shape[0])
        scores[labels == -1] = 1 
        return scores

def get_tableau_scores(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud):
    try:
        step = float(step[0]) if isinstance(step, list) else float(step)
        amount = float(amount[0]) if isinstance(amount, list) else float(amount)
        oldbalanceOrg = float(oldbalanceOrg[0]) if isinstance(oldbalanceOrg, list) else float(oldbalanceOrg)
        newbalanceOrig = float(newbalanceOrig[0]) if isinstance(newbalanceOrig, list) else float(newbalanceOrig)
        oldbalanceDest = float(oldbalanceDest[0]) if isinstance(oldbalanceDest, list) else float(oldbalanceDest)
        newbalanceDest = float(newbalanceDest[0]) if isinstance(newbalanceDest, list) else float(newbalanceDest)
        isFraud = float(isFraud[0]) if isinstance(isFraud, list) else float(isFraud)
        isFlaggedFraud = float(isFlaggedFraud[0]) if isinstance(isFlaggedFraud, list) else float(isFlaggedFraud)
        
        if amount == 0 and oldbalanceOrg == 0 and newbalanceOrig == 0 and oldbalanceDest == 0 and newbalanceDest == 0:
            print("All values are zero, skipping anomaly detection")
            return 0.0
        
        features = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud]], 
                          dtype=np.float64)
        
        detector = EnsembleFraudDetector(contamination=0.3)
        X_processed = detector.preprocess_data(features)  
        scores, _ = detector.train_and_predict(X_processed)
        
        print(f"Input features: {features}")
        print(f"Calculated score: {scores[0]}")
        return float(scores[0])
    except Exception as e:
        print(f"Error in get_tableau_scores: {str(e)}")
        print(f"Input values: step={step}, amount={amount}, oldbalanceOrg={oldbalanceOrg}")
        return 0.0

def get_tableau_clusters(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud):
    try:
        step = float(step[0]) if isinstance(step, list) else float(step)
        amount = float(amount[0]) if isinstance(amount, list) else float(amount)
        oldbalanceOrg = float(oldbalanceOrg[0]) if isinstance(oldbalanceOrg, list) else float(oldbalanceOrg)
        newbalanceOrig = float(newbalanceOrig[0]) if isinstance(newbalanceOrig, list) else float(newbalanceOrig)
        oldbalanceDest = float(oldbalanceDest[0]) if isinstance(oldbalanceDest, list) else float(oldbalanceDest)
        newbalanceDest = float(newbalanceDest[0]) if isinstance(newbalanceDest, list) else float(newbalanceDest)
        isFraud = float(isFraud[0]) if isinstance(isFraud, list) else float(isFraud)
        isFlaggedFraud = float(isFlaggedFraud[0]) if isinstance(isFlaggedFraud, list) else float(isFlaggedFraud)

        if amount == 0 and oldbalanceOrg == 0 and newbalanceOrig == 0 and oldbalanceDest == 0 and newbalanceDest == 0:
            print("All values are zero, skipping clustering")
            return 0

        features = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud]], 
                          dtype=np.float64)

        detector = EnsembleFraudDetector()
        X_processed = detector.preprocess_data(features)
        _, cluster_labels = detector.train_and_predict(X_processed)
        
        print(f"Input features: {features}")
        print(f"Calculated cluster: {cluster_labels[0]}")
        return int(cluster_labels[0])
    except Exception as e:
        print(f"Error in get_tableau_clusters: {str(e)}")
        print(f"Input values: step={step}, amount={amount}, oldbalanceOrg={oldbalanceOrg}")
        return 0

def main():
    """Main execution function."""
    print("Loading data...")
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

    sample_size = 10000
    df = df.sample(n=sample_size, random_state=42)
    
    print("\nData Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {', '.join(df.columns)}")

    model = EnsembleFraudDetector(contamination=0.1)
    X_processed = model.preprocess_data(df)
    scores, clusters = model.train_and_predict(X_processed)
    
    try:
        client = Client('http://localhost:9004/')
        client.deploy('get_tableau_scores', get_tableau_scores, 'Returns anomaly scores', override=True)
        client.deploy('get_tableau_clusters', get_tableau_clusters, 'Returns cluster assignments', override=True)

    except Exception as e:
        print(f"\nError deploying TabPy functions: {str(e)}")

if __name__ == "__main__":
    main()
