import os
import urllib.request
import pandas as pd

def main():
    os.makedirs('data', exist_ok=True)
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    csv_path = "data/diabetes.csv"
    print(f"Downloading PIMA Diabetes dataset from {url}...")
    urllib.request.urlretrieve(url, csv_path)
    
    # Add headers since the original file lacks them
    df = pd.read_csv(csv_path, header=None, names=columns)
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to {csv_path} with {len(df)} records.")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()
