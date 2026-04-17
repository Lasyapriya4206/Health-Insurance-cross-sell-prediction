from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(df):
    df = df.dropna()

    X = df.drop("Response", axis=1)
    y = df["Response"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)