import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# wczytuje dane
df = pd.read_csv("data/sdss_data.csv")

print(df.head())
print(df.info())

features = ['u', 'g', 'r', 'i', 'z', 'redshift'] # cechy
X = df[features]
y = df['class']

print("\nRozkład klas:")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nLiczba próbek treningowych: {len(X_train)}")
print(f"Liczba próbek testowych: {len(X_test)}")

# trening modelu
print("Start")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ewaluacja modelu
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("SDSS Classification Matrix")
plt.show()

# waznosc cech
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 4))
importances.plot(kind='barh', color='teal')
plt.title("Ważność cech w klasyfikacji SDSS")
plt.xlabel("Waga cechy (suma = 1.0)")
plt.tight_layout()
plt.show()

# przykladowy obiekt
def predict_sky_object(u, g, r, i, z, redshift):
    sample = pd.DataFrame([[u, g, r, i, z, redshift]], columns=features)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print(f"\n--- Wynik dla próbki: {prediction} ---")
    for cls, prob in zip(model.classes_, probabilities):
        print(f"{cls}: {prob * 100:.2f}%")

print("\nTest własnej próbki:")
predict_sky_object(u=18.5, g=17.8, r=17.5, i=17.2, z=17.0, redshift=2.45)

# test - brak cechy redshift
print("\n" + "="*40)
print("Trenowanie bez redshiftu")
print("="*40)

features_no_redshift = ['u', 'g', 'r', 'i', 'z']
X_no_redshift = df[features_no_redshift]

X_tr_no_z, X_te_no_z, y_tr_no_z, y_te_no_z = train_test_split(
    X_no_redshift, y, test_size=0.2, random_state=42, stratify=y
)

model_no_z = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_no_z.fit(X_tr_no_z, y_tr_no_z)

y_pred_no_z = model_no_z.predict(X_te_no_z)

print("\nRaport klasyfikacji bez redshiftu:")
print(classification_report(y_te_no_z, y_pred_no_z))

ConfusionMatrixDisplay.from_estimator(model_no_z, X_te_no_z, y_te_no_z, cmap="Oranges")
plt.title("Macierz pomyłek bez redshiftu")
plt.show()