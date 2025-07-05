# 1. Create or overwrite backend/model.py
cat > backend/model.py << 'EOF'
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from backend.validate import load_default_dataset

# 1) Load data
df = load_default_dataset()

# 2) Clean headers & types
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.replace(r"[^\w_]", "", regex=True)
)
df["Eg_eV"] = pd.to_numeric(df["Eg_eV"], errors="coerce")
df = df.dropna(subset=["Composition", "Eg_eV"]).reset_index(drop=True)

# 3) Tag A-site
df["A_site"] = df["Composition"].str.extract(r"^(Cs|MA|FA)", expand=False)

# 4) End-member gaps
E = {
    "CsSnI3": 1.30, "CsPbI3": 1.73,
    "MAPbI3": 1.55, "FAPbI3": 1.48,
    "CsSnCl3": 2.89, "CsSnBr3": 1.79,
}
def get_E(formula):
    for key, val in E.items():
        if key in formula:
            return val
    return np.nan

# 5) Featurize
def featurize(formula, a_site):
    x_sn = formula.count("Sn") / (formula.count("Sn") + formula.count("Pb"))
    x_pb = 1 - x_sn
    veg = x_sn * get_E("CsSnI3") + x_pb * get_E("CsPbI3")
    bow = x_sn * x_pb
    return {
        "Vegard": veg,
        "Bowing": bow,
        "A_MA": int(a_site=="MA"),
        "A_FA": int(a_site=="FA"),
    }

X = pd.DataFrame([
    featurize(row["Composition"], row["A_site"])
    for _, row in df.iterrows()
])
y = df["Eg_eV"]

# 6) Cross-validated regression
model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=0)
maes = -cross_val_score(model, X, y, cv=cv,
                        scoring="neg_mean_absolute_error")
print(f"5-fold CV MAE = {maes.mean():.3f} ± {maes.std():.3f} eV")

# 7) Fit and inspect
model.fit(X, y)
print("Intercept:", model.intercept_)
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.3f}")
EOF

# 2. Stage, commit, and push
git add backend/model.py
git commit -m "feat: add physics-informed regression model with cross-validation"
git push origin main
