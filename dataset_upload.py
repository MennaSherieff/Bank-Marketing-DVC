# download_data.py
from ucimlrepo import fetch_ucirepo
import os

os.makedirs("data/raw", exist_ok=True)

bank_marketing = fetch_ucirepo(id=222)

df = bank_marketing.data.features
df["target"] = bank_marketing.data.targets

df.to_csv("data/raw/bank.csv", index=False)