"""
Run this ONCE to generate the sample dataset:
    python generate_data.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

random.seed(42)
np.random.seed(42)

N = 5000
cities  = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune","Kolkata","Ahmedabad","Jaipur","Surat"]
states  = ["MH","DL","KA","TS","TN","MH","WB","GJ","RJ","GJ"]
cats    = ["electronics","clothing","books","furniture","sports","beauty","toys","grocery","automotive","health"]
payment = ["credit_card","debit_card","upi","boleto","voucher"]
status  = ["delivered","delivered","delivered","delivered","shipped","cancelled"]

start = datetime(2023, 1, 1)

rows = []
for i in range(N):
    cid = f"C{str(i+1).zfill(5)}"
    oid = f"O{str(i+1).zfill(6)}"
    purchase_ts = start + timedelta(days=random.randint(0, 600))
    approved_ts = purchase_ts + timedelta(hours=random.randint(1, 12))
    est_del = purchase_ts + timedelta(days=random.randint(5, 20))
    act_del = est_del + timedelta(days=random.randint(-3, 10)) if random.random() > 0.1 else None

    price  = round(np.random.lognormal(5, 1.2), 2)
    freight= round(price * random.uniform(0.03, 0.15), 2)
    pay_inst = random.choices([1,2,3,6,12], weights=[40,20,15,15,10])[0]
    pay_val  = round((price + freight) * random.uniform(0.98, 1.02), 2)

    city_idx = random.randint(0, len(cities)-1)
    cat  = random.choice(cats)
    ptype= random.choice(payment)
    stat = random.choice(status)

    pname_len = random.randint(20, 60)
    pdesc_len = random.randint(100, 600)
    pphotos   = random.randint(1, 8)
    pweight   = random.randint(100, 10000)
    plength   = random.randint(10, 100)
    pheight   = random.randint(5, 80)
    pwidth    = random.randint(5, 80)

    review = random.randint(1, 5) if random.random() > 0.2 else None
    rows.append([cid, oid, stat, purchase_ts.strftime("%Y-%m-%d %H:%M:%S"),
                 approved_ts.strftime("%Y-%m-%d %H:%M:%S"),
                 est_del.strftime("%Y-%m-%d %H:%M:%S"),
                 act_del.strftime("%Y-%m-%d %H:%M:%S") if act_del else None,
                 est_del.strftime("%Y-%m-%d %H:%M:%S"),
                 cat, i+1, price, freight, ptype, pay_inst, pay_val,
                 cities[city_idx], states[city_idx],
                 pname_len, pdesc_len, pphotos, pweight, plength, pheight, pwidth, review])

cols = ["customer_id","order_id","order_status","order_purchase_timestamp",
        "order_approved_at","shipping_limit_date","order_delivered_customer_date",
        "order_estimated_delivery_date","product_category_name","order_item_id",
        "price","freight_value","payment_type","payment_installments","payment_value",
        "customer_city","customer_state","product_name_length","product_description_length",
        "product_photos_qty","product_weight_g","product_length_cm","product_height_cm",
        "product_width_cm","review_score"]

df = pd.DataFrame(rows, columns=cols)
# Introduce ~5% missing values to make it realistic
for col in ["review_score","product_weight_g","product_description_length"]:
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/ecommerce.csv", index=False)
print(f"✅  Dataset created: data/ecommerce.csv  ({len(df)} rows)")
