"""
Cleaning pipeline for transaction-dataset.csv.
Applies 6 corrections and writes transaction-dataset-cleaned.csv.
(card_name and digitalization_type are left as-is per spec.)
"""
import pandas as pd

SRC = 'transaction-dataset.csv'
DST = 'transaction-dataset-cleaned.csv'

df = pd.read_csv(SRC)
n0 = len(df)
report = {'start_rows': n0}

# 1) Fill major_category = "refund" where status == "refunded"
mask_refund = df['major_category'].isna() & (df['status'] == 'refunded')
report['step1_major_category_filled'] = int(mask_refund.sum())
df.loc[mask_refund, 'major_category'] = 'refund'
report['step1_major_category_remaining_na'] = int(df['major_category'].isna().sum())

# 2) Fill merchant_name via merchant_id lookup, fallback to title
id_to_name = (df.loc[df['merchant_name'].notna(), ['merchant_id','merchant_name']]
                 .drop_duplicates(subset='merchant_id')
                 .set_index('merchant_id')['merchant_name'].to_dict())
mask_mn = df['merchant_name'].isna()
n_mn_before = int(mask_mn.sum())
df.loc[mask_mn, 'merchant_name'] = df.loc[mask_mn, 'merchant_id'].map(id_to_name)
mask_still = df['merchant_name'].isna()
report['step2_merchant_name_filled_via_id']    = n_mn_before - int(mask_still.sum())
df.loc[mask_still, 'merchant_name'] = df.loc[mask_still, 'title']
report['step2_merchant_name_filled_via_title'] = int(mask_still.sum())
report['step2_merchant_name_remaining_na']     = int(df['merchant_name'].isna().sum())

# 3) Remove |amount|>100k on fast-food/grocery merchants
mask_outlier = (df['amount_value'].abs() > 100_000) & (df['major_category'].isin(['restaurants_bars_cafes','groceries']))
report['step3_extreme_outliers_removed'] = int(mask_outlier.sum())
df = df.loc[~mask_outlier].copy()

# 4) Remove zero-amount rows
mask_zero = df['amount_value'] == 0
report['step4_zero_amount_removed'] = int(mask_zero.sum())
df = df.loc[~mask_zero].copy()

# 5) De-duplicate ref_id (keep first)
n_before = len(df)
df = df.drop_duplicates(subset='ref_id', keep='first').copy()
report['step5_ref_id_duplicates_removed'] = n_before - len(df)

# 6) Convert fees from cents to euros
df['fees'] = (df['fees'] / 100.0).round(2)
report['step6_fees_cents_to_euros'] = 'done (column divided by 100, unit now EUR)'

# NOTE: card_name and digitalization_type are intentionally LEFT AS-IS.

df.to_csv(DST, index=False)
report['end_rows'] = len(df)
report['removed_total'] = n0 - len(df)
for k, v in report.items():
    print(f"{k:>50}  {v}")
print("\nWritten to:", DST)
