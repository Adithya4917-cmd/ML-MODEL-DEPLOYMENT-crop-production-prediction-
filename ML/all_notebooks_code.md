# Notebook Code Export


---

## Logistic Regression — Customer

*Source file: `logistic_regression_coustmer.ipynb`*


**Code cell 1:**

```python
import pandas as pd

df = pd.read_csv('customer_purchase_clean.csv')
df.head()
```


**Code cell 2:**

```python
df.shape        # (rows, columns)
df.info()       # column names, data types, non-null counts
df.describe()   # mean, std, min, max, quartiles for numeric columns
```


**Code cell 3:**

```python
df.isna().sum()
```


**Code cell 4:**

```python
df.duplicated().sum()
```


**Code cell 5:**

```python
df['purchased'].value_counts()
```


**Code cell 6:**

```python
df['purchased'].value_counts().plot(kind='bar')
```


**Code cell 7:**

```python
df_encoded = pd.get_dummies(df, columns=['device'], drop_first=True)
df_encoded.head()
```


**Code cell 8:**

```python
X = df_encoded.drop(columns=['customer_id', 'purchased'])
y = df_encoded['purchased']

print(X.shape)
print(y.shape)
```


**Code cell 9:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```


**Code cell 10:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


**Code cell 11:**

```python
Standardizes the numeric feature columns (age, annual_income, time_on_site_min, 
pages_viewed, prior_purchases) so each has mean 0 and std 1 — same idea we covered 
earlier with ram_gb.

Why it matters here (and this is the key difference from your linear regression file):
Logistic regression in sklearn is trained using an iterative optimization 
algorithm (like gradient descent variants), not a plain closed-form solution.
    Unscaled features (like annual_income ranging up to 150,000 vs age ranging 18–65) 
can make this optimization slow to converge or unstable. This is exactly the case 
I mentioned earlier where scaling is not optional — it actually matters for the model 
to train properly.
```


**Code cell 12:**

```python
scaler.fit_transform(X_train) → calculates mean/std from training data,
then scales it.
scaler.transform(X_test) → uses the same mean/std learned from
training data to scale the test set (does NOT recalculate from test data).
```


**Code cell 13:**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```


**Code cell 14:**

```python
print(model.coef_)        # the learned weight for each feature
print(model.intercept_ )  # the bias term (b0)
```


**Code cell 15:**

```python
print(X_train.columns)
#above cloumn coefficents reelate it to this
```


**Code cell 16:**

```python
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(y_pred[:10])
print(y_prob[:10])
```


**Code cell 17:**

```python
import pandas as pd
coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0]
})
print(coef_df.sort_values('coefficient', ascending=False))
```


**Code cell 18:**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```


---

## Logistic Regression — Lung Cancer

*Source file: `logistic_reg_lung_cancer.ipynb`*


**Code cell 1:**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
```


**Code cell 2:**

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```


**Code cell 3:**

```python
print("%%")
```


**Code cell 4:**

```python
D = load_breast_cancer()

X = D.data
y = D.target
print(X)
print(D.feature_names)
print(D.target_names)
```


**Code cell 5:**

```python
import pandas as pd
df = pd.DataFrame(X, columns=D.feature_names)
df['Target'] = y
df.head()
```


**Code cell 6:**

```python
df
```


**Code cell 7:**

```python
X = df.drop("Target",axis=1)
y = df['Target']
```


**Code cell 8:**

```python
y
```


**Code cell 9:**

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
X_train
```


**Code cell 10:**

```python
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```


**Code cell 11:**

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


**Code cell 12:**

```python
model = LogisticRegression()
model.fit(X_train,y_train)
```


**Code cell 13:**

```python
y_pred = model.predict(X_test)
#y_prob = model.predict_proba(X_test)[:, 1]

print(y_pred)
#print(y_prob[:10])
```


**Code cell 14:**

```python
results = pd.DataFrame({
    "actual":y_test.values,
    "Predicted":y_pred
})
```


**Code cell 15:**

```python
results
```


**Code cell 16:**

```python
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
```


**Code cell 17:**

```python
cm = confusion_matrix(y_test,y_pred)
print(cm)
```


---

## Laptop

*Source file: `laptop.ipynb`*


**Code cell 1:**

```python
import pandas as pd
import numpy as np
print('FRANX')
```


**Code cell 2:**

```python
df = pd.read_csv('laptop_data_practice_dirty.csv')
df
```


**Code cell 3:**

```python
df.head()
```


**Code cell 4:**

```python
steps involved are 
-- null values removal
-- correction of data like price havinig """$2423""" (string) into integer 2423
-- duplicates
-- outliers
-- clean Text/Categorical Inconsistencies
"""need not to in order"""
```


**Code cell 5:**

```python
df.info()
```


**Code cell 6:**

```python
df['price'].unique()
```


**Code cell 7:**

```python
df['price'] = df['price'].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)
df['price'] = df['price'].str.replace(' INR', '', regex=False)
df['price'] = df['price'].str.replace('₹', '', regex=False)
df['price'] = df['price'].str.replace('"', '', regex=False)
print("*")
```


**Code cell 8:**

```python
df['price'] = df['price'].fillna("0")
```


**Code cell 9:**

```python
df['price'].unique()
```


**Code cell 10:**

```python
df['price'] = df['price'].astype(int)
```


**Code cell 11:**

```python
df['price'].unique()
```


**Code cell 12:**

```python
#selects all without any zeros
res = df[df['price'] != 0]['price']
print(res)
print(res.median())
```


**Code cell 13:**

```python
median_price = df[df['price'] != 0]['price'].median()
df['price'] = df['price'].replace(0,median_price)
```


**Code cell 14:**

```python
df['price'].unique()
```


**Code cell 15:**

```python
print(df['price'])
```


**Code cell 16:**

```python
df.info()
```


**Code cell 17:**

```python
#now battery_life_hrs
df['battery_life_hrs']
```


**Code cell 18:**

```python
df['battery_life_hrs'].isna().sum()
```


**Code cell 19:**

```python
df['battery_life_hrs'].unique()
```


**Code cell 20:**

```python
df['battery_life_hrs'].mode()
```


**Code cell 21:**

```python
#df[df['battery_life_hrs'] == 8.6]
#for cheking we are gettingg 6 mode values for battery life take median
```


**Code cell 22:**

```python
df['battery_life_hrs'].median()
```


**Code cell 23:**

```python
df['battery_life_hrs'] = df['battery_life_hrs'].fillna(df['battery_life_hrs'].median())
```


**Code cell 24:**

```python
df.info()
```


**Code cell 25:**

```python
df['processor'].head()
```


**Code cell 26:**

```python
# 100% take mode
df['processor'].mode()
```


**Code cell 27:**

```python
df['processor'] = df['processor'].fillna(df['processor'].mode()[0])
```


**Code cell 28:**

```python
df['processor'].isna().sum()
```


**Code cell 29:**

```python
df.info()
```


**Code cell 30:**

```python
finally left os
```


**Code cell 31:**

```python
df['os'].unique()
```


**Code cell 32:**

```python
df['os'].mode()
```


**Code cell 33:**

```python
df['os'] = df['os'].fillna(df['os'].mode()[0])
```


**Code cell 34:**

```python
df['os'].isna().sum()
```


**Code cell 35:**

```python
df['notes']
```


**Code cell 36:**

```python
df['notes'] = df['notes'].fillna('no notes')
```


**Code cell 37:**

```python
df.info()
```


**Code cell 38:**

```python
#finally check for duplicates
df.duplicated().sum()
```


**Code cell 39:**

```python
df[df['laptop_id'].duplicated()==1]
```


**Code cell 40:**

```python
df[df.duplicated()==1]
```


**Code cell 41:**

```python
df[df['laptop_id'].duplicated(keep=False)].sort_values('laptop_id')
```


**Code cell 42:**

```python
df = df.drop_duplicates()
df = df.reset_index(drop=True)
```


**Code cell 43:**

```python
"""
Why add reset_index(drop=True) too?
When you drop rows, pandas keeps the original index numbers — so your row labels will have gaps 
(e.g., 0, 1, 3, 4, 7... instead of 0, 1, 2, 3, 4...). That's not wrong, but it can cause confusion later 
(e.g., if you loop by position, or merge with another DataFrame). reset_index(drop=True) renumbers rows 
cleanly from 0 again. The drop=True part means "don't keep the old index as a new column" — try it without
drop=True once just to see what happens, so you understand why it's there.
"""
```


**Code cell 44:**

```python
df
```


**Code cell 45:**

```python
print(df.shape)          # row count should now be lower than before
print(df.duplicated().sum())   # should be 0
```


**Code cell 46:**

```python
#finally outliers to remove 
```


**Code cell 47:**

```python
#check
df['storage'].unique()
```


**Code cell 48:**

```python
#chheck
df['touchscreen'].unique()
```


**Code cell 49:**

```python
#outliers check in price and battery life
```


**Code cell 50:**

```python
df['price'].unique()
```


**Code cell 51:**

```python
df['price'].describe()
```


**Code cell 52:**

```python
#IQR METHOD REMOVING OUTLIER
```


**Code cell 53:**

```python
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
```


**Code cell 54:**

```python
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(outliers[['laptop_id', 'price']])
```


**Code cell 55:**

```python
#the above two are outliers
```


**Code cell 56:**

```python
# Option A: Remove them
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print("*")
```


**Code cell 57:**

```python
# Option B: Cap them at the bounds instead of removing (keeps row count)
df['price'] = df['price'].clip(lower=lower_bound, upper=upper_bound)
print("*")
```


**Code cell 58:**

```python
#final sanity check
```


**Code cell 59:**

```python
df.info()
df.describe()
```


**Code cell 60:**

```python
#as you can see that min in price is 614 so use brlow method
#you could use this method also
```


**Code cell 61:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(df['ram_gb'], df['price'])
plt.xlabel('RAM (GB)')
plt.ylabel('Price')
plt.title('Price vs RAM')
plt.show()
```


**Code cell 62:**

```python
#using domain knowleedge keeping 5000 as min
```


**Code cell 63:**

```python
df = df[df['price'] >= 5000]
df = df.reset_index(drop=True)
df['price'].min()   # confirm this is now >= 5000
```


**Code cell 64:**

```python
plt.figure(figsize=(8,5))
plt.scatter(df['ram_gb'], df['price'])
plt.xlabel('RAM (GB)')
plt.ylabel('Price')
plt.title('Price vs RAM')
plt.show()
```


**Code cell 65:**

```python
#checking this also
plt.figure(figsize=(8,5))
plt.scatter(df['battery_life_hrs'], df['price'])
plt.xlabel('Battery Life (hrs)')
plt.ylabel('Price')
plt.title('Price vs Battery Life')
plt.show()
```


**Code cell 66:**

```python
#leniar regression
import seaborn as sns
import matplotlib.pyplot as plt
print("*")
```


**Code cell 67:**

```python
df_encoded = pd.get_dummies(df, columns=['processor', 'os', 'touchscreen', 'storage'], drop_first=True)
df_encoded.head()
```


**Code cell 68:**

```python
X = df_encoded.drop(columns=['price', 'laptop_id', 'model_name', 'notes'])
y = df_encoded['price']
```


**Code cell 69:**

```python
X.columns
```


**Code cell 70:**

```python
import sklearn.model_selection as sms
X_train, X_test, y_train, y_test = sms.train_test_split(X, y, test_size=0.2, random_state=42)
#or down 
```


**Code cell 71:**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


**Code cell 72:**

```python
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```


**Code cell 73:**

```python
#correlation to check any more to drop
"""numeric_cols = ['release_year', 'price', 'battery_life_hrs', 'ram_gb'] + \
    [col for col in df_encoded.columns if col.startswith('processor_') 
     or col.startswith('os_') 
     or col.startswith('touchscreen_') 
     or col.startswith('storage_')]"""

corr = df_encoded.corr()
corr

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
```


**Code cell 74:**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```


**Code cell 75:**

```python
y_pred = model.predict(X_test) 
print("*")
```


**Code cell 76:**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
```


**Code cell 77:**

```python
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#comparison_df.head(10)
comparison_df
```


**Code cell 78:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()
```


---

## Logistic Regression — Gym

*Source file: `gym_logistic.ipynb`*


**Code cell 1:**

```python
import numpy as np
import pandas as pd
```


**Code cell 2:**

```python
df = pd.read_csv("gym_membership_dirty.csv")
df
```


**Code cell 3:**

```python
df.info()
```


**Code cell 4:**

```python
df['age'].isna().sum()
```


**Code cell 5:**

```python
df['age'] = df['age'].abs()
print(df['age'])
```


**Code cell 6:**

```python
df['age'] = df['age'].fillna(df['age'].median())
print(df['age'])
```


**Code cell 7:**

```python
df['gender'].unique()
```


**Code cell 8:**

```python
df['gender'] = df['gender'].str.replace('Male','male',regex=False)
df['gender'] = df['gender'].str.replace('Female','female',regex=False)
df['gender'] = df['gender'].str.replace('F','female',regex=False)
df['gender'] = df['gender'].str.replace('M','male',regex=False)
df['gender'] = df['gender'].fillna('unknown')
```


**Code cell 9:**

```python
df['gender'].unique()
```


**Code cell 10:**

```python
print(df['city'].isna().sum())
print(df['membership_type'].isna().sum())
print(df['membership_type'].unique())
print(df['city'].unique())
```


**Code cell 11:**

```python
df['membership_type'] = df['membership_type'].str.strip()
df['membership_type'] = df['membership_type'].str.replace('PREMIUM','Premium',regex=False)
df['membership_type'] = df['membership_type'].str.replace('Standard ','Standard',regex=False)
df['membership_type'] = df['membership_type'].str.replace('basic','Basic',regex=False)
```


**Code cell 12:**

```python
df['city'] = df['city'].str.strip()
df['city'] = df['city'].str.replace('hyderabad','Hyderabad',regex=False)
df['city'] = df['city'].str.replace('Mumbai ','Mumbai',regex=False)
df['city'] = df['city'].str.replace('  Delhi','Delhi',regex=False)
df['city'] = df['city'].str.replace('BANGALORE','Bangalore',regex=False)
```


**Code cell 13:**

```python
print(df['city'].isna().sum())
print(df['membership_type'].isna().sum())
print(df['membership_type'].unique())
print(df['city'].unique())
```


**Code cell 14:**

```python
print(df['price'].unique())
print("-----")
print(df['price'].isna().sum())
```


**Code cell 15:**

```python
"""p = df['price'].str.strip()
p = p.str.replace('Rs ','',regex=False)
p = p.str.replace('₹','',regex=False)
p = p.str.replace('$','',regex=False)
p = p.str.replace(',','',regex=False)"""
```


**Code cell 16:**

```python
"""p = pd.to_numeric(p)
df['price'] = p
df['price'] = df['price'].fillna(df['price'].median())"""
```


**Code cell 17:**

```python
p = df['price'].str.strip()

is_usd = p.str.contains(r'\$', regex=True, na=False)

p = p.str.replace('Rs ', '', regex=False)
p = p.str.replace('₹', '', regex=False)
p = p.str.replace(r'\$', '', regex=True)
p = p.str.replace(',', '', regex=False)

p = pd.to_numeric(p)

p = p.where(~is_usd, p * 83)   # convert $ rows to INR

df['price'] = p
df['price'] = df['price'].fillna(df['price'].median())
```


**Code cell 18:**

```python
print(df['price'])
```


**Code cell 19:**

```python
# --- workout_days_per_week ---
df['workout_days_per_week'] = df['workout_days_per_week'].fillna(df['workout_days_per_week'].median())
df['workout_days_per_week'] = df['workout_days_per_week'].clip(1,7)  # just in case of bad values
```


**Code cell 20:**

```python
# personal_trainer / referral_signup / renewed (Yes/No/Y/N/1/0/null)
yesno_map = {'yes':'Yes','y':'Yes','1':'Yes','no':'No','n':'No','0':'No'}
for col in ['personal_trainer','referral_signup']:
    df[col] = df[col].astype(str).str.strip().str.lower().map(yesno_map)
    df[col] = df[col].fillna(df[col].mode()[0])

df['renewed'] = df['renewed'].astype(str).str.strip().str.lower().map(yesno_map)
df['renewed'] = df['renewed'].fillna(df['renewed'].mode()[0])
df['renewed'] = df['renewed'].map({'Yes':1,'No':0})

# membership_months
df['membership_months'] = df['membership_months'].fillna(df['membership_months'].median())

# satisfaction_score
df['satisfaction_score'] = df['satisfaction_score'].fillna(df['satisfaction_score'].median())

# duplicates
df = df.drop_duplicates()

df.to_csv('gym_membership_cleaned.csv', index=False)
```


**Code cell 21:**

```python
df
```
