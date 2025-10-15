import pandas as pd

# Check training data structure
df = pd.read_csv('artifects/train.csv')
print('Training data columns:')
print(df.columns.tolist())
print('\nSample of first 3 rows:')
print(df.head(3))
print(f'\nTraining data shape: {df.shape}')

# Check for Latitude/Longitude columns
print('\nGeographic data check:')
if 'Latitude' in df.columns:
    print('Latitude column found')
    print(f'Non-null Latitude values: {df["Latitude"].notna().sum()}/{len(df)}')
    if df["Latitude"].notna().sum() > 0:
        print(f'Latitude range: {df["Latitude"].min()} to {df["Latitude"].max()}')
else:
    print('Latitude column NOT found')
    
if 'Longitude' in df.columns:
    print('Longitude column found')
    print(f'Non-null Longitude values: {df["Longitude"].notna().sum()}/{len(df)}')
    if df["Longitude"].notna().sum() > 0:
        print(f'Longitude range: {df["Longitude"].min()} to {df["Longitude"].max()}')
else:
    print('Longitude column NOT found')