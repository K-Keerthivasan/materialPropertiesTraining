import fingerprintMateria
import pandas as pd
import numpy as np

df = pd.read_csv('../data/combinedDataFull.csv')

# print(df["Density"].unique())

# print(df['Density'].apply(lambda x: len(str(x))))

# df.replace(r'^\s^$', np.nan, regex = True, inplace=True)

    
# Drop rows with NaN values in the 'smiles' column
# df.dropna(subset=['SMILES'], inplace=True)

df.dropna(subset=['Tensile stress strength at break'], inplace=True)

GTTFrame = df['Tensile stress strength at break']

fingerprints = df['SMILES'].apply(fingerprintMateria.smiles_to_fingerprint)

output_data = pd.DataFrame(fingerprints.tolist(), columns=[f'fp_{i}' for i in range(len(fingerprints[0]))])

print("Check index alignment: ", output_data.index.equals(GTTFrame.index))

print("Size fingerprints: ", len(output_data), " Size GTTFrame: ", len(GTTFrame))

output_data.reset_index(drop=True, inplace=True)
GTTFrame.reset_index(drop=True, inplace=True)

print("Check index alignment: ", output_data.index.equals(GTTFrame.index))

output_data['Tensile stress strength at break'] = GTTFrame

# print(output_data['Density'].apply(lambda x: len(str(x))))

output_data.to_csv("../fingerprinted_data/fingerprinted_TensileSSAB.csv", index=False)