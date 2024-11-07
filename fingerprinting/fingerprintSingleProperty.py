import fingerprintMateria
import pandas as pd
import sys

numArgs = len(sys.argv)

if(numArgs != 3):
    print("Usage: fingerprintSingleProperty.py INPUTFILE PROPERTYCOLUMNNAME")
    print(numArgs)
    exit()

propertyName = sys.argv[2]
#inputFile = sys.argv[1]

#inputFile = '../data/combinedDataFull.csv'




df = pd.read_csv( "../data/combinedDataFull.csv")
    
# Drop rows with NaN values in the 'desired properties' column
df.dropna(subset=[propertyName], inplace=True)

propertyFrame = df[propertyName]

fingerprints = df['SMILES'].apply(fingerprintMateria.smiles_to_fingerprint)

output_data = pd.DataFrame(fingerprints.tolist(), columns=[f'fp_{i}' for i in range(len(fingerprints[0]))])


output_data.reset_index(drop=True, inplace=True)
propertyFrame.reset_index(drop=True, inplace=True)

output_data[propertyName] = propertyFrame

outputFileName = "../fingerprinted_data/fingerprinted" + propertyName + ".csv"

output_data.to_csv(outputFileName, index=False)