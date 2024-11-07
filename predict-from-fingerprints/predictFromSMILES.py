from pickle import load
import sys
from fingerprinting import fingerprintMateria


numArgs = len(sys.argv)

if(numArgs != 3):
    print("Usage: predictFromSMILES.py MODELFILE SMILESTOPREDICT")
    exit()

smiles = sys.argv[2]
modelFileName = sys.argv[1]


fingerprint = fingerprintMateria.smiles_to_fingerprint(smiles)
# print("Fingerprint: ")
# print(*fingerprint, sep=',')

fingerprintFixed = [fingerprint]


with(open(modelFileName, "rb")) as f:
    model = load(f)

mean_prediction, std_dev = model.predict(fingerprintFixed, return_std=True)
print("Mean prediction: ", mean_prediction[0])
print('Std Deviation: ', std_dev[0])