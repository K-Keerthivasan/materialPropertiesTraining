from tkinter import *
from tkinter import ttk

from pickle import load
import sys
from fingerprinting import fingerprintMateria


def predict(*args):
    try:
        fingerprint = fingerprintMateria.smiles_to_fingerprint(smilesInput.get())

        with(open(modelFile.get(), "rb")) as f:
            model = load(f)
        
        fingerprintFixed = [fingerprint]

        mean_prediction, std_dev = model.predict(fingerprintFixed, return_std=True)
        predictedProperty.set(mean_prediction[0])
        # print("Mean prediction: ", mean_prediction[0])
        # print('Std Deviation: ', std_dev[0])


    except:
        pass


root = Tk()
root.title("Predict Property from SMILES")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


modelFile = StringVar()
# modelFile_entry = ttk.Entry(mainframe, width=20, textvariable=modelFile)
# modelFile_entry.grid(column=2, row=1, sticky=(W, E))
modelFile_entry = ttk.Combobox(mainframe, textvariable=modelFile)
modelFile_entry['values'] = ('densityGPR.pkl', 'tensileSSAB_GPR.pkl', 'glassTransitionGPR.pkl')
modelFile_entry.state(["readonly"])
modelFile_entry.grid(column=2, row=1, sticky=(W, E))

smilesInput = StringVar()
smilesInput_entry = ttk.Entry(mainframe, width=20, textvariable=smilesInput)
smilesInput_entry.grid(column=2, row=2, sticky=(W, E))


predictedProperty = StringVar()
ttk.Label(mainframe,textvariable=predictedProperty).grid(column=2, row=3, sticky=(W, E))



ttk.Button(mainframe, text="Predict", command=predict).grid(column=3, row=3, sticky=W)


ttk.Label(mainframe, text="Model File").grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text="SMILES String").grid(column=3, row=2, sticky=W)

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

smilesInput_entry.focus()

root.bind("<Return>", predict)

root.mainloop()