import fingerprintMateriaAlt
import fingerprintMateria
import pandas as pd
import sys

def get_property_name():
    """ Prompt the user to enter the name of the property column. """
    property_name = input("Enter the property column name: ")
    return property_name


def process_data(property_name):
    """ Process the data to generate fingerprints and save to a CSV file. """
    input_file = "../data/combinedDataFull.csv"

    # Attempt to load the data file and handle possible FileNotFoundError
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        sys.exit(1)

    # Check if the property name exists in the dataframe columns
    if property_name not in df.columns:
        print(f"Error: '{property_name}' does not exist in the data.")
        sys.exit(1)

    # Drop rows with NaN values in the specified property column
    df.dropna(subset=[property_name], inplace=True)

    if df.empty:
        print(f"No data available after dropping rows with missing values in '{property_name}'.")
        sys.exit(1)

    # Generate fingerprints from the SMILES column
    fingerprints = df['SMILES'].apply(fingerprintMateriaAlt.smiles_to_fingerprint)
    #fingerprints = df['SMILES'].apply(fingerprintMateria.smiles_to_fingerprint)

    # Create a DataFrame from the list of fingerprints
    output_data = pd.DataFrame(fingerprints.tolist(), columns=[f'fp_{i}' for i in range(len(fingerprints[0]))])

    # Ensure alignment of indices after dropping rows
    output_data.reset_index(drop=True, inplace=True)
    propertyFrame = df[property_name].reset_index(drop=True)

    # Combine the fingerprints with the property values
    output_data[property_name] = propertyFrame

    # Define the output file path
    output_file_name = "../fingerprinted_data/fingerprinted_" + property_name.replace(' ', '_') + ".csv"

    # Save the combined data to a CSV file
    output_data.to_csv(output_file_name, index=False)
    print(f"Data processed and saved to {output_file_name}")


# Main entry point of the script
if __name__ == "__main__":
    property_name = get_property_name()
    process_data(property_name)
