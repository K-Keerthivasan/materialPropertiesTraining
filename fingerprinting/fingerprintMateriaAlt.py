import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator
from rdkit import DataStructs
import hashlib


def smiles_to_fingerprint(smiles, morgan_bits=512, atom_pair_bits=256, qspr_bits=128, morphological_bits=128,
                          polymer_bits=64):
    try:
        # Convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 1. Atomic scale fingerprints (Morgan and AtomPair)
        atom_fingerprint = calculate_atom_fingerprint(mol, morgan_bits, atom_pair_bits)

        # 2. QSPR fingerprints
        qspr_fingerprint = calculate_qspr_fingerprint(mol, qspr_bits)

        # 3. Morphological descriptors
        morphological_fingerprint = calculate_morphological_fingerprint(mol, morphological_bits)

        # Concatenate all fingerprints
        final_fingerprint = np.concatenate([atom_fingerprint, qspr_fingerprint, morphological_fingerprint])

        return final_fingerprint

    except Exception as e:
        print(f"Error processing SMILES {smiles} string: {str(e)}")
        return None


def calculate_atom_fingerprint(mol, morgan_bits, atom_pair_bits):
    morgan_gen = GetMorganGenerator(radius=2, fpSize=morgan_bits)
    morgan_fp = morgan_gen.GetFingerprint(mol)

    atom_pair_gen = GetAtomPairGenerator(fpSize=atom_pair_bits)
    atom_pair_fp = atom_pair_gen.GetFingerprint(mol)

    morgan_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)

    atom_pair_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(atom_pair_fp, atom_pair_array)

    return np.concatenate([morgan_array, atom_pair_array])


# Using a hashing-based approach instead of random projections
def calculate_qspr_fingerprint(mol, n_bits):
    qspr_features = [
        Descriptors.ExactMolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumRotatableBonds(mol) / mol.GetNumAtoms(),
        Descriptors.RingCount(mol) / mol.GetNumAtoms()
    ]
    qspr_features = (np.array(qspr_features) - np.mean(qspr_features)) / np.std(qspr_features)

    # Hashing each feature into a fixed-length binary vector
    hashed_fingerprint = np.zeros(n_bits, dtype=int)
    for feature in qspr_features:
        hashed_value = int(hashlib.md5(str(feature).encode()).hexdigest(), 16)
        # Convert the hash value to a binary string of fixed length
        for i in range(n_bits):
            hashed_fingerprint[i] ^= (hashed_value >> i) & 1  # XOR to spread bits evenly

    return hashed_fingerprint


"""def calculate_qspr_fingerprint(mol, n_bits):
    qspr_features = [
        Descriptors.ExactMolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumRotatableBonds(mol) / mol.GetNumAtoms(),
        Descriptors.RingCount(mol) / mol.GetNumAtoms()
    ]
    # Normalize features
    qspr_features = (np.array(qspr_features) - np.mean(qspr_features)) / np.std(qspr_features)
    # Use random projection to convert to fixed-length binary string
    random_matrix = np.random.randn(len(qspr_features), n_bits)
    return (np.dot(qspr_features, random_matrix) > 0).astype(int)"""

def calculate_morphological_fingerprint(mol, n_bits):
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)

    shortest_ring_distance = float('inf')
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            if i in ring_atoms and j in ring_atoms:
                path = Chem.GetShortestPath(mol, i, j)
                if len(path) < shortest_ring_distance:
                    shortest_ring_distance = len(path)

    side_chain_atoms = set(range(mol.GetNumAtoms())) - ring_atoms
    fraction_side_chain = len(side_chain_atoms) / mol.GetNumAtoms()

    longest_side_chain = max(len(Chem.GetMolFrags(mol)) for atom in side_chain_atoms)

    morphological_features = [
        shortest_ring_distance if shortest_ring_distance != float('inf') else 0,
        fraction_side_chain,
        longest_side_chain
    ]
    morphological_features = (np.array(morphological_features) - np.mean(morphological_features)) / np.std(
        morphological_features)

    """# Normalize features
    morphological_features = (np.array(morphological_features) - np.mean(morphological_features)) / np.std(morphological_features)
    # Use random projection to convert to fixed-length binary string
    random_matrix = np.random.randn(len(morphological_features), n_bits)
    return (np.dot(morphological_features, random_matrix) > 0).astype(int)"""


    # Hashing each feature into a fixed-length binary vector
    hashed_fingerprint = np.zeros(n_bits, dtype=int)
    for feature in morphological_features:
        hashed_value = int(hashlib.md5(str(feature).encode()).hexdigest(), 16)
        # Convert the hash value to a binary string of fixed length
        for i in range(n_bits):
            hashed_fingerprint[i] ^= (hashed_value >> i) & 1  # XOR to spread bits evenly

    return hashed_fingerprint
