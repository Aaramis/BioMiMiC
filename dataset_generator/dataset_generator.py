# https://projects.volkamerlab.org/teachopencadd/talktorials/T013_query_pubchem.html

import random
import pandas as pd
import numpy as np
import requests
import json
import time
from urllib.parse import quote
from indigo import *
import argparse

random.seed(8)


def get_compound_cid(compound_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/cids/JSON"
    res = requests.get(url)
    data = json.loads(res.content.decode())
    if "IdentifierList" in data:
        cid = data["IdentifierList"]["CID"][0]
        return cid


def get_smiles_from_cid(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSmiles/JSON"
    res = requests.get(url)
    data = json.loads(res.content.decode())
    return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]


def get_chembl_id(name):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?pref_name__iexact={name}&format=json"
    res = requests.get(url)
    data = json.loads(res.content.decode())
    if data["page_meta"]["total_count"] == 0:
        raise ValueError(f"Could not find id for compound: {name}")
    else:
        return data["molecules"][0]["molecule_chembl_id"]


def get_mechanism_by_chembl_id(chembl_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism?parent_molecule_chembl_id__iexact={chembl_id}&format=json"
    res = requests.get(url)
    data = json.loads(res.content.decode())
    mechanism_res = {"binding_site": data["mechanisms"][0]["binding_site_comment"],
                     "mechanism_of_action": data["mechanisms"][0]["mechanism_of_action"]}
    return mechanism_res


def get_similar_activity_molecules_chembl_ids(mechanism):
    if mechanism["binding_site"] == None:
        url = "https://www.ebi.ac.uk/chembl/api/data/mechanism?mechanism_of_action__icontains=" + \
              mechanism["mechanism_of_action"] + "&format=json"
    else:
        url = "https://www.ebi.ac.uk/chembl/api/data/mechanism?mechanism_of_action__icontains=" + \
              mechanism["mechanism_of_action"] + "&binding_site_comment__icontains=" + mechanism["binding_site"] + \
              "&format=json"
    res = requests.get(url)
    data = json.loads(res.content.decode())
    chembl_ids = []
    for recs in data["mechanisms"]:
        chembl_ids.append(recs["parent_molecule_chembl_id"])
    while data["page_meta"]["next"] != None:
        url = "https://www.ebi.ac.uk" + data["page_meta"]["next"]
        res = requests.get(url)
        data = json.loads(res.content.decode())
        for recs in data["mechanisms"]:
            chembl_ids.append(recs["parent_molecule_chembl_id"])
    return chembl_ids


def get_names_from_chembl_ids(chembl_ids):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?molecule_chembl_id__in={','.join(map(str, chembl_ids))}&format=json"
    res = requests.get(url)
    data = json.loads(res.content.decode())
    names = []
    for recs in data["molecules"]:
        names.append(recs["pref_name"])
    while data["page_meta"]["next"] != None:
        url = "https://www.ebi.ac.uk" + data["page_meta"]["next"]
        res = requests.get(url)
        data = json.loads(res.content.decode())
        for recs in data["molecules"]:
            names.append(recs["pref_name"])
    return names


def smiles_from_pubchem_cids(cids):
    """
    Get the canonical SMILES string from the PubChem CIDs.

    Parameters
    ----------
    cids : list
        A list of PubChem CIDs.

    Returns
    -------
    list
        The canonical SMILES strings of the PubChem CIDs.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, cids))}/property/CanonicalSMILES/JSON"
    r = requests.get(url)
    r.raise_for_status()
    return [item["CanonicalSMILES"] for item in r.json()["PropertyTable"]["Properties"]]


def names_from_pubchem_cids(cids):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, cids))}/synonyms/JSON"
    res = requests.get(url)
    res.raise_for_status()
    data = json.loads(res.content.decode())
    names = []
    for compound in data["InformationList"]["Information"]:
        if "Synonym" in compound:
            names.append(compound["Synonym"][0])
        else:
            names.append("Name not found")
    return names


def mws_from_pubchem_cids(cids):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, cids))}/property/MolecularWeight/JSON"
    res = requests.get(url)
    res.raise_for_status()
    data = json.loads(res.content.decode())
    mws = []
    if "PropertyTable" in data:
        for compound in data["PropertyTable"]["Properties"]:
            mws.append(compound["MolecularWeight"])
        mws_float = [float(k) for k in mws]
    else:
        raise ValueError("Could not find molecular weights for the given CIDs")
    return mws_float


def get_random_smiles_from_chembl(n_positives, random_positive_ratio):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?only=molecule_structures__canonical_smiles&format=json"
    res = requests.get(url)
    data = json.loads(res.content.decode())
    random_smiles = []
    try:
        random_smiles.append(data["molecules"][random.randint(0, 19)]["molecule_structures"]["canonical_smiles"])
    except TypeError:
        random_smiles.append(data["molecules"][random.randint(0, 19)]["molecule_structures"]["canonical_smiles"])
    for i in range(n_positives * random_positive_ratio):
        url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?" + \
              f"limit=20&offset=" + str(round(random.random() * int(data["page_meta"]["total_count"]), 0)) + \
              "&only=molecule_structures__canonical_smiles"
        res = requests.get(url)
        data = json.loads(res.content.decode())
        try:
            random_smiles.append(data["molecules"][random.randint(0, 19)]["molecule_structures"]["canonical_smiles"])
        except TypeError:
            random_smiles.append(data["molecules"][random.randint(0, 19)]["molecule_structures"]["canonical_smiles"])
    return random_smiles


def get_cids_from_chembl_smiles(random_smiles):
    random_pubchem_cids = []
    for smiles in random_smiles:
        escaped_smiles = quote(smiles).replace("/", ".")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/cids/JSON"
        res = requests.get(url)
        res.raise_for_status()
        data = json.loads(res.content.decode())
        if "IdentifierList" in data:
            random_pubchem_cids.append(data["IdentifierList"]["CID"][0])
    random_pubchem_cids_clean = [k for k in random_pubchem_cids if k != 0]
    return random_pubchem_cids_clean


def recompute_smiles(compound_list):
    indigo = Indigo()
    indigo_list = [0 for k in range(len(compound_list))]
    new_smiles = [0 for k in range(len(compound_list))]
    for index, compound in enumerate(compound_list):
        indigo_list[index] = indigo.loadMolecule(compound)
    for index, compound in enumerate(indigo_list):
        indigo_list[index].aromatize()
        new_smiles[index] = indigo_list[index].canonicalSmiles()
    return new_smiles



parser = argparse.ArgumentParser(description='Generate datasets fromd drug name')
parser.add_argument('-c', "--compound", type=str, help="Name of compound of interest", default=None)
args = parser.parse_args()

if not args.compound:
    raise ValueError("Please enter a compound name")

compound = args.compound
output_file_name = args.compound + "_dataset.csv"

print("Generating dataset...")
chembl_id = get_chembl_id(compound)
mechanism = get_mechanism_by_chembl_id(chembl_id)
similar_mol = get_similar_activity_molecules_chembl_ids(mechanism)
names = get_names_from_chembl_ids(similar_mol)
pubchem_cids = []
for name in names:
    pubchem_cids.append(get_compound_cid(name))
cids = [item for item in pubchem_cids if item is not None]
pubchem_smiles = smiles_from_pubchem_cids(cids)
pubchem_names = names_from_pubchem_cids(cids)
positive_df = pd.DataFrame({"cid": cids, "name": pubchem_names, "smiles": pubchem_smiles})
positive_df = positive_df.drop_duplicates(subset=["smiles"])
positive_df["std_smiles"] = recompute_smiles(positive_df["smiles"].to_list())
positive_df["compound_of_interest"] = True
positive_df["len_smiles"] = positive_df["std_smiles"].str.len()
positive_df = positive_df.drop(columns=["smiles"])
n_positives = positive_df.shape[0]
random_db = pd.read_csv("random_db.csv", low_memory=False)
random_db = random_db.drop(columns=["Unnamed: 0"])
random_db.columns = ["cid", "name", "std_smiles", "len_smiles"]
len_smiles_boundaries = [positive_df["len_smiles"].mean()-positive_df["len_smiles"].std(),
                         positive_df["len_smiles"].mean()+positive_df["len_smiles"].std()]
random_db = random_db[(random_db["len_smiles"] > len_smiles_boundaries[0]) & \
                      (random_db["len_smiles"] < len_smiles_boundaries[1])]
random_df = random_db.sample(n_positives*9)
random_df["compound_of_interest"] = False
final_df = pd.concat([positive_df, random_df])
final_df = final_df.reset_index(drop=True)
final_df.to_csv(output_file_name)
print("Dataset generation finished !")