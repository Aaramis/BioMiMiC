import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from config_bert import BertConfig 

class SMILESDataset(Dataset):
    """
    Un Dataset personnalisé pour la prédiction des SMILES.
    """
    def __init__(self, smiles_list, tokenizer, max_len):
        """
        Initialise le dataset.
        
        :param smiles_list: Une liste de chaînes SMILES.
        :param tokenizer: Un tokenizeur pré-entraîné.
        :param max_len: La longueur maximale des séquences tokenisées.
        """
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Retourne la taille du dataset."""
        return len(self.smiles_list)

    def __getitem__(self, idx : int):
        """
        Récupère et tokenise un élément SMILES par son index.
        
        :param idx: L'index de l'élément dans le dataset.
        :return: Un dictionnaire de tenseurs correspondant à l'entrée tokenisée.
        """
        smile = self.smiles_list[idx]
        inputs = self.tokenizer(
            smile,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        return inputs

def main(args : argparse.Namespace):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("JuIm/SMILES_BERT")
    model = AutoModelForSequenceClassification.from_pretrained(BertConfig.model_trained)
    model.to(device)

    # datasets = {split: load_dataset("csv", data_files=path, split ='train') for split, path in dataset_paths.items()}
    # max_len = max(len(tokenizer.encode(smile)) for dataset in datasets.values() for smile in dataset[SMILES])

    if args.COCONUT:
        print(f"Screening COCONUT database with {BertConfig.coconut_path} file")
        dataset = load_dataset("csv", data_files=BertConfig.coconut_path, split='train')
        compound_id = dataset["compound_id"] if "compound_id" in dataset.features else None
        labels = None
    else:
        print(f"Predicting test dataset : {BertConfig.test_path} file")
        dataset = load_dataset("csv", data_files=BertConfig.test_path, split='train')
        labels = dataset[BertConfig.label_column] if BertConfig.label_column in dataset.features else None
        compound_id = None

    test_dataset = SMILESDataset(dataset[BertConfig.smiles_column],
                                 tokenizer=tokenizer,
                                 max_len=BertConfig.max_len) 

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BertConfig.batch_size,
                                 shuffle=BertConfig.shuffle,
                                 num_workers=BertConfig.num_workers)

    predictions = []
    proba = []
    for batch in tqdm(test_dataloader, total=len(test_dataset) // BertConfig.batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=-1)  # Appliquer softmax aux logits pour obtenir les probabilités
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        max_prob, max_idx = torch.max(probabilities, dim=-1)
        proba.extend(max_prob.tolist())


    result_df = pd.DataFrame({
        "compound_id": compound_id,
        "SMILES": test_dataset.smiles_list,
        "Predicted_Label": predictions,
        "True_Label": labels,
        "Proba": proba
    })

    result_df.to_csv("./predictions.csv", index=False)
    
    print("Les prédictions ont été sauvegardées dans './predictions.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meta Parameters')

    parser.add_argument('--COCONUT', action='store_true', help="Flag to trigger COCONUT.")
    args = parser.parse_args()

    main(args)