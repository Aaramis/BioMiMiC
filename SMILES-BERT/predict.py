import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm

SMILES='smiles'
LABEL='label'
# TEST_PATH="../datasets/test.csv"
TEST_PATH="../Datasets/COCONUT_DB_std_smiles_filtered.csv"
EVAL_PATH="./Datasets/eval.csv"
TRAIN_PATH="./Datasets/train.csv"
MODEL_TRAINED="../Models/smiles_bert_classifier"
BATCH_SIZE = 500
NUM_WORKERS = 8
MAX_LEN = 58  # Correspond à la taille définit au moment de l'entrainemen du model (Trouver une astuce pour régler le lien)


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

    def __getitem__(self, idx):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("JuIm/SMILES_BERT")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TRAINED)
    model.to(device)

    dataset_paths = {"train": TRAIN_PATH, "eval": EVAL_PATH, "test": TEST_PATH}

    datasets = {split: load_dataset("csv", data_files=path, split ='train') for split, path in dataset_paths.items()}

    # max_len = max(len(tokenizer.encode(smile)) for dataset in datasets.values() for smile in dataset[SMILES])

    labels = datasets["test"][LABEL] if LABEL in datasets["test"].features else None
    compound_id = datasets["test"]["compound_id"] if "compound_id" in datasets["test"].features else None

    test_dataset = SMILESDataset(datasets["test"][SMILES], tokenizer, MAX_LEN)  
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    predictions = []
    for batch in tqdm(test_dataloader, total=len(test_dataset) // BATCH_SIZE):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())


    result_df = pd.DataFrame({
        "compound_id": compound_id,
        "SMILES": test_dataset.smiles_list,
        "Predicted_Label": predictions,
        "True_Label": labels
    })

    result_df.to_csv("./predictions.csv", index=False)
    
    print("Les prédictions ont été sauvegardées dans './predictions.csv'.")

if __name__ == "__main__":
    main()