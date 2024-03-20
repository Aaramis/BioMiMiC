import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.ao import quantization
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from config_bert import BertConfig 


class SMILESDataset(Dataset):
    """
    Un Dataset personnalisé pour la prédiction des SMILES.
    """
    def __init__(self, smiles_list, tokenizer, max_len, cache = None):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache = cache or {} # Permet de garder en cache les tokens

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx : int):
        """
        Récupère et tokenise un élément SMILES par son index.
        
        :param idx: L'index de l'élément dans le dataset.
        :return: Un dictionnaire de tenseurs correspondant à l'entrée tokenisée.
        """
        smile = self.smiles_list[idx]
        if smile in self.cache:
            return self.cache[smile]
        inputs = self.tokenizer(
            smile,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        self.cache[smile] = inputs
        return inputs


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BertConfig.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(BertConfig.model_trained)
    model = quantization.quantize_dynamic(model, dtype=torch.qint8) # Quantization dynamic
    return tokenizer, model


def main(args : argparse.Namespace):

    tokenizer, model = load_model_and_tokenizer()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
            predicted_labels = torch.argmax(logits, dim=-1)
            predicted_proba = probabilities[range(len(predicted_labels)), predicted_labels]
        predictions.extend(predicted_labels.tolist())
        proba.extend(predicted_proba.tolist())

    result_df = pd.DataFrame({
        "compound_id": compound_id,
        "SMILES": test_dataset.smiles_list,
        "Predicted_Label": predictions,
        "True_Label": labels,
        "Proba": proba
    })

    result_df.to_csv(args.prediction_path, index=False)
    
    print(f"Les prédictions ont été sauvegardées dans {args.prediction_path}.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Meta Parameters')
    parser.add_argument('--COCONUT', action='store_true', help="Flag to trigger COCONUT.")
    parser.add_argument('--prediction_path', type=str,
                        default="./predictions.csv",
                        help="Path for prediction output")
    args = parser.parse_args()

    main(args)