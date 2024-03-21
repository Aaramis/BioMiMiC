from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset
from config_bert import BertConfig
import numpy as np
import csv


class SMILESTokenization:
    """
    Classe pour la tokenization et la préparation des datasets SMILES.
    """
    def __init__(self, model_name:str = "JuIm/SMILES_BERT", max_len:int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def tokenize(self, smiles):
        return self.tokenizer(smiles, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

    def load_and_prepare_datasets(self, dataset_paths):
        datasets = {split: load_dataset("csv", data_files=path, split='train') for split, path in dataset_paths.items()}
        tokenized_datasets = {split: dataset.map(lambda examples: self.tokenize(examples[BertConfig.smiles_column]), batched=True) for split, dataset in datasets.items()}
        return tokenized_datasets

class TrainingModule:
    """
    Module pour l'entraînement, la validation et le test du modèle, incluant l'enregistrement des métriques.
    """
    def __init__(self, model_name:str = "JuIm/SMILES_BERT", dataset_paths:str = None):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=SMILESTokenization(model_name).tokenizer)
        self.dataset_paths = dataset_paths or {}

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        softmax_scores = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        if softmax_scores.shape[1] == 2:
            auc_score = roc_auc_score(labels, softmax_scores[:, 1])
        else:
            auc_score = roc_auc_score(labels, softmax_scores, multi_class='ovo', average='macro')
       
        return {
            'accuracy': accuracy_score(labels, predictions),
            'roc_auc': auc_score,
            'precision': precision_score(labels, predictions, average='macro'),
            'recall': recall_score(labels, predictions, average='macro'),
            'f1': f1_score(labels, predictions, average='macro')
        }

    def train(self, tokenized_datasets, training_args):
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['eval'],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[MetricsLoggerCallback()]
        )
        trainer.train()
        self.model.save_pretrained(BertConfig.coconut_f_path)

class MetricsLoggerCallback(TrainerCallback):
    """
    Callback pour enregistrer les métriques de formation dans un fichier CSV après chaque évaluation.
    """
    def __init__(self, csv_file="training_metrics.csv"):
        self.csv_file = csv_file

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            flattened_metrics = {k: v for k, v in metrics.items()}
            mode = 'w' if state.epoch == 1 else 'a'
            with open(self.csv_file, mode=mode, newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=flattened_metrics.keys())
                if mode == 'w':
                    writer.writeheader()
                writer.writerow(flattened_metrics)


def main():
    dataset_paths = {"train": BertConfig.train_path,
                     "eval": BertConfig.eval_path,
                     "test": BertConfig.test_path}

    tokenization = SMILESTokenization(model_name=BertConfig.model_name,
                                      max_len=BertConfig.max_len)
    tokenized_datasets = tokenization.load_and_prepare_datasets(dataset_paths)

    training_args = TrainingArguments(
        output_dir=BertConfig.output_path,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir="./logs",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=10_000,
        save_total_limit=2,
    )

    training_module = TrainingModule()
    training_module.train(tokenized_datasets, training_args)

if __name__ == "__main__":
    main()