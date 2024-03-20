from dataclasses import dataclass


@dataclass
class BertConfig:

    smiles_column: str = 'smiles'
    label_column: str = 'label'
    eval_path: str = "./Datasets/eval.csv"
    train_path: str = "./Datasets/train.csv"
    test_path: str = "./Datasets/train.csv"
    coconut_path: str =  "../Datasets/COCONUT_DB_std_smiles_filtered.csv"
    model_trained: str ="../../SMILES-BERT/smiles_bert_classifier_2"
    batch_size: int = 500
    num_workers: int = 8
    max_len: int = 58
    shuffle: bool = False