from dataclasses import dataclass


@dataclass
class BertConfig:

    smiles_column: str = 'smiles'
    label_column: str = 'label'
    data_path: str =  "../Datasets/ssris_dataset_std_smiles.csv"
    eval_path: str = "./Datasets/eval.csv"
    train_path: str = "./Datasets/train.csv"
    test_path: str = "./Datasets/train.csv"
    coconut_path: str = "../Datasets/COCONUT_DB_std_smiles.csv"
    coconut_f_path: str = "../Datasets/COCONUT_DB_std_smiles_filtered.csv"
    output_path: str = "../Models/smiles_bert_classifier_2"
    model_trained: str = "../../SMILES-BERT/smiles_bert_classifier_2"
    model_name: str = "JuIm/SMILES_BERT"
    batch_size: int = 4
    num_workers: int = 8
    max_len: int = 512
    shuffle: bool = False
