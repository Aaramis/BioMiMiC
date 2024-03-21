from config_bert import BertConfig
import pandas as pd


class DataProcessor:

    def __init__(self, input_file: str) -> None:
        self.input_file = input_file
        self.df = pd.read_csv(input_file, sep=",", index_col=0)
        self.lower_bound = 0
        self.upper_bound = 0

    def calculate_smiles_length(self) -> None:
        self.df['smiles_length'] = self.df['smiles'].apply(len)

    def calculate_bounds(self, factor : int = 1):
        mean_length = self.df['smiles_length'].mean()
        std_length = self.df['smiles_length'].std()
        self.lower_bound = mean_length - factor * std_length
        self.upper_bound = mean_length + factor * std_length

    def filter_coconut_data(self, lower_bound : int, upper_bound :int) -> None:
        self.filtered_df = self.df[(self.df['smiles_length'] >= lower_bound) & 
                                   (self.df['smiles_length'] <= upper_bound)]

    def drop_smiles_length_column(self) -> None:
        self.filtered_df = self.filtered_df.drop(columns=['smiles_length'])

    def save_filtered_data(self, save_path: str):
        self.filtered_df.to_csv(save_path, index=False)


def Process_Coconut():

    processor_data = DataProcessor(BertConfig.data_path)
    processor_data.calculate_smiles_length()
    processor_data.calculate_bounds()

    processor_coconut = DataProcessor(BertConfig.coconut_path)
    processor_coconut.calculate_smiles_length()
    processor_coconut.filter_coconut_data(lower_bound=processor_data.lower_bound,
                                          upper_bound=processor_data.upper_bound)
    processor_coconut.drop_smiles_length_column()
    processor_coconut.save_filtered_data(BertConfig.coconut_f_path)


if __name__ == "__main__":
    Process_Coconut()