import pandas as pd
import os
from torch.utils.data import Dataset
import torch
from sklearn.metrics import f1_score
import sys
import xgboost
import pickle


class PSVDataset3(Dataset):
    def __init__(self, path):
        self.patients = []
        self.labels = []
        self.filenames = []
        for i, filename in enumerate(os.listdir(path)):
            if filename.endswith('.psv'):
                # Load the file into a DataFrame
                self.filenames.append(filename)
                file_path = os.path.join(path, filename)
                df = pd.read_csv(file_path, sep='|')
                df = df.fillna(1)
                if df['SepsisLabel'].sum() > 0:
                    self.labels.append(1)
                    if df['SepsisLabel'][0] == 1:
                        df = pd.DataFrame(df.iloc[0]).T
                    else:
                        index = df.index[df['SepsisLabel'] == 1].tolist()[0]
                        df = df.loc[:index][df['SepsisLabel'] == 0]
                        df = df.tail(1)
                else:
                    self.labels.append(0)
                    df = df.tail(1)
                try:
                    df = df.drop('SepsisLabel', axis=1)
                except:
                    self.wrong_df = df
                    print(i)
                    break
                # feature engineering
                vector = df.values
                self.patients.append(vector)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pat = torch.tensor(self.patients[idx], dtype=torch.float32).unsqueeze(0)
        lab = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pat[0], lab


if __name__ == '__main__':
    arguments = sys.argv
    path = arguments[1]

    test_dataset = PSVDataset3(path)

    #load the model
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)

    #test
    y_pred = model.predict(test_dataset[:][0].squeeze(dim=1))
    files_names = [name[:-4] for name in test_dataset.filenames]
    df = pd.DataFrame({'id': files_names, 'prediction': y_pred})
    df.to_csv('prediction.csv', index=False)
    y_true = test_dataset[:][1]
    print(f1_score(y_true, y_pred, average='binary'))



