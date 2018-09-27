from torch.utils.data import Dataset


class TableDataset(Dataset):
    def __init__(self, X_table, y_table):
        self.X_table = X_table
        self.y_table = y_table

    def __len__(self):
        return len(self.X_table)

    def __getitem__(self, idx):
        X = self.X_table[idx]
        y = self.y_table[idx]
        sample = (X, y)

        return sample
