import numpy as np
import cv2
from torch.utils.data import Dataset


class AdaptationDataset(Dataset):
    def __init__(self, auxs, source_df, target_df, augs, label_cols_list):
        self.source_df = source_df
        self.labels = source_df[label_cols_list].values
        self.target_df = target_df
        self.augs = augs
        self.auxs = auxs
    
    def __getitem__(self, idx: int):
        x_src = self.source_df.at[idx, 'Images']
        x_src = self.read_image(x_src)
        x_src = self.augs(x_src)

        y_src = self.labels[idx]

        tgt_idx = np.random.randint(0,len(self.target_df))
        x_tgt = self.target_df.at[tgt_idx, 'Images']
        x_tgt = self.read_image(x_tgt)
        x_tgt = self.augs(x_tgt)

        xa_src, a_src = self.auxs(x_src)
        xa_tgt, a_tgt = self.auxs(x_tgt)
        
        src = {
            'x': x_src, # original source image
            'y': y_src, # label of source image
            'xa': xa_src, # augmented source image
            'a': a_src # augmentation label
        }

        tgt = {
            'x': x_tgt, # original target image
            'xa': xa_tgt, # augmented target image
            'a': a_tgt # augmentation label
        }
        return src, tgt

    def __len__(self):
        return len(self.source_df)

    @staticmethod
    def read_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


