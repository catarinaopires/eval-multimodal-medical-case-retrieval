import os

import pandas as pd
import torch
from PIL import Image


class ArticleImagesDataset(torch.utils.data.Dataset):
    """Article Images dataset."""

    def __init__(
        self,
        csv_file="figures/images/image-paths.csv",
        data_dir="figures/images/",
        transform=None,
    ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_relative_path = self.images_frame.iloc[idx, 0]
        img_path = os.path.join(self.data_dir, img_relative_path)

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "img_path": img_relative_path}
