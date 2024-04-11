import html
from functools import partial
from pathlib import Path
from typing import Set, Tuple, Sequence
import albumentations as A
import cv2 as cv
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from src.ImageTransforms import ImageTransforms
from src.LabelParser import LabelParser
from src.utils import read_xml
from torch.utils.data import DataLoader

class IAMDataset(Dataset):

    _pad_token = "<PAD>"
    _sos_token = "<SOS>"
    _eos_token = "<EOS>"
    _optional_tokens = ["<END-OF-REGION>", "<MATH>", "<DELETED-TEXT>", "<TABLE>", "<DRAWING>"] # TO BE CHECKED

    MEAN = 0.8275
    STD  = 0.2314
    MAX_WIDTH = 2479
    _transform = None

    def __init__(self, base_dir, sample_set, embedding_loader, writer_id=False):
        self.base_dir = Path(base_dir)
        # assert sample_set in ["train", "test", "val"], f"Sample set {sample_set} not known"
        self.sample_set = sample_set
        self.embedding_loader = embedding_loader
        self.return_writer_id = writer_id
        if not hasattr(self, "data"):
            self.data = self._get_data()

        if self.embedding_loader is None:
            vocab = [self._pad_token, self._sos_token, self._eos_token]
            s = "".join(self.data["target_seq"].tolist())

            vocab += sorted(list(set(s)))
            self.embedding_loader = LabelParser().fit(vocab)

        if not "target_enc" in self.data.columns:
            self.data.insert(
                2,
                "target_enc",
                self.data["target_seq"].apply(
                    lambda x: np.array(
                        self.embedding_loader.encode_labels(
                            [c for c in (x)]
                        )
                    )
                )
            )

        if self._transform is None:
            self._transform = self._get_transform(sample_set)


    def __getitem__(self, item):
        row =  self.data.iloc[item]
        target_enc = row["target_enc"]
        y_upper = row["y_upper"]
        y_lower = row["y_lower"]
        img = cv.imread(row["img_path"], cv.IMREAD_GRAYSCALE)
        img = img[y_upper:y_lower, :]
        assert isinstance(img, np.ndarray), "Image read error"
        if self._transform is not None:
            img = self._transform(image=img)["image"]

        if self.return_writer_id:
            return img, target_enc, row["writer_id"]
        return img, target_enc



    def __len__(self):
        return len(self.data)

    def max_seq_length(self):
        return self.data["target_len"].max()

    def _get_data(self) -> pd.DataFrame:
        data = {
            "img_path": [],
            "img_id": [],
            "target_seq": [],
            "target_len": [],
            "y_upper": [],
            "y_lower": [],
            "writer_id": []
        }

        page_dirs = ["formsA-D","formsE-H", "formsI-Z"]
        xml_dir = self.base_dir / "xml"
        for directory in page_dirs:
            directory = self.base_dir / directory
            for image in directory.iterdir():
                image_id = image.stem
                image_metadata = read_xml(xml_dir / (image_id + ".xml"))


                target_text = []
                for line in image_metadata.iter("line"):
                    target_text += [html.unescape(line.get("text", ""))]

                target_seq = "\n".join(target_text)
                writer_id = image_metadata.attrib["writer-id"]

                lines = image_metadata[1]
                y_upper = int(lines[0].attrib["asy"])
                y_lower = int(lines[-1].attrib["dsy"])

                data["img_path"].append(str(image))
                data["img_id"].append(image_id)
                data["target_seq"].append(target_seq)
                data["target_len"].append(len(target_seq))
                data["y_upper"].append(y_upper)
                data["y_lower"].append(y_lower)
                data["writer_id"].append(int(writer_id))

        return pd.DataFrame(data)

    @staticmethod
    def collate_fn(
            batch,
            pad_val,
            eos_tkn_idx,
            returns_writer_id = False):

        if returns_writer_id:
            imgs, targets, writer_ids = zip(*batch)
        else:
            imgs, targets = zip(*batch)

        img_sizes = [img.shape for img in imgs]
        if not len(set(img_sizes)) == 1:
            hs, ws = zip(*img_sizes)
            pad_fn = A.PadIfNeeded(max(hs), max(ws), border_mode=cv.BORDER_CONSTANT, value=0)

            imgs = [pad_fn(image=im)["image"] for im in imgs]


        seq_lengths = [t.shape[0] for t in targets]
        targets_padded = np.full((len(targets), max(seq_lengths) + 1), pad_val)
        for i, tgt in enumerate(targets):
            targets_padded[i, : seq_lengths[i]] = tgt
            targets_padded[i, seq_lengths[i]] = eos_tkn_idx

        imgs, targets_padded = torch.tensor(np.array(imgs)), torch.tensor(np.array(targets_padded))
        if returns_writer_id:
            # noinspection PyUnboundLocalVariable
            return imgs, targets_padded, torch.tensor(writer_ids)

        return imgs, targets_padded

    def set_transform_pipeline(self, split_name):
        assert split_name in ["train", "test", "val"], "Invalid split name"
        self._transform = self._get_transform(split_name)

    def _get_transform(self, split):

        max_h = (self.data["y_lower"] - self.data["y_upper"]).max()
        transform_pipelines = ImageTransforms((max_h, self.MAX_WIDTH), (self.MEAN, self.STD))
        if split == "train":
            return transform_pipelines.train_trnsf
        elif split in ["test", "val"]:
            return transform_pipelines.test_trnsf
        return None



if __name__ == "__main__":
    dataset = IAMDataset("/Users/tefannastasa/BachelorsWorkspace/BachModels/Playground/HWR/data/raw", "train", None, True)
    eos_tkn_id, sos_tkn_id, pad_tkn_id = dataset.embedding_loader.encode_labels([dataset._eos_token, dataset._sos_token, dataset._pad_token])
    collate_fn = partial(IAMDataset.collate_fn, pad_val=pad_tkn_id, eos_tkn_idx=eos_tkn_id, returns_writer_id=dataset.return_writer_id)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(len(batch[0]))