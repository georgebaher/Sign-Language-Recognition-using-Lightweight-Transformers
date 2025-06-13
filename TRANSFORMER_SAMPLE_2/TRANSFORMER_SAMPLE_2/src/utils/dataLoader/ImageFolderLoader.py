import glob
import os
import pathlib
from typing import Callable, Optional, Tuple

import PIL
import PIL.Image
import torch

import re

import torchvision
import torchvision.transforms.v2


class ImageFolder(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        *,
        transform: Optional[Callable] = None,
        frame_idx_from_filepath: Optional[Callable[[str], int]] = None,
        video_name_from_filepath: Optional[Callable[[str], str]] = None
    ) -> None:
        super(ImageFolder, self).__init__()
        self.root = root
        self.paths = glob.glob(
            os.path.join(
                root,
                '**',
                '*.jpg'
            ),
            recursive=True
        )
        self.paths = list(sorted(self.paths))
        self.transforms = transform
        if self.transforms == None:
            self.transforms = torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True)
            ])

        if frame_idx_from_filepath is not None:
            self._frame_idx_from_filepath = frame_idx_from_filepath
        else:

            self._default_frame_idx_match = re.compile(
                r'.*?_(?P<frame_idx>\d{6})')
            self._frame_idx_from_filepath = self._default_frame_idx_from_filepath

        if video_name_from_filepath is not None:
            self._video_name_from_filepath = video_name_from_filepath
        else:
            self._video_name_from_filepath = self._default_video_name_from_filepath

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[PIL.Image.Image, str, int]:
        path = self.paths[idx]
        img = PIL.Image.open(path, 'r')
        video_name = self._video_name_from_filepath(path)
        frame_idx = self._frame_idx_from_filepath(path)
        return self.transforms(img), video_name, frame_idx

    def _default_frame_idx_from_filepath(self, filepath: str) -> int:
        frame_name = str(pathlib.Path(filepath).stem)
        matches = self._default_frame_idx_match.match(frame_name)
        return int(matches.group('frame_idx'))

    def _default_video_name_from_filepath(self, filepath: str) -> str:
        return str(pathlib.Path(filepath).parent.absolute().stem)
