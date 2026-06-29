import attrs
import torch
import torchvision.transforms.v2.functional as F

from .image import Image

__all__ = ["Focus"]


@attrs.define(frozen=True)
class Focus:
    """画像の正方切り取りを指示する行動 a=(point, zoom)。

    point は [-1, 1] (中心 0, y=+1 が下)、zoom は (0, 1]。 point/zoom
    はスカラーで値等価が自然なため eq=True (既定) のまま hashable とする。
    """

    point: tuple[float, float]
    zoom: float

    def __attrs_post_init__(self) -> None:
        x, y = self.point
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            raise ValueError(f"Focus point must be within [-1, 1], got {self.point}")
        if not (0.0 < self.zoom <= 1.0):
            raise ValueError(f"Focus zoom must be within (0, 1], got {self.zoom}")

    def tensor(self) -> torch.Tensor:
        """行動ベクトル [x, y, zoom] を shape (3,) の float32 tensor で返す (CPU)。"""
        x, y = self.point
        return torch.tensor([x, y, self.zoom], dtype=torch.float32)

    def __call__(self, image: Image) -> Image:
        """この行動で image を正方切り取りした新しい Image を返す (はみ出しは 0 埋め)。"""
        squared = image.square_pad()  # 既に正方なら square_pad が自身を返す
        s = squared.height
        x, y = self.point
        cx = (s / 2) * (1 + x)
        cy = (s / 2) * (1 + y)
        crop = max(1, int(round(self.zoom * s)))
        left = int(round(cx - crop / 2))
        top = int(round(cy - crop / 2))
        return Image(F.crop(squared.tensor, top, left, crop, crop))
