from loguru import logger
import numpy as np
from numpy.typing import NDArray
import torch
import srsly
from torch import Tensor
from torch import nn

from lsdist.dataset.ssch_sample import MAX_NEW_LEN
from lsdist.utils.io import use_dir, use_file
import lsdist.model.backbone as backbone_module
from lsdist.model.net import DiscreteDistributionNet

DEFAULT_PT_CKPT = "saved/ssch/pt_train/pt_ckpt"
DEFAULT_TAG = "20240523A-data-noise-0.8>epoch=108,train_loss=0.0398698,valid_loss=0.102746"


class Predictor:
    def __init__(
        self,
        llm_creator: object,
        dist_model: nn.Module,
        threshold: float = 0.8,
    ) -> None:
        self.llm_creator = llm_creator
        self.dist_model = dist_model
        self.threshold = threshold

    def predict_length(self, inputs, ids=None):
        embedding_last, embedding_mean = self.llm_creator.embed(inputs)
        self.dist_model.eval()
        with torch.no_grad():
            out: Tensor = self.dist_model(embedding_last.float())

        pdf: NDArray = out.softmax(dim=1).cpu().numpy()
        cdf = np.cumsum(pdf, axis=1)
        lengths = (cdf >= self.threshold).argmax(axis=1)
        return lengths

    def predict_pdfs(self, inputs, ids=None):
        embedding_last, embedding_mean = self.llm_creator.embed(inputs)
        self.dist_model.eval()
        with torch.no_grad():
            out: Tensor = self.dist_model(embedding_last.float())

        pdf: NDArray = out.softmax(dim=1).cpu().numpy()
        return pdf

    @classmethod
    def get_dist_model(
        cls, pt_ckpt: str = DEFAULT_PT_CKPT, tag: str = DEFAULT_TAG, device: str = "cuda"
    ):
        pt_ckpt = use_dir(pt_ckpt)
        model_config_dic = srsly.read_yaml(use_file(pt_ckpt / tag / "model_config.yml", new=False))
        logger.debug(f"{model_config_dic=}")
        pt_config_dic = srsly.read_json(use_file(pt_ckpt / tag / "pt_config.json", new=False))
        logger.debug(f"{pt_config_dic=}")

        backbone: backbone_module.TBackbone = getattr(
            backbone_module, model_config_dic["backbone"]["class"]
        )(**model_config_dic["backbone"]["kwargs"])
        model = DiscreteDistributionNet(backbone, max_seq_len=MAX_NEW_LEN)
        model = model.to(device)
        logger.info(f"{model=!r}")

        model_state_fn = use_file(pt_ckpt / tag / "model_state.pt", new=False)
        model.load_state_dict(torch.load(model_state_fn, map_location=device))
        logger.info(f"Load {model_state_fn=}")

        return model
