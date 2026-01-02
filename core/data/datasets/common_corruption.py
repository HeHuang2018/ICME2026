from .base_dataset import TTADatasetBase, DatumRaw
from robustbench.data import load_cifar100c, load_cifar10c, load_imagenet3dcc, load_imagenetc, load_imagenet_c_bar

from tqdm import tqdm, trange
class CorruptionDataset(TTADatasetBase):
    def __init__(self, cfg):
        self.corruptions = cfg.CORRUPTION.TYPE
        self.severity = cfg.CORRUPTION.SEVERITY
        self.corruptions = [self.corruptions] if not isinstance(self.corruptions, list) else self.corruptions
        self.severity = [self.severity] if not isinstance(self.severity, list) else self.severity

        dataset_loaders = {
            "cifar10": load_cifar10c,
            "cifar100": load_cifar100c,
            "imagenet_3dcc": load_imagenet3dcc,
            "imagenet": load_imagenetc,
            "imagenet_c_bar":load_imagenet_c_bar
        }
        self.load_image = dataset_loaders.get(cfg.CORRUPTION.DATASET)

        self.domain_id_to_name = {}
        data_source = []
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption_type in tqdm(enumerate(self.corruptions), desc="Loading Corruption Dataset"):
                d_name = f"{corruption_type}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name
                x_test, y_test = self.load_image(
                    cfg.CORRUPTION.NUM_EX,
                    severity, 
                    cfg.DATA_DIR, 
                    False,
                    [corruption_type]
                )
                for i in range(len(y_test)):
                    data_item = DatumRaw(x_test[i], y_test[i].item(), d_id)
                    data_source.append(data_item)

        super().__init__(cfg, data_source)
