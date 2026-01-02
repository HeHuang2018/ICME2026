import timm
import torch
import torch.nn as nn
import core.model.resnet as Resnet
from core.model.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_D109_MASK

from copy import deepcopy
from robustbench.utils import load_model
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer
from packaging import version
from loguru import logger as log
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
import torchvision
from transformers import BeitFeatureExtractor, Data2VecVisionForImageClassification
import getpass
username = getpass.getuser()
from typing import Union

# def build_model(cfg):
#     if cfg.CORRUPTION.DATASET == "domainnet126":
#         base_model = ResNetDomainNet126(arch=cfg.MODEL.ARCH, checkpoint_path=cfg.MODEL.CKPT_PATH, num_classes=cfg.CORRUPTION.NUM_CLASS)
#     else:
#         try:
#             # load model from torchvision
#             base_model = get_torchvision_model(cfg.MODEL.ARCH, weight_version="IMAGENET1K_V1")
#         except ValueError:
#             try:
#                 # load model from timm
#                 base_model = get_timm_model(cfg.MODEL.ARCH)
#             except ValueError:
#                 try:
#                     # load some custom models
#                     if cfg.MODEL.ARCH == "resnet26_gn":
#                         base_model = resnet26.build_resnet26()
#                         checkpoint = torch.load(cfg.MODEL.CKPT_PATH, map_location="cpu")
#                         base_model.load_state_dict(checkpoint['net'])
#                         base_model = normalize_model(base_model, resnet26.MEAN, resnet26.STD)
#                     else:
#                         raise ValueError(f"Model {cfg.MODEL.ARCH} is not supported!")
#                     logger.info(f"Successfully restored model '{cfg.MODEL.ARCH}' from: {cfg.MODEL.CKPT_PATH}")
#                 except ValueError:
#                     # load model from robustbench
#                     dataset_name = cfg.CORRUPTION.DATASET.split("_")[0]
#                     base_model = load_model(
#                         model_name=cfg.MODEL.ARCH,
#                         dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
#                         threat_model='corruptions'
#                     )


#         # In case of the imagenet variants, wrap a mask around the output layer to get the correct classes
#         if cfg.CORRUPTION.DATASET in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
#             mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")
#             base_model = ImageNetXWrapper(base_model, mask=mask)

#     return base_model.cuda()


def get_transformers_model(model_name="d2v"):
    feature_extractor = BeitFeatureExtractor.from_pretrained(f"/gemini/code/xhy/CTTA/models/data2vec-vision-base-ft1k")
    model = Data2VecVisionForImageClassification.from_pretrained(f"/gemini/code/xhy/CTTA/models/data2vec-vision-base-ft1k")
    return model, feature_extractor

def build_model(cfg):
    if cfg.MODEL.ARCH in ['vit_base_patch16_224']:
        model = get_timm_model(cfg.MODEL.ARCH)
        return model
    if cfg.MODEL.ARCH in ['Standard_VITB_REM']:
        class VisionTransformerExtractor(torchvision.models.vision_transformer.VisionTransformer):
            def __init__(self, **kwargs: Any):
                super().__init__(
                    image_size=224,
                    patch_size=16,
                    num_layers=12,
                    num_heads=12,
                    hidden_dim=768,
                    mlp_dim=3072,
                    **kwargs
                )

            def forward(self, x: torch.Tensor, return_attn: bool = False) -> Any:
                logits = super().forward(x)

                if not return_attn:
                    return logits

                with torch.no_grad():
                    x_processed = self._process_input(x)
                    n = x_processed.shape[0]
                    batch_class_token = self.class_token.expand(n, -1, -1)
                    x_processed = torch.cat([batch_class_token, x_processed], dim=1)

                    intermediate_x = x_processed
                    for layer in self.encoder.layers[:-1]:
                        intermediate_x = layer(intermediate_x)

                    last_block = self.encoder.layers[-1]
                    
                    _, attn_weights = last_block.self_attention(
                        last_block.ln_1(intermediate_x),
                        last_block.ln_1(intermediate_x),
                        last_block.ln_1(intermediate_x),
                        need_weights=True,
                        average_attn_weights=False
                    )
                
                return logits, attn_weights

        class ImageNormalizer(nn.Module):
            def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
                super(ImageNormalizer, self).__init__()
                self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
                self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return (input - self.mean) / self.std

        class NormalizedModel(nn.Module):
            def __init__(self, model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
                super().__init__()
                self.model = model
                self.normalizer = ImageNormalizer(mean, std)

            def forward(self, x: torch.Tensor, **kwargs):
                x_normalized = self.normalizer(x)
                return self.model(x_normalized, **kwargs)
        model_weights = torchvision.models.get_model_weights("vit_b_16")
        model_weights = getattr(model_weights, "IMAGENET1K_V1")
        transform = model_weights.transforms()
        model = VisionTransformerExtractor()
        model.load_state_dict(model_weights.get_state_dict(progress=True))
        model.eval()
        model = NormalizedModel(model, transform.mean, transform.std)

        # model = load_model(
        #     model_name=cfg.MODEL.ARCH,
        #     dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
        #     threat_model='corruptions'
        # )
        return model
    if cfg.MODEL.ARCH in ['PointNet']:
        from core.model.pointnet import PointNetCls
        model = PointNetCls(cfg.CORRUPTION.NUM_CLASS,True)
        checkpoint = torch.load('gemini/code/xhy/CTTA/models/PointNet.pth')
        new_state_dict = checkpoint['model_state']
        new_state_dict = {k.replace("module.model.", ""): v for k, v in new_state_dict.items()}
        model.load_state_dict(new_state_dict)
        return model

    if cfg.CORRUPTION.DATASET in ['gtsrb','eurosat']:
        checkpoint = torch.load(f"gemini/code/xhy/CTTA/models/{cfg.CORRUPTION.DATASET}.pt")
        model = torchvision.models.get_model(cfg.MODEL.ARCH, pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, cfg.CORRUPTION.NUM_CLASS)
        model.load_state_dict(checkpoint['model_state'])

        # normalization = ImageNormalizer(mean=checkpoint['normalization']['mean'], std=checkpoint['normalization']['std'])
        # model = timm.create_model('resnet50', num_classes=cfg.CORRUPTION.NUM_CLASS,  pretrained=True)
        return model

    if cfg.CORRUPTION.DATASET in ['poverty']:
        checkpoint = torch.load(cfg.CKPT_DIR)
        from core.model.resnet18_ms import resnet18_ms
        model = resnet18_ms(1,8)
        new_checkpoint = {}
        for k, v in checkpoint["algorithm"].items():
            name = k.replace('model.', '') 
            new_checkpoint[name] = v
        model.load_state_dict(new_checkpoint)
        return model

    if cfg.CORRUPTION.DATASET == "iwildcam":
        model = torchvision.models.get_model("resnet50")
        checkpoint = torch.load(cfg.CKPT_DIR)
        new_checkpoint = {}
        for k, v in checkpoint["algorithm"].items():
            name = k.replace('model.', '') 
            new_checkpoint[name] = v

        num_classes = 182
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(new_checkpoint)
        return model

    if "data2vec-vision-base-ft1k" in cfg.MODEL.ARCH:
        base_model, feature_extractor = get_transformers_model(cfg.MODEL.ARCH)
        base_model = D2VWrapper(model=base_model, feature_extractor=feature_extractor)
        
        if cfg.CORRUPTION.DATASET == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif cfg.CORRUPTION.DATASET == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif cfg.CORRUPTION.DATASET == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        
        return base_model.cuda()


    if cfg.CORRUPTION.DATASET == "domainnet126":
        log.info("Building DomainNet126 model...")
        return ResNetDomainNet126(
            arch=cfg.MODEL.ARCH,
            checkpoint_path=cfg.MODEL.CKPT_PATH,
            num_classes=cfg.CORRUPTION.NUM_CLASS
        )

    if cfg.CORRUPTION.DATASET in ["camelyon17_v1","camelyon17"]:
        model = torchvision.models.get_model("densenet121")
        checkpoint = torch.load("/gemini/code/xhy/CTTA/models/camelyon17_groupDRO_densenet121_seed0.pth")
        # checkpoint = torch.load(cfg.CKPT_DIR)
        new_checkpoint = {}
        for k, v in checkpoint["algorithm"].items():
            name = k.replace('model.', '') 
            new_checkpoint[name] = v

        num_classes = 2
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model.load_state_dict(new_checkpoint)
        return model

    if cfg.CORRUPTION.DATASET in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        log.info(f"Wrapping model with mask for dataset {cfg.CORRUPTION.DATASET}")
        base_model = get_torchvision_model(cfg.MODEL.ARCH, 'IMAGENET1K_V1')
        mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")
        base_model = ImageNetXWrapper(base_model, mask=mask)
        return base_model.cuda()

    try:
        if cfg.MODEL.ARCH in ['resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']:
            base_model = Resnet.__dict__[cfg.MODEL.ARCH](pretrained=True,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
        elif cfg.MODEL.ARCH in ['resnet50_gn','resnetv2_50d_gn']:
            base_model = timm.create_model('resnet50_gn', pretrained=False,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
            checkpoint = torch.load('/gemini/code/xhy/CTTA/models/imagenet/resnet50_gn_a1h2-8fe6c4d0.pth')
            base_model.load_state_dict(checkpoint)
            log.info("Model created successfully!")

        elif cfg.MODEL.ARCH in ['efficientnet_b4']:
            base_model = timm.create_model('efficientnet_b4', pretrained=True,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
        elif cfg.MODEL.ARCH in ['vit_b_16','swin_b','convnext_tiny','mobilenet_v3_small','mobilenet_v3_large','mobilenet_v2','swin_v2_b','swin_v2_s','efficientnet_v2_s','efficientnet_v2_m','convnext_base','densenet161','densenet121','wide_resnet50_2','wide_resnet101_2','resnext50_32x4d']:
            base_model = get_torchvision_model(cfg.MODEL.ARCH,'IMAGENET1K_V1')
        else:
            base_model = load_model(
                model_name=cfg.MODEL.ARCH,
                dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
                threat_model='corruptions'
            )
    except ValueError:
        base_model = load_model(
            model_name=cfg.MODEL.ARCH,
            dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
            threat_model='corruptions'
        )
        
    return base_model.cuda()

        
class TransformerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

class D2VWrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.normalize = None

    def forward(self, x):
        inputs = {"pixel_values": x}

        x = self.model(**inputs)

        x = x.logits

        return x
    
class D2VSplitWrapper(torch.nn.Module):
    def __init__(self, model, feature_extractor=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.normalize = None

    def forward(self, x):
        inputs = {
            "pixel_values": x,
            "head_mask": None,
            "output_attentions": None,
            "output_hidden_states": None,
            "return_dict": False
        }

        outputs = self.model.model.data2vec_vision(
            **inputs
        )
        
        outputs = outputs[1]

        return outputs

class ImageNetXMaskingLayer(nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]

class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)

def split_up_model(model, arch_name, dataset_name):
    if dataset_name in ['gtsrb','eurosat']:
        # checkpoint = torch.load(f"gemini/code/xhy/CTTA/models/{dataset_name}.pt")
        # normalization = ImageNormalizer(mean=checkpoint['normalization']['mean'], std=checkpoint['normalization']['std'])
        # encoder = nn.Sequential(normalization, *list(model.children())[:-1], *list(model.children())[:-1], nn.Flatten())
        # classifier = model.fc
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        classifier = model.fc
        return encoder, classifier
        
    if dataset_name in ["modelnet40_c"]:
        encoder = nn.Sequential(*list(model.children())[:-7])
        classifier = nn.Sequential(
            model.fc1,
            model.bn1,
            nn.ReLU(),
            model.dropout,
            model.fc2,
            model.dropout,
            model.bn2,
            nn.ReLU(),
            model.fc3,
            nn.LogSoftmax(dim=1)
        )
        return encoder, classifier

    if dataset_name in ["camelyon17_v1"]:
        encoder = nn.Sequential(model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.classifier
        return encoder, classifier

    if dataset_name in ["imagenet_shrtcut"]:
        encoder = nn.Sequential(
            model.conv1, 
            model.bn1, 
            model.relu, 
            model.layer1, 
            model.layer2, 
            model.layer3, 
            model.layer4,
            model.avgpool,
            nn.Flatten()
        )
        classifier = model.classifier
        return encoder, classifier
        
    if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
        mask = eval(f"{dataset_name.upper()}_MASK")
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))
        return encoder, classifier

    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)
    elif arch_name == 'data2vec-vision-base-ft1k':
        encoder = D2VSplitWrapper(model)
        classifier = model.model.classifier
    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "wide_resnet50_2" in arch_name or "resnext50_32x4d" in arch_name:
        encoder = nn.Sequential(*list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "resnet" in arch_name or arch_name in {"Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        classifier = model.fc
    elif "Standard_R50" in arch_name:
        encoder = nn.Sequential(*list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    return encoder, classifier


def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    # check if the specified model name is available in torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    # get the weight object of the specified model and the available weight initialization names
    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    # check if the specified type of weights is available
    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the specified pre-trained weights
    model = torchvision.models.get_model(model_name, weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    log.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    return model

def get_timm_model(model_name: str):
    """
    Restore a pre-trained model from timm: https://github.com/huggingface/pytorch-image-models/tree/main/timm
    Quickstart: https://huggingface.co/docs/timm/quickstart
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in timm. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True)
    log.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # add the corresponding input normalization to the model
    if hasattr(model, "pretrained_cfg"):
        log.info(f"General model information: {model.pretrained_cfg}")
        log.info(f"Adding input normalization to the model using: mean={model.pretrained_cfg['mean']} \t std={model.pretrained_cfg['std']}")
        model = normalize_model(model, mean=model.pretrained_cfg["mean"], std=model.pretrained_cfg["std"])
    else:
        pass
        # raise AttributeError(f"Attribute 'pretrained_cfg' is missing for model '{model_name}' from timm."
        #                      f" This prevents adding the correct input normalization to the model!")
    return model


class ResNetDomainNet126(torch.nn.Module):
    """
    Architecture used for DomainNet-126
    """
    def __init__(self, arch: str = "resnet50", checkpoint_path: str = None, num_classes: int = 126, bottleneck_dim: int = 256):
        super().__init__()

        self.arch = arch
        self.bottleneck_dim = bottleneck_dim
        self.weight_norm_dim = 0

        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = torchvision.models.get_model(self.arch, weights="IMAGENET1K_V1")
            modules = list(model.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            model = torchvision.models.get_model(self.arch, weights="IMAGENET1K_V1")
            model.fc = torch.nn.Linear(model.fc.in_features, self.bottleneck_dim)
            bn = torch.nn.BatchNorm1d(self.bottleneck_dim)
            self.encoder = torch.nn.Sequential(model, bn)
            self._output_dim = self.bottleneck_dim

        self.fc = torch.nn.Linear(self.output_dim, num_classes)

        if self.use_weight_norm:
            self.fc = torch.nn.utils.weight_norm(self.fc, dim=self.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        else:
            log.warning(f"No checkpoint path was specified. Continue with ImageNet pre-trained weights!")

        # add input normalization to the model
        self.encoder = nn.Sequential(ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), self.encoder)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        model_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint.keys() else checkpoint["model"]
        for name, param in model_state_dict.items():
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        log.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[1][0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1][1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.weight_norm_dim >= 0