from torch import optim
import torchvision.transforms as transforms
import torch

from roadmap import transforms as custom_transforms
from roadmap import losses
from roadmap import samplers
from roadmap import datasets
from roadmap import models
from roadmap import engine
from roadmap import utils as lib
from torchsummary import summary
import math


class Getter:
    """
    This class allows to create differents object (model, loss functions, optimizer...)
    based on the config
    """

    def get(self, obj, *args, **kwargs):
        return getattr(self, f"get_{obj}")(*args, **kwargs)

    def get_transform(self, config):
        t_list = []
        for k, v in config.items():
            if hasattr(custom_transforms, k):
                t_list.append(getattr(custom_transforms, k)(**v))
            else:
                t_list.append(getattr(transforms, k)(**v))

        transform = transforms.Compose(t_list)
        lib.LOGGER.info(transform)
        return transform

    def get_optimizer(self, net, config):
        optimizers = {}
        schedulers = {
            "on_epoch": [],
            "on_step": [],
            "on_val": [],
        }

        for opt_cfg in config:
            # 1. Sélection du module cible (Tout le réseau ou une sous-partie)
            if opt_cfg.params is None:
                target_module = net
                name_key = "net"
            else:
                target_module = getattr(net, opt_cfg.params)
                name_key = opt_cfg.params

            # 2. Séparation des paramètres (Poids vs Biais/BN)
            # Cette logique est universelle pour PyTorch
            params_weight = []
            params_bias = []

            for name, param in target_module.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Détection des biais et Batch Normalization
                if 'bias' in name or len(param.shape) == 1: 
                    params_bias.append(param)
                else:
                    params_weight.append(param)

            # 3. Préparation des groupes de paramètres
            # Groupe 1 : Les poids standards (utilisent opt_cfg.kwargs)
            # Groupe 2 : Les biais (utilisent opt_cfg.kwargs + opt_cfg.bias_kwargs s'il existe)
            
            # On récupère les configs
            common_kwargs = opt_cfg.kwargs if opt_cfg.kwargs else {}
            bias_overrides = opt_cfg.bias_kwargs if hasattr(opt_cfg, 'bias_kwargs') and opt_cfg.bias_kwargs else {}
            
            # Fusion intelligente pour les biais : common + override
            bias_kwargs = common_kwargs.copy()
            bias_kwargs.update(bias_overrides)

            param_groups = [
                {'params': params_weight, **common_kwargs},
                {'params': params_bias, **bias_kwargs}
            ]

            # 4. Instanciation de l'optimiseur
            # On ne passe plus **kwargs au constructeur global car tout est dans les param_groups
            optimizer_cls = getattr(optim, opt_cfg.name)
            optimizer = optimizer_cls(param_groups)
            
            optimizers[name_key] = optimizer
            lib.LOGGER.info(f"Optimizer created for {name_key}: {optimizer}")

            # 5. Gestion des Schedulers (Code inchangé, toujours valide)
            if opt_cfg.scheduler_on_epoch is not None:
                schedulers["on_epoch"].append(self.get_scheduler(optimizer, opt_cfg.scheduler_on_epoch))
            if opt_cfg.scheduler_on_step is not None:
                schedulers["on_step"].append(self.get_scheduler(optimizer, opt_cfg.scheduler_on_step))
            if opt_cfg.scheduler_on_val is not None:
                schedulers["on_val"].append(
                    (self.get_scheduler(optimizer, opt_cfg.scheduler_on_val), opt_cfg.scheduler_on_val.key)
                )

        return optimizers, schedulers

    def get_scheduler(self, opt, config):
        # --- Cas Spécifique : Reproduction Boudiaf et al. (Warm Cosine) ---
        if config.name == 'warmcos':
            # On récupère le nombre total d'itérations (T_max) et le warmup
            total_steps = config.kwargs.get('total_steps')
            warmup_steps = config.kwargs.get('warmup_steps', 100) # Défaut à 100 comme dans leur repo
            
            if total_steps is None:
                raise ValueError("Pour 'warmcos', vous devez spécifier 'total_steps' dans kwargs")

            # La formule exacte de leur repo :
            # min((i + 1) / 100, (1 + math.cos(math.pi * i / total_steps)) / 2)
            lr_lambda = lambda step: min(
                (step + 1) / warmup_steps, 
                (1 + math.cos(math.pi * step / total_steps)) / 2
            )
            
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        
        # --- Cas SequentialLR (Ton code existant) ---
        elif config.name == "SequentialLR":
            schedulers_cfg = config.kwargs.schedulers 
            schedulers = [getattr(optim.lr_scheduler, s.name)(opt, **s.kwargs) for s in schedulers_cfg]
            # print("remaining kwargs", config.kwargs) # Debug optionnel
            sch = optim.lr_scheduler.SequentialLR(opt, schedulers=schedulers, milestones=config.kwargs.milestones)
        
        # --- Cas Standard (PyTorch natif) ---
        else:
            sch = getattr(optim.lr_scheduler, config.name)(opt, **config.kwargs)
            
        lib.LOGGER.info(f"Scheduler created: {config.name}")
        return sch

    def get_loss(self, config):
        criterion = []
        for crit in config:
            loss = getattr(losses, crit.name)(**crit.kwargs)
            weight = crit.weight
            lib.LOGGER.info(f"{loss} with weight {weight}")
            criterion.append((loss, weight))
        return criterion

    def get_sampler(self, dataset, config):
        sampler = getattr(samplers, config.name)(dataset, **config.kwargs)
        lib.LOGGER.info(sampler)
        return sampler

    def get_dataset(self, transform, mode, config):
        if (config.name == "InShopDataset") and (mode == "test"):
            dataset = {
                "test": getattr(datasets, config.name)(transform=transform, mode="query", **config.kwargs),
                "gallery": getattr(datasets, config.name)(transform=transform, mode="gallery", **config.kwargs),
            }
            lib.LOGGER.info(dataset)
            return dataset
        elif (config.name == "DyMLDataset") and mode.startswith("test"):
            dataset = {
                "test": getattr(datasets, config.name)(transform=transform, mode="test_query_fine", **config.kwargs),
                "distractor": getattr(datasets, config.name)(transform=transform, mode="test_gallery_fine", **config.kwargs),
            }
            lib.LOGGER.info(dataset)
            return dataset
        elif (config.name == "SfM120kDataset") and (mode == "test"):
            dataset = []
            for dts in config.evaluation:
                test_dts = getattr(datasets, dts.name)(transform=transform, mode="query", **dts.kwargs)
                dataset.append({
                    f"query_{test_dts.city}": test_dts,
                    f"gallery_{test_dts.city}": getattr(datasets, dts.name)(transform=transform, mode="gallery", **dts.kwargs),
                })
            lib.LOGGER.info(dataset)
            return dataset
        else:
            dataset = getattr(datasets, config.name)(
                transform=transform,
                mode=mode,
                **config.kwargs,
            )
            lib.LOGGER.info(dataset)
            return dataset

    def get_model(self, config):
        net = getattr(models, config.name)(**config.kwargs)
    
        if config.freeze_batch_norm:
            lib.LOGGER.info("Freezing batch norm")
            net = lib.freeze_batch_norm(net)
        if config.freeze_pos_embedding:
            lib.LOGGER.info("Freezing pos embeddings")
            net.backbone = lib.freeze_pos_embedding(net.backbone)
        return net

    def get_memory(self, config):
        memory = getattr(engine, config.name)(**config.kwargs)
        lib.LOGGER.info(memory)
        return memory
