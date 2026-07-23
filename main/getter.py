from torch import optim
import torchvision.transforms as transforms
import torch

from main import transforms as custom_transforms
from main import losses
from main import samplers
from main import datasets
from main import models
from main import engine
from main import utils as lib
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
            if opt_cfg.params is None:
                target_module = net
                name_key = "net"
            else:
                target_module = getattr(net, opt_cfg.params)
                name_key = opt_cfg.params

            def split_weight_bias(params_iterator):
                p_weight = []
                p_bias = []
                for name, param in params_iterator:
                    if not param.requires_grad:
                        continue
                    if 'bias' in name or len(param.shape) == 1:
                        p_bias.append(param)
                    else:
                        p_weight.append(param)
                return p_weight, p_bias

            final_param_groups = []

            # Modules listed under 'modules' get their own param group (e.g. a distinct LR for conv1).
            specific_modules_cfg = getattr(opt_cfg, 'modules', [])

            processed_params = set()

            if specific_modules_cfg:
                for mod_cfg in specific_modules_cfg:
                    target_name = mod_cfg.get('name')
                    module_specific_kwargs = mod_cfg.get('kwargs', {})

                    specific_params_list = []
                    for name, param in target_module.named_parameters():
                        if target_name in name:
                            specific_params_list.append((name, param))
                            processed_params.add(param)

                    if specific_params_list:
                        sp_weight, sp_bias = split_weight_bias(specific_params_list)

                        if sp_weight:
                            final_param_groups.append({'params': sp_weight, **module_specific_kwargs})
                        if sp_bias:
                            final_param_groups.append({'params': sp_bias, **module_specific_kwargs})

                        lib.LOGGER.info(f"   -> Applied specific config to module '{target_name}' ({len(specific_params_list)} params)")

            rest_params_list = []
            for name, param in target_module.named_parameters():
                if param not in processed_params:
                    rest_params_list.append((name, param))

            params_weight, params_bias = split_weight_bias(rest_params_list)

            common_kwargs = opt_cfg.kwargs if opt_cfg.kwargs else {}
            bias_overrides = opt_cfg.bias_kwargs if hasattr(opt_cfg, 'bias_kwargs') and opt_cfg.bias_kwargs else {}

            bias_kwargs = common_kwargs.copy()
            bias_kwargs.update(bias_overrides)

            if params_weight:
                final_param_groups.append({'params': params_weight, **common_kwargs})
            if params_bias:
                final_param_groups.append({'params': params_bias, **bias_kwargs})

            optimizer_cls = getattr(optim, opt_cfg.name)
            optimizer = optimizer_cls(final_param_groups)

            optimizers[name_key] = optimizer
            lib.LOGGER.info(f"Optimizer created for {name_key}: {optimizer}")

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
        if config.name == 'warmcos':
            total_steps = config.kwargs.get('total_steps')
            warmup_steps = config.kwargs.get('warmup_steps', 100)

            if total_steps is None:
                raise ValueError("For 'warmcos', you must specify 'total_steps' in kwargs")

            lr_lambda = lambda step: min(
                (step + 1) / warmup_steps,
                (1 + math.cos(math.pi * step / total_steps)) / 2
            )

            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        elif config.name == "SequentialLR":
            schedulers_cfg = config.kwargs.schedulers
            schedulers = [getattr(optim.lr_scheduler, s.name)(opt, **s.kwargs) for s in schedulers_cfg]
            sch = optim.lr_scheduler.SequentialLR(opt, schedulers=schedulers, milestones=config.kwargs.milestones)

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
        elif (config.name in ["Cifar100RetrievalDataset", "ImageNet100Hashing", "Cub200Indomain"]) and (mode == "test"):
            dataset = {
                "test": getattr(datasets, config.name)(transform=transform, mode="query", **config.kwargs),
                "gallery": getattr(datasets, config.name)(transform=transform, mode="gallery", **config.kwargs),
            }
            lib.LOGGER.info(dataset)
            return dataset
        elif config.name ==  "Cifar10Retrieval":
            if mode == 'train':
                return datasets.Cifar10Retrieval(
                    data_dir=config.kwargs.data_dir,
                    mode='train',
                    transform=transform
                )

            elif mode == 'val':
                return {
                    "test": datasets.Cifar10Retrieval(data_dir=config.kwargs.data_dir, mode='val', transform=transform),
                    "gallery": datasets.Cifar10Retrieval(data_dir=config.kwargs.data_dir, mode='database', transform=transform)
                }

            elif mode == 'test':
                return {
                    "test": datasets.Cifar10Retrieval(data_dir=config.kwargs.data_dir, mode='query', transform=transform),
                    "gallery": datasets.Cifar10Retrieval(data_dir=config.kwargs.data_dir, mode='database', transform=transform)
                }
        elif (config.name in ["VOC2012Hashing", "MIRFlickrHashing", "COCOHashing"]) and (mode == "test"):
            dataset = {
                "test": getattr(datasets, config.name)(transform=transform, mode="query", **config.kwargs),
                "gallery": getattr(datasets, config.name)(transform=transform, mode="database", **config.kwargs),
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
