import torch
import torch.nn as nn
from collections import defaultdict
import os

class MBWDinoInstrumentor:
    """Attaches forward/backward hooks to MultiDinoHashingTF to record intermediate features and gradients."""
    def __init__(self, model, save_dir='./analysis_logs'):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.features = defaultdict(dict)
        self.gradients = defaultdict(dict)
        self.hooks = []
        self.batch_counter = 0

        self.target_vit_layers = {
            'low_level': 2,
            'mid_level': 5,
            'high_level': 10
        }

        self.band_names = ['LL', 'LH', 'HL', 'HH']

    def register_hooks(self):
        self.remove_hooks()

        for i, backbone in enumerate(self.model.backbones):
            band_name = self.band_names[i]

            if hasattr(backbone, 'blocks'):
                blocks = backbone.blocks
            elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
                blocks = backbone.encoder.layers
            else:
                 print(f"Warning: Could not locate transformer blocks in backbone {band_name}")
                 continue

            for level_name, block_idx in self.target_vit_layers.items():
                if block_idx < len(blocks):
                    layer = blocks[block_idx]
                    hook_name = f"vit_{band_name}_{level_name}_block{block_idx}"

                    fh = layer.register_forward_hook(self._get_forward_hook(hook_name))
                    self.hooks.append(fh)

                    bh = layer.register_full_backward_hook(self._get_backward_hook(hook_name))
                    self.hooks.append(bh)

        fusion_head = self.model.fusion_head

        for i, proj in enumerate(fusion_head.projections):
             band_name = self.band_names[i]
             hook_name = f"fusion_proj_{band_name}"
             self.hooks.append(proj.register_forward_hook(self._get_forward_hook(hook_name)))
             self.hooks.append(proj.register_full_backward_hook(self._get_backward_hook(hook_name)))

        # nn.MultiheadAttention doesn't expose attn_weights via hooks, so the whole fusion
        # head is hooked instead to capture its Q, K, V and outputs.
        self.hooks.append(fusion_head.register_forward_hook(self._get_fusion_head_forward_hook('fusion_head')))

        print(f"Successfully registered {len(self.hooks)} hooks.")

    def _get_forward_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                 self.features[name] = output[0].detach().cpu()
            else:
                 self.features[name] = output.detach().cpu()
        return hook

    def _get_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients[name] = grad_output[0].detach().cpu()
        return hook

    def _get_fusion_head_forward_hook(self, name):
        def hook(module, input, output):
             if hasattr(module, 'query_tokens'):
                 self.features[f"{name}_query_tokens"] = module.query_tokens.detach().cpu()
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features.clear()
        self.gradients.clear()

    def save_current_state(self, epoch, batch_idx, is_target_batch=True):
        if not is_target_batch:
             self.features.clear()
             self.gradients.clear()
             return

        save_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'features': dict(self.features),
            'gradients': dict(self.gradients)
        }

        filename = os.path.join(self.save_dir, f"analysis_epoch_{epoch}_batch_{batch_idx}.pt")
        torch.save(save_data, filename)
        print(f"Saved instrumentation data to {filename}")

        self.features.clear()
        self.gradients.clear()

class SharedMBWDinoInstrumentor:
    """Instrumentor for SharedDinoHashing: splits the shared batch tensors into sub-bands (LL, LH, HL, HH)."""
    def __init__(self, model, save_dir='./analysis_logs'):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.features = defaultdict(dict)
        self.gradients = defaultdict(dict)
        self.hooks = []

        self.target_vit_layers = {
            'low_level': 2,
            'mid_level': 5,
            'high_level': 10
        }

        self.band_names = ['LL', 'LH', 'HL', 'HH']

    def register_hooks(self):
        self.remove_hooks()

        backbone = self.model.shared_backbone

        if hasattr(backbone, 'blocks'):
            blocks = backbone.blocks
        elif hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
            blocks = backbone.encoder.layers
        else:
             print("Warning: Could not locate transformer blocks in shared backbone")
             blocks = []

        for level_name, block_idx in self.target_vit_layers.items():
            if block_idx < len(blocks):
                layer = blocks[block_idx]

                fh = layer.register_forward_hook(self._get_shared_forward_hook(level_name, block_idx))
                self.hooks.append(fh)

                bh = layer.register_full_backward_hook(self._get_shared_backward_hook(level_name, block_idx))
                self.hooks.append(bh)

        fusion_head = self.model.fusion_head

        for i, proj in enumerate(fusion_head.projections):
             band_name = self.band_names[i]
             hook_name = f"fusion_proj_{band_name}"
             self.hooks.append(proj.register_forward_hook(self._get_standard_forward_hook(hook_name)))
             self.hooks.append(proj.register_full_backward_hook(self._get_standard_backward_hook(hook_name)))

        self.hooks.append(fusion_head.register_forward_hook(self._get_fusion_head_forward_hook('fusion_head')))

        print(f"Successfully registered {len(self.hooks)} hooks on Shared Architecture.")

    def _get_shared_forward_hook(self, level_name, block_idx):
        def hook(module, input, output):
            tensor = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
            chunks = tensor.chunk(4, dim=0)

            for i, band in enumerate(self.band_names):
                name = f"vit_{band}_{level_name}_block{block_idx}"
                self.features[name] = chunks[i]
        return hook

    def _get_shared_backward_hook(self, level_name, block_idx):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                tensor = grad_output[0].detach().cpu()
                chunks = tensor.chunk(4, dim=0)

                for i, band in enumerate(self.band_names):
                    name = f"vit_{band}_{level_name}_block{block_idx}"
                    self.gradients[name] = chunks[i]
        return hook

    def _get_standard_forward_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                 self.features[name] = output[0].detach().cpu()
            else:
                 self.features[name] = output.detach().cpu()
        return hook

    def _get_standard_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients[name] = grad_output[0].detach().cpu()
        return hook

    def _get_fusion_head_forward_hook(self, name):
        def hook(module, input, output):
             if hasattr(module, 'query_tokens'):
                 self.features[f"{name}_query_tokens"] = module.query_tokens.detach().cpu()
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features.clear()
        self.gradients.clear()

    def save_current_state(self, epoch, batch_idx, is_target_batch=True):
        if not is_target_batch:
             self.features.clear()
             self.gradients.clear()
             return

        save_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'features': dict(self.features),
            'gradients': dict(self.gradients)
        }

        filename = os.path.join(self.save_dir, f"analysis_epoch_{epoch}_batch_{batch_idx}.pt")
        torch.save(save_data, filename)
        print(f"Saved shared instrumentation data to {filename}")

        self.features.clear()
        self.gradients.clear()
