import torch
import torch.nn as nn
import torch.optim as optim
import math
import yaml
import os
from roadmap.getter import Getter
from omegaconf import OmegaConf

getter = Getter()
# --- 1. MOCKS (Pour simuler l'environnement Hydra/Roadmap) ---

class MockLogger:
    def info(self, msg):
        print(f"📝 [LOG] {msg}")

class lib:
    LOGGER = MockLogger()

# Classe magique pour transformer les dicts YAML en objets (accès par point)
class HydraConfig(dict):
    def __init__(self, data=None, **kwargs):
        if data is None:
            data = kwargs
        else:
            data.update(kwargs)
        
        for k, v in data.items():
            # Conversion récursive
            if isinstance(v, dict):
                self[k] = HydraConfig(v)
            elif isinstance(v, list):
                self[k] = [HydraConfig(i) if isinstance(i, dict) else i for i in v]
            else:
                self[k] = v

    def __getattr__(self, name):
        # Permet l'accès config.attribut. Si absent, retourne None (comme Hydra parfois)
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value

# --- 2. MODÈLE SIMULÉ (Dummy WCNN) ---
class DummyWCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # On crée une structure qui ressemble à votre WCNN
        self.backbone = nn.Sequential()
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=True) # CIBLE
        self.backbone.layer1 = nn.Linear(64, 10) 

        self.lh_backbone = nn.Sequential()
        self.lh_backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=True) # CIBLE
        
        self.hl_backbone = nn.Sequential()
        self.hl_backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=True) # CIBLE
        
        self.hh_backbone = nn.Sequential()
        self.hh_backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=True) # CIBLE

# --- 3. VOTRE LOGIQUE OPTIMIZER (La version fusionnée) ---
class OptimizerFactory:
    def get_optimizer(self, net, config_list):
        optimizers = {}
        schedulers = {"on_epoch": [], "on_step": [], "on_val": []}

        for opt_cfg in config_list:
            # 1. Module cible
            if opt_cfg.params is None:
                target_module = net
                name_key = "net"
            else:
                target_module = getattr(net, opt_cfg.params)
                name_key = opt_cfg.params

            # Helper séparation Poids/Biais
            def split_weight_bias(params_iterator):
                p_weight, p_bias = [], []
                for name, param in params_iterator:
                    if not param.requires_grad: continue
                    if 'bias' in name or len(param.shape) == 1: p_bias.append(param)
                    else: p_weight.append(param)
                return p_weight, p_bias

            final_param_groups = []
            
            # A. Gestion des Modules Spécifiques (Via YAML 'modules')
            specific_modules_cfg = getattr(opt_cfg, 'modules', [])
            processed_params = set()

            if specific_modules_cfg:
                for mod_cfg in specific_modules_cfg:
                    target_name = mod_cfg.name
                    module_kwargs = mod_cfg.kwargs
                    
                    specific_params_list = []
                    for name, param in target_module.named_parameters():
                        if target_name in name: 
                            specific_params_list.append((name, param))
                            processed_params.add(param)
                    
                    if specific_params_list:
                        sp_weight, sp_bias = split_weight_bias(specific_params_list)
                        if sp_weight:
                            final_param_groups.append({'params': sp_weight, **module_kwargs, 'name': f"Specific_{target_name}_W"})
                        if sp_bias:
                            final_param_groups.append({'params': sp_bias, **module_kwargs, 'name': f"Specific_{target_name}_Bias"})
                        
                        lib.LOGGER.info(f"   -> Config spécifique appliquée à '{target_name}' ({len(specific_params_list)} params)")

            # B. Le Reste (Config Globale)
            rest_params_list = []
            for name, param in target_module.named_parameters():
                if param not in processed_params:
                    rest_params_list.append((name, param))

            params_weight, params_bias = split_weight_bias(rest_params_list)
            
            # Gestion kwargs globaux
            common_kwargs = opt_cfg.kwargs if opt_cfg.kwargs else {}
            bias_overrides = opt_cfg.bias_kwargs if opt_cfg.bias_kwargs else {}
            bias_kwargs = common_kwargs.copy()
            bias_kwargs.update(bias_overrides)

            if params_weight:
                final_param_groups.append({'params': params_weight, **common_kwargs, 'name': "Global_Weight"})
            if params_bias:
                final_param_groups.append({'params': params_bias, **bias_kwargs, 'name': "Global_Bias"})

            # Instanciation
            optimizer_cls = getattr(optim, opt_cfg.name)
            optimizer = optimizer_cls(final_param_groups)
            optimizers[name_key] = optimizer
            
            # Scheduler (Simplifié pour le test)
            if opt_cfg.scheduler_on_step:
                schedulers["on_step"].append(f"Scheduler {opt_cfg.scheduler_on_step.name}")

        return optimizers, schedulers

# --- 4. EXÉCUTION DU TEST ---

# 1. Création d'un faux fichier YAML sur le disque (comme le vôtre)
yaml_content = """
- name: SGD
  params: null
  
  # Config Globale (ImageNet Backbone)
  kwargs:
    lr: 0.02
    momentum: 0.9
    weight_decay: 0.0005
  bias_kwargs:
    weight_decay: 0.0

  # Config Spécifique (Vos conv1)
  modules:
    - name: "conv1."
      kwargs:
        lr: 0.2
        momentum: 0.9
        weight_decay: 0.0005

  scheduler_on_step:
      name: warmcos
      kwargs:
          total_steps: 2820
          warmup_steps: 100
"""

filename = "test_config_generated.yaml"
with open(filename, "w") as f:
    f.write(yaml_content)

print(f"📂 Fichier YAML créé : {filename}")

# 2. Chargement "Style Hydra"
with open(filename, "r") as f:
    raw_data = yaml.safe_load(f)

# Conversion en objets HydraConfig
config_objects = OmegaConf.load("config/optimizer/cub.yaml")  # Dummy load to get structure

# 3. Lancement de la Factory
print("\n🚀 Lancement de la Factory...")
model_configs = OmegaConf.load('config/model/wcnn.yaml')
model = getter.get_model(model_configs)
factory = OptimizerFactory()
optimizers, _ = getter.get_optimizer(model, config_objects)
opt = optimizers['net']
print(opt)

# 4. Inspection des résultats
print("\n🔍 RÉSULTATS D'ANALYSE :")
print("-" * 60)
for i, group in enumerate(opt.param_groups):
    name = group.get('name', 'Unknown')
    lr = group['lr']
    wd = group.get('weight_decay', 'N/A')
    count = len(group['params'])
    
    print(f"Groupe {i} : {name}")
    print(f"   👉 LR: {lr} | WD: {wd} | Params: {count} tenseurs")
    
    # Vérification automatique
    if "Specific" in name and lr == 0.2:
        print("      ✅ SUCCÈS : Boost LR détecté !")
    elif "Global" in name and lr == 0.02:
        print("      ✅ SUCCÈS : LR standard respecté.")
print("-" * 60)

# Nettoyage
os.remove(filename)