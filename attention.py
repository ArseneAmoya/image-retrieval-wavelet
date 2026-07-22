import torch
import matplotlib.pyplot as plt
import argparse

from roadmap.getter import get_model, get_dataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le .ckpt")
    parser.add_argument("--bs", type=int, default=16, help="Taille du batch")
    parser.add_argument("--num_batches", type=int, default=10, help="Nombre de batchs à évaluer")
    args = parser.parse_args()

    print(f"Chargement du checkpoint : {args.config}")
    checkpoint = torch.load(args.config, map_location='cpu', weights_only=False)

    config = checkpoint['cfg']
    net = get_model(config)
    net.load_state_dict(checkpoint['net_state'], strict=True)
    net.cuda()
    net.eval()

    print("Chargement du dataset...")
    dataset_name = config.get("dataset", {}).get("name", "inconnu").lower()
    split = 'query' if 'voc' not in dataset_name else 'val'

    dataset = get_dataset(config, split=split)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    # nn.MultiheadAttention forward returns (output, attention_weights); the hook captures the latter.
    attention_weights_list = []

    def hook_fn(module, input, output):
        weights = output[1].detach().cpu()
        attention_weights_list.append(weights)

    attention_module = net.module.fusion_head.attn if hasattr(net, 'module') else net.fusion_head.attn
    handle = attention_module.register_forward_hook(hook_fn)

    print(f"Extraction des poids d'attention sur {args.num_batches} batchs...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_batches:
                break

            if isinstance(batch, dict):
                img = batch["image"].cuda()
            else:
                img = batch[0].cuda()

            _ = net(img)

    handle.remove()

    all_weights = torch.cat(attention_weights_list, dim=0)
    mean_weights = all_weights.mean(dim=0).squeeze().numpy()

    branches = ['Bande LL', 'Bande LH', 'Bande HL', 'Bande HH']
    branches = branches[:len(mean_weights)]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(branches, mean_weights * 100, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    plt.title("Répartition des Poids d'Attention par Branche Ondelette", fontsize=14, pad=15)
    plt.ylabel("Importance Sémantique (%)", fontsize=12)
    plt.ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2,
                 f'{yval:.1f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    output_filename = "attention_weights_distribution.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nTerminé ! Le graphique a été sauvegardé sous '{output_filename}'.")
    print(f"Valeurs brutes : {mean_weights}")

if __name__ == '__main__':
    main()
