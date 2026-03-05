import torch
def create_label_matrix(labels, other_labels=None):
    labels = labels.squeeze()

    if labels.ndim == 1:
        if other_labels is None:
            return (labels.view(-1, 1) == labels.t()).float()

        return (labels.view(-1, 1) == other_labels.t()).float()

    elif labels.ndim == 2:
        # --- MODIFICATION MULTI-LABEL ---
        if other_labels is None:
            return (torch.matmul(labels.float(), labels.t().float()) > 0).float()
        return (torch.matmul(labels.float(), other_labels.t().float()) > 0).float()
        # --------------------------------
    raise NotImplementedError(f"Function for tensor dimension {labels.ndim} not implemented")
