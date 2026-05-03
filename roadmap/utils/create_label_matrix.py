import torch
def create_label_matrix(labels, other_labels=None):
    # Note/Safeguard:
    # If your background class is stored as class 0 and set to 1 for all images,
    # or if the data loader pads unused slots with 0 and 0 maps to a valid class,
    # torch.matmul will count those background/padding 1s as valid intersections!
    # This causes artificial intersections between ALL images and artificially boosts mAP.
    # Make sure labels corresponding to padding/background are removed or masked!
    
    labels = labels.squeeze()

    if labels.ndim == 1:
        if other_labels is None:
            return (labels.view(-1, 1) == labels.t()).float()

        return (labels.view(-1, 1) == other_labels.t()).float()

    elif labels.ndim == 2:
        # --- MODIFICATION MULTI-LABEL ---
        # Note/Safeguard:
        # If your background class is stored as class 0 and set to 1 for all images,
        # or if the data loader pads unused slots with 0 and 0 maps to a valid class,
        # torch.matmul will count those background/padding 1s as valid intersections!
        # This causes artificial intersections between ALL images and artificially boosts mAP.
        # Make sure labels corresponding to padding/background are removed or masked!
        if other_labels is None:
            return (torch.matmul(labels.float(), labels.t().float()) > 0).float()
        return (torch.matmul(labels.float(), other_labels.t().float()) > 0).float()
        # --------------------------------
    raise NotImplementedError(f"Function for tensor dimension {labels.ndim} not implemented")
