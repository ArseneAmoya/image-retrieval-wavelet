def freeze_pos_embedding(net):
    try:
        net.pos_embed.requires_grad_(False)
    except AttributeError:
        net.base_model.pos_embed.requires_grad_(False)
    return net
