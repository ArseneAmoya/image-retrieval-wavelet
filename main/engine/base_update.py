import os
from contextlib import contextmanager

import torch
import numpy as np
from tqdm import tqdm

import main.utils as lib
from .batch_map import compute_batch_map


def _capture_rng_state():
    return torch.get_rng_state(), (torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


@contextmanager
def _replay_rng_state(state):
    """Temporarily installs a captured RNG state so a forward pass can be replayed
    byte-identically (dropout masks, stochastic sub-band masking, etc.) across two passes."""
    cpu_state, cuda_state = state
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before)


def _split_into_microbatches(batch, sub_batch):
    total = batch["image"].size(0)
    starts = list(range(0, total, sub_batch))
    # Avoid a trailing microbatch of size 1: any BatchNorm layer in the model raises in
    # train mode on a batch of exactly 1 sample. Merge the leftover into the previous chunk.
    if len(starts) > 1 and total - starts[-1] == 1:
        starts.pop()

    micro_batches = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else total
        micro_batch = {"image": batch["image"][start:end], "label": batch["label"][start:end]}
        if "path" in batch:
            micro_batch["path"] = batch["path"][start:end]
        micro_batches.append(micro_batch)
    return micro_batches


def _single_pass_optimization(
    config,
    net,
    batch,
    criterion,
    scaler,
    epoch,
    memory,
    batch_map_calculator=None,
    batch_map_metric=None,
):
    """No sub-batching: identical to a single forward/backward over the whole batch."""
    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
        di = net(batch["image"].cuda())
        labels = batch["label"].cuda()
        label_matrix = lib.create_label_matrix(labels)

        logs = {}
        if batch_map_calculator is not None:
            logs[f"proxy_{batch_map_metric}"] = compute_batch_map(
                batch_map_calculator, batch_map_metric, di, labels,
            )

        if memory:
            memory_embeddings, memory_labels = memory(di.detach(), labels, batch["path"])
            if epoch >= config.memory.activate_after:
                memory_scores = torch.mm(di, memory_embeddings.t())
                memory_label_matrix = lib.create_label_matrix(labels, memory_labels)

        losses = []
        for crit, weight in criterion:
            if hasattr(crit, 'takes_embeddings'):
                if labels.ndim == 1 or (labels.ndim == 2 and labels.size(1) == 1):
                    loss = crit(di, labels.view(-1))
                else:
                    loss = crit(di, labels)
                if memory:
                    if epoch >= config.memory.activate_after:
                        mem_loss = crit(di, labels.view(-1), memory_embeddings, memory_labels.view(-1))

            else:
                scores = torch.mm(di, di.t())
                loss = crit(scores, label_matrix)
                if memory:
                    if epoch >= config.memory.activate_after:
                        mem_loss = crit(memory_scores, memory_label_matrix)

            loss = loss.mean()
            if weight == 'adaptative':
                losses.append(loss)
            else:
                losses.append(weight * loss)

            logs[crit.__class__.__name__] = loss.item()
            if memory:
                if epoch >= config.memory.activate_after:
                    mem_loss = mem_loss.mean()
                    if weight == 'adaptative':
                        losses.append(config.memory.weight * mem_loss)
                    else:
                        losses.append(weight * config.memory.weight * mem_loss)
                    logs[f"memory_{crit.__class__.__name__}"] = mem_loss.item()

    if weight == 'adaptative':
        grads = []
        for i, lss in enumerate(losses):
            g = torch.autograd.grad(lss, net.fc.parameters(), retain_graph=True)
            grads.append(torch.norm(g[0]).item())
        mean_grad = np.mean(grads)
        weights = [mean_grad / g for g in grads]
        losses = [w * lss for w, lss in zip(weights, losses)]
        logs.update({
            f"weight_{crit.__class__.__name__}": w for (crit, _), w in zip(criterion, weights)
        })
        logs.update({
            f"grad_{crit.__class__.__name__}": w for (crit, _), w in zip(criterion, grads)
        })

    actual_net = net.module if hasattr(net, 'module') else net

    if hasattr(actual_net, 'fusion_head') and hasattr(actual_net.fusion_head, 'last_ortho_loss'):
        ortho_loss = actual_net.fusion_head.last_ortho_loss

        if isinstance(ortho_loss, torch.Tensor) and ortho_loss.requires_grad:
            losses.append(ortho_loss)
            logs["Ortho_Loss"] = ortho_loss.item()

    total_loss = sum(losses)
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def _gradient_cached_optimization(
    config,
    net,
    batch,
    criterion,
    scaler,
    epoch,
    memory,
    sub_batch,
    batch_map_calculator=None,
    batch_map_metric=None,
):
    """
    Splits the batch into microbatches of at most `sub_batch` elements to bound peak memory,
    while keeping the loss's view of the batch (pairwise scores, label_matrix, memory-bank
    comparisons) exactly as if it were computed on the whole batch at once. Uses the
    "gradient caching" technique (Gao et al., "Scaling Deep Contrastive Learning Batch Size
    under Memory Limited Setup"):
      1. Forward every microbatch with no_grad to build the full-batch embeddings.
      2. Compute the loss normally on that full-batch embedding tensor.
      3. Backward only down to the embeddings, caching one gradient vector per sample.
      4. Re-forward each microbatch with its graph, and backward it using its slice of the
         cached gradient -- this reproduces the exact full-batch parameter gradient.

    Caveats:
      - `weight == 'adaptative'` loss reweighting needs gradients w.r.t. net.fc through the
        live forward graph, which no longer exists once embeddings are cached; unsupported
        here (raises below). It isn't used by any current config.
      - Any batch statistic computed *inside* the network itself (e.g. MultiDinoHashing's own
        BatchNorm1d before the final sign()) still only sees microbatch-sized statistics per
        forward call -- this fixes the loss, not what happens inside the model's forward.
      - BatchNorm running-stat updates happen on both the no_grad and the with-grad forward of
        each microbatch, so their momentum is effectively applied twice per step. Harmless
        (still legitimate data), just a minor speed-up of running-stat convergence.
    """
    for _, weight in criterion:
        if weight == 'adaptative':
            raise NotImplementedError(
                "weight='adaptative' loss reweighting is incompatible with experience.sub_batch "
                "gradient caching (it needs gradients through the live forward graph, which is "
                "gone once embeddings are cached). Set experience.sub_batch to null (or >= the "
                "batch size) to use adaptive weighting."
            )

    total_batch_size = batch["image"].size(0)
    micro_batches = _split_into_microbatches(batch, sub_batch)
    actual_net = net.module if hasattr(net, 'module') else net

    # --- Pass 1: cheap forward per microbatch, no graph kept, to assemble full-batch embeddings ---
    cached_embeddings = []
    rng_states = []
    with torch.no_grad():
        for micro_batch in micro_batches:
            rng_states.append(_capture_rng_state())
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                cached_embeddings.append(net(micro_batch["image"].cuda()))

    di_full = torch.cat(cached_embeddings, dim=0).float().detach().requires_grad_()
    labels_full = torch.cat([mb["label"] for mb in micro_batches], dim=0).cuda()

    logs = {}

    memory_embeddings, memory_labels = None, None
    if memory:
        paths_full = sum((list(mb.get("path", [])) for mb in micro_batches), [])
        memory_embeddings, memory_labels = memory(di_full.detach(), labels_full, paths_full)

    # --- Loss on the full batch: exactly like the non-split path, pairwise terms see every pair ---
    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
        label_matrix = lib.create_label_matrix(labels_full)
        losses = []
        for crit, weight in criterion:
            if hasattr(crit, 'takes_embeddings'):
                if labels_full.ndim == 1 or (labels_full.ndim == 2 and labels_full.size(1) == 1):
                    loss = crit(di_full, labels_full.view(-1))
                else:
                    loss = crit(di_full, labels_full)
                if memory and epoch >= config.memory.activate_after:
                    mem_loss = crit(di_full, labels_full.view(-1), memory_embeddings, memory_labels.view(-1))
            else:
                scores = torch.mm(di_full, di_full.t())
                loss = crit(scores, label_matrix)
                if memory and epoch >= config.memory.activate_after:
                    memory_scores = torch.mm(di_full, memory_embeddings.t())
                    memory_label_matrix = lib.create_label_matrix(labels_full, memory_labels)
                    mem_loss = crit(memory_scores, memory_label_matrix)

            loss = loss.mean()
            losses.append(weight * loss)
            logs[crit.__class__.__name__] = loss.item()

            if memory and epoch >= config.memory.activate_after:
                mem_loss = mem_loss.mean()
                losses.append(weight * config.memory.weight * mem_loss)
                logs[f"memory_{crit.__class__.__name__}"] = mem_loss.item()

        total_loss = sum(losses)
        logs["total_loss"] = total_loss.item()

    # Backward only as far as di_full: di_full.grad now holds, per sample, exactly the
    # gradient a full-batch backward would have produced.
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    # --- Pass 2: replay each microbatch's forward with its graph, backprop it using its slice
    #     of the cached gradient. Peak memory stays bounded to one microbatch. ---
    start = 0
    for micro_batch, rng_state in zip(micro_batches, rng_states):
        chunk_size = micro_batch["image"].size(0)
        end = start + chunk_size
        grad_slice = di_full.grad[start:end]

        with _replay_rng_state(rng_state):
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                di_chunk = net(micro_batch["image"].cuda())

        backward_tensors = [di_chunk]
        backward_grads = [grad_slice.to(di_chunk.dtype)]

        if hasattr(actual_net, 'fusion_head') and hasattr(actual_net.fusion_head, 'last_ortho_loss'):
            ortho_loss = actual_net.fusion_head.last_ortho_loss
            if isinstance(ortho_loss, torch.Tensor) and ortho_loss.requires_grad:
                # Weighted by this microbatch's share of the batch: exact if ortho_loss is
                # parameter-only (independent of input, e.g. CrossAttentionBottleneckHeadAdvanced),
                # a size-weighted average across microbatches otherwise.
                chunk_weight = chunk_size / total_batch_size
                backward_tensors.append(ortho_loss * chunk_weight)
                backward_grads.append(torch.ones_like(ortho_loss))
                if start == 0:
                    logs["Ortho_Loss"] = ortho_loss.item()

        torch.autograd.backward(backward_tensors, grad_tensors=backward_grads)
        start = end

    if batch_map_calculator is not None:
        logs[f"proxy_{batch_map_metric}"] = compute_batch_map(
            batch_map_calculator, batch_map_metric, di_full.detach(), labels_full,
        )

    return logs


def _batch_optimization(
    config,
    net,
    batch,
    criterion,
    optimizer,
    scaler,
    epoch,
    memory,
    batch_map_calculator=None,
    batch_map_metric=None,
):
    total_batch_size = batch["image"].size(0)
    sub_batch = getattr(config.experience, 'sub_batch', None)

    if not sub_batch or sub_batch >= total_batch_size:
        return _single_pass_optimization(
            config, net, batch, criterion, scaler, epoch, memory,
            batch_map_calculator=batch_map_calculator, batch_map_metric=batch_map_metric,
        )
    if sub_batch < 2:
        raise ValueError(
            f"experience.sub_batch={sub_batch} is too small: any BatchNorm layer in the model "
            "raises in train mode on a batch of exactly 1 sample, so sub_batch must be >= 2."
        )
    return _gradient_cached_optimization(
        config, net, batch, criterion, scaler, epoch, memory, sub_batch,
        batch_map_calculator=batch_map_calculator, batch_map_metric=batch_map_metric,
    )


def base_update(
    config,
    net,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    memory=None,
    batch_map_calculator=None,
    batch_map_metric=None,
):
    meter = lib.DictAverage()
    net.train()
    net.zero_grad()

    iterator = tqdm(loader, disable=os.getenv('TQDM_DISABLE'))
    for i, batch in enumerate(iterator):
        if i > config.experience.step_per_epoch:
            break
        logs = _batch_optimization(
            config,
            net,
            batch,
            criterion,
            optimizer,
            scaler,
            epoch,
            memory,
            batch_map_calculator=batch_map_calculator,
            batch_map_metric=batch_map_metric,
        )

        if config.experience.log_grad:
            grad_norm = lib.get_gradient_norm(net)
            logs["grad_norm"] = grad_norm.item()
        clip_value = getattr(config.experience, 'clip_grad', None)

        if clip_value is not None and clip_value > 0.0:
            if scaler is not None:
                for key, opt in optimizer.items():
                    scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_value)


        for key, opt in optimizer.items():
            if epoch < config.experience.warm_up and key != config.experience.warm_up_key:
                lib.LOGGER.info(f"Warming up @epoch {epoch}")
                continue
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)
        for crit, _ in criterion:
            if hasattr(crit, 'step'):
                if getattr(crit, 'loss_optimizer', None) is not None:
                    if scaler is not None:
                        scaler.step(crit.loss_optimizer)
                        if hasattr(crit, 'custom_step_logic'):
                            crit.custom_step_logic()
                    else:
                        crit.step()
                else:
                    crit.step()

        net.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        for sch in scheduler["on_step"]:
            sch.step()

        if scaler is not None:
            scaler.update()

        meter.update(logs)
        if not os.getenv('TQDM_DISABLE'):
            iterator.set_postfix(meter.avg)
        else:
            if (i + 1) % 50 == 0:
                lib.LOGGER.info(f'Iteration : {i}/{len(loader)}')
                for k, v in logs.items():
                    lib.LOGGER.info(f'Loss: {k}: {v} ')

    for crit, _ in criterion:
        if hasattr(crit, 'epoch_step'):
            crit.epoch_step()
    if hasattr(net, 'epoch_step'):
        net.epoch_step(epoch)
    return meter.avg
