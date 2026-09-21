from os.path import join

import torch

import main.utils as lib


def checkpoint(
    log_dir,
    save_checkpoint,
    net,
    optimizer,
    scheduler,
    scaler,
    epoch,
    seed,
    args,
    score,
    best_model,
    best_score,
    is_best=False,
):
    state_dict = {}
    if torch.cuda.device_count() > 1:
        state_dict["net_state"] = net.module.state_dict()
    else:
        state_dict["net_state"] = net.state_dict()

    state_dict["optimizer_state"] = {key: opt.state_dict() for key, opt in optimizer.items()}

    state_dict["scheduler_on_epoch_state"] = [sch.state_dict() for sch in scheduler["on_epoch"]]
    state_dict["scheduler_on_step_state"] = [sch.state_dict() for sch in scheduler["on_step"]]
    state_dict["scheduler_on_val_state"] = [sch.state_dict() for sch, _ in scheduler["on_val"]]

    if scaler is not None:
        state_dict["scaler_state"] = scaler.state_dict()

    state_dict["epoch"] = epoch
    state_dict["seed"] = seed
    state_dict["config"] = args
    state_dict["score"] = score
    state_dict["best_score"] = best_score
    state_dict["best_model"] = f"{best_model}.ckpt"

    RANDOM_STATE = lib.get_random_state()
    state_dict.update(RANDOM_STATE)

    # Written whenever this epoch is a new best, independent of save_checkpoint's
    # save_model periodicity: save_model only writes a numbered checkpoint every
    # Nth epoch, so without this, whichever epoch is actually best is silently
    # lost unless it happens to land on one of those multiples -- rolling.ckpt
    # doesn't help either, since it's overwritten every epoch and so only ever
    # holds the LAST epoch by the end of training, not the best one.
    if log_dir is None:
        from ray import tune
        torch.save(state_dict, join(tune.get_trial_dir(), "rolling.ckpt"))
        if save_checkpoint or is_best:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                if save_checkpoint:
                    lib.LOGGER.info(f"Checkpoint of epoch {epoch} created")
                    torch.save(state_dict, join(checkpoint_dir, f"epoch_{epoch}.ckpt"))
                if is_best:
                    lib.LOGGER.info(f"New best model (epoch {epoch}, score={best_score}) -- best.ckpt updated")
                    torch.save(state_dict, join(checkpoint_dir, "best.ckpt"))

    else:
        torch.save(state_dict, join(log_dir, 'weights', "rolling.ckpt"))
        if save_checkpoint:
            lib.LOGGER.info(f"Checkpoint of epoch {epoch} created")
            torch.save(state_dict, join(log_dir, 'weights', f"epoch_{epoch}.ckpt"))
        if is_best:
            # Same reasoning as above: guarantees the true best epoch's weights
            # are always on disk, regardless of save_model's periodicity.
            lib.LOGGER.info(f"New best model (epoch {epoch}, score={best_score}) -- best.ckpt updated")
            torch.save(state_dict, join(log_dir, 'weights', "best.ckpt"))
