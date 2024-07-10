"""
Date: December 2023
Author: Jędrzej Smok
Email: jsmok@man.poznan.pl
Github: ai4eosc-psnc
"""

import json
import os
import random
import shutil
import subprocess
from contextlib import nullcontext
from datetime import datetime
from multiprocessing import Process
from pathlib import Path

import mlflow.pytorch
import numpy as np
import torch
import torchmetrics.functional as metrics_F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from integrated_plant_protection import config, paths, utils
from integrated_plant_protection.data_utils import (
    prepare_data,
)
from integrated_plant_protection.models import SmallCNNModel, Unet
from integrated_plant_protection.test_utils import save_intersection


def load_model(path: str, device, Model_Class=Unet, *args):
    model = Model_Class(*args)
    if issubclass(Model_Class, Unet):
        model.load_state_dict(torch.load(path, map_location=device))
    elif issubclass(Model_Class, SmallCNNModel):
        model.load_state_dict(
            torch.load(path, map_location=device)["model_state_dict"]
        )
    else:
        raise ValueError("Invalid model")
    model.eval()
    model.to(device)
    return model


def save_ckpt(
    main_metric_best_flag,
    last_epoch_flag,
    save_path,
    current_epoch,
    model,
    optimizer,
    loss_values,
):
    if main_metric_best_flag or last_epoch_flag:
        print("Save CKPT")
        Path(os.path.join(save_path, "ckpts")).mkdir(
            parents=True, exist_ok=True
        )
        file_name = os.path.join(
            save_path, f"ckpts/model_{str(current_epoch).zfill(4)}.pt"
        )
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_values["train"]["last"].item(),
            },
            file_name,
        )


def mlflow_tensorboard(
    subset: str, CONF, writer, loss_values, current_epoch, metrics_values
):

    if CONF["base"]["is_tensorboard"]:
        writer.add_scalar(
            f"loss/{subset}",
            np.mean(loss_values[subset]["epochs"][current_epoch]),
            current_epoch,
        )
    if CONF["base"]["is_mlflow"]:
        mlflow.log_metric(
            f"{subset}_loss",
            np.mean(loss_values[subset]["epochs"][current_epoch]),
            current_epoch,
        )

    for metric, value in metrics_values[subset].items():
        if CONF["base"]["is_tensorboard"]:
            writer.add_scalar(
                f"{metric}/{subset}",
                np.mean(value["epochs"][current_epoch]),
                current_epoch,
            )
        if CONF["base"]["is_mlflow"]:
            mlflow.log_metric(
                f"{subset}_{metric}",
                np.mean(value["epochs"][current_epoch]),
                current_epoch,
            )


def get_loss_metrics(subset: str, loss_values, current_epoch, metrics_values):
    avg_values = {
        "loss": float(np.mean(loss_values[subset]["epochs"][current_epoch]))
    }
    for k, v in metrics_values[subset].items():
        avg_values[k] = float(np.mean(v["epochs"][current_epoch]))
    return avg_values


def calculate_loss(
    subset: str,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    loss_values,
    current_epoch,
) -> None:
    value = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred.squeeze(), y_true.squeeze().float()
    )
    loss_values[subset]["last"] = value
    loss_values[subset]["epochs"][current_epoch].append(value.item())


def calculate_metrics(
    subset: str,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metrics,
    metrics_values,
    current_epoch,
) -> None:
    for metric in metrics:
        metric_name = (
            metric.__name__
            if hasattr(metric, "__name__")
            else metric.__class__.__name__
        )

        value = metric(torch.squeeze(y_pred), y_true.int(), "binary")

        metrics_values[subset][metric_name]["last"] = value
        metrics_values[subset][metric_name]["epochs"][current_epoch].append(
            value.item()
        )


def launch_tensorboard(port, logdir, host="127.0.0.1"):
    tensorboard_path = shutil.which("tensorboard")
    if tensorboard_path is None:
        raise FileNotFoundError("TensorBoard executable not found.")
    port = int(port) if len(str(port)) >= 4 else 6006
    subprocess.call(
        [
            tensorboard_path,
            "--logdir",
            "{}".format(logdir),
            "--port",
            "{}".format(port),
            "--host",
            "{}".format(host),
        ]
    )


def train_fn(TIMESTAMP, CONF):

    paths.timestamp = TIMESTAMP
    paths.CONF = CONF
    utils.create_dir_tree()

    print("Using CUDA version :", torch.version.cuda)
    print("GPU avail : ", torch.cuda.is_available())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_epoch = 0
    last_epoch_flag = False
    es_counter = 0
    CONF["base"]["is_mlflow"] = CONF["base"]["mlflow"]
    CONF["base"]["is_tensorboard"] = CONF["base"]["tensorboard"]

    model = SmallCNNModel(CONF["constants"]["channels"])
    random.seed(CONF["base"]["seed"])
    torch.manual_seed(CONF["base"]["seed"])
    np.random.seed(CONF["base"]["seed"])

    optimizer = torch.optim.Adam(
        model.parameters(), CONF["base"]["learning_rate"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=CONF["base"]["reduce_lr_factor"],
        patience=CONF["base"]["reduce_lr_patience"],
        verbose=True,
    )

    # LOGGING
    save_path = paths.get_timestamped_dir()
    print("Save path: ", save_path)

    utils.save_conf(CONF)
    if CONF["base"]["is_tensorboard"]:
        port = os.getenv("monitorPORT", 6006)
        port = int(port) if len(str(port)) >= 4 else 6006
        fuser_path = shutil.which("fuser")
        if fuser_path is None:
            subprocess.run(
                [fuser_path, "-k", "{}/tcp".format(port)]
            )  # kill any previous process in that port

        p = Process(
            target=launch_tensorboard,
            args=(port, paths.get_logs_dir()),
            daemon=True,
        )
        p.start()

    if CONF["base"]["is_mlflow"]:
        print(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(CONF["base"]["experiment"])
        mlflow.pytorch.autolog()
        cm = mlflow.start_run(run_name=CONF["base"]["experiment_name"])

        mlflow.log_param("framework", "PyTorch")

    else:
        cm = nullcontext()

    if CONF["base"]["is_tensorboard"]:
        writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

    # DATA
    if CONF["base"]["use_preprocess_model"] == "":
        healthy_paths = CONF["base"]["healthy_data_path"]
        sick_paths = CONF["base"]["sick_data_path"]
    else:
        model_path = os.path.join(
            paths.get_preprocess_models_dir(),
            CONF["base"]["use_preprocess_model"],
        )
        model_unet = load_model(model_path, device, Unet)

        h_output_path = os.path.join(
            os.path.join(
                "/", *CONF["base"]["healthy_data_path"].split("/")[:-1]
            ),
            "healthy_intersection",
        )
        if Path(h_output_path).exists():
            shutil.rmtree(h_output_path)
        print(f"{h_output_path} creating...")
        save_intersection(
            input_dir=CONF["base"]["healthy_data_path"],
            output_dir=h_output_path,
            device=device,
            model=model_unet,
            image_size=CONF["base"]["image_size"],
            batch_size=CONF["base"]["batch_size"],
            num_workers=CONF["constants"]["num_workers"],
        )

        s_output_path = os.path.join(
            os.path.join("/", *CONF["base"]["sick_data_path"].split("/")[:-1]),
            "sick_intersection",
        )
        if Path(s_output_path).exists():
            shutil.rmtree(s_output_path)
        print(f"{s_output_path} creating...")
        save_intersection(
            input_dir=CONF["base"]["sick_data_path"],
            output_dir=s_output_path,
            device=device,
            model=model_unet,
            image_size=CONF["base"]["image_size"],
            batch_size=CONF["base"]["batch_size"],
            num_workers=CONF["constants"]["num_workers"],
        )
        print("\n\nPreprocess finished")
        healthy_paths = h_output_path
        sick_paths = s_output_path

    dataloaders, data_len = prepare_data(
        healthy_paths,
        sick_paths,
        CONF["base"]["image_size"],
        CONF["base"]["batch_size"],
    )

    # LOSS
    loss_values = {}
    for subset in ["train", "val", "test"]:
        loss_values[subset] = {
            "last": torch.Tensor(),
            "epochs": {},
        }

    # METRICS
    metrics_values = {}
    metrics = [
        metrics_F.accuracy,
        metrics_F.recall,
        metrics_F.precision,
        metrics_F.f1_score,
    ]
    main_metric = metrics_F.accuracy.__name__
    main_metric_best = 0
    for subset in ["train", "val", "test"]:
        metrics_values[subset] = {
            (
                metric.__name__
                if hasattr(metric, "__name__")
                else metric.__class__.__name__
            ): {
                "last": torch.Tensor(),
                "epochs": {},
            }
            for metric in metrics
        }
    # Create the model and compile it
    with cm:
        model.to(device)

        summary(
            model,
            (3, CONF["base"]["image_size"], CONF["base"]["image_size"]),
        )

        for epoch in range(CONF["base"]["epochs"]):
            # START EPOCH
            current_epoch = epoch
            for subset in ["train", "val", "test"]:
                loss_values[subset]["epochs"][current_epoch] = []
                for metric in metrics:
                    metric_name = (
                        metric.__name__
                        if hasattr(metric, "__name__")
                        else metric.__class__.__name__
                    )
                    metrics_values[subset][metric_name]["epochs"][
                        current_epoch
                    ] = []

            # TRAIN
            cm_tmp = nullcontext()
            model.train(True)
            with torch.set_grad_enabled(True), tqdm(
                enumerate(dataloaders["train"]), unit="batch"
            ) as tqdm_data, cm_tmp:
                tqdm_data.set_description(f"train: epoch {current_epoch}")
                for batch_idx, (img, label) in tqdm_data:
                    img, label = img.to(device), label.to(device)

                    model.zero_grad()
                    output = model(img)

                    calculate_loss(
                        "train",
                        label,
                        output,
                        loss_values,
                        current_epoch,
                    )

                    loss_values["train"]["last"].backward()
                    optimizer.step()

                    calculate_metrics(
                        "train",
                        label,
                        output,
                        metrics,
                        metrics_values,
                        current_epoch,
                    )

                    tqdm_data.set_postfix(
                        get_loss_metrics(
                            "train",
                            loss_values,
                            current_epoch,
                            metrics_values,
                        )
                    )

            mlflow_tensorboard(
                "train",
                CONF,
                writer,
                loss_values,
                current_epoch,
                metrics_values,
            )

            # VALIDATE
            cm_tmp = nullcontext()
            model.train(False)
            with torch.set_grad_enabled(False), tqdm(
                enumerate(dataloaders["val"]), unit="batch"
            ) as tqdm_data, cm_tmp:
                tqdm_data.set_description(f"val: epoch {current_epoch}")
                for batch_idx, (img, label) in tqdm_data:
                    img, label = img.to(device), label.to(device)

                    model.zero_grad()
                    output = model(img)

                    calculate_loss(
                        "val", label, output, loss_values, current_epoch
                    )

                    calculate_metrics(
                        "val",
                        label,
                        output,
                        metrics,
                        metrics_values,
                        current_epoch,
                    )

                    tqdm_data.set_postfix(
                        get_loss_metrics(
                            "val",
                            loss_values,
                            current_epoch,
                            metrics_values,
                        )
                    )

                scheduler.step(
                    np.mean(loss_values["val"]["epochs"][current_epoch])
                )

            mlflow_tensorboard(
                "val",
                CONF,
                writer,
                loss_values,
                current_epoch,
                metrics_values,
            )

            # check_main_metric
            main_metric_value = np.mean(
                metrics_values["val"][main_metric]["epochs"][current_epoch]
            )
            if main_metric_value > main_metric_best:
                main_metric_best_flag = True
                main_metric_best = main_metric_value
            else:
                main_metric_best_flag = False
            # Save CKPT
            save_ckpt(
                main_metric_best_flag,
                last_epoch_flag,
                save_path,
                current_epoch,
                model,
                optimizer,
                loss_values,
            )

            early_stopping = False
            if CONF["base"]["early_stopping_patience"] > 0:

                if main_metric_best_flag:
                    es_counter = 0
                else:
                    es_counter += 1

                if es_counter > CONF["base"]["early_stopping_patience"]:
                    if CONF["base"]["is_mlflow"]:
                        mlflow.log_metric(
                            "restored_epoch",
                            current_epoch
                            - CONF["base"]["early_stopping_patience"]
                            - 1,
                        )
                        mlflow.log_metric("stopped_epoch", current_epoch)
                    early_stopping = True

            if early_stopping:
                last_epoch_flag = True
                save_ckpt(
                    main_metric_best_flag,
                    last_epoch_flag,
                    save_path,
                    current_epoch,
                    model,
                    optimizer,
                    loss_values,
                )
                break

        # TEST
        cm_tmp = cm
        model.train(False)
        with torch.set_grad_enabled(False), tqdm(
            enumerate(dataloaders["test"]), unit="batch"
        ) as tqdm_data, cm_tmp:
            tqdm_data.set_description(f"{'test'}: epoch {current_epoch}")
            for batch_idx, (img, label) in tqdm_data:
                img, label = img.to(device), label.to(device)

                model.zero_grad()
                output = model(img)

                calculate_loss(
                    "test", label, output, loss_values, current_epoch
                )
                calculate_metrics(
                    "test",
                    label,
                    output,
                    metrics,
                    metrics_values,
                    current_epoch,
                )
                tqdm_data.set_postfix(
                    get_loss_metrics(
                        "test",
                        loss_values,
                        current_epoch,
                        metrics_values,
                    )
                )

        mlflow_tensorboard(
            "test",
            CONF,
            writer,
            loss_values,
            current_epoch,
            metrics_values,
        )
    print(f"Best accuracy: {main_metric_best}")

    stats = {
        "train": get_loss_metrics(
            "train", loss_values, current_epoch, metrics_values
        ),
        "val": get_loss_metrics(
            "val", loss_values, current_epoch, metrics_values
        ),
        "test": get_loss_metrics(
            "test", loss_values, current_epoch, metrics_values
        ),
    }
    stats_dir = paths.get_stats_dir()

    with open(os.path.join(stats_dir, "stats.json"), "w") as outfile:
        json.dump(stats, outfile, sort_keys=True, indent=4)

    # # Saving everything
    # print('Saving data to {} folder.'.format(paths.get_timestamped_dir()))
    # print('Saving training stats ...')
    # stats = {'epoch': history.epoch,
    #          'training time (s)': round(time.time()-t0, 2),
    #          'timestamp': TIMESTAMP}
    # stats.update(history.history)
    # stats = json_friendly(stats)
    # stats_dir = paths.get_stats_dir()
    # with open(os.path.join(stats_dir, 'stats.json'), 'w') as outfile:
    #     json.dump(stats, outfile, sort_keys=True, indent=4)

    # print('Saving the configuration ...')
    # model_utils.save_conf(CONF)

    # print('Saving the model to h5...')
    # fpath = os.path.join(paths.get_checkpoints_dir(), 'final_model.h5')
    # model.save(fpath,
    #            include_optimizer=False)

    # # print('Saving the model to protobuf...')
    # # fpath = os.path.join(paths.get_checkpoints_dir(), 'final_model.proto')
    # # model_utils.save_to_pb(model, fpath)

    print("Finished")


if __name__ == "__main__":

    CONF = config.get_conf_dict()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    train_fn(TIMESTAMP=timestamp, CONF=CONF)
