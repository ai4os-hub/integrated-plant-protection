# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing
the interfacing tasks. In this way you don't mix your true code with
DEEPaaS code and everything is more modular. That is, if you need to write
the predict() function in api.py, you would import your true predict function
and call it from here (with some processing / postprocessing in between
if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at an exemplar
module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

import ast
import base64
import json
import logging
import math
import mimetypes
from pathlib import Path
from random import random
import os
import shutil
import tempfile
import time

from deepaas.model.v2.wrapper import UploadedFile
import mlflow
from tensorboardX import SummaryWriter
from webargs import fields, validate

from . import config, misc


# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parents[1]


def get_metadata():
    """Returns a dictionary containing metadata information about the module.
       DO NOT REMOVE - All modules should have a get_metadata() function

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = config.PROJECT_METADATA
        # TODO: Add dynamic metadata collection here
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


def get_train_args():
    arg_dict = {
        "epoch_num": fields.Int(
            required=False,
            missing=10,
            description="Total number of training epochs",
        ),
    }
    return arg_dict


def train(**kwargs):
    """
    Dummy training. We just sleep for some number of epochs
    (1 epoch = 1 second)
    mimicking some computation taking place.
    We log some random metrics in Tensorboard to mimic monitoring.
    Also log to MLflow, if configuration is enabled.
    """
    # Setup Tensorboard
    logdir = BASE_DIR / "models" / time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(logdir=logdir, flush_secs=1)
    misc.launch_tensorboard(logdir=logdir)

    # Setup Mlflow
    mlflow_vars = [v in os.environ for v in ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_URI"]]
    use_mlflow = all(mlflow_vars)
    if use_mlflow:
        mlflow.set_experiment(experiment_name="ai4os-demo-app")
        _ = mlflow.start_run(run_name="test run")
        mlflow.log_params({"epochs": kwargs["epoch_num"]})

    # Start training loop
    for epoch in range(kwargs["epoch_num"]):
        time.sleep(1.0)
        fake_loss = -math.log(epoch + 1) * (1 + random() * 0.2)  # nosec
        fake_acc = min((1 - 1 / (epoch + 1)) * (1 + random() * 0.1), 1)  # nosec

        # Add metrics to Tensorboard
        writer.add_scalar("scalars/loss", fake_loss, epoch)
        writer.add_scalar("scalars/accuracy", fake_acc, epoch)

        # Add metrics to MLflow
        if use_mlflow:
            mlflow.log_metric("train_loss", fake_loss, step=epoch)
            mlflow.log_metric("train_accuracy", fake_acc, step=epoch)

    writer.close()
    if use_mlflow:
        mlflow.end_run()

    # Save locally a fake model file
    (logdir / "final_model.hdf5").touch()

    return {"status": "done", "final accuracy": 0.9}


def get_predict_args():
    """
    TODO: add more dtypes
    * int with choices
    * composed: list of strs, list of int
    """
    # WARNING: missing!=None has to go with required=False
    # fmt: off
    arg_dict = {
        "demo_str": fields.Str(
            required=False,
            missing="some-string",
            description="test string",
        ),
        "demo_str_choice": fields.Str(
            required=False,
            missing="choice2",
            enum=["choice1", "choice2"],
            description="test multi-choice with strings",
        ),
        "demo_password": fields.String(
            required=False,
            missing="some-string",
            validate=validate.Length(min=1),
            load_only=True,
            metadata={"format": "password"},
        ),
        "demo_int": fields.Int(
            required=False,
            missing=1,
            description="test integer",
        ),
        "demo_int_range": fields.Int(
            required=False,
            missing=50,
            validate=[validate.Range(min=1, max=100)],
            description="test integer is inside a min-max range",
        ),
        "demo_float": fields.Float(
            required=False,
            missing=0.1,
            description="test float",
        ),
        "demo_bool": fields.Bool(
            required=False,
            missing=True,
            description="test boolean",
        ),
        "demo_dict": fields.Str(
            # dicts have to be processed as strings otherwise DEEPaaS Swagger UI
            # throws an error
            required=False,
            missing="{'a': 0, 'b': 1}",
            description="test dictionary",
        ),
        "demo_list_of_floats": fields.List(
            fields.Float(),
            required=False,
            missing=[0.1, 0.2, 0.3],
            description="test list of floats",
        ),
        "demo_image": fields.Field(
            required=True,
            type="file",
            location="form",
            description="test image upload",  # "image" word in description is needed to be parsed by Gradio UI
        ),
        "demo_audio": fields.Field(
            required=True,
            type="file",
            location="form",
            description="test audio upload",  # "audio" word in description is needed to be parsed by Gradio UI
        ),
        "demo_video": fields.Field(
            required=True,
            type="file",
            location="form",
            description="test video upload",  # "video" word in description is needed to be parsed by Gradio UI
        ),
        # Add format type of the response of predict()
        # For demo purposes, we allow the user to receive back either JSON, image or zip.
        # More options for MIME types: https://mimeapplication.net/
        "accept": fields.Str(
            required=False,
            missing="application/json",
            description="Format of the response.",
            validate=validate.OneOf(["application/json", "application/zip", "image/*"]),
        ),
    }
    # fmt: on
    return arg_dict


# @_catch_error
def predict(**kwargs):
    """
    Return same inputs as provided. We also add additional fields
    to test the functionality of the Gradio-based UI [1].
    [1]: https://github.com/ai4os/deepaas_ui
    """
    # Dict are fed as str so have to be converted back
    kwargs["demo_dict"] = ast.literal_eval(kwargs["demo_dict"])

    # Check that the main input types are received in the correct Python type
    arg2type = {
        "demo_str": str,
        "demo_int": int,
        "demo_float": float,
        "demo_bool": bool,
        "demo_dict": dict,
        "demo_image": UploadedFile,
    }

    for k, v in arg2type.items():
        if not isinstance(kwargs[k], v):
            message = (
                f"Key {k} is type {type(kwargs[k])}, not type {v}. \n"
                f"Value: {kwargs[k]}"
            )
            raise Exception(message)

    # Add labels and random probabilities to output as mock
    prob = [random() for _ in range(5)]  # nosec
    kwargs["probabilities"] = [i / sum(prob) for i in prob]
    kwargs["labels"] = ["class2", "class3", "class0", "class1", "class4"]

    # Format the response differently depending on the MIME type selected by the user
    if kwargs["accept"] == "application/json":
        # Read media files and return them back in base64
        for k in ["demo_image", "demo_audio", "demo_video"]:
            with open(kwargs[k].filename, "rb") as f:
                media = f.read()
            media = base64.b64encode(media)  # bytes
            kwargs[k] = media.decode("utf-8")  # string (in utf-8)

        return kwargs

    elif kwargs["accept"] == "application/zip":
        zip_dir = tempfile.TemporaryDirectory()
        zip_dir = Path(zip_dir.name)
        zip_dir.mkdir()

        # Save parameters to JSON file
        with open(zip_dir / "args.json", "w") as f:
            json.dump(kwargs, f, sort_keys=True, indent=4)

        # Copy media files to ZIP folder
        for k in ["demo_image", "demo_audio", "demo_video"]:
            # Try to guess extension, otherwise take last part of content type
            ext = mimetypes.guess_extension(kwargs[k].content_type)
            extension = ext if ext else f".{kwargs[k].content_type.split('/')[-1]}"

            shutil.copyfile(src=kwargs[k].filename, dst=zip_dir / f"{k}{extension}")

        # Pack folder into ZIP file and return it
        shutil.make_archive(zip_dir, format="zip", root_dir=zip_dir)

        return open(f"{zip_dir}.zip", "rb")

    elif kwargs["accept"] == "image/*":
        filepath = kwargs["demo_image"].filename

        return open(filepath, "rb")


# Schema to validate the `predict()` output if accept field is "application/json"
schema = {
    "demo_str": fields.Str(),
    "demo_str_choice": fields.Str(),
    "demo_password": fields.Str(
        metadata={
            "format": "password",
        },
    ),
    "demo_int": fields.Int(),
    "demo_int_range": fields.Int(),
    "demo_float": fields.Float(),
    "demo_bool": fields.Bool(),
    "demo_dict": fields.Dict(),
    "demo_list_of_floats": fields.List(fields.Float()),
    "demo_image": fields.Str(
        description="image"  # description needed to be parsed by UI
    ),
    "demo_audio": fields.Str(
        description="audio"  # description needed to be parsed by UI
    ),
    "demo_video": fields.Str(
        description="video"  # description needed to be parsed by UI
    ),
    "labels": fields.List(fields.Str()),
    "probabilities": fields.List(fields.Float()),
    "accept": fields.Str(),
}
