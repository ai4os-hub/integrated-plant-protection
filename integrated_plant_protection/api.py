# -*- coding: utf-8 -*-
"""
Date: December 2023
Author: Jędrzej Smok (based on code from Ignacio Heredia)
Email: jsmok@man.poznan.pl
Github: ai4eosc-psnc

Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal,
only performing the interfacing
tasks. In this way you don't mix your true code with DEEPaaS code
and everything is more modular.
That is, if you need to write the predict() function in api.py, you
would import your true predict function and call it from here
(with some processing /postprocessing in between if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1]
 and at a canonical exemplar module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
(previously https://github.com/deephdc/demo_app)
"""

import builtins
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from functools import wraps
from pathlib import Path

import confuse
import requests
import torch
from webargs import fields

from integrated_plant_protection import config, paths, test_utils
from integrated_plant_protection.data_utils import mount_nextcloud
from integrated_plant_protection.misc import _catch_error
from integrated_plant_protection.models import SmallCNNModel, Unet
from integrated_plant_protection.train_runfile import train_fn

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate config file using confuse library."""

    def __init__(
        self, config_path: str, template: dict, appname: str = "config"
    ):
        """Object creator.

        Parameters
        ----------
        config_path: str
            Path to config file.
        template: dict
            Validation template.
        appname: str, optional
            Application name.
        """
        self.configuration = confuse.Configuration(appname, read=False)
        self.configuration.set_file(config_path)
        self.config = self.configuration.get(template)

    def get_config(self) -> dict:
        """
        Returns
        -------
        dict
            Validated config file.
        """
        return self.config

    def dump(self) -> str:
        """Useful for saving pretty (interpreted) version of yaml config file.
        Returns
        -------
        str
            Dumped config.
        """
        return self.configuration.dump()


config_template = {
    "base": {
        "batch_size": confuse.Integer(),
        "channels": confuse.TypeTemplate(list),
        "crop_scale": confuse.Number(),
        "epochs": confuse.Integer(),
        "early_stopping_patience": confuse.Integer(),
        "experiment": confuse.String(),
        "experiment_name": confuse.String(),
        "image_size": confuse.Integer(),
        "learning_rate": confuse.Number(),
        "mlflow": confuse.TypeTemplate(bool, default=False),
        "mlflow_params": confuse.TypeTemplate(str),
        "reduce_lr_factor": confuse.Number(),
        "reduce_lr_patience": confuse.Integer(),
        "patch_size": confuse.Integer(),
        "seed": confuse.Integer(),
        "shuffle": confuse.TypeTemplate(bool, default=False),
        "tensorboard": confuse.TypeTemplate(bool, default=False),
        "healthy_data_path": confuse.String(),
        "sick_data_path": confuse.String(),
    }
}

BASE_DIR = Path(__file__).resolve().parents[1]


@_catch_error
def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = {
            "author": config.API_METADATA.get("authors"),
            "author-email": config.API_METADATA.get("author-emails"),
            "description": config.API_METADATA.get("summary"),
            "license": config.API_METADATA.get("license"),
            "version": config.API_METADATA.get("version"),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


# Mount NextCloud folders (if NextCloud is available)
try:
    # mount_nextcloud('rshare:/data/dataset_files', paths.get_splits_dir())
    mount_nextcloud("rshare:/rye", paths.get_images_dir())
    # mount_nextcloud('rshare:/models', paths.get_models_dir())
except Exception as e:
    print(f"Nextcloud: {e}")

# Empty model variables for inference (will be loaded the first time we
# perform inference)
loaded_ts, loaded_ckpt = None, None
(
    model,
    conf,
) = (
    None,
    None,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Additional parameters
allowed_extensions = set(
    ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
)  # allow only certain file extensions


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


def load_inference_model(timestamp=None, ckpt_name=None):
    """
    Load a model for prediction.

    Parameters
    ----------
    * timestamp: str
        Name of the timestamp to use.
        The default is the last timestamp in `./models`.
    * ckpt_name: str
        Name of the checkpoint to use.
        The default is the last checkpoint in `./models/[timestamp]/ckpts`.
    """
    global loaded_ts, loaded_ckpt
    global model, conf
    global device
    # Set the timestamp
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        raise Exception(
            "You have no models in your `./models` folder"
            + "Therefore the API can only be used for training."
        )
    elif timestamp is None:
        timestamp = timestamp_list[-1]
    elif timestamp not in timestamp_list:
        raise ValueError(
            f"Invalid timestamp name: {timestamp}."
            + f"Available timestamp names are: {timestamp_list}"
        )
    paths.timestamp = timestamp
    print("Using TIMESTAMP={}".format(timestamp))

    # Set the checkpoint model to use to make the prediction
    ckpt_list = os.listdir(paths.get_checkpoints_dir())
    ckpt_list = sorted([name for name in ckpt_list if name.endswith(".pt")])
    if not ckpt_list:
        raise Exception(
            f"You have no checkpoints in your `./models/{timestamp}/ckpts`"
            + " folder to be used for inference."
            + " Therefore the API can only be used for training."
        )
    elif ckpt_name is None:
        ckpt_name = ckpt_list[-1]
    elif ckpt_name not in ckpt_list:
        raise ValueError(
            f"Invalid checkpoint name: {ckpt_name}"
            + f" Available checkpoint names are: {ckpt_list}"
        )
    print("Using CKPT_NAME={}".format(ckpt_name))

    # Load training configuration
    conf_path = os.path.join(paths.get_conf_dir(), "conf.json")
    with open(conf_path) as f:
        conf = json.load(f)
        update_with_saved_conf(conf)

    # Load the model
    model = load_model(
        os.path.join(paths.get_checkpoints_dir(), ckpt_name),
        device,
        SmallCNNModel,
        conf["constants"]["channels"],
    )

    # Set the model as loaded
    loaded_ts = timestamp
    loaded_ckpt = ckpt_name


def update_with_saved_conf(saved_conf):
    """
    Update the default YAML configuration
    with the configuration saved from training
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        if group in saved_conf.keys():
            for g_key, g_val in sorted(val.items()):
                if g_key in saved_conf[group].keys():
                    g_val["value"] = saved_conf[group][g_key]

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration
    with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                g_val["value"] = json.loads(user_args[g_key])

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def catch_error(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            pred = f(*args, **kwargs)
            return {"status": "OK", "predictions": pred}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return wrap


def catch_url_error(url_list):

    # Error catch: Empty query
    if not url_list:
        raise ValueError("Empty query")

    for i in url_list:

        if not i.startswith(
            "data:image"
        ):  # don't do the checks for base64 encoded images

            # Error catch: Inexistent url
            try:
                url_type = requests.head(i, timeout=10).headers.get(
                    "content-type"
                )
            except requests.exceptions.Timeout:
                raise ValueError(f"Request to {i} timed out.")
            except Exception:
                raise ValueError(
                    "Failed url connection: "
                    "Check you wrote the url address correctly."
                )

            # Error catch: Wrong formatted urls
            if url_type.split("/")[0] != "image":
                raise ValueError(
                    "Url image format error:"
                    + " Some urls were not in image format."
                    + " Check you didn't uploaded a preview of the image"
                    + " rather than the image itself."
                )


def catch_localfile_error(file_list):

    # Error catch: Empty query
    if not file_list:
        raise ValueError("Empty query")

    # Error catch: Image format error
    for f in file_list:
        extension = os.path.basename(f.content_type).split("/")[-1]
        # extension = mimetypes.guess_extension(f.content_type)
        if extension not in allowed_extensions:
            raise ValueError(
                "Local image format error: "
                + " At least one file is not in a standard image format"
                + f" ({allowed_extensions})."
            )


def warm():
    try:
        load_inference_model()
    except Exception as e:
        print(e)


@catch_error
def predict(**args):

    if args["urls"]:
        args["urls"] = [args["urls"]]  # patch until list is available
        return predict_url(args)

    if not any([args["urls"], args["files"]]) or all(
        [args["urls"], args["files"]]
    ):
        raise Exception(
            "You must provide either 'url' or 'data' in the payload"
        )

    if args["files"]:
        args["files"] = [args["files"]]  # patch until list is available
        return predict_data(args)
    elif args["urls"]:
        args["urls"] = [args["urls"]]  # patch until list is available
        return predict_url(args)


def predict_url(args):
    """
    Function to predict an url
    """
    # Check user configuration
    update_with_query_conf(args)
    conf = config.conf_dict
    catch_url_error(args["urls"])

    # Load model if needed
    if (
        loaded_ts != conf["testing"]["timestamp"]
        or loaded_ckpt != conf["testing"]["ckpt_name"]
    ):
        load_inference_model(
            timestamp=conf["testing"]["timestamp"],
            ckpt_name=conf["testing"]["ckpt_name"],
        )
        conf = config.conf_dict

    model_unet = None
    if conf["base"]["use_preprocess_model"] != "":
        model_path = os.path.join(
            paths.get_preprocess_models_dir(),
            conf["base"]["use_preprocess_model"],
        )
        if Path(model_path).exists():
            model_unet = load_model(model_path, device, Unet)

    # Make the predictions
    result = test_utils.predict(
        model=model,
        model_unet=model_unet,
        device=device,
        file_names=args["urls"],
        filemode="url",
        image_size=conf["base"]["image_size"],
        batch_size=conf["base"]["batch_size"],
        num_workers=conf["constants"]["num_workers"],
    )
    return result


def predict_data(args):
    """
    Function to predict an image in binary format
    """
    # Check user configuration
    update_with_query_conf(args)
    conf = config.conf_dict
    catch_localfile_error(args["files"])

    # Load model if needed
    if (
        loaded_ts != conf["testing"]["timestamp"]
        or loaded_ckpt != conf["testing"]["ckpt_name"]
    ):
        load_inference_model(
            timestamp=conf["testing"]["timestamp"],
            ckpt_name=conf["testing"]["ckpt_name"],
        )
        conf = config.conf_dict

    # Create a list with the path to the images
    filenames = [f.filename for f in args["files"]]

    model_unet = None
    if conf["base"]["use_preprocess_model"] != "":
        model_path = os.path.join(
            paths.get_preprocess_models_dir(),
            conf["base"]["use_preprocess_model"],
        )
        if Path(model_path).exists():
            model_unet = load_model(model_path, device, Unet)
    # Make the predictions
    try:
        result = test_utils.predict(
            model=model,
            model_unet=model_unet,
            device=device,
            file_names=filenames,
            filemode="local",
            image_size=conf["base"]["image_size"],
            batch_size=conf["base"]["batch_size"],
            num_workers=conf["constants"]["num_workers"],
        )
    finally:
        for f in filenames:
            os.remove(f)
    return result


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val["help"] if ("help" in gg_keys) else ""
            type = (
                getattr(builtins, g_val["type"])
                if ("type" in gg_keys)
                else None
            )
            choices = g_val["choices"] if ("choices" in gg_keys) else None

            # Additional info in help string
            help += "\n" + "<font color='#C5576B'> Group name: **{}**".format(
                str(group)
            )
            if choices:
                help += "\n" + "Choices: {}".format(str(choices))
            if type:
                help += "\n" + "Type: {}".format(g_val["type"])
            help += "</font>"

            # Create arg dict
            opt_args = {
                "missing": json.dumps(g_val["value"]),
                "description": help,
                "required": False,
            }
            if choices:
                opt_args["enum"] = [json.dumps(i) for i in choices]

            parser[g_key] = fields.Str(**opt_args)

    return parser


def get_predict_args():

    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([("testing", default_conf["testing"])])

    # Add options for modelname
    timestamp = default_conf["testing"]["timestamp"]
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        timestamp["value"] = ""
    else:
        timestamp["value"] = timestamp_list[-1]
        timestamp["choices"] = timestamp_list

    # Add data and url fields
    parser["files"] = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select the image you want to classify.",
    )

    # Use field.String instead of field.Url because I also want to allow
    # uploading of base 64 encoded data strings
    parser["urls"] = fields.String(
        required=False,
        missing=None,
        description="Select an URL of the image you want to classify.",
    )
    # parser['test_parameter'] = fields.String(required=False,
    #                                missing=None,
    # description="Select an URL of the image you want to classify.")

    # missing action="append" --> append more than one url

    return populate_parser(parser, default_conf)


def get_train_args():

    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict(
        [
            ("base", default_conf["base"]),
            ("general", default_conf["general"]),
        ]
    )

    use_preprocess_model = default_conf["base"]["use_preprocess_model"]
    use_preprocess_model_list = [
        x
        for x in os.listdir(paths.get_preprocess_models_dir())
        if x.endswith(".pth") or x.endswith(".pt")
    ]
    print(f"\n\n\n\n { use_preprocess_model_list}")
    use_preprocess_model_list = sorted(use_preprocess_model_list)
    use_preprocess_model_list.insert(0, "")
    if not use_preprocess_model_list:
        use_preprocess_model["value"] = ""
    else:
        use_preprocess_model["value"] = ""
        use_preprocess_model["choices"] = use_preprocess_model_list

    return populate_parser(parser, default_conf)


def train(**args):
    """
    Train an image classifier
    """
    update_with_query_conf(user_args=args)
    CONF = config.conf_dict
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config.print_conf_table(CONF)
    train_fn(TIMESTAMP=timestamp, CONF=CONF)

    # Sync with NextCloud folders (if NextCloud is available)
    try:
        mount_nextcloud(paths.get_models_dir(), "rshare:/models")
    except Exception as e:
        print(e)

    return {"modelname": timestamp}


schema = {
    "status": fields.Str(),
    "message": fields.Str(),
    "predictions": fields.Field(),
}
