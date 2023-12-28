# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing the interfacing
tasks. In this way you don't mix your true code with DEEPaaS code and everything is
more modular. That is, if you need to write the predict() function in api.py, you
would import your true predict function and call it from here (with some processing /
postprocessing in between if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at a canonical exemplar
module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""

from pathlib import Path
import pkg_resources

from ai4eosc_uc2.misc import _catch_error
import base64, json, math
from random import random
import time
from webargs import fields, validate


BASE_DIR = Path(__file__).resolve().parents[1]


@_catch_error
def get_metadata():
    """
    DO NOT REMOVE - All modules should have a get_metadata() function
    with appropriate keys.
    """
    distros = list(pkg_resources.find_distributions(str(BASE_DIR), only=True))
    if len(distros) == 0:
        raise Exception("No package found.")
    pkg = distros[0]  # if several select first

    meta_fields = {
        "name": None,
        "version": None,
        "summary": None,
        "home-page": None,
        "author": None,
        "author-email": None,
        "license": None,
    }
    meta = {}
    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()  # to avoid inconsistency due to letter cases
        for k in meta_fields:
            if line_low.startswith(k + ":"):
                _, value = line.split(": ", 1)
                meta[k] = value

    return meta


# def warm():
#     pass
#
#
# def get_predict_args():
#     return {}
#
#
# @_catch_error
# def predict(**kwargs):
#     return None
#
#
# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None






def get_predict_args():
    """
    TODO: add more dtypes
    * int with choices
    * composed: list of strs, list of int
    """
    # WARNING: missing!=None has to go with required=False
    # fmt: off
    arg_dict = {
        "demo-str": fields.Str(
            required=False,
            missing="some-string",
        ),
        "demo-str-choice": fields.Str(
            required=False,
            missing="choice2",
            enum=["choice1", "choice2"],
        ),
        "demo-int": fields.Int(
            required=False,
            missing=1,
        ),
        "demo-int-range": fields.Int(
            required=False,
            missing=50,
            validate=[validate.Range(min=1, max=100)],
        ),
        "demo-float": fields.Float(
            required=False,
            missing=0.1,
        ),
        "demo-bool": fields.Bool(
            required=False,
            missing=True,
        ),
        "demo-dict": fields.Str(  # dicts have to be processed as strings
            required=False,
            missing='{"a": 0, "b": 1}',  # use double quotes inside dict
        ),
        "demo-list-of-floats": fields.List(
            fields.Float(),
            required=False,
            missing=[0.1, 0.2, 0.3],
        ),
        "demo-image": fields.Field(
            required=True,
            type="file",
            location="form",
            description="image",  # description needed to be parsed by UI
        ),
        "demo-audio": fields.Field(
            required=True,
            type="file",
            location="form",
            description="audio",  # description needed to be parsed by UI
        ),
        "demo-video": fields.Field(
            required=True,
            type="file",
            location="form",
            description="video",  # description needed to be parsed by UI
        ),
    }
    # fmt: on
    return arg_dict


@_catch_error
def predict(**kwargs):
    """
    Return same inputs as provided. We also add additional fields
    to test the functionality of the Gradio-based UI [1].
       [1]: https://github.com/deephdc/deepaas_ui
    """
    # Dict are fed as str so have to be converted back
    kwargs["demo-dict"] = json.loads(kwargs["demo-dict"])

    # Add labels and random probalities to output as mock
    prob = [random() for _ in range(5)]
    kwargs["probabilities"] = [i / sum(prob) for i in prob]
    kwargs["labels"] = ["class2", "class3", "class0", "class1", "class4"]

    # Read media files and return them back in base64
    for k in ["demo-image", "demo-audio", "demo-video"]:
        with open(kwargs[k].filename, "rb") as f:
            media = f.read()
        media = base64.b64encode(media)  # bytes
        kwargs[k] = media.decode("utf-8")  # string (in utf-8)

    return kwargs


# Schema to validate the `predict()` output
schema = {
    "demo-str": fields.Str(),
    "demo-str-choice": fields.Str(),
    "demo-int": fields.Int(),
    "demo-int-range": fields.Int(),
    "demo-float": fields.Float(),
    "demo-bool": fields.Bool(),
    "demo-dict": fields.Dict(),
    "demo-list-of-floats": fields.List(fields.Float()),
    "demo-image": fields.Str(
        description="image"  # description needed to be parsed by UI
    ),
    "demo-audio": fields.Str(
        description="audio"  # description needed to be parsed by UI
    ),
    "demo-video": fields.Str(
        description="video"  # description needed to be parsed by UI
    ),
    "labels": fields.List(fields.Str()),
    "probabilities": fields.List(fields.Float()),
}