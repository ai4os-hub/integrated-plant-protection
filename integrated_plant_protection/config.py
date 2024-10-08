"""
Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

Modification:
Date: December 2023
Modifier: Jędrzej Smok
Email: jsmok@man.poznan.pl
Github: ai4eosc-psnc
"""

import os
import builtins
import textwrap
import yaml
from importlib import metadata


# Get AI model metadata
API_NAME = "integrated_plant_protection"
API_METADATA = metadata.metadata(API_NAME)  # .json

# Fix metadata for emails from pyproject parsing
_EMAILS = API_METADATA["Author-email"].split(", ")
_EMAILS = map(lambda s: s[:-1].split(" <"), _EMAILS)
API_METADATA["Author-emails"] = dict(_EMAILS)

# Fix metadata for authors from pyproject parsing
_AUTHORS = API_METADATA.get("Author", "").split(", ")
_AUTHORS = [] if _AUTHORS == [""] else _AUTHORS
_AUTHORS += API_METADATA["Author-emails"].keys()
API_METADATA["Authors"] = sorted(_AUTHORS)

homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conf_path = os.path.join(homedir, "etc", "config.yaml")
with open(conf_path, "r") as f:
    CONF = yaml.safe_load(f)


def check_conf(conf=CONF):
    """
    Checks for configuration parameters
    """
    for group, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            gg_keys = g_val.keys()

            if g_val["value"] is None:
                continue

            if "type" in gg_keys:
                var_type = getattr(builtins, g_val["type"])
                if type(g_val["value"]) is not var_type:
                    raise TypeError(
                        "The selected value for {} must be a {}.".format(
                            g_key, g_val["type"]
                        )
                    )

            if ("choices" in gg_keys) and (
                g_val["value"] not in g_val["choices"]
            ):
                raise ValueError(
                    "The selected value for {g_key}"
                    + " is not an available choice."
                )

            if "range" in gg_keys:
                if (g_val["range"][0] is not None) and (
                    g_val["range"][0] > g_val["value"]
                ):
                    raise ValueError(
                        f"The selected value for {g_key}"
                        + " is lower than the minimal possible value."
                    )

                if (g_val["range"][1] != "None") and (
                    g_val["range"][1] < g_val["value"]
                ):
                    raise ValueError(
                        f"The selected value for {g_key}"
                        + " is higher than the maximal possible value."
                    )


check_conf()


def get_conf_dict(conf=CONF):
    """
    Return configuration as dict
    """
    conf_d = {}
    for group, val in conf.items():
        conf_d[group] = {}
        for g_key, g_val in val.items():
            conf_d[group][g_key] = g_val["value"]
    return conf_d


conf_dict = get_conf_dict()


def print_full_conf(conf=CONF):
    """
    Print all configuration parameters (including help, range, choices, ...)
    """
    for group, val in sorted(conf.items()):
        print("=" * 75)
        print("{}".format(group))
        print("=" * 75)
        for g_key, g_val in sorted(val.items()):
            print("{}".format(g_key))
            for gg_key, gg_val in g_val.items():
                print("{}{}".format(" " * 4, gg_key))
                body = "\n".join(
                    [
                        "\n".join(
                            textwrap.wrap(
                                line,
                                width=110,
                                break_long_words=False,
                                replace_whitespace=False,
                                initial_indent=" " * 8,
                                subsequent_indent=" " * 8,
                            )
                        )
                        for line in str(gg_val).splitlines()
                        if line.strip() != ""
                    ]
                )
                print(body)
            print("\n")


def print_conf_table(conf=conf_dict):
    """
    Print configuration parameters in a table
    """
    print("{:<25}{:<30}{:<30}".format("group", "key", "value"))
    print("=" * 75)
    for group, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            print("{:<25}{:<30}{:<15} \n".format(group, g_key, str(g_val)))
        print("-" * 75 + "\n")
