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

import json
import os
from integrated_plant_protection import paths


def create_dir_tree():
    """
    Create directory tree structure
    """
    dirs = paths.get_dirs()
    for d in dirs.values():
        if not os.path.isdir(d):
            print("creating {}".format(d))
            os.makedirs(d)


def remove_empty_dirs():
    basedir = paths.get_base_dir()
    dirs = os.listdir(basedir)
    for d in dirs:
        d_path = os.path.join(basedir, d)
        if not os.listdir(d_path):
            os.rmdir(d_path)


def save_conf(conf):
    """
    Save CONF to a txt file to ease the reading
    and to a json file to ease the parsing.
    Parameters
    ----------
    conf : 1-level nested dict
    """
    save_dir = paths.get_conf_dir()

    # Save dict as json file
    with open(os.path.join(save_dir, "conf.json"), "w") as outfile:
        json.dump(conf, outfile, sort_keys=True, indent=4)

    # Save dict as txt file for easier redability
    txt_file = open(os.path.join(save_dir, "conf.txt"), "w")
    txt_file.write("{:<25}{:<30}{:<30} \n".format("group", "key", "value"))
    txt_file.write("=" * 75 + "\n")
    for key, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            txt_file.write(
                "{:<25}{:<30}{:<15} \n".format(key, g_key, str(g_val))
            )
        txt_file.write("-" * 75 + "\n")
    txt_file.close()
