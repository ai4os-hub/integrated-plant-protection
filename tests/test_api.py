# -*- coding: utf-8 -*-
"""
Its good practice to have tests checking your code runs correctly.
Here we included a dummy test checking the api correctly returns
expected metadata. We suggest to extend this file to include, for
example, test for checking the predict() function is indeed working
as expected.

These tests will run in the Jenkins pipeline after each change
you make to the code.
"""

import unittest, os, shutil, requests, zipfile
import integrated_plant_protection.api as api
from integrated_plant_protection.paths import get_base_dir
from deepaas.model.v2.wrapper import UploadedFile
from integrated_plant_protection import paths


TEST_IMAGE_URL = "https://beta.ibis.apps.psnc.pl/ai4eosc/models/fb8c695c-4b34-4a5c-bef3-1f0fdae6c65f/integrated_plant_protection/test_data/sick_105.JPG"
TEST_MODEL_URL = "https://beta.ibis.apps.psnc.pl/ai4eosc/models/fb8c695c-4b34-4a5c-bef3-1f0fdae6c65f/integrated_plant_protection/test_data/rye-7.zip"
TEST_IMAGES_URL = "https://beta.ibis.apps.psnc.pl/ai4eosc/models/fb8c695c-4b34-4a5c-bef3-1f0fdae6c65f/integrated_plant_protection/test_data/images.zip"


def download_file(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open("./tmp.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {url} to tmp.zip")
        with zipfile.ZipFile("./tmp.zip", "r") as zip_ref:
            zip_ref.extractall(output_path)
        os.remove("./tmp.zip")
        print(f"Deleted ./tmp.zip")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    except zipfile.BadZipFile as e:
        print(f"An error occurred: {e}")
    except OSError as e:
        print(f"An error occurred: {e}")


class TestModelMethods(unittest.TestCase):
    def setUp(self):
        download_file(TEST_MODEL_URL, paths.get_models_dir())
        download_file(TEST_IMAGES_URL, "./data/")
        self.meta = api.get_metadata()
        self.test_image_url = TEST_IMAGE_URL
        self.test_timestamp = '"rye-7"'
        self.test_ckpt = '"rye-7.pt"'
        data_path = os.path.join(get_base_dir(), "data")
        image_path = os.path.join(data_path, "sick_105.jpg")
        tmp_path = os.path.join(data_path, "tmp_file.jpg")
        shutil.copyfile(image_path, tmp_path)
        self.test_image_data = UploadedFile(
            name="data", filename=tmp_path, content_type="image/jpg"
        )

    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)

    def test_model_predict_url(self):
        print("Testing local: predict url ...")
        args = {
            "urls": [self.test_image_url],
            "ckpt_name": self.test_ckpt,
            "timestamp": self.test_timestamp,
        }
        res = api.predict_url(args)
        res_class, res_prob = res
        self.assertEqual(
            res_class[0],
            "sick",
        )
        self.assertGreaterEqual(res_prob[0], 0.5)

    def test_model_predict_data(self):
        print("Testing local: predict data ...")
        args = {
            "files": [self.test_image_data],
            "ckpt_name": self.test_ckpt,
            "timestamp": self.test_timestamp,
        }
        res = api.predict_data(args)
        res_class, res_prob = res
        self.assertEqual(
            res_class[0],
            "sick",
        )
        self.assertGreaterEqual(res_prob[0], 0.5)

    def test_model_train(self):
        print("Testing local: train ...")
        args = {
            "batch_size": "16",
            "image_size": "512",
            "epochs": "1",
            "learning_rate": "0.001",
            "early_stopping_patience": "30",
            "reduce_lr_factor": "0.4",
            "reduce_lr_patience": "8",
            "experiment": '"ai4eosc"',
            "mlflow": "false",
            "mlflow_params": '"Test"',
            "seed": "0",
            "shuffle": "false",
            "tensorboard": "true",
            "experiment_name": '"test"',
            "healthy_data_path": '"./data/images/sick"',
            "sick_data_path": '"./data/images/sick"',
            "use_preprocess_model": '""',
            "base_directory": '"."',
        }
        res = api.train(**args)
        self.assertIn(
            "modelname",
            res.keys(),
        )


if __name__ == "__main__":
    unittest.main()
