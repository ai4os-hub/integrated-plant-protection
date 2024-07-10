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

import unittest, os, shutil

import integrated_plant_protection.api as api
from integrated_plant_protection.paths import get_base_dir
from deepaas.model.v2.wrapper import UploadedFile


class TestModelMethods(unittest.TestCase):
    def setUp(self):
        self.meta = api.get_metadata()
        self.test_image_url = "https://share.services.ai4os.eu/index.php/s/PZP6WYSALsgT7Lf/download/sick_105.jpg"
        self.test_timestamp = '"rye-7"'
        self.test_ckpt = '"model_0006.pt"'
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

    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(
            self.meta["name"].lower().replace("-", "_"),
            "integrated_plant_protection".lower().replace("-", "_"),
        )
        self.assertEqual(self.meta["author"].lower(), "PSNC WODR".lower())
        self.assertEqual(
            self.meta["license"].lower(),
            "MIT".lower(),
        )

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
        self.assertGreaterEqual(res_prob[0], 0.9)

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
        self.assertGreaterEqual(res_prob[0], 0.9)

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
            "healthy_data_path": '"/home/integrated_plant_protection/data/images/dataset1/healthy/images"',
            "sick_data_path": '"/home/integrated_plant_protection/data/images/dataset1/sick/images"',
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
