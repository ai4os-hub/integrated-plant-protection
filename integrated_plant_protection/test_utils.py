"""
Date: December 2023
Author: Jędrzej Smok
Email: jsmok@man.poznan.pl
Github: ai4eosc-psnc
"""

import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from torchvision import transforms

from integrated_plant_protection.data_utils import (
    prepare_preprocess_data,
    prepare_test_data,
)


def _predict(model, model_unet, device, dataloader,resnet):
    res_predictions = []
    res_probs = []
    model.train(False)
    with torch.no_grad():
        for batch in dataloader:
            data = batch.to(device)
            if resnet == 0:
                if model_unet is None:
                    outputs = model(data)
                else:
                    print("\n\nUse preprocessing for predict")
                    target_transform = transforms.Compose(
                        [
                            transforms.Resize((592, 592)),
                            transforms.ToTensor(),
                        ]
                    )
                    intersection_images = _inference_model_batch_optimal(
                        model_unet, data, device, target_transform
                    )
                    outputs = model(intersection_images)
                probs = torch.sigmoid(outputs).numpy(force=True)
                predictions = [
                    "sick" if prob.item() > 0.5 else "healthy" for prob in probs
                ]
                res_predictions.extend(predictions)
                res_probs.extend(probs.flat)
            else:
                if resnet == 101:
                    class_names = np.array(['cercospora', 'unaffected'])
                elif resnet == 34:
                    class_names = np.array(['uneffected', 'brown_rust'])
                else:
                    raise ValueError("Invalid resnet value")
                
                outputs = model(data)
                predicted_indices = outputs.argmax(dim=1)
                class_names_list = [class_names[idx] for idx in predicted_indices]
                scores = outputs.softmax(dim=1).max(dim=1).values
                formatted_scores = [round(score.item(), 3) for score in scores]
                res_predictions.extend(class_names_list)
                res_probs.extend(formatted_scores)
            return res_predictions, [float(p) for p in res_probs]


def predict(
    model,
    model_unet,
    device,
    file_names,
    filemode="local",
    image_size=512,
    batch_size=16,
    num_workers=0,
    resnet=0,
):

    dataloader, _ = prepare_test_data(
        file_names,
        filemode,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        resnet=resnet,
    )
    result = _predict(model, model_unet, device, dataloader, resnet)
    return result


def get_paths(img_dir):
    img_files = os.listdir(img_dir)
    img_files = [os.path.join(img_dir, img) for img in img_files]
    return img_files


@torch.inference_mode()
def _inference_model_batch_optimal(
    model, images_on_device: torch.Tensor, device, target_transformation
) -> torch.Tensor:
    """Return Tensor of transformed intersections on gpu"""
    results = torch.Tensor().to(device)
    outputs = model(images_on_device)
    predicted_masks = (outputs >= 0.5).float() * 255
    for img, mask in zip(images_on_device, predicted_masks):
        mask = mask.squeeze()
        mask = mask.unsqueeze(0).repeat(3, 1, 1)
        res = torch.where(mask == 255, img, 0)
        results = torch.cat((results, torch.unsqueeze(res, 0)), dim=0)
    return results


@torch.inference_mode()
def inference_model(model, images_dataloader, output_path, device):
    results = []
    for batch in images_dataloader:
        images, paths = batch
        images = images.to(device)
        # images = torch.stack(batch, 0)
        outputs = model(images)
        predicted_masks = (outputs >= 0.5).float() * 255

        for img, mask, path in zip(images, predicted_masks, paths):
            mask = mask.squeeze()
            mask = mask.numpy(force=True).astype(np.uint8)
            img = img.permute(1, 2, 0)
            img = (img.numpy(force=True) * 255).astype(np.uint8)
            res = cv2.bitwise_and(img, img, mask=mask)

            res_pil = transforms.functional.to_pil_image(res)
            res_pil.save(f"/{output_path}/{path.split('/')[-1]}")
            results.extend(res)
    return results


def save_intersection(
    input_dir: str,
    output_dir: str,
    device: str,
    model,
    image_size=512,
    batch_size=16,
    num_workers=0,
) -> List[np.ndarray]:
    paths_img_files = get_paths(input_dir)
    images_dataloader, _ = prepare_preprocess_data(
        file_names=paths_img_files,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    inference_model(model, images_dataloader, output_dir, device)
