import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from .common import to_scaled_tensor
from .postprocessing import regression_to_rails

# color jitter
import random
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


class TemporalPathsDataset(Dataset):
    def __init__(
        self,
        imgs_path,
        annotations_path,
        indices,
        config,
        method,
        number_images_used,
        img_crop, # tuple or str or None
        img_aug=False,
        img_rd_flip=True,
        to_tensor=False,
    ):
        """Initializes the dataset for ego-path detection.

        Args:
            imgs_path (str): Path to the images directory.
            annotations_path (str):  Path to the annotations file.
            indices (list): List of indices to use in the dataset.
            config (dict): Data generation configuration.
            method (str): Method to use for ground truth generation ("classification", "regression" or "segmentation").
            number_images_used (int): Number of images, which are feed to the LSTM Layers.
            img_crop (tuple or str or None): Coordinates to use for cropping as dataaugmentaiton on the input image:
                - If str, should be "random" to use random cropping (as dataaugmentation).
                - If tuple, should be the inclusive absolute coordinates (diff_crop_left, diff_crop_right, diff_crop_top) of the fixed region.
                - If None, no cropping is performed. (-> for evaluating with autocrop later on)
            img_aug (bool, optional): Whether to use stochastic image adjustment (brightness, contrast, saturation and hue). Defaults to False.
            img_rd_flip (bool, optional): Whether to use a random flip on images. Defaults to True.
            to_tensor (bool, optional): Whether to return a ready to infer tensor (scaled and possibly resized). Defaults to False.
        """
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.imgs_path = imgs_path
        with open(annotations_path) as json_file:
            self.annotations = json.load(json_file)
        
        #print("Annotations Keys:")
        test_list = list(self.annotations.keys())
        #print("indices list lenght: ", len(test_list))

        output_file = 'test_image_names.txt'

        # Öffne die Datei im Schreibmodus
        with open(output_file, 'w') as file:
            # Schleife durch die Liste
            for image_name in test_list:
                # Schreibe den Bildnamen in die Datei, gefolgt von einem Zeilenumbruch
                file.write(image_name + '\n')
        
        self.imgs = [sorted(self.annotations.keys())[i] for i in indices]

        #print("dataset lenght self_imgs: ", len(self.imgs))

        output_file2 = 'test_self_imgs.txt'

        # Öffne die Datei im Schreibmodus
        with open(output_file2, 'w') as file:
            # Schleife durch die Liste
            for image_name in self.imgs:
                # Schreibe den Bildnamen in die Datei, gefolgt von einem Zeilenumbruch
                file.write(image_name + '\n')
        
        self.config = config
        self.method = method
        self.number_images_used = number_images_used
        self.img_aug = img_aug
        self.img_rd_flip = img_rd_flip

        if (isinstance(img_crop, tuple) and len(img_crop) == 3) or img_crop == "random":
            self.img_crop = img_crop
        else:
            self.img_crop = None


        self.to_tensor = (
            transforms.Compose(
                [
                    to_scaled_tensor,
                    transforms.Resize(self.config["input_shape"][1:][::-1]),
                ]
            )
            if to_tensor
            else None
        )

    def __len__(self):
        return len(self.imgs) #int(len(self.imgs) / 76) # self.sequence_length # - self.sequence_length + 1 # reduced lenght of dataset to prevent index error at last sequence (start_index of last sequence)

    def __getitem__(self, idx):

        images = []
        annotations = []
        #print("GetItem - Index: ", idx)
        
        if idx > (len(self.imgs) - self.number_images_used):
            print("Index: ", idx)
            raise ValueError("!!! ACHTUNG FALSCHER INDEX !!!")

        for i in range(self.number_images_used):
            # reading images
            img_name = self.imgs[idx + i]
            img = Image.open(os.path.join(self.imgs_path, img_name))
            images.append(img)

            # reading annotations
            annotation = self.annotations[img_name]
            annotations.append(annotation)

        # reading in labels of last image of the sequence
        #img_name_last = self.imgs[idx + self.number_images_used - 1]
        #annotation = self.annotations[img_name_last]

        # generating rail masks
        rails_masks = []
        for img, annotation in zip(images, annotations):
            rails_mask = self.generate_rails_mask(img.size, annotation) # generiert die annotations in ein numpy array (methode unverändert von TEP)
            rails_masks.append(rails_mask)
        
        # variables and lists for cropping
        first_img = images[0]               # first image of the sliding window
        first_rails_maks = rails_masks[0]   # first mask of sliding window
        cropped_images = []                 # list of cropped images
        cropped_rails_masks = []            # list of cropped rail masks

        if isinstance(self.img_crop, tuple) and len(self.img_crop) == 3:
            diff_crop_left, diff_crop_right, diff_crop_top = self.img_crop
            # loop over every image to get the crops
            for img, rails_mask in zip(images, rails_masks):
                cropped_image, cropped_rails_mask = self.get_crop(img, rails_mask, diff_crop_left, diff_crop_right, diff_crop_top) # just get the crop of the red rectangle without any random factors
                cropped_images.append(cropped_image)
                cropped_rails_masks.append(cropped_rails_mask)
        elif self.img_crop == "random":
            # get the difference between random and red border
            diff_crop_left, diff_crop_right, diff_crop_top = self.get_diff_random_crop(first_img, first_rails_maks)

            # loop over every image to get the crops
            for img, rails_mask in zip(images, rails_masks):
                cropped_image, cropped_rails_mask = self.get_crop(img, rails_mask, diff_crop_left, diff_crop_right, diff_crop_top)
                cropped_images.append(cropped_image)
                cropped_rails_masks.append(cropped_rails_mask)
        else:
            cropped_images = list(images)           # übernehmen der liste ohne crop --> ganzes Bild
            cropped_rails_masks = list(rails_masks) # übernehmen der liste ohne crop --> ganzes Bild

        if self.img_rd_flip:
            cropped_images, cropped_rails_masks = self.random_flip_lr(cropped_images, cropped_rails_masks) # angepasst auf ein array von images und masks

        # Transforming cropped images to scaled tensors (values between 0 and 1) and resize tensors to input shape (see gloabl.yaml)
        if self.to_tensor:
            tensor_images = []
            for image in cropped_images:
                tensor_image = self.to_tensor(image)
                tensor_images.append(tensor_image)
        
        # Augment image-tensors with color jitter
        if self.img_aug:
            tensor_images = self.apply_color_jitter_to_batch(tensor_images)

        # convert list of 3d tensors to a single 4d tensor
        if self.to_tensor:
            sequence_tensor = torch.stack(tensor_images)  # Erstellt einen neuen Tensor mit einer zusätzlichen Dimension für die Sequenz

        # generate target from the annotation of the last image
        label_last_image = cropped_rails_masks[-1]
        if self.method == "regression":
            path_gt, ylim_gt = self.generate_target_regression(label_last_image)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
                ylim_gt = torch.tensor(ylim_gt)
                return sequence_tensor, path_gt, ylim_gt # returning sequence tensor [seq_len, C, H, W]
            return cropped_images, path_gt, ylim_gt      # returning list of images
        elif self.method == "classification":
            path_gt = self.generate_target_classification(label_last_image)
            if self.to_tensor:
                path_gt = torch.from_numpy(path_gt)
                return sequence_tensor, path_gt          # returning sequence tensor [seq_len, C, H, W]
            return cropped_images, path_gt               # returning list of images
        elif self.method == "segmentation":
            segmentation = self.generate_target_segmentation(label_last_image)
            if self.to_tensor:
                segmentation = segmentation.resize(
                    self.config["input_shape"][1:][::-1], Image.NEAREST
                )
                segmentation = to_scaled_tensor(segmentation)
                return sequence_tensor, segmentation     # returning sequence tensor [seq_len, C, H, W]
            return cropped_images, segmentation          # returning list of images

    def apply_color_jitter_to_batch(self, tensor_images):
        # determine random jitter configuration once
        brightness = random.uniform(max(0, 1 - self.config["brightness"]), 1 + self.config["brightness"])
        contrast = random.uniform(max(0, 1 - self.config["contrast"]), 1 + self.config["contrast"])
        saturation = random.uniform(max(0, 1 - self.config["saturation"]), 1 + self.config["saturation"])
        hue = random.uniform(-self.config["hue"], self.config["hue"])

        def apply_jitter(image_tensor):
            # apply same jitter configuration on one image tensor
            image_tensor = adjust_brightness(image_tensor, brightness)
            image_tensor = adjust_contrast(image_tensor, contrast)
            image_tensor = adjust_saturation(image_tensor, saturation)
            image_tensor = adjust_hue(image_tensor, hue)
            return image_tensor

        # apply same jitter configuration on every image tensor with apply_jitter()
        transformed_images = [apply_jitter(image) for image in tensor_images]
        return transformed_images

    def generate_rails_mask(self, shape, annotation):
        rails_mask = Image.new("L", shape, 0)
        draw = ImageDraw.Draw(rails_mask)
        rails = [np.array(annotation["left_rail"]), np.array(annotation["right_rail"])]
        for rail in rails:
            draw.line([tuple(xy) for xy in rail], fill=1, width=1)
        rails_mask = np.array(rails_mask)
        rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0
        for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]:
            rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0
        return rails_mask

    def get_diff_random_crop(self, img, rails_mask):
        """ This function calculates determines a random crop based on a single image
            and retunrs the difference between the random crop and the red borders (centered + margins) (see fig. 4 in original TEP Paper)
            
            Args:
            img: a single image (can be any image of the sequence, but in this case its the first one of the sliding window)
            rails_maks: the rails_mask of the corresponding image

            Return:
            diff_crop_left, diff_crop_right, diff_crop_top: the difference from the random crop to the red borders
        """
        # extract sides rails coordinates (yellow rectangle in paper fig. 4)
        rails_mask_last_line = rails_mask[-1, :]                        # last line
        rails_mask_last_line_idx = np.nonzero(rails_mask_last_line)[0]  # idxs in last line
        most_left_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][0]   # yellow left
        most_right_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][-1] # yellow right
        # center the crop around the rails (orange rectangle in paper fig. 4)
        base_margin_left = rails_mask_last_line_idx[0] - most_left_rail     # abstand gelbe grenzen zu rails in untersten zeile
        base_margin_right = most_right_rail - rails_mask_last_line_idx[-1]  # abstand gelbe grenzen zu rails in untersten zeile
        max_base_margin = max(base_margin_left, base_margin_right, 0)       # nimm den größeren abstand
        mean_crop_left = rails_mask_last_line_idx[0] - max_base_margin      # füge den abstand hinzu (in die korrekte Richtung)
        mean_crop_right = rails_mask_last_line_idx[-1] + max_base_margin    # füge den abstand hinzu (in die korrekte Richtung)
        # add sides margins (red rectangle in paper fig. 4)
        base_width = mean_crop_right - mean_crop_left + 1
        mean_crop_left -= base_width * self.config["crop_margin_sides"]
        mean_crop_right += base_width * self.config["crop_margin_sides"]
        # bis dahin hat man die grenzen nach links und rechts vom roten Rechteck (äußerstes Rechteck) !!!!!
        # random left crop
        largest_margin = max(mean_crop_left, most_left_rail - mean_crop_left) # entweder der abstand zwischen linker roter und gelber grenze oder zwischen linkem bildrand und linker roter grenze
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_left = round(np.random.normal(mean_crop_left, std_dev))
        if random_crop_left > rails_mask_last_line_idx[0]:
            random_crop_left = 2 * rails_mask_last_line_idx[0] - random_crop_left   # check if rails are in crop
        random_crop_left = max(random_crop_left, 0)                                 # check if crop is in image
        # random right crop
        largest_margin = max(
            mean_crop_right - most_right_rail, img.width - 1 - mean_crop_right
        ) # entweder abstand zwischen rechter roter und rechter gelber grenze oder abstand zwischen rechter roter und rechtem bild rand
        std_dev = largest_margin * self.config["std_dev_factor_sides"]
        random_crop_right = round(np.random.normal(mean_crop_right, std_dev))
        if random_crop_right < rails_mask_last_line_idx[-1]:
            random_crop_right = 2 * rails_mask_last_line_idx[-1] - random_crop_right # check if rails are in crop
        random_crop_right = min(random_crop_right, img.width - 1)                    # check if crop is in image
        # extract top rails coordinates (yellow rectangle in paper fig. 4)
        most_top_rail = np.nonzero(np.sum(rails_mask, axis=1))[0][0] # yellow top
        # add top margin (red rectangle in paper fig. 4)
        rail_height = img.height - most_top_rail
        mean_crop_top = most_top_rail - rail_height * self.config["crop_margin_top"]
        # random top crop
        largest_margin = max(mean_crop_top, img.height - 1 - mean_crop_top)
        std_dev = largest_margin * self.config["std_dev_factor_top"]
        random_crop_top = round(np.random.normal(mean_crop_top, std_dev))
        random_crop_top = max(random_crop_top, 0)               # check if crop is in image
        random_crop_top = min(random_crop_top, img.height - 2)  # check if crop is at least 2 rows

        # calculate the distance from final random_crop to red borders (centered + margin)
        # random_crops: random_crop_left, random_crop_top, random_crop_right + 1
        # red borders: mean_crop_left, mean_crop_top, mean_crop_right
        diff_crop_left = random_crop_left - mean_crop_left
        diff_crop_right = random_crop_right - mean_crop_right
        diff_crop_top = random_crop_top - mean_crop_top

        return diff_crop_left, diff_crop_right, diff_crop_top

    def get_crop(self, img, rails_mask, diff_crop_left, diff_crop_right, diff_crop_top):
        """ This function takes the difference from get_diff_random_crop() and a single image
            and calculates the new crop according to the annotation and the differences.
            This way, a random crop can be used, but consistency along a sequence is kept.
            
            Args:
            img: a single image (can be any image of the sequence, but in this case its the first one)
            rails_maks: the rails_mask of the correscponding image
            diff_crop_left, diff_crop_right, diff_crop_top: the difference from the random crop to the red borders

            Return:
            img, rails_mask: cropped image and rail_mask
        """
        # extract sides rails coordinates (yellow rectangle in paper fig. 4)
        rails_mask_last_line = rails_mask[-1, :]                        # last line
        rails_mask_last_line_idx = np.nonzero(rails_mask_last_line)[0]  # idxs in last line
        most_left_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][0]   # yellow left
        most_right_rail = np.nonzero(np.sum(rails_mask, axis=0))[0][-1] # yellow right
        # center the crop around the rails (orange rectangle in paper fig. 4)
        base_margin_left = rails_mask_last_line_idx[0] - most_left_rail     # abstand gelbe grenzen zu rails in untersten zeile
        base_margin_right = most_right_rail - rails_mask_last_line_idx[-1]  # abstand gelbe grenzen zu rails in untersten zeile
        max_base_margin = max(base_margin_left, base_margin_right, 0)       # nimm den größeren abstand
        mean_crop_left = rails_mask_last_line_idx[0] - max_base_margin      # füge den abstand hinzu (in die korrekte Richtung)
        mean_crop_right = rails_mask_last_line_idx[-1] + max_base_margin    # füge den abstand hinzu (in die korrekte Richtung)
        # add sides margins (red rectangle in paper fig. 4)
        base_width = mean_crop_right - mean_crop_left + 1
        mean_crop_left -= base_width * self.config["crop_margin_sides"]
        mean_crop_right += base_width * self.config["crop_margin_sides"]
        # bis dahin hat man die grenzen nach links und rechts vom roten Rechteck (äußerstes Rechteck) !!!!!

        # extract top rails coordinates (yellow rectangle in paper fig. 4)
        most_top_rail = np.nonzero(np.sum(rails_mask, axis=1))[0][0] # yellow top
        # add top margin (red rectangle in paper fig. 4)
        rail_height = img.height - most_top_rail
        mean_crop_top = most_top_rail - rail_height * self.config["crop_margin_top"]

        # red borders: mean_crop_left, mean_crop_right
        # differences: diff_crop_left, diff_crop_right, diff_crop_top
        new_crop_left = mean_crop_left + diff_crop_left
        new_crop_right = mean_crop_right + diff_crop_right
        new_crop_top = mean_crop_top + diff_crop_top

        new_crop_left = round(new_crop_left)
        new_crop_right = round(new_crop_right)
        new_crop_top = round(new_crop_top)

        # new left crop
        if new_crop_left > rails_mask_last_line_idx[0]:
            new_crop_left = 2 * rails_mask_last_line_idx[0] - new_crop_left   # check if rails are in crop
        new_crop_left = max(new_crop_left, 0)                                 # check if crop is in image
        # new right crop
        if new_crop_right < rails_mask_last_line_idx[-1]:
            new_crop_right = 2 * rails_mask_last_line_idx[-1] - new_crop_right # check if rails are in crop
        new_crop_right = min(new_crop_right, img.width - 1)                    # check if crop is in image
        # new top crop
        new_crop_top = max(new_crop_top, 0)               # check if crop is in image
        new_crop_top = min(new_crop_top, img.height - 2)  # check if crop is at least 2 rows

        # crop image and mask
        img = img.crop(
            (new_crop_left, new_crop_top, new_crop_right + 1, img.height)
        )
        rails_mask = rails_mask[
            new_crop_top:, new_crop_left : new_crop_right + 1
        ]
        return img, rails_mask

    def resize_mask(self, mask, shape):
        height_factor = (shape[0] - 1) / (mask.shape[0] - 1)
        width_factor = (shape[1] - 1) / (mask.shape[1] - 1)
        resized_mask = np.zeros(shape, dtype=np.uint8)
        for i in range(resized_mask.shape[0]):
            row_mask = mask[round(i / height_factor), :]
            row_idx = np.nonzero(row_mask)[0]
            if len(row_idx) == 2:
                resized_mask[i, np.round(row_idx * width_factor).astype(int)] = 1
        return resized_mask

    # für ein bild
    #def random_flip_lr(self, img, rails_mask):
    #    if np.random.rand() < 0.5:
    #        img = ImageOps.mirror(img)
    #        rails_mask = np.fliplr(rails_mask)
    #    return img, rails_mask

    # für array von bildern und masks
    def random_flip_lr(self, imgs, rails_masks):
        flipped_imgs = []
        flipped_rails_masks = []
        flip = np.random.rand() < 0.5  # Entscheide einmal, ob geflippt wird oder nicht
        if flip:
            for rm in rails_masks:
                rm = np.fliplr(rm)
                flipped_rails_masks.append(rm)
            for img in imgs:
                img = ImageOps.mirror(img)
                flipped_imgs.append(img)
            return flipped_imgs, flipped_rails_masks
        return imgs, rails_masks

    def generate_target_regression(self, rails_mask):
        unvalid_rows = np.where(np.sum(rails_mask, axis=1) != 2)[0]
        ylim_target = (
            float(1 - (unvalid_rows[-1] + 1) / rails_mask.shape[0])
            if len(unvalid_rows) > 0
            else 1.0
        )
        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], rails_mask.shape[1])
        )
        traj_target = np.array(
            [np.zeros(self.config["anchors"]), np.ones(self.config["anchors"])],
            dtype=np.float32,
        )
        for i in range(self.config["anchors"]):  # it's possible to vectorize this loop
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            if len(rails_points) != 2:
                break
            rails_points_normalized = rails_points / (rails_mask.shape[1] - 1)
            traj_target[:, i] = rails_points_normalized
        return traj_target, ylim_target

    def generate_target_classification(self, rails_mask):
        rails_mask = self.resize_mask(
            rails_mask, (self.config["anchors"], self.config["classes"])
        )
        target = (
            np.ones((2, self.config["anchors"]), dtype=int) * self.config["classes"]
        )
        for i in range(self.config["anchors"]):
            row = rails_mask.shape[0] - 1 - i
            rails_points = np.nonzero(rails_mask[row, :])[0]
            if len(rails_points) != 2:
                break
            target[:, i] = rails_points
        return target

    def generate_target_segmentation(self, rails_mask):
        target = np.zeros_like(rails_mask, dtype=np.uint8)
        row_indices, col_indices = np.nonzero(rails_mask)
        range_rows = np.arange(row_indices.min(), row_indices.max() + 1)
        for row in reversed(range_rows):
            rails_points = col_indices[row_indices == row]
            if len(rails_points) != 2:
                break
            target[row, rails_points[0] : rails_points[1] + 1] = 255
        return Image.fromarray(target)

    def get_perspective_weight_limit(self, percentile, logger):
        logger.info("\nCalculating perspective weight limit...")
        weights = []
        # original TEP: Perspective weight limit: 19.00
        # temporal_dataset: Perspective weight limit: 19.64
        #limit = 19.64
        for i in range(len(self)-self.number_images_used+1):
            print("i: ", i)
            #print("self[i]: ", self[i])
            _, traj, ylim = self[i]
            # _ --> images: Tensor.size([10, 3, 512, 512])
            # traj --> trajectory from last image
            # ylim --> ylim from last image
            #print("--------------")
            #print("_: ", _)
            #print("_.shape: ", _.shape)
            #print("--------------")
            #print("traj: ", traj)
            #print("traj.shape: ", traj.shape)
            #print("--------------")
            #print("ylim: ", ylim)
            #print("--------------")
            rails = regression_to_rails(traj.numpy(), ylim.item())
            #print("rails: ", rails)
            left_rail, right_rail = rails
            #print("rails:")
            #print("left_rail: ", left_rail)
            #print("right_rail: ", right_rail)
            rail_width = right_rail[:, 0] - left_rail[:, 0]
            #print("rail_width: ", rail_width)
            #print("rail_width.len(): ", len(rail_width))
            weight = 1 / rail_width
            #print("weight: ", weight)
            #print("weight.len: ", len(weight))
            weights += weight.tolist()
            #print("weights: ", weights)
            #print("weights.len: ", len(weights))
        limit = np.percentile(sorted(weights), percentile)
        logger.info(f"Perspective weight limit: {limit:.2f}")
        return limit
