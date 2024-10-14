import numpy as np
from PIL import Image

# Aufzeichnen:
# - original
# - verbesserter
# - verbesserter ohne überschreiben des avgs
# - vielleicht höhere crop margins (margins mit side 0.5 und top 0.2 ausprobieren)

class Autocropper:
    def __init__(self, config, coeff=0.01):
    #def __init__(self, config, coeff=0.8):
        self.coeff = coeff  # running average coefficient
        self.crop_margin_sides = config["crop_margin_sides"]
        self.crop_margin_top = config["crop_margin_top"]
        self.n = 0
        self.avg = None
        self.crop_coords = None

    def __call__(self):
        return tuple(self.crop_coords) if self.crop_coords is not None else None

    def rails_coords(self, pred):
        coords = None
        if isinstance(pred, list):
            rails = np.array(pred)
            if rails.size > 0:
                coords = (
                    np.min(rails[0, :, 0]).item(),
                    np.min(rails[:, :, 1]).item(),
                    np.max(rails[1, :, 0]).item(),
                )
        elif isinstance(pred, Image.Image):
            mask = np.array(pred)
            mask = np.nonzero(mask)
            if mask[0].size > 0:
                coords = (
                    np.min(mask[1]).item(),
                    np.min(mask[0]).item(),
                    np.max(mask[1]).item(),
                )
        return coords

    def update(self, img_shape, pred):
        rails_coords = self.rails_coords(pred)
        if rails_coords is None:
            return
        if self.n == 0: # first update
            self.avg = [0, 0, img_shape[0]]                        # ganzes Bild
            self.crop_coords = [0, 0, img_shape[0], img_shape[1]]  # ganzes Bild
            #self.avg = [0, img_shape[1]*0.9, img_shape[0]]                          # nur die unteren 10 % des bildes am anfang
            #self.crop_coords = [0, img_shape[1]*0.9, img_shape[0], img_shape[1]]    # nur die unteren 10 % des bildes am anfang
        if rails_coords[1] >= (self.crop_coords[1] + 0.6 * (img_shape[1] - self.crop_coords[1])): # reset regel
            self.n = 0
            self.avg = [img_shape[0]/3, img_shape[1]/2, 2*img_shape[0]/3] # left=1/3, top=1/2, right=2/3
            self.crop_coords = [img_shape[0]/3, img_shape[1]/2, 2*img_shape[0]/3, img_shape[1]]
        else:
            for i in range(3):
                #self.avg[i] = (self.avg[i] * self.n + rails_coords[i]) / (self.n + 1) # TEP-Net Original
                self.avg[i] = (1 - self.coeff) * self.avg[i] + self.coeff * rails_coords[i] # EWMA -> running average
        new_left = min(rails_coords[0], self.avg[0])  # self.avg to prevent collapse
        new_right = max(rails_coords[2], self.avg[2])
        new_top = min(rails_coords[1], self.avg[1])

        # fixing aspect ratio depending on crop height
        if False:
            #print("test-aspect ratio")
            crop_height = img_shape[1] - new_top
            #ratio_factor = img_shape[0] / img_shape[1] # image width / image height (16:9 => 1,7778)
            ratio_factor = 1 # für quadrat
            crop_width = new_right - new_left
            crop_aspect_ratio = crop_width / crop_height # crop aspect ratio
            if crop_aspect_ratio < ratio_factor:
                # calculating new crop with right aspect ratio
                crop_middle = new_left + (crop_width / 2)
                fixed_crop_width = crop_height * ratio_factor
                new_left = crop_middle - (fixed_crop_width / 2)
                new_right = crop_middle + (fixed_crop_width / 2)

        # avg überschreiben damit es nicht gleich wieder zurückspringen kann sondern sich nur langsam annähert
        self.avg[0] = new_left
        self.avg[2] = new_right
        self.avg[1] = new_top

        # margins & image borders
        margin_sides = self.crop_margin_sides * (new_right - new_left)
        margin_top = self.crop_margin_top * (img_shape[1] - new_top)
        new_coords = (
            max(new_left - margin_sides, 0),
            max(new_top - margin_top, 0),
            min(new_right + margin_sides, img_shape[0]),
        )

        for i in range(3):
            self.crop_coords[i] = int(
                new_coords[i]
            ) # new_coords übernehmen
        self.n += 1
