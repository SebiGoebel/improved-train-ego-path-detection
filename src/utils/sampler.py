from torch.utils.data import Sampler

class TemporalSamplerIteratingSequenceSingleUsedImages(Sampler):
    """Initializes the sampler of the dataloader.
       The reason for this class is to have a batch of used_images which get fed to the model simultaniously.
       this batch of used_images should iterate through the sequence but should not go out of range of a sequnce.
       The TemporalSamplerIteratingSequenceSingleUsedImages class allows the dataloader to have the same used images with different dataaugmentations in one batch.
       (with batch-size 8 -> 1 sequence with different data augementations)

       Args:
            data_source:     Dataset from the dataset class.
            batch_size:      Batch-size.
            sequence_length: Defines the overall length of the sequences of the dataset (76 images)
            num_images:      The number of images feed to the model simultaniouly. (10 images)
    """
    def __init__(self, data_source, batch_size, sequence_length, num_images):
        self.data_source = data_source
        print("length: datasource: ", len(self.data_source))
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_images = num_images
    
    def __iter__(self):
        for sequence in range(len(self.data_source) // self.sequence_length): # durch das dataset
            base_index_of_sequence = sequence * self.sequence_length
            for index_of_image_in_sequence in range(self.sequence_length - self.num_images + 1): # durch die sequence
                for _ in range(self.batch_size): # gleiche used images so oft wie batch_size
                    #print("base index if images used: ", base_index_of_sequence + index_of_image_in_sequence)
                    yield base_index_of_sequence + index_of_image_in_sequence

    def __len__(self):
        return (self.sequence_length - self.num_images) * (len(self.data_source) // self.sequence_length) * self.batch_size # dataloader teilt sowieso nochmal durch die batch_size --> * batch_size, da man nicht den dataset auf die batches aufteilt sondern die sequencen mit der match_size multipliziert (bzw. eine sequence so oft wie die batch_size verwendet)

class TemporalSamplerIteratingSequence(Sampler):
    """Initializes the sampler of the dataloader.
       The reason for this class is to have a batch of used_images which get fed to the model simultaniously.
       this batch of used_images should iterate through the sequence but should not go out of range of a sequnce.
       The TemporalSamplerIteratingSequence class allows the dataloader to have different sequences in one batch.
       (with batch-size 8 -> 8 different sequences)

       Args:
            data_source:     Dataset from the dataset class.
            batch_size:      Batch-size.
            sequence_length: Defines the overall length of the sequences of the dataset (76 images)
            num_images:      The number of images feed to the model simultaniouly. (10 images)
    """
    def __init__(self, data_source, batch_size, sequence_length, num_images):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_images = num_images
    
    def __iter__(self):
        for _ in range(self.batch_size): # immer unterschiedliche used images in einer batch
            for sequence in range(len(self.data_source) // self.sequence_length): # durch das dataset
                base_index_of_sequence = sequence * self.sequence_length
                for index_of_image_in_sequence in range(self.sequence_length - self.num_images + 1): # durch die sequence
                    #print("base index if images used: ", base_index_of_sequence + index_of_image_in_sequence)
                    yield base_index_of_sequence + index_of_image_in_sequence

    def __len__(self):
        return (self.sequence_length - self.num_images) * (len(self.data_source) // self.sequence_length) # dataloader teilt sowieso nochmal durch die batch_size --> kein * batch_size weil man da ja alle scenen in batches auf teilt
    


class TemporalSamplerSingleSequence(Sampler):
    """Initializes the sampler of the dataloader.
       The reason for this class is to always jump to the first image of the next sequence and not just to the next image.
       The TemporalSamplerSingleSequence class allows the dataloader to have the same sequences with different dataaugmentations in one batch.
       (with batch-size 8 -> 1 sequence with different data augementations)

        Args:
            data_source: Dataset from the dataset class.
            batch_size:  Batch-size.
            skip_step:   The number of images the dataloader jumps forward.
    """
    def __init__(self, data_source, batch_size, skip_step):
        self.data_source = data_source
        self.batch_size = batch_size
        self.skip_step = skip_step

    def __iter__(self):
        n = len(self.data_source)
        print("Sampler n: ", n)
        index = 0
        while index < n:
            for _ in range(self.batch_size):
                yield index
                print("index: ", index)
            index += self.skip_step

    def __len__(self):
        return (len(self.data_source) // self.skip_step) * self.batch_size

class TemporalSampler(Sampler):
    """Initializes the sampler of the dataloader.
       The reason for this class is to always jump to the first image of the next sequence and not just to the next image.
       The TemporalSampler class allows the dataloader to have different sequences in one batch.
       (with batch-size 8 -> 8 different sequences)

        Args:
            data_source: Dataset from the dataset class.
            batch_size:  Batch-size.
            skip_step:   The number of images the dataloader jumps forward.
    """
    def __init__(self, data_source, batch_size, skip_step):
        self.data_source = data_source
        self.batch_size = batch_size
        self.skip_step = skip_step

    def __iter__(self):
        n = len(self.data_source)
        index = 0
        while index < n:
            for _ in range(self.batch_size):
                if index < n:
                    yield index
                    index += self.skip_step

    def __len__(self):
        return (len(self.data_source) + (self.skip_step - 1)) // self.skip_step * self.batch_size
