# Original code from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/utils.py#L186-L260
import numpy as np


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch, slices=None):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch, slices)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch, slices):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch, slices=None):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        if slices is None:
            step_size = int(batch.shape[2] / 4)
            slices = list(range(batch.shape[2]))[::step_size]
            # slices = [batch.shape[-3] // 2]  # get the middle slice
        elif isinstance(slices, int):
            slices = [slices]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    for slice_idx in slices:
                        tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                        img = batch[batch_idx, channel_idx, slice_idx, ...]
                        tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            for batch_idx in range(batch.shape[0]):
                for slice_idx in slices:
                    tag = tag_template.format(name, batch_idx, 0, slice_idx)
                    img = batch[batch_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - img.float().min()) / np.ptp(img))
