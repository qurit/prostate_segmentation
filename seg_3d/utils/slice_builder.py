class SliceBuilder:
    """
    original code from
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/datasets/utils.py#L40

    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_datasets, label_datasets, patch_shape, stride_shape, weight_dataset, **kwargs):
        """
        :param raw_datasets: ndarray of raw data
        :param label_datasets: ndarray of ground truth labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param weight_dataset: ndarray of weights for the labels
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_datasets[0], patch_shape, stride_shape)
        if label_datasets is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            self._label_slices = self._build_slices(label_datasets[0], patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset[0], patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.
        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'

    @staticmethod
    def remove_halo(patch, index, shape, patch_halo):
        """
        Remove `pad_width` voxels around the edges of a given patch.
        """
        assert len(patch_halo) == 3

        def _new_slices(slicing, max_size, pad):
            if slicing.start == 0:
                p_start = 0
                i_start = 0
            else:
                p_start = pad
                i_start = slicing.start + pad

            if slicing.stop == max_size:
                p_stop = None
                i_stop = max_size
            else:
                p_stop = -pad if pad != 0 else 1
                i_stop = slicing.stop - pad

            return slice(p_start, p_stop), slice(i_start, i_stop)

        D, H, W = shape

        i_c, i_z, i_y, i_x = index
        p_c = slice(0, patch.shape[0])

        p_z, i_z = _new_slices(i_z, D, patch_halo[0])
        p_y, i_y = _new_slices(i_y, H, patch_halo[1])
        p_x, i_x = _new_slices(i_x, W, patch_halo[2])

        patch_index = (p_c, p_z, p_y, p_x)
        index = (i_c, i_z, i_y, i_x)
        return patch[patch_index], index

