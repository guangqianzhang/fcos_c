import collections
from torchvision import transforms
# from deploy.preprocess_image import preprocess_image
from register import register

class Compose:
    """Compose multiple transforms sequentially. The pipeline registry of
    mmdet3d separates with mmdet, however, sometimes we may need to use mmdet's
    pipeline. So the class is rewritten to be able to use pipelines from both
    mmdet3d and mmdet.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        # assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform=register[transform['type']]()
                # transform= preprocess_image(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
