import numpy as np

from imgaug import augmenters as iaa


def augment(batch, kinds, random_state):
    ''' perform data augmetation on the image data in batch.
    '''
    if len(batch['input'].shape) != 4:
        return batch
    assert len(batch['input'].shape) == 4
    batch_aug = {}
    batch_aug.update(batch)
    seq = iaa.SomeOf(
        (0, None),
        [augmentations[kind] for kind in kinds],
        random_order=True,
        random_state=random_state,
    )
    batch_aug['input'] = seq.augment_images(batch_aug['input'])
    return batch_aug

augmentations = {
    'translate_px': iaa.Affine(translate_px={"x": (-4, 4), "y": (-4, 4)}),
    'fliplr': iaa.Fliplr(0.5),
}
