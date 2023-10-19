from .detector3d_template import Detector3DTemplate

from .voxel_rcnn import VoxelRCNN
from .voxel_rcnn_ei import VoxelRCNN_EI
__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'VoxelRCNN': VoxelRCNN,
    'VoxelRCNN_EI': VoxelRCNN_EI
}

def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
