from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .occupancy_loss import OccupancyLoss
from .bce_loss import BinaryCrossEntropyLoss, PixelDistributionLoss
from .render_loss import RGBLoss, DepthLoss, RenderedDepthLoss, DistillLoss, MaskLoss
from .reproj_loss import MultiViewReprojLoss, TemporalViewReprojLoss
