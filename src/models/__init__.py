from .bert import BertClf
from .video import VideoClf
from .latefusion_video import MultimodalLateFusionClf_video

MODELS = {
    "bert": BertClf,
    'video': VideoClf,
    'latefusion_video': MultimodalLateFusionClf_video,
}

def get_model(args):
    return MODELS[args.model](args)
