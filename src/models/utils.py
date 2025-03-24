


from src.models.modules import Video_Encoder, VariancePredictor, Mel_Decoder
from src.models.flow import FlowSpecDecoder

def load_video_encoder(cfg):
    video_encoder = Video_Encoder(cfg)
    return video_encoder

def load_variance_decoder(cfg):
    variance_decoder = VariancePredictor(cfg)
    return variance_decoder

def load_mel_decoder(cfg):
    mel_decoder = Mel_Decoder(cfg)
    return mel_decoder

def load_flow_postnet(cfg):
    postnet = FlowSpecDecoder(cfg)
    return postnet


