import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class AMED_predictor(torch.nn.Module):
    def __init__(
        self,
        hidden_dim              = 128,
        output_dim              = 1, 
        bottleneck_input_dim    = 64, 
        bottleneck_output_dim   = 4, 
        noise_channels          = 8,  
        embedding_type          = 'positional',
        dataset_name            = None,
        img_resolution          = None,
        num_steps               = None,
        sampler_tea             = None, 
        sampler_stu             = None, 
        M                       = None,
        guidance_type           = None,      
        guidance_rate           = None,
        schedule_type           = None,
        schedule_rho            = None,
        afs                     = False,
        scale_dir               = 0,
        scale_time              = 0,
        max_order               = None,
        predict_x0              = True,
        lower_order_final       = True,
    ):
        super().__init__()
        assert sampler_stu in ['amed', 'dpm', 'dpmpp', 'euler', 'ipndm']
        assert sampler_tea in ['heun', 'dpm', 'dpmpp', 'euler', 'ipndm']
        assert scale_dir >= 0
        assert scale_time >= 0
        self.dataset_name = dataset_name
        self.img_resolution = img_resolution
        self.num_steps = num_steps
        self.sampler_stu = sampler_stu
        self.sampler_tea = sampler_tea
        self.M = M
        self.guidance_type = guidance_type
        self.guidance_rate = guidance_rate
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.scale_dir = scale_dir
        self.scale_time = scale_time
        self.max_order = max_order
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        
        init = dict(init_mode='xavier_uniform')
        
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_layer0 = Linear(in_features=noise_channels, out_features=noise_channels, **init)
        
        self.enc_layer0 = Linear(bottleneck_input_dim, hidden_dim)
        self.enc_layer1 = Linear(hidden_dim, bottleneck_output_dim)
        
        self.fc_r = Linear(2 * noise_channels + bottleneck_output_dim, output_dim)
        if self.scale_dir:
            self.fc_scale_dir = Linear(2 * noise_channels + bottleneck_output_dim, output_dim)
        if self.scale_time:
            self.fc_scale_time = Linear(2 * noise_channels + bottleneck_output_dim, output_dim)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, unet_bottleneck, t_cur, t_next, class_labels=None):
        # Encode the current and next time steps, then concatenate them
        emb = self.map_noise(t_cur.reshape(1,))
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = silu(self.map_layer0(emb)).repeat(unet_bottleneck.shape[0], 1)
        emb1 = self.map_noise(t_next.reshape(1,))
        emb1 = emb1.reshape(emb1.shape[0], 2, -1).flip(1).reshape(*emb1.shape) # swap sin/cos
        emb1 = silu(self.map_layer0(emb1)).repeat(unet_bottleneck.shape[0], 1)
        emb = torch.cat((emb, emb1), dim=1)
        
        # Encode the U-Net bottlenect and concatenate it with the time-embedding
        unet_bottleneck = unet_bottleneck.reshape(unet_bottleneck.shape[0], -1)
        unet_bottleneck = self.enc_layer0(unet_bottleneck)
        unet_bottleneck = silu(unet_bottleneck)
        unet_bottleneck = self.enc_layer1(unet_bottleneck)
        out = torch.cat((unet_bottleneck, emb), dim=1)

        r = self.fc_r(out)
        r = self.sigmoid(r)

        if self.scale_dir:
            scale_dir = self.fc_scale_dir(out)
            scale_dir = self.sigmoid(scale_dir) / (1 / (2 * self.scale_dir)) + (1 - self.scale_dir)
            if not self.scale_time:
                return r, scale_dir

        if self.scale_time:
            scale_time = self.fc_scale_time(out)
            scale_time = self.sigmoid(scale_time) / (1 / (2 * self.scale_time)) + (1 - self.scale_time)
            if not self.scale_dir:
                return r, scale_time
            else:
                return r, scale_dir, scale_time

        return r
