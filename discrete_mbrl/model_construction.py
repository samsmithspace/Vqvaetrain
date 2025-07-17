import os
import sys
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from training_helpers import log_param_updates
from shared.models import *
from shared.trainers import *
from shared.models.iris_models import Encoder as IrisEncoder, Decoder as IrisDecoder, EncoderDecoderConfig

DISCRETE_ENCODER_TYPES = {'vqvae', 'dae', 'softmax_ae', 'hard_fta_ae'}
CONTINUOUS_ENCODER_TYPES = {'ae', 'vae', 'soft_vqvae', 'fta_ae'}

MODEL_VARS = [
    'embedding_dim', 'latent_dim', 'filter_size', 'codebook_size',
    'ae_model_type', 'ae_model_version', 'trans_model_type', 'trans_model_version',
    'trans_hidden', 'trans_depth', 'stochastic', 'extra_info',
    'repr_sparsity', 'sparsity_type', 'vq_trans_1d_conv']

AE_MODEL_VARS = [
    'embedding_dim', 'latent_dim', 'filter_size', 'codebook_size',
    'ae_model_type', 'ae_model_version', 'extra_info', 'repr_sparsity', 'sparsity_type']


def make_ae_layers(input_dim, embedding_dim, version='2'):
    """Create encoder/decoder layers based on version"""
    if len(input_dim) <= 1:
        # Dense AE
        hidden_sizes = [512, 512, 256, 256]
        latent_dim = embedding_dim or hidden_sizes[-1]
        n_features = np.array(input_dim).squeeze()
        encoder = nn.Sequential(*create_dense_layers(n_features, latent_dim, hidden_sizes[:-1]))
        decoder = nn.Sequential(*create_dense_layers(latent_dim, n_features, hidden_sizes[:-1][::-1]))
        return encoder, decoder

    # CNN-based AE
    if version == '1':
        return _make_ae_v1(input_dim, embedding_dim)
    elif version == '2':
        return _make_ae_v2(input_dim, embedding_dim)
    elif version in ['3', '4']:
        return _make_iris_ae(input_dim, embedding_dim, version)
    elif version == 'nature':
        return _make_nature_ae(input_dim, embedding_dim, vanilla=True)
    elif version == 'nature_ae':
        return _make_nature_ae(input_dim, embedding_dim, vanilla=False)
    else:
        raise ValueError(f'Invalid AE version: {version}')


def _make_ae_v1(input_dim, embedding_dim):
    n_channels = input_dim[0]
    filters = (8, 5) if input_dim[1] == 84 else (6, 3) if input_dim[1] == 56 else (8, 5)
    strides = (3, 2) if input_dim[1] == 84 else (2, 2) if input_dim[1] == 56 else (3, 2)
    padding = 1

    encoder_conv = nn.Sequential(
        nn.Conv2d(n_channels, 16, filters[0], strides[0], padding),
        nn.ReLU(),
        nn.Conv2d(16, 32, filters[1], strides[1]),
        nn.ReLU(),
        nn.Conv2d(32, embedding_dim, 3, 1),
        nn.ReLU())

    test_input = torch.ones(1, *input_dim)
    mid_shape = encoder_conv(test_input).shape[1:]
    print(f'AE mid filter shape: {mid_shape[1:]}')

    encoder = nn.Sequential(
        encoder_conv,
        nn.AdaptiveAvgPool2d(8),
        ResidualBlock(embedding_dim, embedding_dim),
        ResidualBlock(embedding_dim, embedding_dim))

    decoder_layers = [
        nn.ReLU(),
        ResidualBlock(embedding_dim, embedding_dim),
        ResidualBlock(embedding_dim, embedding_dim),
        nn.AdaptiveAvgPool2d(mid_shape[1:]),
        nn.ConvTranspose2d(embedding_dim, 32, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, filters[1], strides[1]),
        nn.ReLU(),
        nn.ConvTranspose2d(16, n_channels, filters[0], strides[0], padding)]

    out_shape = nn.Sequential(encoder, *decoder_layers)(test_input).shape[1:]
    if list(out_shape) != list(input_dim):
        decoder_layers.extend([
            nn.AdaptiveAvgPool2d(input_dim[1:]),
            nn.ReLU(),
            ResidualBlock(n_channels, n_channels),
            nn.Conv2d(n_channels, n_channels, 1)])

    return encoder, nn.Sequential(*decoder_layers)


def _make_ae_v2(input_dim, embedding_dim):
    embedding_dim = embedding_dim or 128
    channels = (input_dim[0], 64, 128, embedding_dim)
    filters = (8, 6, 4)
    strides = (2, 2, 2) if input_dim[1] not in (48, 56, 64) else (2, 2, 1)
    if input_dim[1] == 54:
        strides = (2, 1, 2)
    padding = (1, 0, 0)

    encoder_layers = []
    for i in range(len(filters)):
        encoder_layers.extend([
            nn.Conv2d(channels[i], channels[i + 1], filters[i], strides[i], padding[i]),
            nn.ReLU()])

    test_input = torch.ones(1, *input_dim)
    mid_shape = nn.Sequential(*encoder_layers)(test_input).shape[1:]
    print(f'AE mid filter shape: {mid_shape[1:]}')

    encoder_layers.extend([
        nn.AdaptiveAvgPool2d(8),
        ResidualBlock(embedding_dim, embedding_dim)])

    decoder_layers = [
        ResidualBlock(embedding_dim, embedding_dim),
        nn.AdaptiveAvgPool2d(mid_shape[1:])]

    for i in reversed(range(len(filters))):
        decoder_layers.extend([
            nn.ConvTranspose2d(channels[i + 1], channels[i], filters[i], strides[i], padding[i]),
            nn.ReLU()])

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


def _make_iris_ae(input_dim, embedding_dim, version):
    res = input_dim[1]
    config = EncoderDecoderConfig(
        resolution=res, in_channels=input_dim[0], z_channels=embedding_dim, ch=64,
        ch_mult=[1, 1, 1, 1, 1], num_res_blocks=2,
        attn_resolutions=[res // 4, res // 8], out_ch=input_dim[0], dropout=0.0)

    if version == '4':
        config.downsamples = [True, True, True, False, False]

    return IrisEncoder(config), IrisDecoder(config)


def _make_nature_ae(input_dim, embedding_dim, vanilla=False):
    if len(input_dim) <= 1:
        raise ValueError('Nature AE requires 2D+ input')

    embedding_dim = embedding_dim or 64
    channels = (input_dim[0], 32, 64, embedding_dim)
    filters, strides = (8, 4, 3), (4, 2, 1)
    padding = (0, 0, 0) if vanilla or input_dim[1] != 64 else (2, 0, 0)

    encoder_layers = []
    for i in range(len(filters)):
        encoder_layers.extend([
            nn.Conv2d(channels[i], channels[i + 1], filters[i], strides[i], padding[i]),
            nn.ReLU()])

    test_input = torch.ones(1, *input_dim)
    mid_shape = nn.Sequential(*encoder_layers)(test_input).shape[1:]
    print(f'AE mid filter shape: {mid_shape[1:]}')

    if not vanilla:
        encoder_layers.append(nn.AdaptiveAvgPool2d(8))

    decoder_layers = []
    if not vanilla:
        decoder_layers.append(nn.AdaptiveAvgPool2d(mid_shape[1:]))

    for i in reversed(range(len(filters))):
        decoder_layers.append(nn.ConvTranspose2d(channels[i + 1], channels[i], filters[i], strides[i], padding[i]))
        if i > 0:
            decoder_layers.append(nn.ReLU())

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


# Fixed VQ-VAE Implementation
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Match saved model structure
        self._embedding = nn.Embedding(n_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1 / n_embeddings, 1 / n_embeddings)

        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings))
        self.register_buffer('_ema_w', torch.zeros(n_embeddings, embedding_dim))

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).reshape(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

    def encode(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.reshape(input_shape[0], -1)

    def decode_and_reconstruct(self, indices, decoder, spatial_dims):
        flat_indices = indices.reshape(-1)
        quantized_flat = self._embedding(flat_indices)
        batch_size = indices.shape[0]

        if len(spatial_dims) == 2:
            H, W = spatial_dims
            quantized = quantized_flat.reshape(batch_size, H, W, self.embedding_dim)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
        else:
            S = spatial_dims[0]
            quantized = quantized_flat.reshape(batch_size, S, self.embedding_dim)
            quantized = quantized.permute(0, 2, 1).contiguous()

        return decoder(quantized)


class FixedVQVAE(nn.Module):
    def __init__(self, encoder, decoder, codebook_size, embedding_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # Get spatial dimensions
        with torch.no_grad():
            test_input = torch.randn(1, 3, 48, 48)
            encoded_output = encoder(test_input)

        if len(encoded_output.shape) == 4:
            self.spatial_dims = encoded_output.shape[2:]
            self.n_latent_embeds = encoded_output.shape[2] * encoded_output.shape[3]
        else:
            self.spatial_dims = (encoded_output.shape[2],)
            self.n_latent_embeds = encoded_output.shape[2]

        self.n_embeddings = codebook_size
        self.quantizer = VectorQuantizerEMA(codebook_size, embedding_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, perplexity = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded, vq_loss, perplexity

    def encode(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            return self.quantizer.encode(encoded)

    def decode(self, indices):
        with torch.no_grad():
            return self.quantizer.decode_and_reconstruct(indices, self.decoder, self.spatial_dims)


def args_update(args, key, value):
    if hasattr(args, 'wandb') and args.wandb:
        args.update({key: value}, allow_val_change=True)
    else:
        setattr(args, key, value)
    log_param_updates(args, {key: value})


def safe_getattr(args, attr, default=None):
    try:
        return getattr(args, attr, default)
    except (KeyError, AttributeError):
        return default


def construct_ae_model(input_dim, args, load=True, latent_activation=False):
    """Construct autoencoder model"""
    new_hash = make_model_hash(args, model_vars=AE_MODEL_VARS, exp_type='encoder')
    args_update(args, 'ae_model_hash', new_hash)

    if args.ae_model_type in ('identity', 'flatten'):
        if args.ae_model_type == 'flatten':
            model = FlattenModel(input_dim)
        else:
            model = IdentityModel(input_dim, embedding_dim=args.embedding_dim)

        args_update(args, 'final_latent_dim', model.latent_dim)
        print(f'Constructed {args.ae_model_type} model with {model.latent_dim} latents')
        return model, None

    # Create encoder/decoder
    encoder, decoder = make_ae_layers(input_dim, args.embedding_dim, args.ae_model_version)

    # Determine encoder type
    test_input = torch.ones(1, *input_dim)
    encoder_out_shape = encoder(test_input).shape[1:]
    encoder_type = 'dense' if len(encoder_out_shape) == 1 else 'cnn'

    # Construct model based on type
    if args.ae_model_type in CONTINUOUS_ENCODER_TYPES:
        model, trainer = _construct_continuous_model(
            input_dim, encoder, decoder, args, latent_activation, encoder_type)
    else:
        model, trainer = _construct_discrete_model(
            input_dim, encoder, decoder, args, encoder_type)

    # Load model if requested
    if load and model is not None:
        try:
            load_model(model, args, exp_type='encoder', model_vars=AE_MODEL_VARS, model_hash=args.ae_model_hash)
        except Exception as e:
            print(f"Could not load model: {e}. Using fresh initialization.")

    return model, trainer


def _construct_continuous_model(input_dim, encoder, decoder, args, latent_activation, encoder_type):
    """Construct continuous encoder models"""
    if args.ae_model_type in ('ae', 'vae', 'fta_ae'):
        stochastic = args.ae_model_type == 'vae'
        fta = args.ae_model_type == 'fta_ae'
        fta_params = {
            'tiles': args.fta_tiles, 'bound_low': args.fta_bound_low,
            'bound_high': args.fta_bound_high, 'eta': args.fta_eta
        }

        args_update(args, 'codebook_size', None)
        model = AEModel(input_dim, latent_dim=args.latent_dim, encoder=encoder,
                        decoder=decoder, stochastic=stochastic, fta=fta,
                        fta_params=fta_params, latent_activation=latent_activation)
        args_update(args, 'final_latent_dim', model.latent_dim)
        print(f'Constructed {args.ae_model_type.upper()} with {model.latent_dim}-dim latent space')

        TrainerClass = AETrainer if args.ae_model_type in ('ae', 'fta_ae') else VAETrainer

    elif args.ae_model_type == 'soft_vqvae':
        n_latents = args.latent_dim if encoder_type == 'dense' else None
        model = VQVAEModel(
            input_dim, codebook_size=args.codebook_size, embedding_dim=args.embedding_dim,
            encoder=encoder, decoder=decoder, n_latents=n_latents, quantized_enc=True,
            sparsity=args.repr_sparsity, sparsity_type=args.sparsity_type)
        args_update(args, 'final_latent_dim', model.n_latent_embeds * args.codebook_size)
        print(f'Constructed Soft VQVAE with {model.n_latent_embeds} latents and {args.codebook_size} codebook entries')
        TrainerClass = VQVAETrainer

    trainer = TrainerClass(model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip)
    return model, trainer


def _construct_discrete_model(input_dim, encoder, decoder, args, encoder_type):
    """Construct discrete encoder models"""
    n_latents = args.latent_dim if encoder_type == 'dense' else None

    if args.ae_model_type == 'vqvae':
        print("🔧 Creating VQ-VAE model")
        model = FixedVQVAE(encoder, decoder, args.codebook_size, args.embedding_dim)
        total_latent_dim = model.n_latent_embeds * args.codebook_size
        args_update(args, 'final_latent_dim', total_latent_dim)
        print(f'Constructed VQ-VAE with {model.n_latent_embeds} latents and {args.codebook_size} codebook entries')

    elif args.ae_model_type == 'dae':
        model = DAEModel(input_dim, encoder=encoder, decoder=decoder)
        args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
        print(f'Constructed DAE with {np.prod(model.encoder_out_shape[1:])} latents')

    elif args.ae_model_type == 'softmax_ae':
        model = SoftmaxAEModel(input_dim, codebook_size=args.codebook_size,
                               encoder=encoder, decoder=decoder, n_latents=n_latents)
        args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
        print(f'Constructed softmax AE with {model.encoder_out_shape[1:]} latents')

    elif args.ae_model_type == 'hard_fta_ae':
        model = HardFTAAEModel(input_dim, codebook_size=args.codebook_size,
                               encoder=encoder, decoder=decoder, n_latents=n_latents)
        args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
        print(f'Constructed hard FTA AE with {model.encoder_out_shape[1:]} latents')

    TrainerClass = VQVAETrainer if args.ae_model_type == 'vqvae' else AETrainer
    trainer = TrainerClass(model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip)
    return model, trainer


def construct_trans_model(encoder, args, act_space, load=True):
    """Construct transition model"""
    new_hash = make_model_hash(args, model_vars=MODEL_VARS, exp_type='trans_model')
    args_update(args, 'trans_model_hash', new_hash)

    if args.e2e_loss and args.trans_model_type != 'continuous':
        raise ValueError('End-to-end loss only supported for continuous models!')

    if args.trans_model_type == 'discrete':
        use_soft_embeds = safe_getattr(args, 'use_soft_embeds', False) or safe_getattr(encoder, 'quantized_enc', False)
        trans_model = DiscreteTransitionModel(
            encoder.n_latent_embeds, encoder.n_embeddings, encoder.embedding_dim,
            act_space, hidden_sizes=[args.trans_hidden] * args.trans_depth,
            stochastic=args.stochastic, stoch_hidden_sizes=[256, 256],
            discretizer_hidden_sizes=[256], use_soft_embeds=use_soft_embeds,
            return_logits=safe_getattr(encoder, 'quantized_enc', False))

        args_update(args, 'final_latent_dim', encoder.n_latent_embeds * encoder.n_embeddings)
        trainer = DiscreteTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip)

    elif args.trans_model_type == 'continuous':
        if hasattr(encoder, 'latent_dim'):
            latent_dim = encoder.latent_dim
        elif hasattr(encoder, 'n_latent_embeds') and hasattr(encoder, 'embedding_dim'):
            latent_dim = encoder.n_latent_embeds * encoder.embedding_dim
        else:
            latent_dim = 512

        trans_model = ContinuousTransitionModel(
            latent_dim, act_space, hidden_sizes=[args.trans_hidden] * args.trans_depth,
            stochastic=args.stochastic, stoch_hidden_sizes=[256, 256],
            discretizer_hidden_sizes=[256])

        args_update(args, 'final_latent_dim', latent_dim)
        trainer = ContinuousTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip, e2e_loss=args.e2e_loss)
    else:
        raise ValueError(f'Unknown trans_model_type: {args.trans_model_type}')

    if load:
        try:
            load_model(trans_model, args, exp_type='trans_model', model_vars=MODEL_VARS,
                       model_hash=args.trans_model_hash)
        except Exception as e:
            print(f"Could not load transition model: {e}. Using fresh initialization.")

    return trans_model, trainer


def make_model_hash(args=None, model_vars=MODEL_VARS, **kwargs):
    """Create MD5 hash of model parameters"""
    if args is not None:
        args_dict = vars(args).get('_items', vars(args))
        for param in model_vars:
            if param in args_dict:
                kwargs[param] = args_dict[param]

    # Convert numpy types to native Python types
    kwargs = {k: int(v) if isinstance(v, (np.int32, np.int64)) else v for k, v in kwargs.items()}

    dhash = hashlib.md5()
    encoded = json.dumps(kwargs, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


MODEL_SAVE_FORMAT = './models/{}/model_{}.pt'


def save_model(model, args, model_hash=None, model_vars=MODEL_VARS, **kwargs):
    """Save model to disk"""
    if model_hash is None:
        model_hash = make_model_hash(args, model_vars=model_vars, **kwargs)

    save_path = MODEL_SAVE_FORMAT.format(args.env_name, model_hash).replace(':', '-')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to "{save_path}"')
    return model_hash


def load_model(model, args, model_hash=None, return_hash=False, model_vars=MODEL_VARS, **kwargs):
    """Load model from disk"""
    if model_hash is None:
        model_hash = make_model_hash(args, model_vars=model_vars, **kwargs)

    file_path = MODEL_SAVE_FORMAT.format(args.env_name, model_hash).replace(':', '-')

    if not os.path.exists(file_path):
        print(f'No model found at "{file_path}", not loading')
        return (model, model_hash) if return_hash else model

    print(f'Model found at "{file_path}", loading')
    try:
        state_dict = torch.load(file_path, map_location=args.device)
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully!")
    except RuntimeError as e:
        print(f'Failed to load model: {e}')
        raise e

    return (model, model_hash) if return_hash else model