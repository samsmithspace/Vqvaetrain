import os
import sys
import warnings

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hashlib
import json
from torch import nn
from training_helpers import log_param_updates
from shared.models import *
from shared.trainers import *
from shared.models.iris_models import \
    Encoder as IrisEncoder, Decoder as IrisDecoder, EncoderDecoderConfig

DISCRETE_ENCODER_TYPES = set(['vqvae', 'dae', 'softmax_ae', 'hard_fta_ae'])
CONTINUOUS_ENCODER_TYPES = set(['ae', 'vae', 'soft_vqvae', 'fta_ae'])

MODEL_VARS = [
    'embedding_dim', 'latent_dim', 'filter_size', 'codebook_size',
    'ae_model_type', 'ae_model_version', 'trans_model_type', 'trans_model_version',
    'trans_hidden', 'trans_depth', 'stochastic', 'extra_info',
    'repr_sparsity', 'sparsity_type', 'vq_trans_1d_conv']
AE_MODEL_VARS = [
    'embedding_dim', 'latent_dim', 'filter_size', 'codebook_size',
    'ae_model_type', 'ae_model_version', 'extra_info', 'repr_sparsity',
    'sparsity_type']


def make_ae_v1(input_dim, embedding_dim, filter_shape=(8, 8)):
    n_channels = input_dim[0]

    if input_dim[1] == 84:
        filters = (8, 5)
        strides = (3, 2)
        extra_padding = 1
    elif input_dim[1] == 56:
        filters = (6, 3)
        strides = (2, 2)
        extra_padding = 1
    else:
        filters = (8, 5)
        strides = (3, 2)
        extra_padding = 1

    encoder_p1 = nn.Sequential(
        nn.Conv2d(n_channels, 16, filters[0], strides[0], extra_padding),
        nn.ReLU(),
        nn.Conv2d(16, 32, filters[1], strides[1]),
        nn.ReLU(),
        nn.Conv2d(32, embedding_dim, 3, 1),
        nn.ReLU())
    test_input = torch.ones([1] + list(input_dim), dtype=torch.float32)
    out_shape = encoder_p1(test_input).shape[1:]
    mid_filter_shape = out_shape[1:]
    print('AE mid filter shape:', mid_filter_shape)

    encoder = nn.Sequential(
        encoder_p1,
        nn.AdaptiveAvgPool2d(filter_shape),
        ResidualBlock(embedding_dim, embedding_dim),
        ResidualBlock(embedding_dim, embedding_dim))
    decoder_p1 = nn.Sequential(
        nn.ReLU(),
        ResidualBlock(embedding_dim, embedding_dim),
        ResidualBlock(embedding_dim, embedding_dim),
        nn.AdaptiveAvgPool2d(mid_filter_shape),
        nn.ConvTranspose2d(embedding_dim, 32, 3, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, filters[1], strides[1]),
        nn.ReLU(),
        nn.ConvTranspose2d(16, n_channels, filters[0], strides[0], extra_padding))

    out_shape = decoder_p1(encoder(test_input)).shape[1:]

    extra_layers = []
    if list(out_shape) != list(input_dim):
        extra_layers.append(nn.AdaptiveAvgPool2d(input_dim[1:]))
    extra_layers.append(nn.ReLU())
    extra_layers.append(ResidualBlock(n_channels, n_channels))
    extra_layers.append(nn.Conv2d(n_channels, n_channels, 1, 1))
    decoder = nn.Sequential(decoder_p1, *extra_layers)

    return encoder, decoder


def make_dense_ae_v2(input_dim, latent_dim=None, hidden_sizes=[512, 512, 256, 256]):
    if latent_dim is None:
        latent_dim = hidden_sizes[-1]
        hidden_sizes = hidden_sizes[:-1]
    n_features = np.array(input_dim).squeeze()
    encoder = nn.Sequential(*create_dense_layers(
        n_features, out_features=latent_dim,
        hidden_sizes=hidden_sizes))
    decoder = nn.Sequential(*create_dense_layers(
        latent_dim, out_features=n_features,
        hidden_sizes=hidden_sizes[::-1]))
    return encoder, decoder


def make_ae_v2(input_dim, embedding_dim=None, filter_size=None):
    embedding_dim = embedding_dim or 128

    if len(input_dim) <= 1:
        return make_dense_ae_v2(input_dim)

    channels = (input_dim[0], 64, 128, embedding_dim)
    filters = (8, 6, 4)
    strides = (2, 2, 2)
    padding = (1, 0, 0)

    if input_dim[1] in (48, 56, 64):
        strides = (2, 2, 1)
    elif input_dim[1] == 54:
        strides = (2, 1, 2)

    encoder_layers = []
    decoder_layers = []

    for i in range(len(filters)):
        encoder_layers.append(nn.Conv2d(
            channels[i], channels[i + 1], filters[i], strides[i], padding[i]))
        encoder_layers.append(nn.ReLU())
    encoder_p1 = nn.Sequential(*encoder_layers)

    test_input = torch.ones([1] + list(input_dim), dtype=torch.float32)
    out_shape = encoder_p1(test_input).shape[1:]
    mid_filter_shape = out_shape[1:]
    print('AE mid filter shape:', mid_filter_shape)

    if filter_size:
        encoder_layers.append(nn.AdaptiveAvgPool2d(filter_size))
        encoder_layers.append(ResidualBlock(embedding_dim, embedding_dim))
        decoder_layers.append(ResidualBlock(embedding_dim, embedding_dim))
        decoder_layers.append(nn.AdaptiveAvgPool2d(mid_filter_shape))

    for i in reversed(range(len(filters))):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], filters[i], strides[i], padding[i]))
        decoder_layers.append(nn.ReLU())

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


def make_ae_v3(input_dim, embedding_dim=None, filter_size=None):
    res = input_dim[1]
    config = EncoderDecoderConfig(
        resolution=res,
        in_channels=input_dim[0],
        z_channels=embedding_dim,
        ch=64,
        ch_mult=[1, 1, 1, 1, 1],
        num_res_blocks=2,
        attn_resolutions=[res / (2 ** 2), res / (2 ** 3)],
        out_ch=input_dim[0],
        dropout=0.0
    )

    encoder = IrisEncoder(config)
    decoder = IrisDecoder(config)
    return encoder, decoder


def make_ae_v4(input_dim, embedding_dim=None, filter_size=None):
    res = input_dim[1]
    config = EncoderDecoderConfig(
        resolution=res,
        in_channels=input_dim[0],
        z_channels=embedding_dim,
        ch=64,
        ch_mult=[1, 1, 1, 1, 1],
        downsamples=[True, True, True, False, False],
        num_res_blocks=2,
        attn_resolutions=[res / (2 ** 2), res / (2 ** 3)],
        out_ch=input_dim[0],
        dropout=0.0
    )

    encoder = IrisEncoder(config)
    decoder = IrisDecoder(config)
    return encoder, decoder


def make_nature_ae(input_dim, embedding_dim=None, filter_size=None, vanilla=False):
    embedding_dim = embedding_dim or 64

    if len(input_dim) <= 1:
        raise ValueError('Input dim must be at least 2D for Nature AE')

    channels = (input_dim[0], 32, 64, embedding_dim)
    filters = (8, 4, 3)
    strides = (4, 2, 1)

    if not vanilla and input_dim[1] == 64:
        padding = (2, 0, 0)
    else:
        padding = (0, 0, 0)

    encoder_layers = []
    decoder_layers = []

    for i in range(len(filters)):
        encoder_layers.append(nn.Conv2d(
            channels[i], channels[i + 1], filters[i], strides[i], padding[i]))
        encoder_layers.append(nn.ReLU())
    encoder_p1 = nn.Sequential(*encoder_layers)

    test_input = torch.ones([1] + list(input_dim), dtype=torch.float32)
    out_shape = encoder_p1(test_input).shape[1:]
    mid_filter_shape = out_shape[1:]
    print('AE mid filter shape:', mid_filter_shape)

    if filter_size and not vanilla:
        encoder_layers.append(nn.AdaptiveAvgPool2d(filter_size))
        decoder_layers.append(nn.AdaptiveAvgPool2d(mid_filter_shape))

    for i in reversed(range(len(filters))):
        decoder_layers.append(nn.ConvTranspose2d(
            channels[i + 1], channels[i], filters[i], strides[i], padding[i]))
        if i > 0:
            decoder_layers.append(nn.ReLU())

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


def make_ae(input_dim, embedding_dim, filter_size, version='2'):
    version = str(version)
    if version == '1':
        return make_ae_v1(input_dim, embedding_dim, (filter_size, filter_size))
    elif version == '2':
        return make_ae_v2(input_dim, embedding_dim, filter_size)
    elif version == '3':
        return make_ae_v3(input_dim, embedding_dim, filter_size)
    elif version == '4':
        return make_ae_v4(input_dim, embedding_dim, filter_size)
    elif version == 'nature_ae':
        return make_nature_ae(input_dim, embedding_dim, filter_size, vanilla=False)
    elif version == 'nature':
        return make_nature_ae(input_dim, embedding_dim, filter_size, vanilla=True)
    raise ValueError(f'Invalid AE version, {version}')


def args_update(args, key, value):
    if args.wandb:
        args.update({key: value}, allow_val_change=True)
    else:
        setattr(args, key, value)
    log_param_updates(args, {key: value})


def safe_getattr(args, attr, default=None):
    try:
        return getattr(args, attr, default)
    except KeyError:
        return default


def construct_ae_model(input_dim, args, load=True, latent_activation=False):
    new_hash = make_model_hash(args, model_vars=AE_MODEL_VARS, exp_type='encoder')
    model = None
    args_update(args, 'ae_model_hash', new_hash)

    if args.ae_model_type not in ('identity', 'flatten'):
        encoder, decoder = make_ae(
            input_dim, args.embedding_dim, args.filter_size, version=args.ae_model_version)
        test_input = torch.ones(1, *input_dim, dtype=torch.float32)
        encoder_out_shape = encoder(test_input).shape[1:]
        encoder_type = 'dense' if (len(encoder_out_shape) == 1) else 'cnn'

    if args.ae_model_type in CONTINUOUS_ENCODER_TYPES:
        if args.ae_model_type in ('ae', 'vae', 'fta_ae'):
            stochastic = args.ae_model_type == 'vae'
            fta = args.ae_model_type == 'fta_ae'
            fta_params = {
                'tiles': args.fta_tiles,
                'bound_low': args.fta_bound_low,
                'bound_high': args.fta_bound_high,
                'eta': args.fta_eta
            }

            args_update(args, 'codebook_size', None)
            model = AEModel(input_dim, latent_dim=args.latent_dim, encoder=encoder,
                            decoder=decoder, stochastic=stochastic, fta=fta, fta_params=fta_params,
                            latent_activation=latent_activation)
            args_update(args, 'final_latent_dim', model.latent_dim)
            print(f'Constructed {args.ae_model_type.upper()} with {args.final_latent_dim}-dim latent space')

            TrainerClass = AETrainer if args.ae_model_type in ('ae', 'fta_ae') else VAETrainer
            trainer = TrainerClass(model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip)

        elif args.ae_model_type == 'soft_vqvae':
            n_latents = args.latent_dim if encoder_type == 'dense' else None
            model = VQVAEModel(
                input_dim, codebook_size=args.codebook_size, embedding_dim=args.embedding_dim,
                encoder=encoder, decoder=decoder, n_latents=n_latents, quantized_enc=True,
                sparsity=args.repr_sparsity, sparsity_type=args.sparsity_type)
            args_update(args, 'final_latent_dim', model.n_latent_embeds * args.codebook_size)
            print(
                f'Constructed Soft VQVAE with {model.n_latent_embeds} latents and {args.codebook_size} codebook entries')
            trainer = VQVAETrainer(model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip)

        if load:
            load_model(model, args, exp_type='encoder', model_vars=AE_MODEL_VARS, model_hash=args.ae_model_hash)

    elif args.ae_model_type in DISCRETE_ENCODER_TYPES:
        TrainerClass = VQVAETrainer if args.ae_model_type in ('vqvae', 'soft_vqvae') else AETrainer
        n_latents = args.latent_dim if encoder_type == 'dense' else None

        if args.ae_model_type == 'vqvae':
            print("🔧 Creating VQ-VAE model")

            # Test encoder to get spatial dimensions
            test_input = torch.ones(1, *input_dim, dtype=torch.float32)
            with torch.no_grad():
                encoded_output = encoder(test_input)

            # Calculate spatial dimensions
            if len(encoded_output.shape) == 4:  # [B, C, H, W]
                spatial_dims = encoded_output.shape[2:]
                n_latent_positions = spatial_dims[0] * spatial_dims[1]
            elif len(encoded_output.shape) == 3:  # [B, C, S]
                spatial_dims = (encoded_output.shape[2],)
                n_latent_positions = encoded_output.shape[2]
            else:
                raise ValueError(f"Unexpected encoder output shape: {encoded_output.shape}")

            print(f"Encoder output shape: {encoded_output.shape}")
            print(f"Spatial dimensions: {spatial_dims}")
            print(f"Latent positions: {n_latent_positions}")

            # Simple working VQ-VAE implementation
            class SimpleVQVAE(nn.Module):
                def __init__(self, encoder, decoder, codebook_size, embedding_dim, commitment_cost=0.25):
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder
                    self.codebook_size = codebook_size
                    self.embedding_dim = embedding_dim
                    self.commitment_cost = commitment_cost
                    self.spatial_dims = spatial_dims
                    self.n_latent_embeds = n_latent_positions
                    self.n_embeddings = codebook_size

                    # VQ Embedding table
                    self.embeddings = nn.Embedding(codebook_size, embedding_dim)
                    nn.init.uniform_(self.embeddings.weight, -1 / codebook_size, 1 / codebook_size)

                def forward(self, x):
                    # Encode
                    encoded = self.encoder(x)
                    original_shape = encoded.shape

                    # Reshape for quantization
                    if len(encoded.shape) == 4:  # [B, C, H, W]
                        B, C, H, W = encoded.shape
                        flat_encoded = encoded.permute(0, 2, 3, 1).reshape(-1, C)
                    else:  # [B, C, S]
                        B, C, S = encoded.shape
                        flat_encoded = encoded.permute(0, 2, 1).reshape(-1, C)

                    # Quantization
                    distances = (torch.sum(flat_encoded ** 2, dim=1, keepdim=True)
                                 + torch.sum(self.embeddings.weight ** 2, dim=1)
                                 - 2 * torch.matmul(flat_encoded, self.embeddings.weight.t()))

                    encoding_indices = torch.argmin(distances, dim=1)
                    quantized = self.embeddings(encoding_indices)

                    # Reshape back
                    if len(original_shape) == 4:
                        quantized = quantized.reshape(B, H, W, C).permute(0, 3, 1, 2)
                    else:
                        quantized = quantized.reshape(B, S, C).permute(0, 2, 1)

                    # Straight-through estimator
                    quantized = encoded + (quantized - encoded).detach()

                    # Decode
                    decoded = self.decoder(quantized)

                    # Losses
                    e_latent_loss = F.mse_loss(quantized.detach(), encoded)
                    q_latent_loss = F.mse_loss(quantized, encoded.detach())
                    loss = q_latent_loss + self.commitment_cost * e_latent_loss

                    # Perplexity approximation
                    avg_probs = torch.mean(F.softmax(-distances, dim=-1), dim=0)
                    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

                    return decoded, loss, perplexity

                def encode(self, x):
                    with torch.no_grad():
                        encoded = self.encoder(x)

                        if len(encoded.shape) == 4:
                            B, C, H, W = encoded.shape
                            flat_encoded = encoded.permute(0, 2, 3, 1).reshape(-1, C)
                            spatial_size = H * W
                        else:
                            B, C, S = encoded.shape
                            flat_encoded = encoded.permute(0, 2, 1).reshape(-1, C)
                            spatial_size = S

                        distances = (torch.sum(flat_encoded ** 2, dim=1, keepdim=True)
                                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                                     - 2 * torch.matmul(flat_encoded, self.embeddings.weight.t()))

                        encoding_indices = torch.argmin(distances, dim=1)
                        return encoding_indices.reshape(B, spatial_size)

                def decode(self, indices):
                    if len(indices.shape) == 2:  # [B, spatial_positions]
                        quantized_flat = self.embeddings(indices.flatten())
                        B = indices.shape[0]

                        if len(self.spatial_dims) == 2:  # 2D spatial
                            H, W = self.spatial_dims
                            quantized = quantized_flat.reshape(B, H, W, self.embedding_dim).permute(0, 3, 1, 2)
                        else:  # 1D spatial
                            S = self.spatial_dims[0]
                            quantized = quantized_flat.reshape(B, S, self.embedding_dim).permute(0, 2, 1)
                    else:
                        quantized = indices

                    return self.decoder(quantized)

            model = SimpleVQVAE(
                encoder=encoder,
                decoder=decoder,
                codebook_size=args.codebook_size,
                embedding_dim=args.embedding_dim,
                commitment_cost=0.25
            )

            total_latent_dim = n_latent_positions * args.codebook_size
            args_update(args, 'final_latent_dim', total_latent_dim)
            print(f'Constructed VQ-VAE with {n_latent_positions} latents and {args.codebook_size} codebook entries')
            print(f'Total latent dimension: {total_latent_dim}')

        elif args.ae_model_type == 'dae':
            model = DAEModel(input_dim, encoder=encoder, decoder=decoder)
            args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
            print(
                f'Constructed DAE with {np.prod(model.encoder_out_shape[1:])} latents and {model.n_channels} codebook entries')

        elif args.ae_model_type == 'softmax_ae':
            model = SoftmaxAEModel(
                input_dim, codebook_size=args.codebook_size,
                encoder=encoder, decoder=decoder, n_latents=n_latents)
            args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
            print(f'Constructed softmax AE with {model.encoder_out_shape[1:]} latents')

        elif args.ae_model_type == 'hard_fta_ae':
            model = HardFTAAEModel(
                input_dim, codebook_size=args.codebook_size,
                encoder=encoder, decoder=decoder, n_latents=n_latents)
            args_update(args, 'final_latent_dim', np.prod(model.encoder_out_shape))
            print(f'Constructed hard FTA AE with {model.encoder_out_shape[1:]} latents')

        if load and args.ae_model_type == 'vqvae':
            try:
                load_model(model, args, exp_type='encoder', model_vars=AE_MODEL_VARS, model_hash=args.ae_model_hash)
            except:
                print("Could not load VQ-VAE model, using fresh initialization")
        elif load and args.ae_model_type != 'vqvae':
            load_model(model, args, exp_type='encoder', model_vars=AE_MODEL_VARS, model_hash=args.ae_model_hash)

        trainer = TrainerClass(model, lr=args.learning_rate, log_freq=-1, grad_clip=args.ae_grad_clip)

    elif args.ae_model_type == 'flatten':
        model = FlattenModel(input_dim)
        args_update(args, 'final_latent_dim', model.latent_dim)
        print(f'Constructed Flatten model with {model.latent_dim} latents')
        trainer = None

    elif args.ae_model_type == 'identity':
        model = IdentityModel(input_dim, embedding_dim=args.embedding_dim)
        args_update(args, 'final_latent_dim', model.latent_dim)
        print(f'Constructed Identity model with {model.latent_dim} latents')
        trainer = None

    return model, trainer


def construct_trans_model(encoder, args, act_space, load=True):
    new_hash = make_model_hash(args, model_vars=MODEL_VARS, exp_type='trans_model')
    trans_model = None
    args_update(args, 'trans_model_hash', new_hash)

    if args.e2e_loss and args.trans_model_type != 'continuous':
        raise ValueError('End-to-end loss only supported for continuous models!')

    if args.trans_model_type == 'discrete':
        if args.trans_model_version == '1':
            use_soft_embeds = safe_getattr(args, 'use_soft_embeds', False) \
                              or safe_getattr(encoder, 'quantized_enc', False)
            trans_model = DiscreteTransitionModel(
                encoder.n_latent_embeds, encoder.n_embeddings, encoder.embedding_dim,
                act_space, hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic, stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256], use_soft_embeds=use_soft_embeds,
                return_logits=safe_getattr(encoder, 'quantized_enc', False))
        args_update(args, 'final_latent_dim', encoder.n_latent_embeds * encoder.n_embeddings)
        if load:
            load_model(trans_model, args, exp_type='trans_model', model_vars=MODEL_VARS,
                       model_hash=args.trans_model_hash)
        trans_trainer = DiscreteTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip)

    elif args.trans_model_type == 'continuous':
        if args.trans_model_version == '1':
            if hasattr(encoder, 'latent_dim'):
                latent_dim = encoder.latent_dim
            elif hasattr(encoder, 'n_latent_embeds') and hasattr(encoder, 'embedding_dim'):
                latent_dim = encoder.n_latent_embeds * encoder.embedding_dim
            else:
                latent_dim = 512

            trans_model = ContinuousTransitionModel(
                latent_dim, act_space,
                hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic,
                stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256]
            )

        args_update(args, 'final_latent_dim', latent_dim)
        if load:
            load_model(trans_model, args, exp_type='trans_model', model_vars=MODEL_VARS,
                       model_hash=args.trans_model_hash)
        trans_trainer = ContinuousTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip, e2e_loss=args.e2e_loss)

    elif args.trans_model_type == 'shared_vq':
        def logits_to_state(logits):
            logits = logits.view(logits.shape[0], encoder.n_embeddings, encoder.n_latent_embeds)
            with torch.no_grad():
                mask = encoder.sparsity_mask if encoder.sparsity_enabled else None
                quantized = encoder.quantizer(logits, mask)[1]
            states = quantized.view(logits.shape[0], -1)
            return states

        if args.trans_model_version == '1':
            trans_model = ContinuousTransitionModel(
                encoder.latent_dim, act_space,
                hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic,
                stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256],
                logits_to_state_func=logits_to_state
            )
        args_update(args, 'final_latent_dim', encoder.latent_dim)
        if load:
            load_model(trans_model, args, exp_type='trans_model', model_vars=MODEL_VARS,
                       model_hash=args.trans_model_hash)
        trans_trainer = ContinuousTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, grad_clip=args.ae_grad_clip)

    elif args.trans_model_type == 'universal_vq':
        if args.trans_model_version == '1':
            use_soft_embeds = safe_getattr(args, 'use_soft_embeds', False) \
                              or safe_getattr(encoder, 'quantized_enc', False)
            embed_snap_enc = encoder if args.vq_trans_state_snap else None

            if args.extra_info and 'scale_embeds' in args.extra_info:
                codebook = encoder.get_codebook()
                zeros = torch.zeros_like(codebook)
                zeros_count = (codebook.isclose(zeros)).sum(dim=1)

                if (zeros_count < codebook.shape[1] - 1).any():
                    raise ValueError('Codebook must have one or less non-zero per row!')
                elif (zeros_count == codebook.shape[1]).any():
                    n_zero_rows = (zeros_count == codebook.shape[1]).sum()
                    warnings.warn(f'Codebook has {n_zero_rows} rows with all zeros!')

                scale_factor = 1.0 / codebook.sum(dim=1)
                scale_factor = scale_factor.unsqueeze(0)
            else:
                scale_factor = None

            embed_grad_hook = args.extra_info and 'embed_grad_hook' in args.extra_info
            rand_mask = args.extra_info and 'rand_mask' in args.extra_info

            trans_model = UniversalVQTransitionModel(
                encoder.n_latent_embeds, encoder.n_embeddings, encoder.embedding_dim,
                act_space, hidden_sizes=[args.trans_hidden] * args.trans_depth,
                stochastic=args.stochastic, stoch_hidden_sizes=[256, 256],
                discretizer_hidden_sizes=[256], use_soft_embeds=use_soft_embeds,
                use_1d_conv=args.vq_trans_1d_conv, embed_snap_encoder=embed_snap_enc,
                embed_scale_factor=scale_factor, embed_grad_hook=embed_grad_hook,
                rand_mask=rand_mask)
        args_update(args, 'final_latent_dim', encoder.n_latent_embeds * encoder.n_embeddings)
        if load:
            load_model(trans_model, args, exp_type='trans_model', model_vars=MODEL_VARS,
                       model_hash=args.trans_model_hash)
        trans_trainer = UniversalVQTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            log_norms=args.log_norms, loss_type=args.vq_trans_loss_type, grad_clip=args.ae_grad_clip)

    elif args.trans_model_type == 'transformer':
        if args.trans_model_version == '1':
            trans_model = TransformerTransitionModel(
                encoder.codebook_size, encoder.embedding_dim, act_space,
                num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                dim_feedforward=1024, dropout=0.2, stochastic=args.stochastic)
        args_update(args, 'final_latent_dim', encoder.n_latent_embeds * encoder.codebook_size)
        if load:
            load_model(trans_model, args, exp_type='trans_model', model_vars=MODEL_VARS,
                       model_hash=args.trans_model_hash)
        trans_trainer = TransformerTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            grad_clip=args.ae_grad_clip)

    elif args.trans_model_type == 'transformerdec':
        if args.trans_model_version == '1':
            trans_model = TransformerDecTransitionModel(
                encoder.codebook_size, encoder.embedding_dim, act_space,
                num_heads=4, num_decoder_layers=6, dim_feedforward=256,
                dropout=0.1, stochastic=args.stochastic)
        args_update(args, 'final_latent_dim', encoder.n_latent_embeds * encoder.codebook_size)
        if load:
            load_model(trans_model, args, exp_type='trans_model', model_vars=MODEL_VARS,
                       model_hash=args.trans_model_hash)
        trans_trainer = TransformerTransitionTrainer(
            trans_model, encoder=encoder, lr=args.trans_learning_rate, log_freq=-1,
            grad_clip=args.ae_grad_clip)

    else:
        raise ValueError(f'No trans_model_type, "{args.trans_model_type}"!')

    return trans_model, trans_trainer


def make_model_hash(args=None, model_vars=MODEL_VARS, **kwargs):
    """MD5 hash of a dictionary."""
    args_dict = vars(args)
    args_dict = args_dict.get('_items', args_dict)
    if args is not None:
        for model_param in model_vars:
            if model_param in args_dict:
                kwargs[model_param] = args_dict[model_param]
    dhash = hashlib.md5()
    kwargs = {k: int(v) if isinstance(v, (np.int32, np.int64)) \
        else v for k, v in kwargs.items()}
    encoded = json.dumps(dict(kwargs), sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


MODEL_SAVE_FORMAT = './models/{}/model_{}.pt'


def save_model(model, args, model_hash=None, model_vars=MODEL_VARS, **kwargs):
    if model_hash is None:
        model_hash = make_model_hash(args, model_vars=model_vars, **kwargs)
    save_path = MODEL_SAVE_FORMAT.format(args.env_name, model_hash)
    save_path = save_path.replace(':', '-')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to "{save_path}"')
    return model_hash


def load_model(model, args, model_hash=None, return_hash=False, model_vars=MODEL_VARS, **kwargs):
    if model_hash is None:
        model_hash = make_model_hash(args, model_vars=model_vars, **kwargs)
    file_path = MODEL_SAVE_FORMAT.format(args.env_name, model_hash)
    file_path = file_path.replace(':', '-')
    if not os.path.exists(file_path):
        print(f'No model found at "{file_path}", not loading')
        model = None
    else:
        print(f'Model found at "{file_path}", loading')
        try:
            state_dict = torch.load(file_path, map_location=args.device)
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully!")
        except RuntimeError as e:
            print(f'Failed to load model at {file_path}!')
            print(f'Error details: {e}')
            raise e

    if return_hash:
        return model, model_hash
    return model