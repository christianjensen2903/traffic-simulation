from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from gymnasium import spaces
from typing import Callable, Tuple
from sumo_env import TrafficColor, LaneType
from stable_baselines3.common.policies import ActorCriticPolicy


class Convolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.silu = nn.SiLU()
        self.batch = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch(out)
        out = self.silu(out)
        return out


class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.silu = nn.SiLU()
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.batch2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = out + x
        out = self.silu(out)
        return out


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        input_channels: int,
        hidden_channels: int,
        hidden_kernel_size: int,
        hidden_stride: int,
        blocks: int = 4,
        device="cpu",
    ):
        super(FeatureExtractor, self).__init__(observation_space, features_dim=1)

        self.lane_embedding = nn.Linear(len(LaneType), 15)

        self.color_linear = nn.Linear(len(TrafficColor), 15)

        self.device = device

        layers = []
        layers.append(
            Convolutional_Block(
                input_channels,
                hidden_channels,
                hidden_kernel_size,
                stride=hidden_stride,
            )
        )
        for _ in range(blocks):
            layers.append(
                Residual_Block(
                    hidden_channels,
                    hidden_channels,
                    hidden_kernel_size,
                    hidden_stride,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Embed the lanes
        lanes = observations["legs"]
        embedded_lanes = self.lane_embedding(lanes).unsqueeze(3)

        # Embed the colors
        colors = observations["signals"]
        embedded_colors = self.color_linear(colors).unsqueeze(3)

        # Concatenate the embeddings with the vehicles (batch, channels, 15, 4)
        vehicles = observations["vehicles"]

        # Concatenate the embeddings with the vehicles
        concatenated = torch.cat([embedded_lanes, embedded_colors, vehicles], dim=-1)

        # Permutes the tensor to (batch, 6, 15, channels)
        concatenated = concatenated.permute(0, 3, 1, 2)

        x = self.layers(concatenated)

        return x


class Head(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm2d(output_channels),
            activation,
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CustomNetwork(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        hidden_kernel_size: int,
        hidden_stride: int,
        feature_dim: int = 2,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = Head(
            input_channels=hidden_channels,
            output_channels=feature_dim,
            kernel_size=hidden_kernel_size,
            stride=hidden_stride,
        )
        # Value network
        self.value_net = Head(
            input_channels=hidden_channels,
            output_channels=feature_dim,
            kernel_size=hidden_kernel_size,
            stride=hidden_stride,
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of torche specified network.
            If all layers are shared, torchen ``latent_policy == latent_value``
        """
        x = features

        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        policy = self.policy_net(features)

        return policy

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            hidden_channels=self.net_arch["hidden_channels"],
            hidden_kernel_size=self.net_arch["hidden_kernel_size"],
            hidden_stride=self.net_arch["hidden_stride"],
            feature_dim=self.net_arch["feature_dim"],
            last_layer_dim_pi=self.net_arch["last_layer_dim_pi"],
            last_layer_dim_vf=self.net_arch["last_layer_dim_vf"],
        )
