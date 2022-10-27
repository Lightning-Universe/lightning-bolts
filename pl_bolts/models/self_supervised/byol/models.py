from typing import Tuple, Union

from torch import Tensor, nn

from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class MLP(nn.Module):
    """MLP architecture used as projectors in online and target networks and predictors in the online network.

    Args:
        input_dim (int, optional): Input dimension. Defaults to 2048.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 4096.
        output_dim (int, optional): Output dimension. Defaults to 256.

    Note:
        Default values for input, hidden, and output dimensions are based on values used in BYOL.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256) -> None:

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SiameseArm(nn.Module):
    """SiameseArm consolidates the encoder and projector networks of BYOL's symmetric architecture into a single
    class.

    Args:
        encoder (Union[str, nn.Module], optional): Online and target network encoder architecture.
            Defaults to "resnet50".
        encoder_out_dim (int, optional): Output dimension of encoder. Defaults to 2048.
        projector_hidden_dim (int, optional): Online and target network projector network hidden dimension.
            Defaults to 4096.
        projector_out_dim (int, optional): Online and target network projector network output dimension.
            Defaults to 256.
    """

    def __init__(
        self,
        encoder: Union[str, nn.Module] = "resnet50",
        encoder_out_dim: int = 2048,
        projector_hidden_dim: int = 4096,
        projector_out_dim: int = 256,
    ) -> None:

        super().__init__()

        if isinstance(encoder, str):
            self.encoder = torchvision_ssl_encoder(encoder)
        else:
            self.encoder = encoder

        self.projector = MLP(encoder_out_dim, projector_hidden_dim, projector_out_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.encoder(x)[0]
        z = self.projector(y)
        return y, z

    def encode(self, x: Tensor) -> Tensor:
        """Returns the encoded representation of a view. This method does not calculate the projection as in the
        forward method.

        Args:
            x (Tensor): sample to be encoded
        """
        return self.encoder(x)[0]
