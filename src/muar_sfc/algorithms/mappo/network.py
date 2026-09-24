import torch
import torch.nn as nn
from torch.distributions import Categorical


def init_weights(m):
    """Inicialização ortogonal padrão para redes MAPPO."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=math_gain(m))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def math_gain(m):
    return 1.414  # sqrt(2) para ReLU / Tanh


class RegionalActorNetwork(nn.Module):
    """
    Rede de Ator Descentralizado para cada Controlador de Região de Borda.
    Executa inferência local ultrarrápida (< 5 ms) com suporte a Action Masking.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.action_head.weight.data, gain=0.01)
        if self.action_head.bias is not None:
            nn.init.constant_(self.action_head.bias.data, 0.0)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        """
        Retorna os logits e a distribuição de probabilidade sobre as ações locais válidas.
        """
        features = self.net(obs)
        logits = self.action_head(features)

        if action_mask is not None:
            # Substitui logits de ações inválidas por valor muito negativo
            mask_bool = action_mask.bool()
            # Garante que pelo menos uma ação seja válida para evitar NaNs
            all_invalid = (~mask_bool).all(dim=-1, keepdim=True)
            if all_invalid.any():
                mask_bool = mask_bool | all_invalid
            neg_inf = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            logits = torch.where(mask_bool, logits, neg_inf)

        dist = Categorical(logits=logits)
        return dist

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor | None = None
    ):
        """Avalia ações durante o passo de atualização do PPO."""
        dist = self.forward(obs, action_mask)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()
        return action_log_probs, dist_entropy


class CentralizedCriticNetwork(nn.Module):
    """
    Crítico Centralizado (CTDE) que observa o estado global da rede inteira
    (recursos de todas as regiões, balanceamento de carga global, etc.).
    Usado apenas durante a fase de treino para estimar V(s_t).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Inicialização ortogonal da última camada
        last_linear = self.net[-1]
        nn.init.orthogonal_(last_linear.weight.data, gain=1.0)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias.data, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estima o valor escalar de estado V(s)."""
        return self.net(state).squeeze(-1)
