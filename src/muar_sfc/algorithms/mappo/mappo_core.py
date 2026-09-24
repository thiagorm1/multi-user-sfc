from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from muar_sfc.algorithms.mappo.network import CentralizedCriticNetwork, RegionalActorNetwork


class MAPPOBuffer:
    """Buffer de rollouts para o treinamento de múltiplos agentes com crítico centralizado."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.obs: list[dict[int, np.ndarray]] = []
        self.states: list[np.ndarray] = []
        self.actions: list[dict[int, int]] = []
        self.action_log_probs: list[dict[int, float]] = []
        self.action_masks: list[dict[int, np.ndarray]] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def insert(
        self,
        obs: dict[int, np.ndarray],
        state: np.ndarray,
        actions: dict[int, int],
        action_log_probs: dict[int, float],
        action_masks: dict[int, np.ndarray],
        reward: float,
        value: float,
        done: bool,
    ):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(actions)
        self.action_log_probs.append(action_log_probs)
        self.action_masks.append(action_masks)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.obs.clear()
        self.states.clear()
        self.actions.clear()
        self.action_log_probs.clear()
        self.action_masks.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)


class MAPPOTrainer:
    """
    Controlador central de treinamento e inferência do MAPPO (CTDE).
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 5e-4,
        clip_param: float = 0.2,
        ppo_epoch: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = torch.device(device)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Atores descentralizados para cada agente regional
        self.actors = nn.ModuleList([
            RegionalActorNetwork(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])

        # Crítico Centralizado global
        self.critic = CentralizedCriticNetwork(state_dim).to(self.device)

        # Otimizadores
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor, eps=1e-5)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    def select_actions(
        self,
        obs: dict[int, np.ndarray],
        action_masks: dict[int, np.ndarray],
        deterministic: bool = False,
    ) -> tuple[dict[int, int], dict[int, float], float]:
        """
        Execução Descentralizada: cada ator avalia sua própria observação.
        O crítico é consultado apenas para obter o valor estimado durante rollouts.
        """
        actions = {}
        action_log_probs = {}

        with torch.no_grad():
            for i in range(self.n_agents):
                o_t = torch.as_tensor(
                    obs[i], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                m_t = torch.as_tensor(
                    action_masks[i], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                dist = self.actors[i](o_t, action_mask=m_t)

                act = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()

                actions[i] = int(act.item())
                action_log_probs[i] = float(dist.log_prob(act).item())

        return actions, action_log_probs

    def get_value(self, state: np.ndarray) -> float:
        """Estimação pelo crítico centralizado."""
        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            val = self.critic(s_t)
            return float(val.item())

    def train(self, buffer: MAPPOBuffer) -> dict[str, float]:
        """
        Atualização dos gradientes do MAPPO utilizando GAE e Clipped Objective.
        """
        if len(buffer) == 0:
            return {}

        rewards = np.array(buffer.rewards, dtype=np.float32)
        values = np.array(buffer.values, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.float32)

        # Cálculo do GAE (Generalized Advantage Estimation)
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0.0
        n_steps = len(buffer)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        # Normalização de vantagens
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Converte buffers para tensores
        t_states = torch.as_tensor(
            np.array(buffer.states), dtype=torch.float32, device=self.device
        )
        t_returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        t_advs = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.ppo_epoch):
            # 1. Atualização do Crítico Centralizado
            val_preds = self.critic(t_states)
            critic_loss = nn.functional.mse_loss(val_preds, t_returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            total_critic_loss += critic_loss.item()

            # 2. Atualização dos Atores Descentralizados
            for i in range(self.n_agents):
                agent_obs = torch.as_tensor(
                    np.array([b[i] for b in buffer.obs]), dtype=torch.float32, device=self.device
                )
                agent_masks = torch.as_tensor(
                    np.array([b[i] for b in buffer.action_masks]),
                    dtype=torch.float32,
                    device=self.device,
                )
                agent_acts = torch.as_tensor(
                    np.array([b[i] for b in buffer.actions]),
                    dtype=torch.int64,
                    device=self.device,
                )
                old_log_probs = torch.as_tensor(
                    np.array([b[i] for b in buffer.action_log_probs]),
                    dtype=torch.float32,
                    device=self.device,
                )

                new_log_probs, entropy = self.actors[i].evaluate_actions(
                    agent_obs, agent_acts, agent_masks
                )
                ratios = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratios * t_advs
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * t_advs
                actor_loss = -torch.min(surr1, surr2).mean() - (self.entropy_coef * entropy)

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                self.actor_optimizers[i].step()
                total_actor_loss += actor_loss.item()

        return {
            "actor_loss": total_actor_loss / (self.ppo_epoch * self.n_agents),
            "critic_loss": total_critic_loss / self.ppo_epoch,
        }

    def save(self, filepath: str | Path):
        """Salva os pesos do modelo treinado."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str | Path):
        """Carrega os pesos salvos."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Pesos do MAPPO não encontrados em: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint["actors"][i])
        self.critic.load_state_dict(checkpoint["critic"])
