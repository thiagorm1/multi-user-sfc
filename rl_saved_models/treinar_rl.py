import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from muar_sfc.algorithms.environments.env_da_rsppo import SFC_AllocationEnv_DARSPPO
from muar_sfc.algorithms.environments.environment import SFC_AllocationEnv

# ==============================================================================
#      REGISTRO DE AMBIENTES
# ==============================================================================
# Importe todos os seus ambientes aqui
from muar_sfc.algorithms.environments.hephaestus_env import SFC_AllocationEnv_hephaestus

# Importe outros como Kuririn ou Replic quando necessário e adicione ao dicionário abaixo
from muar_sfc.utils.salvar_var import carregar_lista

# Dicionário que mapeia uma string de argumento para a classe do ambiente correspondente
ENV_REGISTRY = {
    "hephaestus": SFC_AllocationEnv_hephaestus,
    "darsppo": SFC_AllocationEnv_DARSPPO,
    "default": SFC_AllocationEnv,
    # "kuririn": SFC_AllocationEnv_Kuririn,
    # "replic": SFC_AllocationEnv_Replic,
}

# ==============================================================================
#      FUNÇÃO PARA CARREGAR O AMBIENTE
# ==============================================================================
def carregar_dados_do_ambiente(env_class):
    """
    Carrega os dados e inicializa o ambiente dinamicamente baseado na classe fornecida.
    """
    try:
        list_graph = []
        list_sfc = []
        for i in range(1, 5):
            list_graph = list_graph + carregar_lista(f"list_graph{i}")
            list_sfc = list_sfc + carregar_lista(f"list_sfc{i}")

    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se que os arquivos de dados existem.")
        return None

    valid_nodes = []
    if list_graph and list_graph[0]:
        for node in list_graph[0].nodes():
            if list_graph[0].nodes[node]["type"] == "server":
                valid_nodes.append(node)
    valid_nodes.append("M")

    # Instancia a classe que foi passada como argumento
    env = env_class(list_graph=list_graph, list_sfc=list_sfc, valid_nodes=valid_nodes)
    return env


# ==============================================================================
#               FLUXO PRINCIPAL DE TREINAMENTO
# ==============================================================================

def main():
    # --- PARSER DE ARGUMENTOS DE LINHA DE COMANDO ---
    parser = argparse.ArgumentParser(description="Treinamento unificado de RL para Alocação SFC")
    parser.add_argument(
        "--env",
        type=str,
        choices=list(ENV_REGISTRY.keys()),
        required=True,
        help="Escolha o ambiente para treinar"
    )
    parser.add_argument(
        "--no-masking",
        action="store_true",
        help="Desativa o MaskablePPO e usa o PPO padrão"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=150_000,
        help="Número adicional de passos de treinamento"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Número de episódios para o teste final"
    )
    args = parser.parse_args()

    env_name = args.env
    use_masking = not args.no_masking
    env_class = ENV_REGISTRY[env_name]

    # --- 1. DEFINIÇÃO DOS DIRETÓRIOS ---
    log_dir = Path("logs/")
    tensorboard_log_dir = Path("tensorboard_logs/")
    save_dir = Path("rl_saved_models/")

    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. CRIAÇÃO DOS AMBIENTES ---
    print(f"Iniciando com o ambiente '{env_name}' (sem paralelismo).")

    train_env = carregar_dados_do_ambiente(env_class)
    train_env = Monitor(train_env, str(log_dir / f"train_{env_name}"))

    eval_env = carregar_dados_do_ambiente(env_class)
    eval_env = Monitor(eval_env, str(log_dir / f"eval_{env_name}"))

    # --- 3. CARREGAR MODELO EXISTENTE OU CRIAR UM NOVO ---
    prefix = "MASKABLEPPO" if use_masking else "PPO"
    model_name = f"{prefix}_{env_name}_allocation_model.zip"
    model_log_name = f"{prefix}_{env_name.capitalize()}_Allocation"

    ModelClass = MaskablePPO if use_masking else PPO
    print(f"Configurado para usar {ModelClass.__name__}.")

    final_model_path = save_dir / model_name

    if final_model_path.exists():
        print(f"Modelo encontrado em '{final_model_path}'.\nCarregando para continuar o train")
        model = ModelClass.load(final_model_path, env=train_env)
        new_logger = configure(str(tensorboard_log_dir), ["stdout", "tensorboard"])
        model.set_logger(new_logger)
    else:
        print(f"Nenhum modelo salvo encontrado.Novo train para {model_name}")
        model = ModelClass("MultiInputPolicy",
                           train_env, verbose=1,
                           tensorboard_log=str(tensorboard_log_dir))

    # --- 4. TREINAMENTO (NOVO OU CONTINUADO) ---
    CallbackClass = MaskableEvalCallback if use_masking else EvalCallback
    print(f"Usando {CallbackClass.__name__}.")

    eval_callback = CallbackClass(
        eval_env,
        log_path=str(log_dir),
        eval_freq=1000,
        n_eval_episodes=30,
        deterministic=False,
        render=False,
    )

    print(f"--- Iniciando/Continuando o treinamento por mais {args.timesteps} passos ---")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=model_log_name,
        reset_num_timesteps=False,
    )
    print("--- Treinamento finalizado ---")

    # --- 5. SALVAR O MODELO ATUALIZADO ---
    model.save(final_model_path)
    print(f"\nModelo final salvo em: {final_model_path}")

    # --- 6. TESTE COM O MODELO FINAL ---
    print(f"\n--- Iniciando teste com o modelo {model_name} em {args.episodes} episódios ---")

    all_rewards = []
    successful_runs = 0
    total_latency_on_success = 0.0

    for i in range(args.episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            if use_masking:
                action_masks = eval_env.env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, _info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        all_rewards.append(total_reward)
        if eval_env.env.success:
            successful_runs += 1
            total_latency_on_success += eval_env.env.latency_used

        if (i + 1) % 100 == 0:
            success_status = eval_env.env.success
            print(
                f"Episódio {i + 1}/{args.episodes} concluído. "
                f"Recompensa: {total_reward:.2f}, Sucesso: {success_status}"
            )

    # --- 7. CÁLCULO E EXIBIÇÃO DAS MÉTRICAS DE DESEMPENHO ---
    print(f"\n--- Métricas de Desempenho ({args.episodes} execuções) ---")
    success_rate = (successful_runs / args.episodes) * 100 if args.episodes > 0 else 0
    mean_reward = np.mean(all_rewards) if all_rewards else 0
    std_reward = np.std(all_rewards) if all_rewards else 0
    variance = np.var(all_rewards) if all_rewards else 0
    average_latency = total_latency_on_success / successful_runs if successful_runs > 0 else 0

    print(f"Taxa de Sucesso: {success_rate:.2f}% ({successful_runs}/{args.episodes})")
    print(f"Recompensa Média: {mean_reward:.2f}")
    print(f"Desvio Padrão da Recompensa: {std_reward:.2f} (Variância: {variance:.2f})")
    print(f"Latência Média (apenas em sucessos): {average_latency:.2f}")


if __name__ == "__main__":
    main()
