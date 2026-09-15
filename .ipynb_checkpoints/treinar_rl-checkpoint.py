import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
# NOVO: Importação para reconfigurar o logger
from stable_baselines3.common.logger import configure

from salvar_var import carregar_lista
from muar_sfc.algorithms.environment import SFC_AllocationEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks

# ==============================================================================
#      FUNÇÃO PARA CARREGAR O AMBIENTE (Seu código original, sem alterações)
# ==============================================================================
def carregar_dados_do_ambiente():
    """
    Carrega os dados e inicializa o ambiente SFC_AllocationEnv.
    """
    try:
        list_graph = carregar_lista("list_graph")
        list_sfc = carregar_lista("list_sfc")
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se que os arquivos de dados existem.")
        return None

    # list_sfc = []
    # session_id = '1'
    # list_aux = []
    # if list_aux_sfc:
    #     for idx, sfc in enumerate(list_aux_sfc):
    #         session_sfc = sfc.id.split("_")[-1]
    #         if session_sfc != session_id:
    #             list_sfc.append(list_aux)
    #             list_aux = []
    #             session_id = sfc.id.split("_")[-1]
    #         list_aux.append(sfc)
    #     list_sfc.append(list_aux)

    valid_nodes = []
    if list_graph and list_graph[0]:
        for node in list_graph[0].nodes():
            if list_graph[0].nodes[node]["type"] == "server":
                valid_nodes.append(node)
    valid_nodes.append("M")
    
    # pesos = {"cpu": 5, "cache": 5, "lat":  2, "band": 3}
    # while True:
    #     print(len(list_graph))
    env = SFC_AllocationEnv(list_graph=list_graph, list_sfc=list_sfc, valid_nodes=valid_nodes)
    env.reset()
    return env


# ==============================================================================
#               FLUXO PRINCIPAL DE TREINAMENTO (COM ALTERAÇÕES)
# ==============================================================================

if __name__ == '__main__':
    # --- 1. DEFINIÇÃO DOS DIRETÓRIOS ---
    log_dir = "logs/"
    tensorboard_log_dir = "tensorboard_logs/"
    save_dir = "rl_saved_models/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. CRIAÇÃO DOS AMBIENTES ---
    num_cpu = 10
    print(f"Iniciando com {num_cpu} processos paralelos.")
    train_env = make_vec_env(carregar_dados_do_ambiente, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    eval_env = carregar_dados_do_ambiente()
    eval_env = Monitor(eval_env)
    
    # --- 3. CARREGAR MODELO EXISTENTE OU CRIAR UM NOVO ---
    model_name = "ppo_allocation_model.zip"
    final_model_path = os.path.join(save_dir, model_name)

    if os.path.exists(final_model_path):
        # Se o modelo existir, carrega e prepara para continuar o treinamento
        print(f"Modelo salvo encontrado em '{final_model_path}'. Carregando para continuar o treinamento...")
        model = MaskablePPO.load(final_model_path, env=train_env)
        # Reconfigura o logger para que o TensorBoard continue registrando
        new_logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])
        model.set_logger(new_logger)
    else:
        # Se o modelo não existir, cria um novo
        print("Nenhum modelo salvo encontrado. Iniciando novo treinamento...")
        model = MaskablePPO(
            "MultiInputPolicy",
            train_env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir
        )

    # --- 4. TREINAMENTO (NOVO OU CONTINUADO) ---
    # Este bloco agora é executado em ambos os casos
    
    # Configuração do Callback de Avaliação
    eval_callback = MaskableEvalCallback(
        eval_env,
        log_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )
    
    # Define quantos passos de treinamento adicionais serão executados
    additional_timesteps = 150_000
    
    print(f"--- Iniciando/Continuando o treinamento por mais {additional_timesteps} passos ---")
    # model.learn(
    #     total_timesteps=additional_timesteps,
    #     callback=eval_callback,
    #     tb_log_name="MaskablePPO_SFC_Allocation_Parallel",
    #     reset_num_timesteps=False  # ESSENCIAL: Não reseta o contador de passos
    # )
    print("--- Treinamento finalizado ---")

    # --- 5. SALVAR O MODELO ATUALIZADO ---
    model.save(os.path.join(save_dir, "ppo_allocation_model"))
    print(f"\nModelo final salvo em: {final_model_path}")

    # --- 6. TESTE COM O MODELO FINAL ---
    print("\n--- Iniciando teste com o modelo em 200 episódios ---")
    
    num_episodes = 1000
    all_rewards = []
    successful_runs = 0
    cont = 0

    total_latency_on_success = 0.0

    for i in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action_masks = eval_env.env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            total_reward += reward
            done = terminated or truncated

        all_rewards.append(total_reward)
        # CORREÇÃO: Usa get_attr para acessar o atributo do ambiente original de forma segura
        if eval_env.env.success:
            successful_runs += 1
            # MODIFICAÇÃO: Soma a latência usada se o episódio foi um sucesso
            total_latency_on_success += eval_env.env.latency_used

        if (i + 1) % 100 == 0:
            success_status = eval_env.env.success
            print(f"Episódio {i + 1}/{num_episodes} concluído. Recompensa: {total_reward:.2f}, Sucesso: {success_status}")

    # --- 7. CÁLCULO E EXIBIÇÃO DAS MÉTRICAS DE DESEMPENHO ---
    print(f"\n--- Métricas de Desempenho ({num_episodes} execuções) ---")

    success_rate = (successful_runs / num_episodes) * 100 if num_episodes > 0 else 0
    mean_reward = np.mean(all_rewards) if all_rewards else 0
    std_reward = np.std(all_rewards) if all_rewards else 0
    variance = np.var(all_rewards) if all_rewards else 0

    # MODIFICAÇÃO: Calcula a latência média (evitando divisão por zero)
    average_latency = total_latency_on_success / successful_runs if successful_runs > 0 else 0

    print(f"Taxa de Sucesso: {success_rate:.2f}% ({successful_runs}/{num_episodes})")
    print(f"Recompensa Média: {mean_reward:.2f}")
    print(f"Desvio Padrão da Recompensa: {std_reward:.2f} (Variância: {variance:.2f})")
    # MODIFICAÇÃO: Exibe a latência média
    print(f"Latência Média (apenas em sucessos): {average_latency:.2f}")
