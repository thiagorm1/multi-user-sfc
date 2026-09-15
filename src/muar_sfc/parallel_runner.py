import argparse
import subprocess
import time
from datetime import datetime as dt
from multiprocessing import Pool


def run_process(cmd_list: list[str]) -> None:
    """Executa um comando em um subprocesso de forma segura."""
    # subprocess.run aceita uma lista de argumentos, evitando falhas de string
    subprocess.run(cmd_list, check=False)


def process_callback(process_name: list[str]) -> None:
    """Função chamada quando um processo termina."""
    pass


def main() -> None:
    """Função principal (Entry point para o CLI)."""
    # =========================================================================
    # ARGUMENTOS
    # =========================================================================
    parser = argparse.ArgumentParser(description="Select MUAR arguments")
    parser.add_argument("--n_sessions", type=int, default=50)
    parser.add_argument("--n_players", type=int, default=6)
    parser.add_argument("--time", type=int, default=120)
    parser.add_argument("--eco_effi_ratio", type=float, default=0.7)
    parser.add_argument("--sfc", type=str, default="on")
    args, unknown = parser.parse_known_args()

    # ========================================================================
    # CONFIGURAÇÃO DE EXECUÇÃO
    # =========================================================================
    RUN_MODE = "batch"
    BATCH_ALGS = ["msf", "musfico", "vegeta", "greedyb"]
    BATCH_FAIL_TARGETS = ["low_risk", "med_risk", "high_risk", "all"]
    BATCH_TOTAL_RUNS = 20 // len(BATCH_FAIL_TARGETS)
    BATCH_PARALLEL_RUNS = 20

    number_of_fails = ["3"]
    CRASH_AT_TIME = ["400", "520", "640"]
    avas = ["0.99"]

    # =========================================================================
    # LÓGICA DE GERAÇÃO DE COMANDOS
    # =========================================================================
    begin = dt.now()
    cmds = []

    print(f"[INFO] Modo: {RUN_MODE}")
    print(f"[INFO] Algoritmos: {BATCH_ALGS}")
    print(f"[INFO] Cenários: {BATCH_FAIL_TARGETS}")
    print(f"[INFO] Configuração: 3 falha em T={CRASH_AT_TIME}s")
    print(f"[INFO] Total de Jobs: {len(BATCH_ALGS) * len(BATCH_FAIL_TARGETS) * BATCH_TOTAL_RUNS}")

    for alg_name in BATCH_ALGS:
        for fail_target in BATCH_FAIL_TARGETS:
            for _i in range(BATCH_TOTAL_RUNS):
                for a in avas:
                    for n in number_of_fails:
                        # Construção moderna usando listas
                        command = [
                            "muar-sim", # Chamando o CLI recém-criado
                            "--n_sessions", str(args.n_sessions),
                            "--alg", alg_name,
                            "--fail_target", fail_target,
                            "--sfc", str(args.sfc),
                            "--ava", str(a),
                            "--number_of_fails", str(n),
                            "--crash_at", *CRASH_AT_TIME, # Desempacota a lista
                            "--n_players", str(args.n_players),
                            "--sfc_lifetime", str(args.time),
                            "--eco_effi_ratio", str(args.eco_effi_ratio),
                            "--verbose", "n"
                        ]
                        cmds.append(command)

    # =========================================================================
    # EXECUÇÃO DO POOL
    # =========================================================================
    print(f"[INFO] Iniciando pool com {BATCH_PARALLEL_RUNS} processos...")
    print("[AVISO] Rodando tudo simultaneamente. Monitore o uso de CPU/RAM.")

    pool = Pool(processes=BATCH_PARALLEL_RUNS)

    for _i, command in enumerate(cmds):
        pool.apply_async(run_process, (command,), callback=lambda c=command: process_callback(c))
        time.sleep(1.0)

    pool.close()
    pool.join()

    duration = dt.now() - begin
    print("\n==================================================")
    print(f"Execução finalizada em: {duration}")
    print("==================================================")


if __name__ == "__main__":
    main()
