import argparse
import subprocess
from datetime import datetime as dt


def run_process(cmd_list: list[str]) -> None:
    """Executa um comando em um subprocesso de forma sequencial."""
    print(f"[{dt.now()}] Iniciando processo: {' '.join(cmd_list)}")
    # subprocess.run aguarda a execução terminar naturalmente (comportamento sequencial)
    subprocess.run(cmd_list, check=False)
    print(f"[{dt.now()}] Finalizado.\n")


def main() -> None:
    """Função principal (Entry point para o CLI)."""
    horas = 1
    simul_em_hora = 3.6
    num_simul = int(horas * simul_em_hora)

    parser = argparse.ArgumentParser(description="Select MUAR arguments")
    parser.add_argument("--n_sessions", type=int, help="(int) number of sessions", default=50)
    parser.add_argument("--n_players", type=int, help="(int) number of players", default=6)
    parser.add_argument("--repetition", type=int, help="(int) repetitions", default=2)
    parser.add_argument("--sfc", type=str, help="(str) on or off", default="on")
    parser.add_argument("--alg", type=str, help="(str) algorithm name", default="greedyb")
    parser.add_argument("--verbose", type=str, help="verbose log", default="n")
    parser.add_argument("--time", type=int, help="(int) total time for the simulation", default=120)
    args = parser.parse_args()

    begin = dt.now()

    cmds = []
    avas = ["0.99"]
    number_of_fails = ["20"]

    for a in avas:
        for n in number_of_fails:
            for _ in range(args.repetition):
                # Construção moderna usando listas
                command = [
                    "muar-sim", # Chamando o CLI
                    "--n_sessions", str(args.n_sessions),
                    "--alg", args.alg,
                    "--sfc", args.sfc,
                    "--ava", str(a),
                    "--number_of_fails", str(n),
                    "--n_players", str(args.n_players),
                    "--time", str(args.time)
                ]
                cmds.append(command)

    for idx, command in enumerate(cmds):
        print(f"=== Executando Simulação {idx + 1}/{len(cmds)} ===")
        run_process(command)

    duration = dt.now() - begin
    print("Processing time:", duration)
    print("Finished")


if __name__ == "__main__":
    main()
