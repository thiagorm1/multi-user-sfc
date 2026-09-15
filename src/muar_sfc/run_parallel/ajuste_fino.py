import argparse
import os
import time
from datetime import datetime as dt
from itertools import product
from multiprocessing import Pool


def run_process(process):
    """Executa um comando em um subprocesso."""
    os.system(f"python {process}")
    print(process)


if __name__ == "__main__":
    # Análise de argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Select MUAR arguments")
    parser.add_argument("--n_sessions", type=int, help="(int) number of sessions", default=50)
    parser.add_argument("--threads", type=int, help="(int) number of cores to use", default=16)
    parser.add_argument("--repetition", type=int, help="(int) repetitions", default=1)
    parser.add_argument("--sfc", type=str, help="(str) on or off", default="on")
    parser.add_argument("--alg", type=str, help="(str) algorithm name", default="ga")

    parser.add_argument("--share", type=str, help="(str) whether to share sfs or not", default="y")
    parser.add_argument(
        "--shareband", type=str, help="(str) whether to share sfs or not", default="y"
    )
    parser.add_argument(
        "--time", type=int, help="(int) the total time for the simulation in seconds", default=120
    )
    args = parser.parse_args()

    pool = Pool(processes=args.threads)
    begin = dt.now()

    # Construção dos comandos a serem executados
    cmd = []

    numeros = [1.0, 2.0, 3.0, 4.0]
    tamanho_vetor = 4

    # Gerando todas as combinações possíveis
    combinacoes = list(product(numeros, repeat=tamanho_vetor))
    combinacoes_formatadas_sem_espacos = [
        str(list(combinacao)).replace(" ", "") for combinacao in combinacoes
    ]

    for permutacao in combinacoes_formatadas_sem_espacos:
        for _ in range(args.repetition):
            command = (
                "./muar.py"
                + " --n_sessions "
                + str(args.n_sessions)
                + " --alg "
                + args.alg
                + " --costs_parameter "
                + str(permutacao)
                + " --sfc "
                + args.sfc
                + " --time "
                + str(args.time)
            )
            cmd.append(command)

    print(cmd)

    # Executar cada comando com um atraso de 1 segundo entre eles
    for command in cmd:
        pool.apply_async(run_process, (command,))
        time.sleep(0.5)  # Atraso antes de iniciar o próximo comando

    pool.close()  # Nenhum outro trabalho será adicionado
    pool.join()  # Esperar por todos os processos terminarem

    duration = dt.now() - begin
    print("Processing time:", duration)
    print("Finished")
