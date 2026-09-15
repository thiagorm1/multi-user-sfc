import os
import argparse
from datetime import datetime as dt
import time
import sys  # <--- 1. IMPORTE O MÓDULO SYS

def run_process(process):
    """Executa um comando em um subprocesso de forma sequencial."""
    print(f"[{dt.now()}] Iniciando processo: {process}")
    # --- 2. MODIFIQUE ESTA LINHA ---
    # ANTES: os.system(f'python {process}')
    # DEPOIS:
    os.system(f'{sys.executable} {process}')
    # -------------------------------
    print(f"[{dt.now()}] Finalizado: {process}\n")

if __name__ == '__main__':
    # Análise de argumentos de linha de comando
    horas=1
    simul_em_hora = 3.6
    num_simul = int(horas*simul_em_hora)
    parser = argparse.ArgumentParser(description='Select MUAR arguments')
    # ... (o resto do seu código permanece o mesmo) ...
    parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
    parser.add_argument('--n_players', type=int, help='(int) number of players', default=4)
    parser.add_argument('--repetition', type=int, help='(int) repetitions', default=num_simul)
    parser.add_argument('--sfc', type=str, help='(str) on or off', default='on')
    parser.add_argument('--alg', type=str, help='(str) algorithm name', default='kuririn_ppo')
    parser.add_argument('--verbose', type=str, help='verbose log', default='n')
    parser.add_argument('--time', type=int, help='(int) the total time for the simulation in seconds', default=120)
    args = parser.parse_args()
    begin = dt.now()

    # Construção dos comandos a serem executados
    cmd = []
    # avas = ['1.0', '0.99', '0.97', '0.95']
    avas = ['1.0']
    number_of_fails = ['3']
    for a in avas:
        for n in number_of_fails:
            for _ in range(args.repetition):
                command = './main.py' + \
                    ' --n_sessions ' + str(args.n_sessions) + \
                    ' --alg ' + args.alg + \
                    ' --sfc ' + args.sfc + \
                    ' --ava ' + str(a) + \
                    ' --number_of_fails ' + str(n) + \
                    ' --n_players ' + str(args.n_players) + \
                    ' --time ' + str(args.time)
                cmd.append(command)

    # Executar cada comando SEQUENCIALMENTE
    for idx, command in enumerate(cmd):
        print(f"=== Executando Simulação {idx + 1}/{len(cmd)} ===")
        run_process(command)

    duration = dt.now() - begin
    print('Processing time:', duration)
    print('Finished')