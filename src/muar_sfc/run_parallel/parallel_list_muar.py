import argparse
import os
from datetime import datetime as dt
from multiprocessing import Pool

# command line arguments

parser = argparse.ArgumentParser(description="Select MUAR arguments")
parser.add_argument("--n_sessions", type=int, help="(int) number of sessions", default=50)
parser.add_argument("--alg", type=str, help="(str) algorithm name", default="dp")
parser.add_argument("--threads", type=int, help="(int) number of cores to use", default=10)
parser.add_argument("--repetition", type=str, help="(int) repetitions", default=5)
parser.add_argument("--sfc", type=str, help="(str) on or off", default="on")
parser.add_argument("--share", type=str, default="y")
parser.add_argument(
    "--time", type=int, help="(int) the total time for the simulation in seconds", default=120
)

args = parser.parse_args()

pool_size = args.threads
repetition = int(args.repetition)


def run_process(process):
    os.system(f"python {process}")
    print(process)


if __name__ == "__main__":
    pool = Pool(processes=pool_size)
    begin = dt.now()
    cmd = ()
    for _i in range(0, repetition):
        if args.sfc == "on":
            cmd += (
                "./muar.py"
                + " --n_sessions "
                + str(args.n_sessions)
                + " --alg "
                + str(args.alg)
                + " --sfc on"
                + " --time "
                + str(args.time),
            )
        elif args.sfc == "off":
            cmd += (
                "./muar.py"
                + " --n_sessions "
                + str(args.n_sessions)
                + " --alg "
                + str(args.alg)
                + " --sfc off"
                + " --time "
                + str(args.time),
            )

    print(cmd)
    print()
    pool.map(run_process, cmd)
    duration = dt.now() - begin
    print("Processing time:", duration)
    print("Finished")
