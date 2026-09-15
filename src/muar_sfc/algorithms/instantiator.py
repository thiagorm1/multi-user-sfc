from muar_sfc.algorithms.betweenness_centrality_algorithm import BetweennessCentralityAlgorithm
from muar_sfc.algorithms.bruno_alg import BrunoAlg
from muar_sfc.algorithms.bruno_alg_2 import BrunoAlgNew
from muar_sfc.algorithms.darsppo import DARSPPO
from muar_sfc.algorithms.dynamic_programming_algorithm import DynamicProgrammingAlgorithm
from muar_sfc.algorithms.genetic_alg import Genetic
from muar_sfc.algorithms.greedy_algorithm import GreedyAlgorithm
from muar_sfc.algorithms.greedy_boosted import GreedyOptAlgorithm
from muar_sfc.algorithms.hephaestus import hephaestus
from muar_sfc.algorithms.k_shortest_paths_algorithm import KShortestPathsAlgorithm
from muar_sfc.algorithms.kuririn import Kuririn
from muar_sfc.algorithms.msf import MSF
from muar_sfc.algorithms.musfico import Musfico
from muar_sfc.algorithms.new_alg import NewAlg
from muar_sfc.algorithms.nfvsdn import Goku
from muar_sfc.algorithms.osfem import Osfem
from muar_sfc.algorithms.replic import REPLIC
from muar_sfc.algorithms.rodrigo_alg import Rodrigo
from muar_sfc.algorithms.vegeta import Vegeta


class AlgorithmInstantiator:
    def instantiate_algorithm(self, type):
        if type == "musfico":
            alg = Musfico()
        elif type == "new_alg":
            alg = NewAlg()
        elif type == "dp":
            alg = DynamicProgrammingAlgorithm()
        elif type == "g":
            alg = GreedyAlgorithm()
        elif type == "greedyb":
            alg = GreedyOptAlgorithm()
        elif type == "k":
            alg = KShortestPathsAlgorithm(5)
        elif type == "b":
            alg = BetweennessCentralityAlgorithm()
        elif type == "msf":
            alg = MSF()
        elif type == "bruno":
            alg = BrunoAlg()
        elif type == "brunonew":
            alg = BrunoAlgNew()
        elif type == "rodrigo":
            alg = Rodrigo()
        elif type == "ga":
            alg = Genetic()
        elif type == "osfem":
            alg = Osfem()
        elif type == "goku":
            alg = Goku()
        elif type == "vegeta":
            alg = Vegeta()
        elif "kuririn" in type:
            if "MaskablePPO" in type:
                alg = Kuririn("MASKABLEPPO")
            elif "PPO" in type:
                alg = Kuririn("PPO")

            if "DQN" in type:
                alg = Kuririn("DQN")
        elif "darsppo" in type:
            if "MaskablePPO" in type:
                alg = DARSPPO("MASKABLEPPO")
            elif "PPO" in type:
                alg = DARSPPO("PPO")

            if "DQN" in type:
                alg = DARSPPO("DQN")
        elif "hephaestus" in type:
            if "MaskablePPO" in type:
                alg = hephaestus("MASKABLEPPO")
            elif "PPO" in type:
                alg = hephaestus("PPO")

            if "DQN" in type:
                alg = hephaestus("DQN")
        elif "replic" in type:
            if "DQN" in type:
                alg = REPLIC("DQN")
            elif "PPO" in type:
                alg = REPLIC("PPO")
            else:
                alg = REPLIC("MASKABLEPPO")
            

            

        else:
            raise ValueError("algorithm not found")
        return alg
