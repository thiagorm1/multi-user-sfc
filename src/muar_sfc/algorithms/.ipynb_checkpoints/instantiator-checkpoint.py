from muar_sfc.algorithms.greedy_boosted import GreedyOptAlgorithm
from muar_sfc.algorithms.nfvsdn import Goku
from muar_sfc.algorithms.osfem import Osfem
from muar_sfc.algorithms.genetic_alg import Genetic
from muar_sfc.algorithms.random_algorithm import RandomAlgorithm
from muar_sfc.algorithms.greedy_algorithm import GreedyAlgorithm
from muar_sfc.algorithms.dynamic_programming_algorithm import DynamicProgrammingAlgorithm
from muar_sfc.algorithms.k_shortest_paths_algorithm import KShortestPathsAlgorithm
from muar_sfc.algorithms.betweenness_centrality_algorithm import BetweennessCentralityAlgorithm
from muar_sfc.algorithms.musfico import Musfico  
from muar_sfc.algorithms.msf import MSF
from muar_sfc.algorithms.new_alg import NewAlg
from muar_sfc.algorithms.bruno_alg import BrunoAlg
from muar_sfc.algorithms.bruno_alg_2 import BrunoAlgNew
from muar_sfc.algorithms.rodrigo_alg import Rodrigo
from muar_sfc.algorithms.vegeta import Vegeta
from muar_sfc.algorithms.Kuririn_Prototype import Kuririn

class AlgorithmInstantiator:
    def instantiate_algorithm(self, type):
        if type == 'musfico':
            alg = Musfico()
        elif type =='new_alg':
            alg = NewAlg()
        elif type == 'dp':
            alg = DynamicProgrammingAlgorithm()
        elif type == 'g':
            alg = GreedyAlgorithm()  
        elif type == 'greedyb':
            alg = GreedyOptAlgorithm()  
        elif type == 'k':
            alg = KShortestPathsAlgorithm(5)
        elif type == 'b':
            alg = BetweennessCentralityAlgorithm()
        elif type == 'msf':
            alg = MSF()
        elif type =='bruno':
            alg = BrunoAlg()
        elif type =='brunonew':
            alg = BrunoAlgNew()
        elif type =='rodrigo':
            alg = Rodrigo()
        elif type ==  'ga':
            alg = Genetic()
        elif type ==  'osfem':
            alg = Osfem()
        elif type ==  'goku':
            alg = Goku()
        elif type ==  'vegeta':
            alg = Vegeta()
        elif type == 'kuririn':
            alg = Kuririn('PPO')
        else:
            raise ValueError('algorithm not found')
        return alg