import threading
import numpy as np
from loguru import logger

class PoissonEmitter:
    def __init__(self, lam: float, initial_delay: float = 0.0):
        self.lam = lam
        self.initial_delay = initial_delay
        self.current_time = 0.0
        self.next_time = 0.0
        self.is_output = False
        self.callback = None
        self.args = None
        self.is_stop = True
        self.timer: threading.Timer | None = None

    def start(self, func, *args):
        logger.info("Gerador de emissão iniciado.")
        self.is_stop = False
        self.callback = func
        
        # EAFP: Tentamos acessar o primeiro argumento. Se não existir, alocamos None.
        try:
            self.args = args[0]
        except IndexError:
            self.args = None

        if self.initial_delay > 0:
            self.timer = threading.Timer(self.initial_delay, self.reset_timer)
            self.timer.daemon = True  # Blindagem contra vazamento de threads
            self.timer.start()
        else:
            self.reset_timer()

    def reset_timer(self):
        if self.is_stop:
            return
            
        self.is_output = False
        interval = np.random.poisson(self.lam)
        logger.debug(f"Intervalo calculado: {interval}s")
        
        self.timer = threading.Timer(interval, self.run)
        self.timer.daemon = True  # Blindagem contra vazamento de threads
        self.timer.start()

    def run(self):
        if self.is_stop:
            return
            
        self.is_output = True
        if self.callback:
            self.callback(self.args)
            
        self.reset_timer()

    def stop(self):
        self.is_stop = True
        
        # EAFP: Tenta cancelar, caso o timer ainda seja None, captura a exceção de forma explícita.
        try:
            self.timer.cancel()
        except AttributeError:
            logger.warning("Tentativa de parada sem um timer instanciado.")
            
        logger.info("Gerador de emissão finalizado de forma limpa.")