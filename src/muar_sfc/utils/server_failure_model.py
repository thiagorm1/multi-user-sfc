import numpy as np


class ServerFailureModel:
    """
    Calculates the probability of server failure based on hardware and software models.

    This class combines a Weibull distribution for hardware wear-out and a linear
    aging model for software degradation to determine the likelihood of a server
    failure at any given time.
    """

    def __init__(self, eta: float, beta: float, alpha: float, lambda_base: float):
        """
        Initializes the ServerFailureModel with parameters for hardware and software failure.

        Args:
            eta (float): The characteristic life (scale parameter η) for the Weibull distribution.
            beta (float): The shape parameter (β) for the Weibull distribution.
                          For edge environments, a value > 1 (e.g., 1.5 to 2.5) is recommended
                          to model wear-out.
            alpha (float): The aging factor (α) for the software aging model, representing
                           the rate of degradation.
            lambda_base (float): The base failure rate (λ_base) for the software model,
                                 representing sporadic bugs.
        """
        if beta <= 1.0:
            print(
                f"Warning: beta = {beta} <= 1. This models a constant or "
                f"decreasing failure rate, not wear-out."
            )

        self.eta = eta
        self.beta = beta
        self.alpha = alpha
        self.lambda_base = lambda_base

    def weibull_hazard_rate(self, t: float) -> float:
        """
        Calculates the instantaneous hardware failure rate (Hazard Rate) using the Weibull model.

        Formula: λ(t) = (β / η) * (t / η)^(β - 1)

        Args:
            t (float): The current time (age) of the hardware.

        Returns:
            float: The instantaneous probability of hardware failure.
        """
        if t < 0:
            return 0.0
        return (self.beta / self.eta) * np.power(t / self.eta, self.beta - 1)

    def software_aging_rate(self, t_uptime: float) -> float:
        """
        Calculates the instantaneous software failure rate based on a linear aging model.

        Formula: λ_software(t_uptime) = λ_base + α * t_uptime

        Args:
            t_uptime (float): The time since the last server reboot.

        Returns:
            float: The instantaneous probability of software failure.
        """
        if t_uptime < 0:
            return self.lambda_base
        return self.lambda_base + self.alpha * t_uptime

    def check_failure(
        self,
        global_time: float,
        uptime_time: float,
        step_duration: float,
        cpu_stress_factor: float = 1.0,
    ) -> bool:
        """
        Determines if a server fails in a given time step.

        This method combines hardware and software failure rates, applies a CPU stress
        factor, and uses a probabilistic approach to decide if a failure occurs.

        Args:
            global_time (float): The total simulation time (hardware age).
            uptime_time (float): The time since the last reboot (software age).
            step_duration (float): The duration of the simulation tick.
            cpu_stress_factor (float, optional): A multiplier (0.0 to 1.0) to adjust
                                                 failure rates based on CPU load. Defaults to 1.0.

        Returns:
            bool: True if a failure occurred, False otherwise.
        """
        # Calculate individual hazard rates
        hw_rate = self.weibull_hazard_rate(global_time)
        sw_rate = self.software_aging_rate(uptime_time)

        # Combine rates (assuming independence) and apply stress factor
        total_hazard_rate = (hw_rate + sw_rate) * cpu_stress_factor

        # Calculate probability of failure during the step
        # For a small step_duration, P(failure) ≈ λ(t) * Δt
        prob_of_failure = total_hazard_rate * step_duration

        # Clamp the probability to a valid range [0, 1]
        prob_of_failure = np.clip(prob_of_failure, 0, 1)

        # "Weighted Roulette Wheel" decision
        return np.random.rand() < prob_of_failure


if __name__ == "__main__":
    # --- Example Usage ---

    # Parameters for an industrial edge server
    # High reliability hardware, but subject to wear-out (beta > 1)
    # Software that degrades moderately over time
    server_model = ServerFailureModel(
        eta=50000,  # Characteristic life of 50,000 hours
        beta=1.5,  # Wear-out failure mode
        alpha=0.00001,  # Software degrades slowly
        lambda_base=0.0001,  # Low base rate for software bugs
    )

    print("--- Simulating Server Failure ---")

    global_time = 0
    uptime_time = 0
    step_duration = 1  # Simulate in 1-hour steps
    total_steps = 10000

    for _t in range(total_steps):
        global_time += step_duration
        uptime_time += step_duration

        # Simulate varying CPU stress
        cpu_stress = np.random.uniform(0.5, 1.0)

        if server_model.check_failure(
            global_time, uptime_time, step_duration, cpu_stress_factor=cpu_stress
        ):
            print(
                f"Server FAILED at global_time={global_time}h, "
                f"uptime={uptime_time}h (CPU Stress: {cpu_stress:.2f})"
            )
            # On failure, uptime is reset
            uptime_time = 0

        # Random reboots (maintenance)
        if np.random.rand() < 0.001:
            print(f"INFO: Server rebooted for maintenance at global_time={global_time}h")
            uptime_time = 0

    print(f"--- Simulation Finished after {total_steps} hours ---")
