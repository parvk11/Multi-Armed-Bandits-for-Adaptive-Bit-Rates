"""
Stateless Thompson Sampling (Bayesian Bandit) for ABR.

Algorithm: Model each arm's reward as random variable. 
Maintain posterior distribution (Beta) for each arm.
Sample from posteriors and pick arm with highest sample. 

"""

import numpy as np
from collections import deque

# ============================================================================
# GLOBAL STATE
# ============================================================================
class ThompsonSamplingMABState:
    """Thompson Sampling (Beta-Bernoulli) bandit."""
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        
        # Beta distribution parameters for each arm
        # Reward is treated as Bernoulli (mapped to [0,1])
        # Beta(alpha, beta) posterior
        self.alpha = np.ones(n_actions, dtype=np.float32)  # Success counts
        self.beta = np. ones(n_actions, dtype=np.float32)   # Failure counts
        
        # For variation penalty
        self.past_quality_bitrates = deque(maxlen=10)
        
        self.total_pulls = 0
    
    def select_action(self) -> int:
        """Thompson Sampling:  sample from posterior and pick best."""
        # Sample from Beta distribution for each arm
        samples = np. array([
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_actions)
        ])
        
        # Pick arm with highest sample
        return np.argmax(samples)
    
    def update(self, action: int, reward: float):
        """
        Update Beta posterior for chosen action.
        
        Reward is assumed to be in [0, 1].  We treat it as a Bernoulli
        outcome:  if reward > 0.5, count as success; else failure.
        
        For continuous rewards, this is a simplification.  Alternative: 
        use Gaussian posterior or normalize rewards to [0,1].
        """
        self.total_pulls += 1
        
        # Convert continuous reward to binary (threshold at 0.5)
        # More sophisticated: use Gaussian-Beta conjugate pair
        if reward >= 0.5:
            self.alpha[action] += 1.0
        else:
            self.beta[action] += 1.0


_bandit = None


# CLIENT MESSAGE
class ClientMessage: 
    """Message from simulator to ABR algorithm."""
    pass


# REWARD COMPUTATION
def compute_reward(
    quality:  int,
    quality_bitrates: list,
    buffer_level: float,
    buffer_max: float,
    quality_coeff: float,
    rebuffer_coeff: float,
    variation_coeff: float,
    past_quality_bitrate:  float = None,
) -> float:
    n_qualities = len(quality_bitrates)
    bitrate = quality_bitrates[quality]
    
    max_bitrate = max(quality_bitrates)
    quality_score = (quality / max(n_qualities - 1, 1)) * quality_coeff
    
    buffer_utilization = buffer_level / buffer_max
    rebuffering_penalty = 0.0
    
    if buffer_utilization < 0.5:
        rebuffering_penalty = rebuffer_coeff * (bitrate / max_bitrate) * (0.5 - buffer_utilization)
    
    variation_penalty = 0.0
    if past_quality_bitrate is not None and past_quality_bitrate > 0:
        bitrate_change = abs(bitrate - past_quality_bitrate)
        variation_penalty = variation_coeff * (bitrate_change / max_bitrate)
    
    reward = quality_score - rebuffering_penalty - variation_penalty
    return reward

# STUDENT ENTRYPOINT
def student_entrypoint(message:  ClientMessage) -> int:
    """
    args:
        message: ClientMessage with available qualities
    
    returns:
        Quality level (0 to num_qualities - 1)
    """
    global _bandit
    
    # Initialize on first call
    if _bandit is None: 
        _bandit = ThompsonSamplingMABState(n_actions=message.quality_levels)
    
    # Select action using Thompson Sampling
    quality = _bandit.select_action()
    
    # Clamp to valid range
    quality = max(0, min(quality, message.quality_levels - 1))
    
    return int(quality)


def update_bandit_reward(message: ClientMessage, quality: int):
    # Update bandit with reward signal.
    global _bandit
    
    if _bandit is None:
        return
    
    reward = compute_reward(
        quality=quality,
        quality_bitrates=message.quality_bitrates,
        buffer_level=message.buffer_seconds_until_empty,
        buffer_max=message.buffer_max_size,
        quality_coeff=message.quality_coefficient,
        rebuffer_coeff=message.rebuffering_coefficient,
        variation_coeff=message.variation_coefficient,
        past_quality_bitrate=_bandit.past_quality_bitrates[-1] if _bandit.past_quality_bitrates else None,
    )
    
    _bandit.update(quality, reward)
    _bandit.past_quality_bitrates.append(message. quality_bitrates[quality])


def reset_bandit():
    global _bandit
    _bandit = None