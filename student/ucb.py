"""
Stateless UCB (Upper Confidence Bound) Multi-Armed Bandit for ABR. 

Algorithm: Balances exploration and exploitation using confidence bounds.
Selects arm with highest upper confidence bound.

"""

import numpy as np
from collections import deque
import math

# ============================================================================
# GLOBAL STATE
# ============================================================================
class UCBMABState:
    
    def __init__(self, n_actions: int, c: float = 1.0):
        self.n_actions = n_actions
        self.c = c  # Confidence radius coefficient
        
        # Empirical mean reward for each arm
        self.estimates = np.zeros(n_actions, dtype=np.float32)
        self.counts = np.zeros(n_actions, dtype=np.int32)
        
        # Total pulls across all arms
        self.total_pulls = 0
        
        # For variation penalty
        self.past_quality_bitrates = deque(maxlen=10)
    
    def select_action(self) -> int:
        """Select arm with highest UCB value."""
        # UCB = mean + c * sqrt(ln(N) / n_i)
        # where N = total pulls, n_i = pulls for arm i
        
        ucb_values = np.zeros(self.n_actions)
        
        for i in range(self.n_actions):
            if self.counts[i] == 0:
                # Unvisited arm:  infinite UCB (explore first)
                ucb_values[i] = float('inf')
            else:
                mean = self.estimates[i]
                radius = self.c * math.sqrt(math.log(self.total_pulls + 1) / self.counts[i])
                ucb_values[i] = mean + radius
        
        # Select arm with highest UCB (break ties randomly)
        best_value = ucb_values.max()
        best_arms = np. where(ucb_values == best_value)[0]
        return np.random.choice(best_arms)
    
    def update(self, action: int, reward: float):
        """Update empirical mean for chosen action."""
        self.counts[action] += 1
        self.total_pulls += 1
        self.estimates[action] += (reward - self.estimates[action]) / self.counts[action]


_bandit = None


# CLIENT MESSAGE
class ClientMessage: 
    """Message from simulator to ABR algorithm."""
    pass


# REWARD COMPUTATION
def compute_reward(
    quality: int,
    quality_bitrates:  list,
    buffer_level:  float,
    buffer_max:  float,
    quality_coeff: float,
    rebuffer_coeff: float,
    variation_coeff: float,
    past_quality_bitrate: float = None,
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
def student_entrypoint(message: ClientMessage) -> int:
    """
    args: 
        message: ClientMessage with available qualities
    
    returns:
        Quality level (0 to num_qualities - 1)
    """
    global _bandit
    
    # Initialize on first call
    if _bandit is None: 
        _bandit = UCBMABState(n_actions=message.quality_levels, c=1.0)
    
    # Select action using UCB
    quality = _bandit.select_action()
    
    # Clamp to valid range
    quality = max(0, min(quality, message.quality_levels - 1))
    
    return int(quality)


def update_bandit_reward(message: ClientMessage, quality: int):
    global _bandit
    
    if _bandit is None:
        return
    
    reward = compute_reward(
        quality=quality,
        quality_bitrates=message.quality_bitrates,
        buffer_level=message.buffer_seconds_until_empty,
        buffer_max=message.buffer_max_size,
        quality_coeff=message.quality_coefficient,
        rebuffer_coeff=message. rebuffering_coefficient,
        variation_coeff=message.variation_coefficient,
        past_quality_bitrate=_bandit. past_quality_bitrates[-1] if _bandit.past_quality_bitrates else None,
    )
    
    _bandit.update(quality, reward)
    _bandit.past_quality_bitrates.append(message.quality_bitrates[quality])


def reset_bandit():
    global _bandit
    _bandit = None