#!/usr/bin/env python3
"""
Stateless Epsilon-Greedy Multi-Armed Bandit for ABR. 

Algorithm: Maintains empirical mean reward for each quality level. 
Selects quality using epsilon-greedy (explore with probability epsilon, 
exploit with probability 1-epsilon).

This bandit has NO context about network conditions, buffer level, or 
throughput.  It only learns:  "which quality level gives best average reward?"

Reward computation: quality_score - rebuffering_penalty - variation_penalty
"""

import numpy as np
from collections import deque

# ============================================================================
# GLOBAL STATE (persistent across chunks)
# ============================================================================
class EpsilonGreedyMABState:
    """Stateless epsilon-greedy bandit."""
    
    def __init__(self, n_actions:  int, epsilon: float = 0.1, initial_reward: float = 0.0):
        self.n_actions = n_actions
        self.epsilon = epsilon
        
        # Empirical mean reward for each arm
        self.estimates = np.full(n_actions, initial_reward, dtype=np.float32)
        self.counts = np.zeros(n_actions, dtype=np.int32)
        
        # For computing variation penalty
        self.past_quality_bitrates = deque(maxlen=10)
        
        # Episode counter
        self.episode_count = 0
    
    def select_action(self, training:  bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random. randint(self.n_actions)
        else:
            # Exploit: best arm (break ties randomly)
            best_value = self.estimates. max()
            best_arms = np.where(self.estimates == best_value)[0]
            return np.random.choice(best_arms)
    
    def update(self, action: int, reward: float):
        """Update empirical mean for chosen action."""
        self.counts[action] += 1
        # Incremental mean update
        self.estimates[action] += (reward - self.estimates[action]) / self.counts[action]
    
    def get_action(self) -> int:
        """Greedy action (no exploration)."""
        best_value = self.estimates.max()
        best_arms = np.where(self.estimates == best_value)[0]
        return np.random.choice(best_arms)


_bandit = None


# ============================================================================
# CLIENT MESSAGE
# ============================================================================
class ClientMessage:
    """Message from simulator to ABR algorithm."""
    pass


# ============================================================================
# REWARD COMPUTATION
# ============================================================================
def compute_reward(
    quality:  int,
    quality_bitrates: list,
    buffer_level: float,
    buffer_max:  float,
    quality_coeff: float,
    rebuffer_coeff: float,
    variation_coeff: float,
    past_quality_bitrate: float = None,
) -> float:
    """
    Compute reward for this quality choice.
    
    Reward = quality_score - rebuffering_risk - variation_penalty
    
    Note: This is an estimate of future QoE, computed before we know
    the actual download time.  We use buffer level as a proxy for
    rebuffering risk.
    """
    n_qualities = len(quality_bitrates)
    bitrate = quality_bitrates[quality]
    
    # Quality score:  normalized by max quality
    max_bitrate = max(quality_bitrates)
    quality_score = (quality / max(n_qualities - 1, 1)) * quality_coeff
    
    # Rebuffering penalty: higher quality + low buffer = higher risk
    # Heuristic: rebuffer_risk = (bitrate / buffer_level) if buffer is low
    buffer_utilization = buffer_level / buffer_max
    rebuffering_penalty = 0.0
    
    if buffer_utilization < 0.5:
        # Low buffer: penalize high quality
        rebuffering_penalty = rebuffer_coeff * (bitrate / max_bitrate) * (0.5 - buffer_utilization)
    
    # Variation penalty: penalize switching away from past quality
    variation_penalty = 0.0
    if past_quality_bitrate is not None and past_quality_bitrate > 0:
        bitrate_change = abs(bitrate - past_quality_bitrate)
        variation_penalty = variation_coeff * (bitrate_change / max_bitrate)
    
    reward = quality_score - rebuffering_penalty - variation_penalty
    return reward


# ============================================================================
# STUDENT ENTRYPOINT
# ============================================================================
def student_entrypoint(message: ClientMessage) -> int:
    """
    Stateless Epsilon-Greedy MAB for ABR.
    
    This algorithm:
    - Has NO knowledge of network conditions, buffer state, or throughput
    - Only learns: "which quality level gives the best average reward?"
    - Uses epsilon-greedy exploration/exploitation
    
    Args:
        message: ClientMessage with available qualities and coefficients
    
    Returns:
        Quality level (0 to num_qualities - 1)
    """
    global _bandit
    
    # Initialize on first call
    if _bandit is None:
        # Optimistic initialization: assume all qualities start with reward = 0.5
        _bandit = EpsilonGreedyMABState(
            n_actions=message.quality_levels,
            epsilon=0.1,  # Explore 10% of the time
            initial_reward=0.0
        )
    
    # Select action (quality) using epsilon-greedy
    quality = _bandit.select_action(training=True)
    
    # Clamp to valid range
    quality = max(0, min(quality, message.quality_levels - 1))
    
    return int(quality)


def update_bandit_reward(message: ClientMessage, quality: int):
    """
    Called after download completes to update bandit with reward signal.
    
    This would need to be called from simulator or wrapped in a separate
    evaluation loop that tracks actual outcomes.
    """
    global _bandit
    
    if _bandit is None:
        return
    
    reward = compute_reward(
        quality=quality,
        quality_bitrates=message.quality_bitrates,
        buffer_level=message.buffer_seconds_until_empty,
        buffer_max=message.buffer_max_size,
        quality_coeff=message. quality_coefficient,
        rebuffer_coeff=message.rebuffering_coefficient,
        variation_coeff=message.variation_coefficient,
        past_quality_bitrate=_bandit.past_quality_bitrates[-1] if _bandit. past_quality_bitrates else None,
    )
    
    _bandit.update(quality, reward)
    _bandit.past_quality_bitrates.append(message.quality_bitrates[quality])


def reset_bandit():
    """Reset bandit for a new episode."""
    global _bandit
    _bandit = None