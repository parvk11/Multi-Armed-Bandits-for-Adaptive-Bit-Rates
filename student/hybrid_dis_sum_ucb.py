from typing import List, Dict, Tuple
from SMPyBandits.Policies import CUSUM_UCB, SlidingWindowUCB, DiscountedUCB
import math

bandits = {}
last_quality = None
previous_chunk_info = None


class ClientMessage:
    total_seconds_elapsed: float
    previous_throughput: float
    buffer_seconds_per_chunk: float
    buffer_seconds_until_empty: float
    buffer_max_size: float
    quality_levels: int
    quality_bitrates: List[float]
    upcoming_quality_bitrates: List[List[float]]
    quality_coefficient: float
    variation_coefficient: float
    rebuffering_coefficient: float


def discretize_state(client: ClientMessage) -> Tuple[str, str]:
    """Discretize the continuous state space into buckets."""
    # Discretize throughput
    if client.previous_throughput > 0:
        throughput_mbps = client.previous_throughput / 125  # kB/s to Mbps
    else:
        throughput_mbps = 1.0
    
    if throughput_mbps < 1.5:
        throughput_state = "LOW"
    elif throughput_mbps < 3.0:
        throughput_state = "MED"
    elif throughput_mbps < 5.0:
        throughput_state = "HIGH"
    else:
        throughput_state = "VERY_HIGH"

    buffer_ratio = client.buffer_seconds_until_empty / client.buffer_max_size
    # Buffer
    if buffer_ratio < 0.1:
        buffer_state = "CRIT_LOW"
    elif buffer_ratio < 0.3:
        buffer_state = "LOW"
    elif buffer_ratio < 0.6:
        buffer_state = "MED"
    elif buffer_ratio < 0.9:
        buffer_state = "HIGH"
    else:
        buffer_state = "FULL"
    
    return throughput_state, buffer_state


def get_or_create_bandit(state: Tuple[str, str], n_arms: int):
    """Get or create a bandit for a specific state."""
    global bandits
    
    state_key = f"{state[0]}_{state[1]}"
    
    if state_key not in bandits:
        # print(f"Creating new bandit for state {state_key}")
        
       
        bandit = DiscountedUCB(
            nbArms=n_arms,
            alpha=0.5,
            gamma=0.94,
            useRealDiscount=True
        )
       
        bandit.startGame()
        bandits[state_key] = bandit
    
    return bandits[state_key]


def student_entrypoint(client_message: ClientMessage) -> int:
    """
    State-based bandit using a Discounted-UCB algorithm
    Each (throughput, buffer) state has its own bandit.
    The bandit chooses the quality level (action) for that state.
    
    """
    global last_quality, previous_chunk_info
    
    # Give reward for previous action
    if previous_chunk_info is not None and client_message.previous_throughput > 0:
        # prev_chunk_size_kb = previous_chunk_info['chunk_size'] * 1024
        # download_time = prev_chunk_size_kb / client_message.previous_throughput
        prev_bitrate_mbps = previous_chunk_info['quality']
        download_time = min(prev_bitrate_mbps / client_message.previous_throughput, client_message.buffer_seconds_until_empty)

        
        reward = compute_reward(
            quality=previous_chunk_info['quality'],
            chunk_size_mb=previous_chunk_info['chunk_size'],
            download_time=download_time,
            buffer_before=previous_chunk_info['buffer_before'],
            client_message=client_message
        )
        
        # Give reward to the bandit that made the decision
        prev_state_key = f"{previous_chunk_info['state'][0]}_{previous_chunk_info['state'][1]}"
        if prev_state_key in bandits:
            bandits[prev_state_key].getReward(previous_chunk_info['action'], reward)
            # print(f"State {prev_state_key}: Q{previous_chunk_info['action']} reward={reward:.2f}")
    
    # Determine current state
    state = discretize_state(client_message)
    
    # Get the bandit for this state
    bandit = get_or_create_bandit(state, client_message.quality_levels)
    
    # Choose action using this state's bandit
    action = bandit.choice()
    quality = int(action)
    
    # Apply safety override: if buffer is critically low, force lowest quality
    buffer_ratio = client_message.buffer_seconds_until_empty / client_message.buffer_max_size
    if buffer_ratio < 0.05:
        quality = max(0, quality-1)  # Force Q0
        # print(f"  [SAFETY] Buffer at {buffer_ratio:.1%}, forcing Q0")
    
    # Store info for next iteration
    previous_chunk_info = {
        'action': quality,
        'quality': quality,
        'chunk_size': client_message.quality_bitrates[quality],
        'buffer_before': client_message.buffer_seconds_until_empty,
        'state': state
    }
    
    last_quality = quality
    
    return quality


def compute_reward(quality: int, chunk_size_mb: float, download_time: float,
                   buffer_before: float, client_message: ClientMessage) -> float:
    """
    Compute reward for a given chunk based on metrics.
    Reward is calculated as:
        (Quality Coefficient) * (Bitrate in Mbps)
        - (Rebuffering Coefficient) * (Rebuffering Time in seconds)
        - (Variation Coefficient) * (Quality Variation in Mbps)
    
    """
    global last_quality
    
    # Convert chunk size to bitrate (Mbps)
    bitrate_mbps = (chunk_size_mb * 8) / client_message.buffer_seconds_per_chunk
    
    # Calculate rebuffering time (seconds)
    rebuffer_time = max(0, download_time - buffer_before)
    
    # Calculate quality variation (Mbps)
    variation = 0
    if last_quality is not None:
        prev_bitrate = (client_message.quality_bitrates[previous_chunk_info['quality']] * 8) / client_message.buffer_seconds_per_chunk
        variation = abs(bitrate_mbps - prev_bitrate)
    
    # Raw QoE reward (no normalization!)
    # Higher bitrate = good, rebuffering = bad, variation = bad
    reward = (
        client_message.quality_coefficient * bitrate_mbps
        - client_message.rebuffering_coefficient * rebuffer_time
        - client_message.variation_coefficient * variation
    )

    return reward


def reset_state():
    """Call this between test files to reset state."""
    global bandits, last_quality, previous_chunk_info
    bandits = {}
    last_quality = None
    previous_chunk_info = None