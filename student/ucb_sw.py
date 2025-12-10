from typing import List, Union
from SMPyBandits.Policies import CUSUM_UCB, SlidingWindowUCB, DiscountedUCB
import math

bandit = None
initialized = False
last_action = None
last_quality = None
last_chunk_size = None
previous_chunk_info = None  # Store info about the chunk we just downloaded


class ClientMessage:
    """
    This class will be filled out and passed to student_entrypoint for your algorithm.
    """
    total_seconds_elapsed: float
    previous_throughput: float

    buffer_current_fill: float
    buffer_seconds_per_chunk: float
    buffer_seconds_until_empty: float
    buffer_max_size: float

    quality_levels: int
    quality_bitrates: List[float]
    upcoming_quality_bitrates: List[List[float]]

    quality_coefficient: float
    variation_coefficient: float
    rebuffering_coefficient: float


def student_entrypoint(client_message: ClientMessage) -> int:
    """
    Adaptive bitrate selection using multi-armed bandit algorithms.
    
    Args:
        client_message: ClientMessage holding the parameters for this chunk and current client state.
        bandit_algo: Which MAB algorithm to use ('CUSUM-UCB', 'SlidingWindow-UCB', 'Discounted-UCB')
    
    Returns:
        int: Quality choice in the range [0 ... quality_levels - 1] inclusive.
    """
    global last_quality, last_action, last_chunk_size, bandit, initialized, previous_chunk_info

    # Initialize bandit on first call
    maybe_initialize_bandit(client_message)

    # Give reward for the previous action if we have the information
    if previous_chunk_info is not None and client_message.previous_throughput > 0:
        # Calculate actual download time for the previous chunk
        prev_chunk_size_kb = previous_chunk_info['chunk_size'] * 1024  # MB to kB
        download_time = prev_chunk_size_kb / client_message.previous_throughput  # seconds
        
        # Calculate the reward based on actual performance
        reward = compute_reward(
            quality=previous_chunk_info['quality'],
            chunk_size_mb=previous_chunk_info['chunk_size'],
            download_time=download_time,
            buffer_before=previous_chunk_info['buffer_before'],
            client_message=client_message
        )
        
        bandit.getReward(previous_chunk_info['action'], reward)
        # print(f"Gave reward {reward:.4f} for action {previous_chunk_info['action']} "
            #   f"(quality {previous_chunk_info['quality']}, download_time={download_time:.2f}s)")

    # Choose the next action
    action = bandit.choice()
    quality = int(action)
    
    # Store information about this chunk for next iteration's reward calculation
    previous_chunk_info = {
        'action': action,
        'quality': quality,
        'chunk_size': client_message.quality_bitrates[quality],  # in MB
        'buffer_before': client_message.buffer_seconds_until_empty
    }
    
    last_quality = quality
    last_action = action
    last_chunk_size = client_message.quality_bitrates[quality]
    
    # print(f"Selected quality {quality} (chunk size: {last_chunk_size:.2f} MB)")
    
    return quality


def compute_reward(quality: int, chunk_size_mb: float, download_time: float, 
                   buffer_before: float, client_message: ClientMessage) -> float:
    """
    Compute reward for a chunk selection.
    
    Args:
        quality: Quality level that was chosen
        chunk_size_mb: Size of the chunk in MB
        download_time: Actual time to download in seconds
        buffer_before: Buffer level before download
        client_message: Current client state
    
    Returns:
        Normalized reward in [0, 1]
    """
    global last_quality
    
    # Convert chunk size to bitrate (Mbps)
    bitrate = (chunk_size_mb * 8) / client_message.buffer_seconds_per_chunk
    
    # Calculate rebuffering time
    rebuffer = max(0, download_time - buffer_before)
    
    # Calculate quality variation penalty
    smoothness = 0
    if last_quality is not None:
        # Compare bitrates, not chunk sizes
        prev_bitrate = (client_message.quality_bitrates[last_quality] * 8) / client_message.buffer_seconds_per_chunk
        curr_bitrate = bitrate
        smoothness = abs(curr_bitrate - prev_bitrate)
    
    # Calculate raw reward using QoE formula
    raw_reward = (
        client_message.quality_coefficient * bitrate
        - client_message.rebuffering_coefficient * rebuffer
        - client_message.variation_coefficient * smoothness
    )
    
    # print(f"  Bitrate: {bitrate:.2f} Mbps, Rebuffer: {rebuffer:.2f}s, "
    #       f"Smoothness: {smoothness:.2f}, Raw reward: {raw_reward:.2f}")
    
    # Normalize reward to [0, 1] using global min/max estimates
    R_max = client_message.quality_coefficient * (max(client_message.quality_bitrates) * 8 / client_message.buffer_seconds_per_chunk)
    
    # Use more realistic bounds for normalization
    max_expected_rebuffer = 10  # seconds
    max_expected_smoothness = (max(client_message.quality_bitrates) - min(client_message.quality_bitrates)) * 8 / client_message.buffer_seconds_per_chunk
    
    R_min = (
        client_message.quality_coefficient * (min(client_message.quality_bitrates) * 8 / client_message.buffer_seconds_per_chunk)
        - client_message.rebuffering_coefficient * max_expected_rebuffer
        - client_message.variation_coefficient * max_expected_smoothness
    )
    
    # Normalize
    if R_max - R_min > 0:
        normalized_reward = (raw_reward - R_min) / (R_max - R_min)
    else:
        normalized_reward = 0.5
    
    # Clip to [0, 1]
    normalized_reward = max(0, min(1, normalized_reward))
    
    # print(f"  Normalized reward: {normalized_reward:.4f} (R_min={R_min:.2f}, R_max={R_max:.2f})")
    
    return normalized_reward


def maybe_initialize_bandit(client: ClientMessage):
    """Initialize the bandit algorithm on first call."""
    global bandit, initialized
    
    if initialized:
        return
    
    # print(f"Initializing {bandit_algo} bandit with {client.quality_levels} arms...")
    
    n_arms = client.quality_levels
    
  
    bandit = SlidingWindowUCB.SWUCB(
        nbArms=n_arms,
        tau=50,  # Window size - adjust based on expected change frequency
        alpha=1.0
    )
 
   
    
    bandit.startGame()
    initialized = True
    # print("Bandit initialized successfully")