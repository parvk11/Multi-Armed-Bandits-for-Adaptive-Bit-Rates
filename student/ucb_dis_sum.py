from typing import List, Union
from SMPyBandits.Policies import DiscountedUCB
import math

bandit = None
initialized = False
last_action = None
last_quality = None
last_chunk_size = None
previous_chunk_info = None  # Store info about the last chunk


class ClientMessage:
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
    '''
    Adaptive bitrate selection using CUSUM-UCB bandit algorithm.    
    :param client_message: Description
    :type client_message: ClientMessage
    :return: Description
    :rtype: int
    '''
    global last_quality, last_action, last_chunk_size
    global bandit, initialized, previous_chunk_info

    # Initialize bandit on first call
    maybe_initialize_bandit(client_message)


    if previous_chunk_info is not None and client_message.previous_throughput > 0:
        chunk_size_mb = previous_chunk_info["chunk_size"]
        chunk_kb = chunk_size_mb * 1024   # MB → kB
        download_time = chunk_kb / client_message.previous_throughput

        reward = compute_reward(
            quality=previous_chunk_info["quality"],
            chunk_size_mb=chunk_size_mb,
            download_time=download_time,
            buffer_before=previous_chunk_info["buffer_before"],
            client_message=client_message
        )

        bandit.getReward(previous_chunk_info["action"], reward)


    action = bandit.choice()
    quality = int(action)

    # Store current chunk for next round’s reward
    previous_chunk_info = {
        "action": action,
        "quality": quality,
        "chunk_size": client_message.quality_bitrates[quality],
        "buffer_before": client_message.buffer_seconds_until_empty
    }

    last_action = action
    last_quality = quality
    last_chunk_size = client_message.quality_bitrates[quality]

    return quality



def compute_reward(quality: int, chunk_size_mb: float, download_time: float,
                   buffer_before: float, client_message: ClientMessage) -> float:
    """
    Raw, unnormalized reward:
    
        reward =  qc * bitrate_mbps
                - rc * rebuffer_seconds
                - vc * variation_mbps
    """
    global last_quality

    # Bitrate in Mbps
    bitrate_mbps = (chunk_size_mb * 8) / client_message.buffer_seconds_per_chunk

    # Rebuffer time
    rebuffer_time = max(0, download_time - buffer_before)

    # Smoothness penalty (variation)
    variation = 0
    if last_quality is not None:
        prev_bitrate = (
            client_message.quality_bitrates[last_quality] * 8 /
            client_message.buffer_seconds_per_chunk
        )
        variation = abs(bitrate_mbps - prev_bitrate)

    # Raw QoE reward (no normalization!)
    reward = (
        client_message.quality_coefficient * bitrate_mbps
        - client_message.rebuffering_coefficient * rebuffer_time
        - client_message.variation_coefficient * variation
    )

    return reward



def maybe_initialize_bandit(client: ClientMessage):
    """Initialize the bandit algorithm on first call."""
    global bandit, initialized
    
    if initialized:
        return
    
    # print(f"Initializing {bandit_algo} bandit with {client.quality_levels} arms...")
    
    n_arms = client.quality_levels
    
    # if bandit_algo == 'SlidingWindow-UCB':
    #     bandit = SlidingWindowUCB.SWUCB(
    #         nbArms=n_arms,
    #         tau=50,  # Window size - adjust based on expected change frequency
    #         alpha=1.0
    #     )
   
    bandit = DiscountedUCB(
        nbArms=n_arms,
        alpha=0.5,       # Exploration parameter
        gamma=0.94,
        useRealDiscount=True  # Discount factor

    )
   
    
    bandit.startGame()
    initialized = True
    # print("Bandit initialized successfully")