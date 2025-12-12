from typing import List

# Adapted from code by Zach Peats

# ======================================================================================================================
# Do not touch the client message class!
# ======================================================================================================================

last_quality = None
last_throughput = None

class ClientMessage:
	"""
	This class will be filled out and passed to student_entrypoint for your algorithm.
	"""
	total_seconds_elapsed: float	  # The number of simulated seconds elapsed in this test
	previous_throughput: float		  # The measured throughput for the previous chunk in kB/s

	buffer_current_fill: float		    # The number of kB currently in the client buffer
	buffer_seconds_per_chunk: float     # Number of seconds that it takes the client to watch a chunk. Every
										# buffer_seconds_per_chunk, a chunk is consumed from the client buffer.
	buffer_seconds_until_empty: float   # The number of seconds of video left in the client buffer. A chunk must
										# be finished downloading before this time to avoid a rebuffer event.
	buffer_max_size: float              # The maximum size of the client buffer. If the client buffer is filled beyond
										# maximum, then download will be throttled until the buffer is no longer full

	# The quality bitrates are formatted as follows:
	#
	#   quality_levels is an integer reflecting the # of quality levels you may choose from.
	#
	#   quality_bitrates is a list of floats specifying the number of kilobytes the upcoming chunk is at each quality
	#   level. Quality level 2 always costs twice as much as quality level 1, quality level 3 is twice as big as 2, and
	#   so on.
	#       quality_bitrates[0] = kB cost for quality level 1
	#       quality_bitrates[1] = kB cost for quality level 2
	#       ...
	#
	#   upcoming_quality_bitrates is a list of quality_bitrates for future chunks. Each entry is a list of
	#   quality_bitrates that will be used for an upcoming chunk. Use this for algorithms that look forward multiple
	#   chunks in the future. Will shrink and eventually become empty as streaming approaches the end of the video.
	#       upcoming_quality_bitrates[0]: Will be used for quality_bitrates in the next student_entrypoint call
	#       upcoming_quality_bitrates[1]: Will be used for quality_bitrates in the student_entrypoint call after that
	#       ...
	#
	quality_levels: int
	quality_bitrates: List[float]
	upcoming_quality_bitrates: List[List[float]]

	# You may use these to tune your algorithm to each user case! Remember, you can and should change these in the
	# config files to simulate different clients!
	#
	#   User Quality of Experience =    (Average chunk quality) * (Quality Coefficient) +
	#                                   -(Number of changes in chunk quality) * (Variation Coefficient)
	#                                   -(Amount of time spent rebuffering) * (Rebuffering Coefficient)
	#
	#   *QoE is then divided by total number of chunks
	#
	quality_coefficient: float
	variation_coefficient: float
	rebuffering_coefficient: float
# ======================================================================================================================


# Your helper functions, variables, classes here. You may also write initialization routines to be called
# when this script is first imported and anything else you wish.


def student_entrypoint(client_message: ClientMessage, mab_algo=None):
	"""
	Baseline adaptive bitrate algorithm.

	Args:
		client_message : ClientMessage holding the parameters for this chunk and current client state.

	:return: float Your quality choice. Must be one in the range [0 ... quality_levels - 1] inclusive.
	"""
	
	global last_quality, last_throughput

    # Step 1: Estimate available bandwidth
	if client_message.previous_throughput > 0:
		estimated_throughput = client_message.previous_throughput
	elif last_throughput is not None:
		estimated_throughput = last_throughput
	else:
		# fallback: pick the lowest bitrate initially
		estimated_throughput = client_message.quality_bitrates[0]

	last_throughput = estimated_throughput

	# Step 2: Compute a safe target bitrate based on buffer
	buffer_ratio = client_message.buffer_seconds_until_empty / client_message.buffer_max_size
	# If buffer is low, be conservative
	if buffer_ratio < 0.3:
		safety_factor = 0.6
	elif buffer_ratio < 0.6:
		safety_factor = 0.8
	else:
		safety_factor = 1.0

	safe_bandwidth = estimated_throughput * safety_factor

	# Step 3: Pick the highest quality below the safe bandwidth
	chosen_quality = 0
	for i, br in enumerate(client_message.quality_bitrates):
		if br <= safe_bandwidth:
			chosen_quality = i
		else:
			break

	# Optional: avoid large jumps
	if last_quality is not None:
		if abs(chosen_quality - last_quality) > 1:
			# limit to 1-step change
			if chosen_quality > last_quality:
				chosen_quality = last_quality + 1
			else:
				chosen_quality = last_quality - 1

	last_quality = chosen_quality
	return chosen_quality