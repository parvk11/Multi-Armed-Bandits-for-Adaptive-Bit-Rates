## Class Structure

	total_seconds_elapsed: float	  # The number of simulated seconds elapsed in this test
	previous_throughput: float		  # The measured throughput for the previous chunk in kB/s

	buffer_current_fill: float		    # The number of kB currently in the client buffer
	buffer_seconds_per_chunk: float     # Number of seconds that it takes the client to watch a chunk. Every
										# buffer_seconds_per_chunk, a chunk is consumed from the client buffer.
	buffer_seconds_until_empty: float   # The number of seconds of video left in the client buffer. A chunk must
										# be finished downloading before this time to avoid a rebuffer event.
	buffer_max_size: float              # The maximum size of the client buffer. If the client buffer is filled beyond
										# maximum, then download will be throttled until the buffer is no longer full
	quality_levels is an integer reflecting the # of quality levels you may choose from.
	quality_bitrates is a list of floats specifying the number of kilobytes the upcoming chunk is at each quality level. Quality level 2 always costs twice as much as quality level 1, quality level 3 is twice as big as 2, and
	so on.
	    - quality_bitrates[0] = kB cost for quality level 1
		- quality_bitrates[1] = kB cost for quality level 2
	
	  upcoming_quality_bitrates is a list of quality_bitrates for future chunks. Each entry is a list of
	  quality_bitrates that will be used for an upcoming chunk. Use this for algorithms that look forward multiple chunks in the future. Will shrink and eventually become empty as streaming approaches the end of the video.
	- upcoming_quality_bitrates[0]: Will be used for quality_bitrates in the next student_entrypoint call
	-upcoming_quality_bitrates[1]: Will be used for quality_bitrates in the student_entrypoint call after tha
	
	quality_levels: int
	quality_bitrates: List[float]
	upcoming_quality_bitrates: List[List[float]]


	## 

	#   User Quality of Experience =    (Average chunk quality) * (Quality Coefficient) +
	#                                   -(Number of changes in chunk quality) * (Variation Coefficient)
	#                                   -(Amount of time spent rebuffering) * (Rebuffering Coefficient)
	#
	#   *QoE is then divided by total number of chunks
	#
	quality_coefficient: float
	variation_coefficient: float
	rebuffering_coefficient: float