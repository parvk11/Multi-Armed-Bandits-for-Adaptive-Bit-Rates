import os
import simulator
from importlib import reload
import sys


TEST_DIRECTORY = './tests'

STUDENT_DIR = './student'

student_files = [f.strip('.py') for f in os.listdir(STUDENT_DIR) if f.endswith('.py')]

def main(student_algo):
	"""
	Runs simulator and student algorithm on all tests in TEST_DIRECTORY
	Args:
		student_algo : Student algorithm to run
		mab_algo: Multi-armed bandit algorithm to run
	"""

	# Run main loop, print output
	test_results = {}
	sum_qoe = 0
	print(f'\nTesting student algorithm {student_algo}')
	for test in os.listdir(TEST_DIRECTORY):
		reload(simulator)
		quality, variation, rebuff, qoe = simulator.main(os.path.join(TEST_DIRECTORY, test), student_algo, False, False)
		print(f'\tTest {test: <12}:'
			  f' Total Quality {quality:8.2f},'
			  f' Total Variation {variation:8.2f},'
			  f' Rebuffer Time {rebuff:8.2f},'
			  f' Total QoE {qoe:8.2f}')
		sum_qoe += qoe
		test_results[test] = (quality, variation, rebuff, qoe)
	
	average_qoe = sum_qoe / len(os.listdir(TEST_DIRECTORY))

	print(f'\n\tAverage QoE over all tests: {sum_qoe / len(os.listdir(TEST_DIRECTORY)):.2f}')
	return test_results, average_qoe

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print('Usage: python mab_tester.py <student_algorithm.py> OR python mab_tester.py ALL')
		exit()
	if sys.argv[1] == 'ALL':
		algorithm_results = {}
		for student_file in student_files:
			test_results, average_qoe = main(student_file)
			algorithm_results[student_file] = (test_results, average_qoe)
		print('\nSummary of all student algorithms:')
		for student_file, (test_results, average_qoe) in algorithm_results.items():
			print(f'\nStudent Algorithm: {student_file}')
			for test, (quality, variation, rebuff, qoe) in test_results.items():
				print(f'\tTest {test: <12}:'
					  f' Total Quality {quality:8.2f},'
					  f' Total Variation {variation:8.2f},'
					  f' Rebuffer Time {rebuff:8.2f},'
					  f' Total QoE {qoe:8.2f}')
			print(f'\tAverage QoE: {average_qoe:.2f}')
		
	