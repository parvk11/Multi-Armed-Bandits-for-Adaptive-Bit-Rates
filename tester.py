import os
import simulator
from importlib import reload
import sys

TEST_DIRECTORY = './tests'


def main(student_algo: str, mab_algo):
    """
    Runs simulator and student algorithm on all tests in TEST_DIRECTORY
    Args:
        student_algo : Student algorithm to run
        mab_algo: Multi-armed bandit algorithm to run
    """

    # Run main loop, print output
    sum_qoe = 0
    print(f'\nTesting student algorithm {student_algo} with MAB algorithm {mab_algo}')
    for test in os.listdir(TEST_DIRECTORY):
        reload(simulator)
        quality, variation, rebuff, qoe = simulator.main(os.path.join(TEST_DIRECTORY, test), student_algo, mab_algo, False, False)
        print(f'\tTest {test: <12}:'
              f' Total Quality {quality:8.2f},'
              f' Total Variation {variation:8.2f},'
              f' Rebuffer Time {rebuff:8.2f},'
              f' Total QoE {qoe:8.2f}')
        sum_qoe += qoe

    print(f'\n\tAverage QoE over all tests: {sum_qoe / len(os.listdir(TEST_DIRECTORY)):.2f}')


if __name__ == "__main__":
    assert len(sys.argv) >= 3 and (sys.argv[2] in ['CUSUM-UCB', 'SlidingWindow-UCB', 'Discounted-UCB'] or sys.argv[2] == 'RUN_ALL'), f'Proper usage: python3 {sys.argv[0]} [student_algo or RUN_ALL] [MAB_algo or RUN_ALL]'
    if sys.argv[1] != 'RUN_ALL' and sys.argv[2] != 'RUN_ALL':
        main(sys.argv[1], mab_algo=sys.argv[2])
    elif sys.argv[1] != 'RUN_ALL':
        for mab_algo in ['CUSUM-UCB', 'SlidingWindow-UCB', 'Discounted-UCB']:
            print(f'\n\nRunning student algorithm {sys.argv[1]} with MAB algorithm {mab_algo}:\n')
            main(sys.argv[1], mab_algo=mab_algo)
    else:
        for mab_algo in ['CUSUM-UCB', 'SlidingWindow-UCB', 'Discounted-UCB']:
            print(f'\n\nRunning all student algorithms with MAB algorithm {mab_algo}:\n')
            for algo in os.listdir('./student'):
                if algo[:len('student')] != 'student':
                    continue
                name = algo[len('student'):].split('.')[0]
                # mab_algo = None if name == '2' else mab_algo
                main(name, mab_algo=mab_algo)