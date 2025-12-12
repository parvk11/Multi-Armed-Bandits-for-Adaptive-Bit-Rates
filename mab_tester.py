import os
import simulator
from importlib import reload
import sys
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
TEST_DIRECTORY = "./tests"
STUDENT_DIR = "./student"
N_TRIALS = 10

student_files = [f.strip(".py") for f in os.listdir(STUDENT_DIR) if f.endswith(".py")]
print(f"Found student algorithms: {student_files}")

# ----------------------------------------------------------
# DISPLAY NAMES (PAPER-FRIENDLY)
# ----------------------------------------------------------
ALGO_DISPLAY_NAMES = {
    "ucb": "UCB",
    "ucb_sw": "Sliding-Window UCB",
    "ucb_dis_sum": "Discounted UCB",
    "ucb_cu_sum": "CUSUM-UCB",
    "epsilon_greed": "Epsilon-Greedy",
    "thompson": "Thompson Sampling",
    "hybrid_cumul_sum_ucb": "Hybrid CUSUM-UCB",
    "hybrid_sw_ucb": "Hybrid SW-UCB",
    "hybrid_dis_sum_ucb": "Hybrid Discounted UCB",
}

TEST_DISPLAY_NAMES = {
    "lo_avg_lo_var.ini": "Low BW / Low Variability",
    "lo_avg_mi_var.ini": "Low BW / Medium Variability",
    "lo_avg_hi_var.ini": "Low BW / High Variability",
    "mi_avg_lo_var.ini": "Medium BW / Low Variability",
    "mi_avg_mi_var.ini": "Medium BW / Medium Variability",
    "mi_avg_hi_var.ini": "Medium BW / High Variability",
    "hi_avg_lo_var.ini": "High BW / Low Variability",
    "hi_avg_mi_var.ini": "High BW / Medium Variability",
    "hi_avg_hi_var.ini": "High BW / High Variability",
}


def algo_name(a):
    return ALGO_DISPLAY_NAMES.get(a, a.upper())


def test_name(t):
    return TEST_DISPLAY_NAMES.get(t, t.replace(".ini", ""))


def ordered_tests():
    """Consistent left-to-right ordering for plots."""
    order = [
        "lo_avg_lo_var.ini", "lo_avg_mi_var.ini", "lo_avg_hi_var.ini",
        "med_avg_lo_var.ini", "med_avg_mi_var.ini", "med_avg_hi_var.ini",
        "hi_avg_lo_var.ini", "hi_avg_mi_var.ini", "hi_avg_hi_var.ini",
    ]
    return [t for t in order if t in os.listdir(TEST_DIRECTORY)]


# ----------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------
def plot_heatmap_and_bar(algorithm_results):
    algorithms = []
    avg_qoes = []
    test_names = ordered_tests()

    qoe_matrix = []

    for algo, (test_results, avg_qoe, _) in algorithm_results.items():
        algorithms.append(algo_name(algo))
        avg_qoes.append(avg_qoe)
        qoe_matrix.append([test_results[t][3] for t in test_names])

    qoe_matrix = np.array(qoe_matrix)

    # --- HEATMAP ---
    plt.figure(figsize=(12, 6))
    plt.imshow(qoe_matrix, aspect="auto")
    plt.colorbar(label="Average QoE")

    plt.xticks(
        np.arange(len(test_names)),
        [test_name(t) for t in test_names],
        rotation=30,
        ha="right"
    )
    plt.yticks(np.arange(len(algorithms)), algorithms)

    plt.xlabel("Network Scenario")
    plt.ylabel("Algorithm")
    plt.title("Average QoE Across Algorithms and Network Conditions")
    plt.tight_layout()
    plt.show()

    # --- BAR CHART ---
    plt.figure(figsize=(9, 5))
    plt.bar(algorithms, avg_qoes)
    plt.ylabel("Average QoE")
    plt.title("Overall Average QoE by Algorithm")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


def plot_qoe_cdf(algorithm_results):
    """
    CDF of *final episode QoE* values across all tests and trials.
    """
    plt.figure(figsize=(7, 5))

    for algo, (_, _, qoe_trials) in algorithm_results.items():
        all_qoes = []

        for test in qoe_trials:
            for episode_curve in qoe_trials[test]:
                all_qoes.append(np.sum(episode_curve))

        all_qoes = np.array(all_qoes)
        sorted_qoes = np.sort(all_qoes)
        cdf = np.arange(1, len(sorted_qoes) + 1) / len(sorted_qoes)

        plt.plot(sorted_qoes, cdf, label=algo_name(algo), linewidth=2)

    plt.xlabel("Total QoE per Episode")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Episode-Level QoE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# def plot_qoe_boxplot(algorithm_results):
#     labels = []
#     data = []

#     for algo, (_, _, qoe_trials) in algorithm_results.items():
#         episode_qoes = []

#         for test in qoe_trials:
#             for episode_curve in qoe_trials[test]:
#                 episode_qoes.append(np.sum(episode_curve))

#         labels.append(algo_name(algo))
#         data.append(np.array(episode_qoes))

#     plt.figure(figsize=(8, 5))
#     plt.boxplot(data, tick_labels=labels, showfliers=True)
#     plt.ylabel("Total QoE per Episode")
#     plt.title("QoE Distribution Across Algorithms")
#     plt.xticks(rotation=25, ha="right")
#     plt.grid(axis="y", alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def plot_qoe_boxplot(algorithm_results):
    """
    Boxplot of mean episode-level QoE.
    Each point = average QoE over all chunks in one episode (trial for a specific test case).
    """
    labels = []
    data = []

    for algo, (_, _, qoe_trials) in algorithm_results.items():
        episode_mean_qoes = []

        for test in qoe_trials:
            for episode_curve in qoe_trials[test]:
                episode_mean_qoes.append(np.mean(episode_curve))

        labels.append(algo_name(algo))
        data.append(np.array(episode_mean_qoes))

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.ylabel("Mean QoE per Episode")
    plt.title("Distribution of Episode-Level Mean QoE")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(algorithm_results, test):
    plt.figure(figsize=(8, 5))

    for algo, (_, _, qoe_trials) in algorithm_results.items():
        curves = qoe_trials[test]
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]

        mean_curve = np.mean(curves, axis=0)
        plt.plot(mean_curve, label=algo_name(algo), linewidth=2)

    plt.xlabel("Chunk Index")
    plt.ylabel("Instantaneous QoE")
    plt.title(f"QoE Learning Curve — {test_name(test)}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# MAIN EVALUATION LOOP
# ----------------------------------------------------------
def main(student_algo):
    print(f"\nRunning algorithm: {student_algo}")

    per_test_curves = {t: [] for t in ordered_tests()}
    results_sum = {t: np.zeros(4) for t in ordered_tests()}

    for trial in range(N_TRIALS):
        for test in ordered_tests():
            reload(simulator)
            quality, variation, rebuff, qoe, chunk_qoe = simulator.main(
                os.path.join(TEST_DIRECTORY, test),
                student_algo,
                False,
                False
            )

            results_sum[test] += np.array([quality, variation, rebuff, qoe])
            per_test_curves[test].append(chunk_qoe)

    test_results = {
        t: tuple(results_sum[t] / N_TRIALS)
        for t in results_sum
    }

    avg_qoe = np.mean([v[3] for v in test_results.values()])
    return test_results, avg_qoe, per_test_curves


# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mab_tester.py <student_algo.py> OR ALL")
        sys.exit(1)

    if sys.argv[1] == "ALL":
        algorithm_results = {}

        for student_algo in student_files:
            test_results, avg_qoe, curves = main(student_algo)
            print(f"Completed testing for {student_algo}: Average QoE = {avg_qoe:.2f}")
            algorithm_results[student_algo] = (test_results, avg_qoe, curves)

        plot_heatmap_and_bar(algorithm_results)
        plot_qoe_cdf(algorithm_results)
        plot_qoe_boxplot(algorithm_results)

        for test in ordered_tests():
            plot_learning_curves(algorithm_results, test)

    else:
        main(sys.argv[1])
