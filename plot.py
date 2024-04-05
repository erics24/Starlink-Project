import pickle
import numpy as np
import matplotlib.pyplot as plt


def analyze(log_file_list, label_list):
    assert len(log_file_list) == len(label_list)
    log_data_list = []
    for log_filepath in log_file_list:
        with open(log_filepath, "rb") as log_file:
            log_data = pickle.load(log_file)
            log_data_list.append(log_data)

    # Format print for table
    print("\nArea count:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{}".format(label_list[log_i], log_data["n_area"]))

    print("\nAverage ISL link count:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["isl_link_cnt_over_time"]),
                                    np.std(log_data["isl_link_cnt_over_time"])))

    print("\nAverage ISL link change rate:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["isl_link_changes_over_time"]),
                                    np.std(log_data["isl_link_changes_over_time"])))

    print("\nAverage sat-to-gateway link count:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["stg_link_cnt_over_time"]),
                                    np.std(log_data["stg_link_cnt_over_time"])))

    print("\nAverage sat-to-gateway link change rate:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["stg_link_changes_over_time"]),
                                    np.std(log_data["stg_link_changes_over_time"])))

    print("\nAverage sat-to-sat connectivity:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["isl_link_cnt_over_time"]) / log_data["n_sat"],
                                    np.std(log_data["isl_link_cnt_over_time"]) / log_data["n_sat"]))

    print("\nAverage sat-to-gateway connectivity:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["stg_link_cnt_over_time"]) / log_data["n_sat"],
                                    np.std(log_data["stg_link_cnt_over_time"]) / log_data["n_sat"]))

    print("\nAverage gateway-to-sat connectivity:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["stg_link_cnt_over_time"]) / log_data["n_gate"],
                                    np.std(log_data["stg_link_cnt_over_time"]) / log_data["n_gate"]))

    print("\nAverage inter-area connection count:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["area_conn_cnt_over_time"]),
                                    np.std(log_data["area_conn_cnt_over_time"])))

    print("\nAverage inter-area connection change rate:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["area_conn_changes_over_time"]),
                                    np.std(log_data["area_conn_changes_over_time"])))

    print("\nAverage intra-area update rate:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["intra_area_changes_over_time"]),
                                    np.std(log_data["intra_area_changes_over_time"])))

    print("\nMaximum node graph diameter:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{}".format(label_list[log_i], log_data["node_graph_diameter"]))

    print("\nMaximum area-level graph diameter:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{}".format(label_list[log_i], log_data["area_graph_diameter"]))

    print("\nAverage sat count per area:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["n_sat_per_area"]),
                                    np.std(log_data["n_sat_per_area"])))

    print("\nAverage gateway count per area:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["n_gate_per_area"]),
                                    np.std(log_data["n_gate_per_area"])))

    print("\nAverage area count assigned to each sat:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["n_area_per_sat"]),
                                    np.std(log_data["n_area_per_sat"])))

    print("\nAverage area count assigned to each gateway:")
    for log_i, log_data in enumerate(log_data_list):
        print("{}:\t{} ({})".format(label_list[log_i],
                                    np.mean(log_data["n_area_per_gate"]),
                                    np.std(log_data["n_area_per_gate"])))

    # Plot timeline figure
    # [1] ISL/STG link count over time
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of links')
    ax.plot(np.arange(len(log_data_list[0]["isl_link_cnt_over_time"])),
            log_data_list[0]["isl_link_cnt_over_time"], label='LISL Links', color='blue')
    ax.plot(np.arange(len(log_data_list[0]["isl_link_cnt_over_time"])),
            [np.mean(log_data_list[0]["isl_link_cnt_over_time"])] * len(log_data_list[0]["isl_link_cnt_over_time"]),
            "--", color='red')
    ax.legend()
    fig.savefig("sec_5_1_1_1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of links')
    ax.plot(np.arange(len(log_data_list[0]["stg_link_cnt_over_time"])),
            log_data_list[0]["stg_link_cnt_over_time"], label='Satellite-Gateway Links', color='orange')
    ax.plot(np.arange(len(log_data_list[0]["stg_link_cnt_over_time"])),
            [np.mean(log_data_list[0]["stg_link_cnt_over_time"])] * len(log_data_list[0]["stg_link_cnt_over_time"]),
            "--", color='red')
    ax.legend()
    fig.savefig("sec_5_1_1_2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # [2] Area connection count over time
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of inter-area connections')
    for log_i, log_data in enumerate(log_data_list):
        ax.plot(np.arange(len(log_data["area_conn_cnt_over_time"])),
                log_data["area_conn_cnt_over_time"], label=label_list[log_i])
    ax.legend()
    fig.savefig("sec_5_1_2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # [3] Area with routing table updates count over time
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of intra-area routing updates')
    ax.set_ylim(0, 50)
    for log_i, log_data in enumerate(log_data_list):
        ax.plot(np.arange(len(log_data["intra_area_changes_over_time"])),
                log_data["intra_area_changes_over_time"], label=label_list[log_i])
    ax.legend()
    fig.savefig("sec_5_1_3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot Histogram (Optional)
    # [1] #sat/gateway per area
    # [2] #area assigned per sat/gateway

    # Plot CDF
    # Also list latency/hop count mean and std
    # [1] all sat to nearest gateway: latency CDF, hop count CDF
    print("\nAverage latency between sat to nearest gateway:")
    line_style = ["-", "-.", "--"]
    fig, ax = plt.subplots()
    ax.set_xlabel('Latency (milliseconds)')
    ax.set_ylabel('Cumulative Probability')
    best_sat_to_gate_latency = log_data_list[0]["full_latency_mat"][:log_data_list[0]["n_sat"], log_data_list[0]["n_sat"]:]
    best_sat_to_nearest_gate_latency = np.min(best_sat_to_gate_latency, axis=1)
    print("{}:\t{} ({})".format("Theoretical Optimal",
                                np.mean(best_sat_to_nearest_gate_latency),
                                np.std(best_sat_to_nearest_gate_latency)))
    ax.ecdf(best_sat_to_nearest_gate_latency, ls=':', label="Theoretical Optimal", lw=0.5, color='red')
    for log_i, log_data in enumerate(log_data_list):
        sat_to_gate_latency = log_data["latency_mat"][:log_data["n_sat"], log_data["n_sat"]:]
        sat_to_nearest_gate_latency = np.min(sat_to_gate_latency, axis=1)
        print("{}:\t{} ({})\t{}% increase compared to optimal".format(
            label_list[log_i],
            np.mean(sat_to_nearest_gate_latency),
            np.std(sat_to_nearest_gate_latency),
            (np.mean(sat_to_nearest_gate_latency) / np.mean(best_sat_to_nearest_gate_latency) - 1.) * 100))
        ax.ecdf(sat_to_nearest_gate_latency, ls=line_style[log_i], label=label_list[log_i], lw=1.)
    ax.legend()
    fig.savefig("sec_5_2_1_1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nAverage hop count between sat to nearest gateway:")
    line_style = ["-", "-.", "--"]
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of hops')
    ax.set_ylabel('Cumulative Probability')
    best_sat_to_gate_hop = log_data_list[0]["full_hop_cnt_mat"][:log_data_list[0]["n_sat"], log_data_list[0]["n_sat"]:]
    best_sat_to_nearest_gate_hop = np.min(best_sat_to_gate_hop, axis=1)
    print("{}:\t{} ({})".format("Theoretical Optimal",
                                np.mean(best_sat_to_nearest_gate_hop),
                                np.std(best_sat_to_nearest_gate_hop)))
    ax.ecdf(best_sat_to_nearest_gate_hop, ls=':', label="Theoretical Optimal", lw=0.5, color='red')
    for log_i, log_data in enumerate(log_data_list):
        sat_to_gate_hop = log_data["hop_cnt_mat"][:log_data["n_sat"], log_data["n_sat"]:]
        sat_to_nearest_gate_hop = np.min(sat_to_gate_hop, axis=1)
        print("{}:\t{} ({})\t{}% increase compared to optimal".format(
            label_list[log_i],
            np.mean(sat_to_nearest_gate_hop),
            np.std(sat_to_nearest_gate_hop),
            (np.mean(sat_to_nearest_gate_hop) / np.mean(best_sat_to_nearest_gate_hop) - 1.) * 100))
        ax.ecdf(sat_to_nearest_gate_hop, ls=line_style[log_i], label=label_list[log_i], lw=1.)
    ax.legend()
    fig.savefig("sec_5_2_1_2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # [2] all sat to all sat: latency CDF, hop count CDF
    print("\nAverage latency between sat to sat:")
    line_style = ["-", "-.", "--"]
    fig, ax = plt.subplots()
    ax.set_xlabel('Latency (milliseconds)')
    ax.set_ylabel('Cumulative Probability')
    best_sat_to_sat_latency = log_data_list[0]["full_latency_mat"][:log_data_list[0]["n_sat"], :log_data_list[0]["n_sat"]]
    best_sat_to_sat_latency = best_sat_to_sat_latency[best_sat_to_sat_latency > 0.].flatten()
    print("{}:\t{} ({})".format("Theoretical Optimal",
                                np.mean(best_sat_to_sat_latency),
                                np.std(best_sat_to_sat_latency)))
    ax.ecdf(best_sat_to_sat_latency, ls=':', label="Theoretical Optimal", lw=0.5, color='red')
    for log_i, log_data in enumerate(log_data_list):
        sat_to_sat_latency = log_data["latency_mat"][:log_data["n_sat"], :log_data["n_sat"]]
        sat_to_sat_latency = sat_to_sat_latency[sat_to_sat_latency > 0.].flatten()
        print("{}:\t{} ({})\t{}% increase compared to optimal".format(
            label_list[log_i],
            np.mean(sat_to_sat_latency),
            np.std(sat_to_sat_latency),
            (np.mean(sat_to_sat_latency) / np.mean(best_sat_to_sat_latency) - 1.) * 100))
        ax.ecdf(sat_to_sat_latency, ls=line_style[log_i], label=label_list[log_i], lw=1.)
    ax.legend()
    fig.savefig("sec_5_2_2_1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nAverage hop count between sat to nearest gateway:")
    line_style = ["-", "-.", "--"]
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of hops')
    ax.set_ylabel('Cumulative Probability')
    best_sat_to_sat_hop = log_data_list[0]["full_hop_cnt_mat"][:log_data_list[0]["n_sat"], :log_data_list[0]["n_sat"]]
    best_sat_to_sat_hop = best_sat_to_sat_hop[best_sat_to_sat_hop > 0].flatten()
    print("{}:\t{} ({})".format("Theoretical Optimal",
                                np.mean(best_sat_to_sat_hop),
                                np.std(best_sat_to_sat_hop)))
    ax.ecdf(best_sat_to_sat_hop, ls=':', label="Theoretical Optimal", lw=0.5, color='red')
    for log_i, log_data in enumerate(log_data_list):
        sat_to_sat_hop = log_data["hop_cnt_mat"][:log_data["n_sat"], :log_data["n_sat"]]
        sat_to_sat_hop = sat_to_sat_hop[sat_to_sat_hop > 0].flatten()
        print("{}:\t{} ({})\t{}% increase compared to optimal".format(
            label_list[log_i],
            np.mean(sat_to_sat_hop),
            np.std(sat_to_sat_hop),
            (np.mean(sat_to_sat_hop) / np.mean(best_sat_to_sat_hop) - 1.) * 100))
        ax.ecdf(sat_to_sat_hop, ls=line_style[log_i], label=label_list[log_i], lw=1.)
    ax.legend()
    fig.savefig("sec_5_2_2_2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(per_sec_data, ylabel, fname):
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    p = ax.plot(np.arange(len(per_sec_data)), per_sec_data, '-')
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    log_files = ["log_1_0.pkl", "log_3_1.pkl", "log_1_1.pkl"]
    labels = ["100% Static", "75% Static, 25% Dynamic", "50% Static, 50% Dynamic"]
    analyze(log_files, labels)
