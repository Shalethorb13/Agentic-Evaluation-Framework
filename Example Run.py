if __name__ == "__main__":
    sample_data = [
        {
            "id": "A1",
            "prompt": "List three causes of latency in distributed databases and explain briefly.",
            "response": "Network congestion, disk I/O bottlenecks, and poor query design. "
                        "Congestion adds delays, disks slow writes, queries cause full scans."
        },
        {
            "id": "A2",
            "prompt": "Explain reservoir sampling with Python code.",
            "response": "Reservoir sampling picks k random items from a stream. "
                        "In Python, you maintain a list of k items and replace randomly "
                        "as the stream grows.",
            "reference": "Reservoir sampling: keep first k items, then for i>=k pick random j in [0,i]; "
                         "if j<k replace r[j]."
        },
        {
            "id": "A3",
            "prompt": "Summarize the novel 'The Lost City' in two sentences.",
            "response": "The Lost City is about explorers discovering ruins. "
                        "It claims discovery in 1820 by a fictional Sir John, which is unverified."
        },
    ]

    report, board = evaluate_responses(sample_data)

    print("Detailed Report")
    print(report.to_string(index=False))
    print("\nLeaderboard")
    print(board[["id", "composite_score"]].to_string(index=False))

    # Save outputs
    report.to_csv("agentic_report.csv", index=False)
    board.to_csv("agentic_leaderboard.csv", index=False)
    print("\nFiles saved: agentic_report.csv, agentic_leaderboard.csv")
