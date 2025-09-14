demo = [
    {
        "id": "A1",
        "prompt": "List three causes of latency in distributed databases and explain briefly.",
        "response": "Network congestion, disk I/O bottlenecks, and poor query design. Congestion adds delays, disks slow writes, queries cause scans."
    },
    {
        "id": "A2",
        "prompt": "Explain reservoir sampling with Python code.",
        "response": "Reservoir sampling picks k random items from a stream. In Python, you maintain a list of k items and replace randomly as the stream grows.",
        "reference": "Reservoir sampling: keep first k items, then for i>=k pick j in [0,i]; if j<k replace r[j]."
    },
    {
        "id": "A3",
        "prompt": "Summarize the novel 'The Lost City' in two sentences.",
        "response": "The Lost City is about explorers discovering ruins. It claims discovery in 1820 by a fictional Sir John, which is unverified."
    }
]

# Initialize evaluator and generate report
evaluator = AgenticEvaluator(demo, ref_key="reference")
report_df, leaderboard_df = evaluator.evaluate_all()

print("Detailed Report")
print(report_df)
print("\nLeaderboard")
print(leaderboard_df[["id", "composite_score"]])
    report.to_csv("agentic_report.csv", index=False)
    board.to_csv("agentic_leaderboard.csv", index=False)
    print("\nFiles saved: agentic_report.csv, agentic_leaderboard.csv")
