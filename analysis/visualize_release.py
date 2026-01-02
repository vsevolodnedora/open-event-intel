import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_slack_log(filepath: str) -> pd.DataFrame:
    # 1) Read file
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    # 2) How we detect the start of a new event/message
    def is_event_start(line: str) -> bool:
        return (
            ":red_circle:" in line
            or ":white_check_mark:" in line
            or "Successful bidding for" in line
        )

    # 3) Group lines into event blocks
    events = []
    current_block = None

    for line in lines:
        if is_event_start(line):
            # Close previous block (if any)
            if current_block is not None:
                events.append(current_block)
            current_block = [line]
        else:
            # Continue current block
            if current_block is not None:
                current_block.append(line)

    # Append the last one
    if current_block is not None:
        events.append(current_block)

    # 4) Classify block and extract date + error flag
    def classify_event(block_lines):
        block = "\n".join(block_lines)

        # ---- event_name classification ----
        if "Strategy execution check failed for check_strategy_status" in block:
            event_name = "Strategy Execution Failed"
        elif "idc_mvp_preprocessing failed in task preprocess_idc_mvp_data" in block:
            event_name = "IDC MVP Preprocessing Failed"
        elif "Retrieved IDA1-Granularity.QUARTER_HOURLY auction" in block:
            event_name = "Successful Data Retrieval"
        elif "aggregate_bidfiles succeeded" in block:
            event_name = "Aggregation Successful"
        elif "disaggregate_auction_results" in block:
            if "failed" in block:
                event_name = "Disaggregation Failed"
            elif "succeeded" in block:
                event_name = "Disaggregation Successful"
            else:
                event_name = "Disaggregation"
        elif "Successful bidding for" in block:
            event_name = "Bidding Successful"
        elif "failed in task check_positions" in block:
            event_name = "Position Check Failed"
        else:
            # Fallbacks in case a new pattern appears
            if ":red_circle:" in block:
                event_name = "Unknown Error"
            elif ":white_check_mark:" in block:
                event_name = "Unknown Success"
            else:
                event_name = "Unknown"

        # ---- date extraction: prefer 'Execution Date', else 'Delivery Day' ----
        m_exec = re.search(r"Execution Date:\s*(.*)", block)
        if m_exec:
            date_str = m_exec.group(1).strip()
            date = pd.to_datetime(date_str, errors="coerce")
        else:
            m_deliv = re.search(r"Delivery Day:\s*(.*)", block)
            if m_deliv:
                date_str = m_deliv.group(1).strip()
                # offset by 1 day back (e.g., -1 day)
                date = pd.to_datetime(date_str, errors="coerce") - pd.Timedelta(days=1)
            else:
                date = pd.NaT

        # ---- error flag ----
        is_error = ":red_circle:" in block

        return event_name, date, is_error

    rows = []
    for ev in events:
        event_name, date, is_error = classify_event(ev)
        rows.append(
            {
                "event_name": event_name,
                "date": date,          # datetime64[ns]
                "is_error": is_error,  # True if :red_circle: present
            }
        )

    df = pd.DataFrame(rows, columns=["event_name", "date", "is_error"])
    return df


# Point this to your Slack-export .txt file
input_path = "slack_channel.txt"

df = parse_slack_log(input_path)

# --- Plot: number of errors per day with weekend shading ---

# Ensure date is datetime (no-op if already correct)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Aggregate to daily error counts (including zeros)
daily_errors = (
    df.set_index("date")["is_error"]
    .resample("D")
    .sum()
    .fillna(0)
    .astype(int)
)

fig, ax = plt.subplots(figsize=(10, 5))

# Line plot of error counts
ax.plot(daily_errors.index, daily_errors.values, marker="o", linewidth=2)

# Gray shaded areas for weekends
start = daily_errors.index.min().normalize()
end = daily_errors.index.max().normalize()
current = start

while current <= end:
    if current.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        ax.axvspan(
            current,
            current + pd.Timedelta(days=1),
            facecolor="lightgray",
            alpha=0.3,
        )
    current += pd.Timedelta(days=1)

ax.set_title("Number of Errors per Day")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Errors")
ax.grid(True, which="major", linestyle="--", alpha=0.5)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Show the table in the console
print(df.to_string(index=False))

# Optionally dump to CSV for further analysis
df.to_csv("slack_events.csv", index=False)
print("\nSaved to slack_events.csv")
