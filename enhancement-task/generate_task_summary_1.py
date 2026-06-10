import pandas as pd

INPUT_FILE = "taskout1.xlsx"
OUTPUT_FILE = "tasksummary.xlsx"

# Read source file
df = pd.read_excel(INPUT_FILE)

# Clean column names
df.columns = df.columns.str.strip()

# Remove blank task names
df = df[df["Task Name"].notna()]
df = df[df["Task Name"].astype(str).str.strip() != ""]

# Standardize status values
df["Completed On Time"] = (
    df["Completed On Time"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Aggregate counts
summary = (
    df.groupby(
        ["Task Name", "Completed On Time"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Ensure columns exist
if "DELAYED" not in summary.columns:
    summary["DELAYED"] = 0

if "ON TIME" not in summary.columns:
    summary["ON TIME"] = 0

# Rename columns
summary = summary.rename(
    columns={
        "DELAYED": "DelayedCount",
        "ON TIME": "OnTimeCount"
    }
)

# Keep only required columns
summary = summary[
    [
        "Task Name",
        "DelayedCount",
        "OnTimeCount"
    ]
]

# Sort by task name
summary = summary.sort_values("Task Name")

# Remove ugly pandas column header
summary.columns.name = None

# Write output
summary.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\nTASK SUMMARY GENERATED\n")
print(summary.to_string(index=False))
print(f"\nOutput File : {OUTPUT_FILE}")