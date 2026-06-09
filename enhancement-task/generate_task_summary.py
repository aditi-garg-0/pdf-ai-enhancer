import pandas as pd
from pathlib import Path


class TaskSummaryGenerator:

    INPUT_FILE = "taskout.xlsx"
    OUTPUT_FILE = "tasksummary.xlsx"

    REQUIRED_COLUMNS = [
        "Process",
        "Task Name",
        "Task Date",
        "Completed On Time"
    ]

    @classmethod
    def validate_columns(cls, dataframe):

        missing_columns = (
            set(cls.REQUIRED_COLUMNS)
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: "
                f"{', '.join(missing_columns)}"
            )

    @classmethod
    def clean_data(cls, dataframe):

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        cls.validate_columns(dataframe)

        dataframe["Process"] = (
            dataframe["Process"]
            .ffill()
        )

        dataframe = dataframe[
            dataframe["Task Name"].notna()
        ]

        dataframe = dataframe[
            dataframe["Task Name"]
            .astype(str)
            .str.strip()
            .ne("")
        ]

        dataframe = dataframe[
            ~dataframe["Process"]
            .astype(str)
            .str.startswith(
                "Total",
                na=False
            )
        ]

        dataframe["Process"] = (
            dataframe["Process"]
            .astype(str)
            .str.strip()
        )

        dataframe["Task Name"] = (
            dataframe["Task Name"]
            .astype(str)
            .str.strip()
        )

        dataframe["Completed On Time"] = (
            dataframe["Completed On Time"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        return dataframe

    @classmethod
    def generate_summary(cls, dataframe):

        summary = (
            dataframe
            .groupby(
                [
                    "Process",
                    "Task Name",
                    "Completed On Time"
                ]
            )
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        if "ON TIME" not in summary.columns:
            summary["ON TIME"] = 0

        if "DELAYED" not in summary.columns:
            summary["DELAYED"] = 0

        summary = summary.rename(
            columns={
                "Process": "ProcessGroupName",
                "DELAYED": "DelayedCount",
                "ON TIME": "OnTimeCount"
            }
        )

        summary = summary[
            [
                "ProcessGroupName",
                "Task Name",
                "DelayedCount",
                "OnTimeCount"
            ]
        ]

        summary = summary.sort_values(
            by=[
                "ProcessGroupName",
                "Task Name"
            ]
        )

        summary.reset_index(
            drop=True,
            inplace=True
        )

        return summary

    @classmethod
    def write_output(
        cls,
        summary_dataframe
    ):

        with pd.ExcelWriter(
            cls.OUTPUT_FILE,
            engine="openpyxl"
        ) as writer:

            summary_dataframe.to_excel(
                writer,
                sheet_name="Task Summary",
                index=False
            )

    @classmethod
    def run(cls):

        input_path = Path(
            cls.INPUT_FILE
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: "
                f"{cls.INPUT_FILE}"
            )

        dataframe = pd.read_excel(
            cls.INPUT_FILE
        )

        dataframe = cls.clean_data(
            dataframe
        )

        summary = cls.generate_summary(
            dataframe
        )

        cls.write_output(
            summary
        )

        print("\nTASK SUMMARY GENERATED\n")
        print(summary)
        print(
            f"\nOutput File : "
            f"{cls.OUTPUT_FILE}"
        )


if __name__ == "__main__":
    TaskSummaryGenerator.run()