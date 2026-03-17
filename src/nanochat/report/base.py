import re
import shutil
import os

import datetime

from nanochat.report.utils import slugify, extract_timestamp, EXPECTED_FILES, extract, chat_metrics, generate_header


class BaseReport:
    """Base class for report generation.

    Provides the directory structure and no-op implementations of log, reset,
    and generate. Subclasses override these to produce actual output.
    """
    def __init__(self, report_dir: str) -> None:
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def _log_path(self, section: str) -> str:
        return os.path.join(self.report_dir, f"{slugify(section)}.md")

    def _report_path(self) -> str:
        return os.path.join(self.report_dir, "report.md")

    def log(self, section: str, data: list[object]) -> str:
        return self._log_path(section)

    def reset(self):
        pass

    def generate(self) -> str:
        return self._report_path()


class Report(BaseReport):
    """Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir: str) -> None:
        super().__init__(report_dir)
        
    def log(self, section: str, data: list[object]) -> str:
        """Log a section of data to the report."""
        file_path = self._log_path(section)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def _read_header(self, report_dir: str) -> tuple[datetime.datetime | None, str, str]:
        """Read header.md and return (start_time, bloat_data, content)."""
        header_file = os.path.join(report_dir, "header.md")
        if not os.path.exists(header_file):
            print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?")
            return None, "[bloat data missing]", ""
        with open(header_file, "r", encoding="utf-8") as f:
            content = f.read()
        start_time = extract_timestamp(content, "Run started:")
        bloat_match = re.search(r"### Bloat\n(.*?)\n\n", content, re.DOTALL)
        bloat_data = bloat_match.group(1) if bloat_match else ""
        return start_time, bloat_data, content

    def _process_sections(self, out_file, report_dir: str) -> tuple[datetime.datetime | None, dict]:
        """Write each section to out_file, return (end_time, final_metrics)."""
        end_time = None
        final_metrics = {}
        for file_name in EXPECTED_FILES:
            section_file = os.path.join(report_dir, file_name)
            if not os.path.exists(section_file):
                print(f"Warning: {section_file} does not exist, skipping")
                continue
            with open(section_file, "r", encoding="utf-8") as f:
                section = f.read()
            if "rl" not in file_name:
                end_time = extract_timestamp(section, "timestamp:")
            if file_name == "base-model-evaluation.md":
                final_metrics["base"] = extract(section, "CORE")
            if file_name == "chat-evaluation-sft.md":
                final_metrics["sft"] = extract(section, chat_metrics)
            if file_name == "chat-evaluation-rl.md":
                final_metrics["rl"] = extract(section, "GSM8K")
            out_file.write(section)
            out_file.write("\n")
        return end_time, final_metrics

    def _write_summary(self, out_file, final_metrics: dict, bloat_data: str, start_time, end_time) -> None:
        """Write the summary table and wall clock time to out_file."""
        out_file.write("## Summary\n\n")
        out_file.write(bloat_data)
        out_file.write("\n\n")
        all_metrics = set()
        for stage_metrics in final_metrics.values():
            all_metrics.update(stage_metrics.keys())
        all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
        stages = ["base", "sft", "rl"]
        metric_width = 15
        value_width = 8
        header = f"| {'Metric'.ljust(metric_width)} |"
        for stage in stages:
            header += f" {stage.upper().ljust(value_width)} |"
        out_file.write(header + "\n")
        separator = f"|{'-' * (metric_width + 2)}|"
        for stage in stages:
            separator += f"{'-' * (value_width + 2)}|"
        out_file.write(separator + "\n")
        for metric in all_metrics:
            row = f"| {metric.ljust(metric_width)} |"
            for stage in stages:
                value = final_metrics.get(stage, {}).get(metric, "-")
                row += f" {str(value).ljust(value_width)} |"
            out_file.write(row + "\n")
        out_file.write("\n")
        if start_time and end_time:
            total_seconds = int((end_time - start_time).total_seconds())
            hours, minutes = total_seconds // 3600, (total_seconds % 3600) // 60
            out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
        else:
            out_file.write("Total wall clock time: unknown\n")

    def generate(self) -> str:
        """Generate the final report."""
        report_file = self._report_path()
        print(f"Generating report to {report_file}")
        start_time, bloat_data, header_content = self._read_header(self.report_dir)
        with open(report_file, "w", encoding="utf-8") as out_file:
            out_file.write(header_content if header_content else "")
            end_time, final_metrics = self._process_sections(out_file, self.report_dir)
            self._write_summary(out_file, final_metrics, bloat_data, start_time, end_time)
        print("Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """Reset the report."""
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove report.md if it exists
        report_file = self._report_path()
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")

