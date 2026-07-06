import csv
from pathlib import Path

from openpyxl import Workbook


def table_to_markdown(headers, rows) -> str:
    if not headers:
        return ""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def save_table_xlsx(headers, rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    if headers:
        ws.append(headers)
    for row in rows:
        ws.append(row)
    wb.save(path)


def save_table_csv(headers, rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(rows)
