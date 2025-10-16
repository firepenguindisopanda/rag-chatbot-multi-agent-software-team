"""Google Sheets helper for exporting evaluation reports.

This module provides a thin wrapper around gspread + google-auth to append rows
or JSON blobs to a Google Sheets worksheet. It expects a Google Service Account
JSON credentials file (or pasted JSON) and requires the spreadsheet ID and the
worksheet name.

Usage:
    from assignment_evaluator.google_sheets import SheetsClient

    client = SheetsClient.from_service_account_info(json.loads(sa_json))
    client.append_evaluation_row(spreadsheet_id, worksheet_name, row_values)

Note: This file intentionally keeps the API surface small for maintainability.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    # The project may not have dependencies installed in this environment.
    gspread = None  # type: ignore
    Credentials = None  # type: ignore


class SheetsClient:
    SCOPE = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]

    def __init__(self, client: Any):
        self.client = client

    @classmethod
    def from_service_account_file(cls, file_path: str) -> "SheetsClient":
        if gspread is None:
            raise RuntimeError("gspread/google-auth not installed")
        creds = Credentials.from_service_account_file(file_path, scopes=cls.SCOPE)
        gc = gspread.authorize(creds)
        return cls(gc)

    @classmethod
    def from_service_account_info(cls, info: Dict[str, Any]) -> "SheetsClient":
        if gspread is None:
            raise RuntimeError("gspread/google-auth not installed")
        creds = Credentials.from_service_account_info(info, scopes=cls.SCOPE)
        gc = gspread.authorize(creds)
        return cls(gc)

    def append_evaluation_row(self, spreadsheet_id: str, worksheet_name: str, row: List[Any]) -> None:
        """Append a single row to the worksheet (creates worksheet if missing).

        Row should be a list of values matching the expected columns.
        """
        sh = self.client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

        ws.append_row(row, value_input_option="USER_ENTERED")

    def append_json(self, spreadsheet_id: str, worksheet_name: str, data: Dict[str, Any]) -> None:
        """Append a JSON blob by flattening to key/value pairs in a row.

        This is a simple serializer: keys become column headers if the sheet is empty.
        """
        sh = self.client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

        # Ensure header
        existing = ws.get_all_values()
        if not existing:
            headers = list(data.keys())
            ws.append_row(headers, value_input_option="USER_ENTERED")
            values = [data.get(h, "") for h in headers]
            ws.append_row(values, value_input_option="USER_ENTERED")
        else:
            headers = existing[0]
            values = [data.get(h, "") for h in headers]
            ws.append_row(values, value_input_option="USER_ENTERED")

    def validate_spreadsheet_access(self, spreadsheet_id: str) -> bool:
        """Validate that the service account can open the spreadsheet and list worksheets."""
        try:
            sh = self.client.open_by_key(spreadsheet_id)
            _ = sh.worksheets()
            return True
        except Exception:
            return False

    def append_student_rows(self, spreadsheet_id: str, worksheet_name: str, report: Dict[str, Any], column_mapping: Dict[str, str]) -> None:
        """Append rows for each student/file using a column mapping.

        column_mapping: mapping of column_name -> dotted path into report or a small expression
        Example mapping:
            {
                'student_id': 'submission.student_id',
                'filename': 'file',
                'correctness_score': 'results.correctness.llm_result.score'
            }
        The implementation supports simple dotted paths and will produce a header row if sheet is empty.
        """
        sh = self.client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=max(10, len(column_mapping)))

        # Ensure header
        existing = ws.get_all_values()
        headers = list(column_mapping.keys())
        if not existing:
            ws.append_row(headers, value_input_option="USER_ENTERED")

        # Prepare rows from report - if report contains multiple student files, iterate
        # We try to detect student files list; fallback to a single blob
        rows = []

        # If report contains a 'files' or 'submission_files' key, use that
        files_list = None
        if isinstance(report.get('submission'), dict) and report['submission'].get('files'):
            files_list = report['submission']['files']
        elif isinstance(report.get('files'), list):
            files_list = report['files']

        def _resolve_path(ctx: Dict[str, Any], path: str):
            parts = path.split('.') if path else []
            cur = ctx
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return ''
            return cur if cur is not None else ''

        if files_list:
            for f in files_list:
                row = []
                # Build a minimal context for path resolution where 'file' points to filename
                ctx = dict(report)
                ctx['file'] = f
                for col, path in column_mapping.items():
                    val = _resolve_path(ctx, path)
                    row.append(val)
                rows.append(row)
        else:
            # Single-row export - mapping resolved against whole report
            row = []
            for col, path in column_mapping.items():
                val = _resolve_path(report, path)
                row.append(val)
            rows.append(row)

        # Append rows
        for r in rows:
            ws.append_row(r, value_input_option="USER_ENTERED")


def check_gspread_available() -> bool:
    return gspread is not None and Credentials is not None
