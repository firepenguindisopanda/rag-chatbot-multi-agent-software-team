import json
from typing import Optional

import streamlit as st

from assignment_evaluator.google_sheets import SheetsClient, check_gspread_available


@st.dialog("Google Sheets Settings", width="medium")
def sheets_settings_dialog(prefix: Optional[str] = None) -> None:
    """Streamlit dialog for Google Sheets settings.

    Uses Session State for persistence. This replaces the old non-modal form so
    the UI opens a true modal dialog (Streamlit >=1.50).
    """
    form_key = f"gs_form_{prefix or 'default'}"

    sa_input_type = st.radio("Service Account Input", ["Paste JSON", "Upload JSON file"], index=0, key=f"{form_key}_sa_type")
    if sa_input_type == "Paste JSON":
        sa_text = st.text_area("Service Account JSON", value=st.session_state.get('gs_service_account', ''), height=180, key=f"{form_key}_sa_text")
        st.session_state['gs_service_account'] = sa_text
    else:
        sa_file = st.file_uploader("Upload Service Account JSON", type=["json"], key=f"{form_key}_sa_file")
        if sa_file:
            try:
                sa_text = sa_file.read().decode('utf-8')
                st.session_state['gs_service_account'] = sa_text
                st.success("Service account JSON loaded")
            except Exception as e:
                st.error(f"Failed to read service account file: {e}")

    sid = st.text_input("Spreadsheet ID", value=st.session_state.get('gs_spreadsheet_id', ''), key=f"{form_key}_sid")
    sws = st.text_input("Worksheet name", value=st.session_state.get('gs_worksheet', 'Sheet1'), key=f"{form_key}_sws")

    st.markdown("---")
    st.markdown("#### Column mapping (JSON). Map column_name -> dotted path in report or 'file' for filename")
    mapping_text = st.text_area("Column mapping", value=st.session_state.get('gs_column_mapping', '{"student_id":"file_metrics.student_id","filename":"file","avg_score":"file_metrics.avg_score","ai_confidence":"file_metrics.ai_confidence"}'), height=120, key=f"{form_key}_mapping")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("Save"):
            # persist and close dialog
            st.session_state['gs_spreadsheet_id'] = st.session_state.get(f"{form_key}_sid", sid)
            st.session_state['gs_worksheet'] = st.session_state.get(f"{form_key}_sws", sws)
            st.session_state['gs_column_mapping'] = st.session_state.get(f"{form_key}_mapping", mapping_text)
            st.success("Settings saved")
            # closing dialog will happen on rerun since dialog function returns
            st.rerun()
    with cols[1]:
        if st.button("Preflight Check"):
            # Validate gspread presence early
            if not check_gspread_available():
                st.error("gspread/google-auth are not installed in this environment.")
            else:
                # Try to parse provided SA JSON; fall back to local creds.json if empty
                sa_json = st.session_state.get('gs_service_account', '').strip()
                info = None
                try:
                    if sa_json:
                        info = json.loads(sa_json)
                    else:
                        # Try to read creds.json file in project root
                        try:
                            with open('creds.json', 'r', encoding='utf-8') as f:
                                info = json.load(f)
                                st.info("Using creds.json from project root for preflight")
                        except FileNotFoundError:
                            st.error("No service account JSON provided and creds.json not found in project.")
                except Exception as e:
                    st.error(f"Failed to parse service account JSON: {e}")
                    info = None

                if info:
                    try:
                        client = SheetsClient.from_service_account_info(info)
                        sid_val = st.session_state.get('gs_spreadsheet_id', '') or st.session_state.get(f"{form_key}_sid", '')
                        if not sid_val:
                            st.error("Spreadsheet ID is empty")
                        else:
                            ok = client.validate_spreadsheet_access(sid_val)
                            if ok:
                                st.success("✅ Service account can access the spreadsheet")
                            else:
                                st.error("❌ Service account cannot access the spreadsheet - check spreadsheet ID and sharing")
                    except Exception as e:
                        st.error(f"Preflight check failed: {e}")
    with cols[2]:
        if st.button("Close"):
            st.rerun()


def show_sheets_settings_modal(prefix: Optional[str] = None) -> None:
    """Helper to open the dialog (keeps existing import usage)."""
    # Only one dialog can be open per run; call the dialog function which will render a modal
    sheets_settings_dialog(prefix)
