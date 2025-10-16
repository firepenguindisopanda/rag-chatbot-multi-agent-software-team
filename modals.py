import json
from typing import Optional

import streamlit as st

from assignment_evaluator.google_sheets import SheetsClient, check_gspread_available


def show_sheets_settings_modal(prefix: Optional[str] = None) -> None:
    """Render a reusable settings 'modal' form for Google Sheets settings.

    This uses a form and session_state to approximate a modal dialog in Streamlit
    versions that lack the native `st.modal` API.
    """
    key_prefix = f"{prefix}_" if prefix else "gs_"
    form_key = f"gs_form_{prefix or 'default'}"

    st.markdown("### Google Sheets Settings")
    with st.form(form_key):
        sa_input_type = st.radio("Service Account Input", ["Paste JSON", "Upload JSON file"], index=0, key=f"{form_key}_sa_type")
        if sa_input_type == "Paste JSON":
            sa_text = st.text_area("Service Account JSON", value=st.session_state.get('gs_service_account',''), height=180, key=f"{form_key}_sa_text")
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

        sid = st.text_input("Spreadsheet ID", value=st.session_state.get('gs_spreadsheet_id',''), key=f"{form_key}_sid")
        sws = st.text_input("Worksheet name", value=st.session_state.get('gs_worksheet','Sheet1'), key=f"{form_key}_sws")

        st.markdown("---")
        st.markdown("#### Column mapping (JSON). Map column_name -> dotted path in report or 'file' for filename")
        mapping_text = st.text_area("Column mapping", value=st.session_state.get('gs_column_mapping','{"student_id":"submission.student_id","filename":"file","avg_score":"avg_score"}'), height=120, key=f"{form_key}_mapping")

        submitted = st.form_submit_button("Save Settings")
        if submitted:
            # Persist into session state and rerun so main UI picks up values
            st.session_state['gs_spreadsheet_id'] = st.session_state.get(f"{form_key}_sid", sid)
            st.session_state['gs_worksheet'] = st.session_state.get(f"{form_key}_sws", sws)
            st.session_state['gs_column_mapping'] = st.session_state.get(f"{form_key}_mapping", mapping_text)
            st.success("Settings saved")
            st.experimental_rerun()

    # Preflight check button outside the form area (so it can be used independently)
    if st.button("Preflight Check (Validate access)", key=f"{form_key}_preflight"):
        try:
            if not check_gspread_available():
                st.error("gspread/google-auth are not installed in this environment.")
                return

            info = json.loads(st.session_state.get('gs_service_account', '{}'))
            client = SheetsClient.from_service_account_info(info)
            sid = st.session_state.get('gs_spreadsheet_id') or st.session_state.get(f"{form_key}_sid")
            if not sid:
                st.error("Spreadsheet ID is empty")
                return
            ok = client.validate_spreadsheet_access(sid)
            if ok:
                st.success("✅ Service account can access the spreadsheet")
            else:
                st.error("❌ Service account cannot access the spreadsheet - check spreadsheet ID and sharing")
        except Exception as e:
            st.error(f"Preflight check failed: {e}")
