#!/usr/bin/env bash
cd "$(dirname "$0")"
streamlit run app/main.py --server.maxUploadSize 2000
