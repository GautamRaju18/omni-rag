"""Streamlit entry point.

uv run streamlit run src/omnirag/frontend/app.py    # the API must run (OMNIRAG_FRONTEND_API_URL)
"""

from omnirag.frontend.pages import main

main()
