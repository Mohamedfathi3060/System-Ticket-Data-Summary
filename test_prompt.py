import sys
from llm_summary import _build_user_prompt, _build_system_prompt

tickets = [{"ORDER_NUMBER": "123", "ACCEPTANCE_TIME": "2024-01-01"}, {"ORDER_NUMBER": "124", "ACCEPTANCE_TIME": "2024-01-02", "CUSTOMER_COUNT": "1", "PROCESSING_STATUS": "Open"}]
print("--- SYSTEM PROMPT ---")
print(_build_system_prompt())
print("--- USER PROMPT ---")
print(_build_user_prompt("CUST-001", "Broadband", tickets))
