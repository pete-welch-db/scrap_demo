"""
Databricks Genie API Client
============================
Client for interacting with Databricks Genie conversational AI.

API Reference: https://docs.databricks.com/api/azure/workspace/genie
"""

import requests
import time
import streamlit as st
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from config import databricks_config, genie_config


@dataclass
class GenieResponse:
    """Structured response from Genie."""
    content: str  # The text answer
    sql_query: Optional[str] = None  # The generated SQL
    query_description: Optional[str] = None  # Description of the query
    data_table: Optional[str] = None  # Formatted markdown table
    row_count: int = 0
    suggested_questions: Optional[List[str]] = None


class GenieClient:
    """Client for Databricks Genie API."""

    def __init__(self):
        from config import app_config
        import os

        self.base_url = f"https://{databricks_config.host}/api/2.0/genie/spaces"
        self.sql_base_url = f"https://{databricks_config.host}/api/2.0/sql/statements"
        self.space_id = genie_config.space_id

        # Get token based on environment
        token = self._get_token(app_config)

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.conversation_id: Optional[str] = None
        self.debug = False

    def _get_token(self, app_config) -> str:
        """Get authentication token based on environment."""
        import os

        if app_config.is_local():
            # LOCAL: Use personal access token
            print("[DEBUG] GenieClient: Using LOCAL auth")
            return databricks_config.token
        else:
            # DBX: Get token from SDK Config OAuth M2M
            print("[DEBUG] GenieClient: Using DBX service principal auth")
            from databricks.sdk.core import Config

            cfg = Config(host=databricks_config.host)
            cfg.authenticate()
            token = cfg.oauth_token().access_token

            if not token:
                raise ValueError("Failed to get token from service principal authentication")

            return token
    
    def _log(self, message: str):
        """Log debug messages."""
        if self.debug:
            st.write(f"🔍 DEBUG: {message}")
    
    def _make_request(self, method: str, url: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request."""
        self._log(f"Request: {method} {url}")
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, timeout=120)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=json_data, timeout=120)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            self._log(f"Response status: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self._log(f"Request error: {str(e)}")
            return {"error": str(e)}
    
    def _genie_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a Genie API request."""
        url = f"{self.base_url}/{self.space_id}/{endpoint}"
        return self._make_request(method, url, json_data)
    
    def start_conversation(self, content: str) -> Tuple[str, str, Optional[GenieResponse]]:
        """Start a new conversation with Genie."""
        response = self._genie_request("POST", "start-conversation", {"content": content})
        
        if "error" in response:
            return None, None, GenieResponse(content=f"❌ Error: {response['error']}")
        
        self.conversation_id = response.get("conversation_id")
        message_id = response.get("message_id")
        
        self._log(f"Started conversation: {self.conversation_id}, message: {message_id}")
        
        if self.conversation_id and message_id:
            result = self._poll_for_result(message_id)
            return self.conversation_id, message_id, result
        
        return self.conversation_id, message_id, GenieResponse(content="Failed to start conversation")
    
    def send_message(self, content: str) -> Tuple[str, Optional[GenieResponse]]:
        """Send a message and get response."""
        if not self.conversation_id:
            _, message_id, result = self.start_conversation(content)
            return message_id, result
        
        response = self._genie_request(
            "POST", 
            f"conversations/{self.conversation_id}/messages",
            {"content": content}
        )
        
        if "error" in response:
            if "404" in str(response.get("error", "")) or "not found" in str(response.get("error", "")).lower():
                self.conversation_id = None
                _, message_id, result = self.start_conversation(content)
                return message_id, result
            return None, GenieResponse(content=f"❌ Error: {response['error']}")
        
        message_id = response.get("message_id")
        
        if message_id:
            result = self._poll_for_result(message_id)
            return message_id, result
        
        return message_id, GenieResponse(content="Failed to send message")
    
    def _poll_for_result(self, message_id: str, max_attempts: int = 60, delay: float = 2.0) -> GenieResponse:
        """Poll for message result."""
        self._log(f"Polling for message: {message_id}")
        
        processing_statuses = [
            "EXECUTING_QUERY", "PENDING", "FILTERING_RESULTS", "SUBMITTED", "RUNNING",
            "FILTERING_CONTEXT", "ASKING_AI", "PENDING_WAREHOUSE", "QUEUED"
        ]
        
        for attempt in range(max_attempts):
            response = self._genie_request(
                "GET",
                f"conversations/{self.conversation_id}/messages/{message_id}"
            )
            
            if "error" in response:
                return GenieResponse(content=f"❌ Error polling: {response['error']}")
            
            status = response.get("status", "").upper()
            self._log(f"Attempt {attempt + 1}: status = {status}")
            
            if status == "COMPLETED":
                return self._extract_response(response)
            elif status in ["FAILED", "CANCELLED"]:
                error_msg = response.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", "Unknown error")
                return GenieResponse(content=f"❌ Query failed: {error_msg}")
            else:
                time.sleep(delay)
        
        return GenieResponse(content="⏱️ Request timed out. Check the Genie space directly.")
    
    def _fetch_statement_result(self, statement_id: str) -> Optional[Dict]:
        """Fetch query results using statement ID."""
        self._log(f"Fetching statement results: {statement_id}")
        
        url = f"{self.sql_base_url}/{statement_id}"
        response = self._make_request("GET", url)
        
        if "error" in response:
            self._log(f"Statement fetch error: {response['error']}")
            return None
        
        return response
    
    def _extract_response(self, response: Dict[str, Any]) -> GenieResponse:
        """Extract and format the response."""
        self._log(f"Extracting response from: {list(response.keys())}")
        
        result = GenieResponse(content="")
        
        # Get text content
        content = response.get("content", "")
        if content:
            result.content = content
        
        # Process attachments
        attachments = response.get("attachments", [])
        self._log(f"Found {len(attachments)} attachments")
        
        for i, attachment in enumerate(attachments):
            keys = list(attachment.keys())
            self._log(f"Attachment {i}: keys={keys}")
            
            # Extract query info
            if "query" in attachment:
                query_info = attachment["query"]
                result.sql_query = query_info.get("query", "")
                result.query_description = query_info.get("description", "")
                self._log(f"Found query, description: {result.query_description[:50] if result.query_description else 'none'}...")
            
            # Extract suggested questions
            if "suggested_questions" in attachment:
                suggestions = attachment["suggested_questions"]
                if isinstance(suggestions, list):
                    result.suggested_questions = [q.get("question", q) if isinstance(q, dict) else str(q) for q in suggestions[:3]]
        
        # Fetch actual query results using statement_id
        query_result = response.get("query_result", {})
        statement_id = query_result.get("statement_id")
        
        if statement_id:
            self._log(f"Found statement_id: {statement_id}")
            statement_result = self._fetch_statement_result(statement_id)
            
            if statement_result:
                result.data_table, result.row_count = self._format_statement_result(statement_result)
        
        # Build content if empty
        if not result.content and result.query_description:
            result.content = result.query_description
        
        return result
    
    def _format_statement_result(self, statement_result: Dict) -> Tuple[Optional[str], int]:
        """Format statement result as markdown table."""
        self._log(f"Formatting statement result with keys: {list(statement_result.keys())}")
        
        # Get result data
        result_data = statement_result.get("result", {})
        manifest = statement_result.get("manifest", {})
        
        # Get schema from manifest
        schema = manifest.get("schema", {})
        columns = schema.get("columns", [])
        
        # Get data
        data_chunk = result_data.get("data_array", [])
        if not data_chunk:
            chunk = result_data.get("chunk", {})
            data_chunk = chunk.get("data_array", [])
        
        row_count = result_data.get("row_count", len(data_chunk))
        
        self._log(f"Statement: {len(columns)} columns, {row_count} rows, data={len(data_chunk)}")
        
        if not columns or not data_chunk:
            return None, 0
        
        # Build table
        col_names = [col.get("name", f"col_{i}") for i, col in enumerate(columns)]
        
        table = "| " + " | ".join(col_names) + " |\n"
        table += "| " + " | ".join(["---"] * len(col_names)) + " |\n"
        
        for row in data_chunk[:15]:  # Show up to 15 rows
            formatted_row = []
            for j, val in enumerate(row):
                if val is None:
                    formatted_row.append("-")
                elif isinstance(val, (int, float)):
                    # Check if it's a cost/money column
                    col_name = col_names[j].lower() if j < len(col_names) else ""
                    if any(word in col_name for word in ["cost", "amount", "total", "sum", "price"]):
                        formatted_row.append(f"${val:,.2f}")
                    elif isinstance(val, float):
                        formatted_row.append(f"{val:,.2f}")
                    else:
                        formatted_row.append(f"{val:,}")
                else:
                    formatted_row.append(str(val)[:50])  # Truncate long strings
            table += "| " + " | ".join(formatted_row) + " |\n"
        
        if row_count > 15:
            table += f"\n*Showing 15 of {row_count} rows*"
        
        return table, row_count
    
    def format_response_markdown(self, response: GenieResponse) -> str:
        """Format GenieResponse as markdown for display."""
        parts = []
        
        # Main content/description
        if response.content:
            parts.append(response.content)
        
        # Data table
        if response.data_table:
            parts.append(f"\n**Results** ({response.row_count} rows):\n")
            parts.append(response.data_table)
        
        return "\n\n".join(parts)
    
    def reset_conversation(self):
        """Reset the conversation."""
        self.conversation_id = None


# Singleton
_genie_client: Optional[GenieClient] = None

def get_genie_client() -> GenieClient:
    """Get or create the Genie client singleton."""
    global _genie_client
    if _genie_client is None:
        _genie_client = GenieClient()
    return _genie_client
