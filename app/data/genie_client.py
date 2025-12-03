"""
Databricks Genie API Client
============================
Client for interacting with Databricks Genie conversational AI.

API Reference: https://docs.databricks.com/api/azure/workspace/genie
"""

import requests
import time
import streamlit as st
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from config import databricks_config, genie_config


@dataclass
class GenieMessage:
    """Represents a message in a Genie conversation."""
    role: str  # "user" or "assistant"
    content: str
    query: Optional[str] = None  # SQL query if generated
    attachments: Optional[list] = None  # Any attachments/results


class GenieClient:
    """Client for Databricks Genie API."""
    
    def __init__(self):
        self.base_url = f"https://{databricks_config.host}/api/2.0/genie/spaces"
        self.space_id = genie_config.space_id
        self.headers = {
            "Authorization": f"Bearer {databricks_config.token}",
            "Content-Type": "application/json"
        }
        self.conversation_id: Optional[str] = None
        self.debug = False  # Set to True for debugging
    
    def _log(self, message: str):
        """Log debug messages."""
        if self.debug:
            st.write(f"🔍 DEBUG: {message}")
    
    def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request to Genie."""
        url = f"{self.base_url}/{self.space_id}/{endpoint}"
        
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
            result = response.json()
            self._log(f"Response keys: {list(result.keys())}")
            return result
        except requests.exceptions.RequestException as e:
            self._log(f"Request error: {str(e)}")
            return {"error": str(e)}
    
    def start_conversation(self, content: str) -> Tuple[str, str, Optional[str]]:
        """
        Start a new conversation with Genie.
        
        Args:
            content: The user's question/message
            
        Returns:
            Tuple of (conversation_id, message_id, initial_response)
        """
        response = self._make_request("POST", "start-conversation", {
            "content": content
        })
        
        if "error" in response:
            return None, None, f"❌ Error: {response['error']}"
        
        self.conversation_id = response.get("conversation_id")
        message_id = response.get("message_id")
        
        self._log(f"Started conversation: {self.conversation_id}, message: {message_id}")
        
        # Poll for the response
        if self.conversation_id and message_id:
            result = self._poll_for_result(message_id)
            return self.conversation_id, message_id, result
        
        return self.conversation_id, message_id, "Failed to start conversation"
    
    def send_message(self, content: str) -> Tuple[str, Optional[str]]:
        """
        Send a follow-up message in an existing conversation.
        
        Args:
            content: The user's follow-up question
            
        Returns:
            Tuple of (message_id, response)
        """
        if not self.conversation_id:
            # Start a new conversation if none exists
            _, message_id, result = self.start_conversation(content)
            return message_id, result
        
        response = self._make_request(
            "POST", 
            f"conversations/{self.conversation_id}/messages",
            {"content": content}
        )
        
        if "error" in response:
            # If conversation expired, start a new one
            if "404" in str(response.get("error", "")) or "not found" in str(response.get("error", "")).lower():
                self.conversation_id = None
                _, message_id, result = self.start_conversation(content)
                return message_id, result
            return None, f"❌ Error: {response['error']}"
        
        message_id = response.get("message_id")
        
        if message_id:
            result = self._poll_for_result(message_id)
            return message_id, result
        
        return message_id, "Failed to send message"
    
    def _poll_for_result(self, message_id: str, max_attempts: int = 60, delay: float = 2.0) -> str:
        """
        Poll for the result of a message.
        
        Args:
            message_id: The message ID to poll for
            max_attempts: Maximum polling attempts (60 * 2s = 2 minutes max)
            delay: Delay between attempts in seconds
            
        Returns:
            The assistant's response text
        """
        self._log(f"Polling for message: {message_id}")
        
        # Valid in-progress statuses
        processing_statuses = [
            "EXECUTING_QUERY", "PENDING", "FILTERING_RESULTS", "SUBMITTED", "RUNNING",
            "FILTERING_CONTEXT", "ASKING_AI", "PENDING_WAREHOUSE", "QUEUED"
        ]
        
        for attempt in range(max_attempts):
            response = self._make_request(
                "GET",
                f"conversations/{self.conversation_id}/messages/{message_id}"
            )
            
            if "error" in response:
                return f"❌ Error polling: {response['error']}"
            
            status = response.get("status", "").upper()
            self._log(f"Attempt {attempt + 1}: status = {status}")
            
            if status == "COMPLETED":
                return self._extract_response(response)
            elif status in ["FAILED", "CANCELLED"]:
                error_msg = response.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", "Unknown error")
                return f"❌ Query failed: {error_msg}"
            elif status.upper() in [s.upper() for s in processing_statuses]:
                # Still processing, wait and retry
                time.sleep(delay)
            else:
                # Unknown status, wait anyway (might be new status)
                self._log(f"Unknown status: {status}")
                time.sleep(delay)
        
        return "⏱️ Request timed out after 2 minutes. The query may still be running in Databricks - check the Genie space directly."
    
    def _extract_response(self, response: Dict[str, Any]) -> str:
        """Extract the response text from a completed message."""
        self._log(f"Extracting response from: {list(response.keys())}")
        
        result_parts = []
        
        # Check for message content (text response from Genie)
        content = response.get("content", "")
        if content and isinstance(content, str) and len(content) > 0:
            result_parts.append(content)
        
        # Check for query_result directly in response (common in newer API)
        if "query_result" in response:
            query_result = response.get("query_result", {})
            self._log(f"Found direct query_result with keys: {list(query_result.keys()) if query_result else 'empty'}")
            table_result = self._format_query_result(query_result)
            if table_result:
                result_parts.append(table_result)
        
        # Process attachments
        attachments = response.get("attachments", [])
        self._log(f"Found {len(attachments)} attachments")
        
        for i, attachment in enumerate(attachments):
            attach_type = attachment.get("type", "")
            self._log(f"Attachment {i}: type='{attach_type}', keys={list(attachment.keys())}")
            
            # Handle TEXT attachment
            if attach_type.upper() == "TEXT" or "text" in attachment:
                text_obj = attachment.get("text", {})
                if isinstance(text_obj, str):
                    if text_obj not in result_parts:
                        result_parts.append(text_obj)
                elif isinstance(text_obj, dict):
                    text_content = text_obj.get("content", "")
                    if text_content and text_content not in result_parts:
                        result_parts.append(text_content)
            
            # Handle QUERY attachment
            if attach_type.upper() == "QUERY" or "query" in attachment:
                query_info = attachment.get("query", {})
                if isinstance(query_info, dict):
                    query_text = query_info.get("query", "")
                    description = query_info.get("description", "")
                    
                    if description:
                        result_parts.append(f"**Analysis:** {description}")
                    if query_text:
                        result_parts.append(f"\n```sql\n{query_text}\n```")
            
            # Handle QUERY_RESULT attachment
            if attach_type.upper() == "QUERY_RESULT" or "query_result" in attachment:
                query_result = attachment.get("query_result", {})
                table_result = self._format_query_result(query_result)
                if table_result and table_result not in result_parts:
                    result_parts.append(table_result)
        
        if result_parts:
            return "\n\n".join(result_parts)
        
        # If no results found, return debug info
        self._log(f"No results extracted. Full response: {response}")
        return "🤔 I received a response but couldn't parse it. Try asking in a different way, or check the Genie space directly in Databricks."
    
    def _format_query_result(self, query_result: Dict[str, Any]) -> Optional[str]:
        """Format a query result as markdown table."""
        if not query_result:
            return None
        
        self._log(f"Formatting query_result with keys: {list(query_result.keys())}")
        
        # Check for statement_response format (newer API)
        statement_response = query_result.get("statement_response", {})
        if statement_response:
            result_data = statement_response.get("result", {})
            if result_data:
                query_result = result_data
        
        # Get columns
        columns = query_result.get("columns", [])
        
        # Get data - try multiple locations
        data_rows = query_result.get("data_array", [])
        if not data_rows:
            result_data = query_result.get("result", {})
            data_rows = result_data.get("data_array", [])
        if not data_rows:
            data_chunk = query_result.get("data_chunk", {})
            data_rows = data_chunk.get("data_array", [])
        
        row_count = query_result.get("row_count", len(data_rows))
        
        self._log(f"Query result: {len(columns)} columns, {row_count} rows, data_rows={len(data_rows)}")
        
        if not columns or row_count == 0:
            return None
        
        result_parts = []
        
        # Add summary for single-value results
        if row_count == 1 and len(columns) == 1 and data_rows:
            val = data_rows[0][0] if data_rows[0] else None
            if val is not None and isinstance(val, (int, float)):
                col_name = columns[0].get("name", "value") if isinstance(columns[0], dict) else str(columns[0])
                if "cost" in col_name.lower() or "amount" in col_name.lower() or "total" in col_name.lower():
                    result_parts.append(f"**Answer:** ${val:,.2f}")
                else:
                    result_parts.append(f"**Answer:** {val:,.2f}")
        
        # Format as markdown table (first 10 rows)
        display_rows = data_rows[:10]
        
        # Header
        col_names = []
        for col in columns:
            if isinstance(col, dict):
                col_names.append(col.get("name", "col"))
            else:
                col_names.append(str(col))
        
        table = "| " + " | ".join(col_names) + " |\n"
        table += "| " + " | ".join(["---"] * len(col_names)) + " |\n"
        
        # Rows
        for row in display_rows:
            formatted_row = []
            for val in row:
                if val is None:
                    formatted_row.append("-")
                elif isinstance(val, float):
                    if abs(val) >= 1000:
                        formatted_row.append(f"${val:,.2f}")
                    else:
                        formatted_row.append(f"{val:.2f}")
                else:
                    formatted_row.append(str(val))
            table += "| " + " | ".join(formatted_row) + " |\n"
        
        result_parts.append(f"\n**Results** ({row_count} rows):\n\n{table}")
        
        if row_count > 10:
            result_parts.append(f"\n*Showing first 10 of {row_count} rows*")
        
        return "\n".join(result_parts)
    
    def get_conversation_messages(self) -> list:
        """Get all messages in the current conversation."""
        if not self.conversation_id:
            return []
        
        response = self._make_request(
            "GET",
            f"conversations/{self.conversation_id}/messages"
        )
        
        if "error" in response:
            return []
        
        return response.get("messages", [])
    
    def reset_conversation(self):
        """Reset the conversation to start fresh."""
        self.conversation_id = None


# Singleton instance
_genie_client: Optional[GenieClient] = None


def get_genie_client() -> GenieClient:
    """Get or create the Genie client singleton."""
    global _genie_client
    if _genie_client is None:
        _genie_client = GenieClient()
    return _genie_client
