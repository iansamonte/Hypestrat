"""
Base Agent — Abstract base class for all IAN.AI Ventures agents.
Uses Anthropic SDK with tool use and an agentic loop.
"""

from __future__ import annotations
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import anthropic

from src.models.company_state import AgentResult, Action, CompanyState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base agent with Anthropic client, tool use loop, and structured results.

    Subclasses must implement:
      - get_tools(): return list of Anthropic tool defs
      - handle_tool_call(name, input, state): execute a tool, return string result
      - get_system_prompt(): return the system prompt string
      - run_cycle(state): orchestrate one agent cycle, return list[Action]
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    MAX_ITERATIONS = 10

    def __init__(
        self,
        name: str,
        role: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.name = name
        self.role = role
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._last_result: Optional[AgentResult] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return Anthropic-format tool definitions for this agent."""
        ...

    @abstractmethod
    def handle_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any], state: CompanyState
    ) -> str:
        """Execute a tool call and return a string result."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...

    @abstractmethod
    def run_cycle(self, state: CompanyState) -> List[Action]:
        """
        Run one agent cycle against the company state.
        Returns a list of Actions taken.
        """
        ...

    # ------------------------------------------------------------------
    # Core agentic loop
    # ------------------------------------------------------------------

    def run(self, task: str, context: Dict[str, Any], state: CompanyState) -> AgentResult:
        """
        Run the agent on a task using the Anthropic tool-use loop.

        The loop continues until:
          - The model returns stop_reason == "end_turn" (no more tool calls)
          - Max iterations reached
        """
        result = AgentResult(agent_name=self.name, task=task)
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": self._build_user_message(task, context),
            }
        ]

        tools = self.get_tools()
        iterations = 0

        try:
            while iterations < self.MAX_ITERATIONS:
                iterations += 1
                logger.debug(f"[{self.name}] Iteration {iterations}, messages={len(messages)}")

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.get_system_prompt(),
                    tools=tools,
                    messages=messages,
                )

                # Collect text output
                text_parts = []
                tool_use_blocks = []

                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_use_blocks.append(block)

                if text_parts:
                    result.output += "\n".join(text_parts) + "\n"

                # If no tool calls or end_turn, we're done
                if response.stop_reason == "end_turn" or not tool_use_blocks:
                    break

                # Process tool calls
                tool_results = []
                for tb in tool_use_blocks:
                    tool_name = tb.name
                    tool_input = tb.input if isinstance(tb.input, dict) else {}
                    logger.debug(f"[{self.name}] Calling tool: {tool_name}({tool_input})")

                    try:
                        tool_result_str = self.handle_tool_call(tool_name, tool_input, state)
                    except Exception as e:
                        tool_result_str = f"Tool error: {e}"
                        logger.warning(f"[{self.name}] Tool {tool_name} error: {e}")

                    # Record as an Action
                    action = Action(
                        agent=self.name,
                        action_type=tool_name,
                        description=f"{tool_name}: {json.dumps(tool_input)[:120]}",
                        result=tool_result_str[:200],
                        impact="positive",
                    )
                    result.actions_taken.append(action)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tb.id,
                        "content": tool_result_str,
                    })

                # Add assistant turn + tool results to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            result.success = True

        except anthropic.APIError as e:
            result.success = False
            result.error = str(e)
            result.output += f"\n[API Error: {e}]"
            logger.error(f"[{self.name}] API error: {e}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.output += f"\n[Error: {e}]"
            logger.exception(f"[{self.name}] Unexpected error")

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, task: str, context: Dict[str, Any]) -> str:
        """Combine task + context into a user message string."""
        context_str = json.dumps(context, indent=2, default=str)
        return (
            f"## Task\n{task}\n\n"
            f"## Company Context\n```json\n{context_str}\n```"
        )

    def _actions_to_text(self, actions: List[Action]) -> str:
        if not actions:
            return "No actions taken."
        lines = []
        for a in actions:
            lines.append(f"- [{a.action_type}] {a.description[:80]}")
        return "\n".join(lines)

    def log_cycle_start(self, state: CompanyState) -> None:
        logger.info(f"[{self.name}] Starting cycle — Month {state.month} | MRR ${state.mrr:,.0f}")

    def log_cycle_end(self, actions: List[Action]) -> None:
        logger.info(f"[{self.name}] Cycle complete — {len(actions)} actions taken")
