"""Prompt templates and system instructions for AI-powered Mermaid.js diagram generation."""

from typing import Any, Dict, List, Optional


class DiagramPromptBuilder:
    """Builds structured prompts for generating grounded Mermaid.js diagrams from document context."""

    SYSTEM_PROMPT = """You are an expert technical diagram generation assistant.
Your task is to analyze the supplied document context and generate accurate, syntactically valid Mermaid.js diagrams based on the user's request.

STRICT INSTRUCTIONS:
1. Use the supplied document context as your primary and authoritative source of truth.
2. Understand the relationships, processes, entities, flows, and hierarchies described in the document.
3. Identify or adhere to the requested Mermaid diagram type.
4. Generate strictly valid Mermaid.js syntax.
5. Keep node labels concise, professional, and meaningful.
6. Crucial Mermaid Syntax Rule: Always wrap node text containing special characters (parentheses, brackets, hyphens, colons, slashes) in double quotes.
   Example: id1["Client (Browser)"] --> id2["Server (Node.js)"]
7. Use appropriate arrows and directional relationships.
8. Preserve the technical meaning, steps, and sequence of the source document.
9. Never invent facts, components, or relationships not supported by the supplied context.
10. Return Mermaid code inside a fenced ```mermaid ... ``` code block.
11. If the provided context does not contain sufficient information to generate the requested diagram, explicitly output:
    "The provided document does not contain enough information to generate this diagram reliably."
"""

    DIAGRAM_TYPE_INSTRUCTIONS = {
        "flowchart": """- Use 'flowchart {direction}' as the diagram declaration.
- Direction should be TD (top-to-bottom) or LR (left-to-right).
- Use clear alphanumeric node IDs with double-quoted descriptive labels: nodeA["Step 1: Description"].
- For decision / conditional logic, use rhombus nodes: cond{{"Condition Check?"}}.
- Connect nodes with valid arrows: --> or -- "label" -->.
- Keep the flow structured, logical, and easy to read.""",

        "mindmap": """- Use 'mindmap' as the diagram declaration.
- Start with a single root concept: root((Central Concept))
- Organize subtopics hierarchically using indentation (2 or 4 spaces per level).
- Wrap labels in double quotes if they contain special characters.
- Example:
  mindmap
    root((Cloud Computing))
      Service Models
        IaaS
        PaaS
        SaaS
      Deployment Models
        Public
        Private
        Hybrid""",

        "sequenceDiagram": """- Use 'sequenceDiagram' as the diagram declaration.
- Explicitly declare participants at the top: participant A as "Client", participant B as "Server".
- Represent chronological interactions using directional arrows:
  - Synchronous request: A->>B: Request
  - Asynchronous / response: B-->>A: Response
- Use notes where helpful: Note over A,B: Description.""",

        "classDiagram": """- Use 'classDiagram' as the diagram declaration.
- Define classes, their key fields, and methods:
  class ClassName {
    +String attribute
    +method()
  }
- Represent relationships: Inheritance (<|--), Composition (*--), Aggregation (o--), Association (-->).""",

        "erDiagram": """- Use 'erDiagram' as the diagram declaration.
- Define entities and their attributes:
  ENTITY_NAME {
    string id PK
    string name
  }
- Define relationships: ENTITY_A ||--o{ ENTITY_B : "relates to".""",

        "stateDiagram": """- Use 'stateDiagram-v2' as the diagram declaration.
- Show entry with [*] --> InitialState.
- Show state transitions: State1 --> State2 : event / condition.
- Show termination with FinalState --> [*].""",

        "timeline": """- Use 'timeline' as the diagram declaration.
- Format with title and eras/events:
  timeline
    title Evolution / Timeline
    section Period 1
      Milestone A : Event details
      Milestone B : Event details
    section Period 2
      Milestone C : Event details""",

        "gantt": """- Use 'gantt' as the diagram declaration.
- Specify dateFormat and title.
- Group into sections and sequential tasks with durations or dates.""",

        "gitGraph": """- Use 'gitGraph' as the diagram declaration.
- Use commit, branch, checkout, merge commands representing the workflow.""",

        "architecture": """- Use 'flowchart {direction}' or 'architecture-beta' with subgraphs to represent system architecture components, layers, and communication boundaries.
- Group related services using subgraph:
  subgraph Data_Layer ["Data Layer"]
    db[("Main Database")]
  end""",
    }

    @classmethod
    def build_generation_prompt(
        cls,
        user_query: str,
        pages_context: List[Dict[str, Any]],
        diagram_type: str = "auto",
        direction: str = "TD",
    ) -> str:
        """Constructs the prompt containing user request, diagram guidelines, and PDF evidence."""
        type_key = diagram_type.lower()
        if type_key in ("flowchart td", "flowchart lr"):
            direction = "LR" if "lr" in type_key else "TD"
            type_key = "flowchart"
        elif type_key in ("sequence", "sequence diagram"):
            type_key = "sequenceDiagram"
        elif type_key in ("class", "class diagram"):
            type_key = "classDiagram"
        elif type_key in ("er", "erd", "entity relationship diagram"):
            type_key = "erDiagram"
        elif type_key in ("state", "state diagram"):
            type_key = "stateDiagram"
        elif type_key in ("mind map", "mindmap"):
            type_key = "mindmap"

        guidelines = cls.DIAGRAM_TYPE_INSTRUCTIONS.get(
            type_key,
            "Identify the most appropriate diagram type (flowchart, mindmap, sequenceDiagram, classDiagram, erDiagram, stateDiagram-v2, timeline) matching the user's intent.",
        )
        if "{direction}" in guidelines:
            guidelines = guidelines.format(direction=direction)

        evidence_blocks = []
        if not pages_context:
            evidence_blocks.append("No document context available.")
        else:
            for page in pages_context:
                doc_id = page.get("document_id", "unknown")
                page_num = page.get("page_number", "N/A")
                evidence_blocks.append(f"--- Document: {doc_id} | Page: {page_num} ---")
                for chunk in page.get("text_chunks", []):
                    c_id = chunk.get("chunk_id", "c")
                    content = chunk.get("content", "").strip()
                    evidence_blocks.append(f"[Chunk {c_id}]: {content}")

        context_str = "\n".join(evidence_blocks)

        prompt = f"""User Diagram Request:
"{user_query}"

Selected Diagram Type: {diagram_type.upper()} (Direction: {direction})

Diagram Type Syntax Guidelines:
{guidelines}

Supplied PDF Evidence:
{context_str}

Task:
Analyze the supplied PDF evidence. Synthesize the entities, actions, states, or hierarchical concepts relevant to the request.
Output ONLY the valid Mermaid.js code inside a fenced ```mermaid ... ``` code block.
Do not wrap the code in HTML or markdown outside the mermaid code block.
Every node label containing parentheses or special characters MUST be enclosed in double quotes.
"""
        return prompt

    @classmethod
    def build_correction_prompt(
        cls,
        original_query: str,
        failed_mermaid_code: str,
        error_message: str,
        pages_context: List[Dict[str, Any]],
    ) -> str:
        """Constructs a repair prompt to fix invalid Mermaid syntax."""
        return f"""The previous Mermaid.js code failed syntax validation or rendering with the following error:
Error: {error_message}

Original User Request:
"{original_query}"

Faulty Mermaid Code:
```mermaid
{failed_mermaid_code}
```

Instructions for Fix:
1. Carefully inspect the syntax error.
2. Ensure node IDs are valid alphanumeric strings without illegal characters.
3. Ensure every node label containing brackets, parentheses, colons, or punctuation is strictly wrapped in double quotes: nodeA["Clean Label (Note)"]
4. Ensure all subgraphs, blocks, and quotes are properly closed.
5. Return the corrected, fully working Mermaid code inside a fenced ```mermaid ... ``` code block.
"""

