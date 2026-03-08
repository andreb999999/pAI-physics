from __future__ import annotations
from typing import Optional, Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import litellm

import json
from ..utils import extract_content_between_markers


class GenerateIdeaToolInput(BaseModel):
    task_description: Optional[str] = Field(default=None, description="Description of the research task, can contain details about the task, the domain, the dataset, the model, the evaluation, the benchmark, the implementation, the code, the results, the analysis, the conclusion, etc.")
    seed_ideas_json: Optional[str] = Field(default=None, description="JSON string containing seed ideas for context (optional)")


class GenerateIdeaTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "GenerateIdeaTool"
    description: str = "Generates research ideas by prompting LLM. Takes task description and optional seed ideas for context."
    args_schema: Type[BaseModel] = GenerateIdeaToolInput
    model_id: str = ""

    def __init__(self, model=None, **kwargs: Any):
        """
        Initialize the tool with access to the LLM model.

        Args:
            model: The LLM model to use for generating ideas
        """
        model_id = model if isinstance(model, str) else getattr(model, 'model', str(model)) if model else ""
        super().__init__(model_id=model_id, **kwargs)

    def _run(self, task_description: str = "", seed_ideas_json: str = "") -> str:
        """
        Generate a complete research idea by calling the LLM.

        Args:
            task_description: Task description for context
            seed_ideas_json: JSON string containing existing seed ideas (optional)

        Returns:
            JSON string containing the generated research idea
        """
        try:
            # Parse seed ideas
            seed_ideas = json.loads(seed_ideas_json) if seed_ideas_json else []

            # Create the formatted string of previous ideas
            prev_ideas_string = "\n\n".join([json.dumps(idea) for idea in seed_ideas])

            # Create the idea generation prompt
            prompt = f"""{task_description}

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Now, come up with an impactful and creative research idea based on the task description above.
Your new idea should not be more complex than those you have already generated.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first thoroughly discuss your intuitions and motivations for your proposed idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments.
Be as specific as possible, including technical details if necessary.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Technical_Details": A precise and verbose technical description of the proposed improvement, using specific technical language and avoiding vague terms. BE SURE TO DEFINE BEFORE YOU USE NONSTANDARD TERMINOLOGY. WHENEVER POSSIBLE, USE MATHEMATICAL LANGUAGE TO AVOID AMBIGUITY.
- "Rationale": An extremely detailed explanation of why the proposed experiment can be expected to improve from the baseline model. Carefully explaining the logic behind every step of reasoning you make. Avoid making unjustified claims about improvement.
- "Implementation_Plan": A plan of steps to implement the experiment described above.

Be cautious and critical in your output.

This JSON will be automatically parsed, so ensure the format is precise. CRITICAL JSON FORMATTING RULES:
- USE PROPER JSON ESCAPING: Escape quotes (\"), backslashes (\\\\), and newlines (\\n)
- AVOID UNICODE MATHEMATICAL SYMBOLS: Instead of τ, π, θ use tau, pi, theta
- AVOID RAW UNICODE ESCAPES: Don't use \\u03c4 - use "tau" instead
- USE ASCII CHARACTERS ONLY in field values to prevent parsing errors
- Example: Instead of "Let τ = (s_0, a_0)" write "Let tau = (s_0, a_0)"
"""

            # Call the LLM model to generate the idea
            if not self.model_id:
                return json.dumps({"error": "No model provided to GenerateIdeaTool"})

            # Generate response using the model
            messages = [{"role": "user", "content": prompt}]
            response = litellm.completion(model=self.model_id, messages=messages)
            response_content = response.choices[0].message.content

            # Extract JSON content using the utility function
            json_content = extract_content_between_markers(response_content, "```json", "```")

            if json_content:
                try:
                    # Parse and validate the JSON
                    idea_dict = json.loads(json_content)
                    return json.dumps(idea_dict, indent=2)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"Invalid JSON extracted: {str(e)}", "raw_content": json_content[:500]})
            else:
                # No JSON block found, try to parse entire response as JSON
                try:
                    idea_dict = json.loads(response_content)
                    return json.dumps(idea_dict, indent=2)
                except json.JSONDecodeError:
                    return json.dumps({
                        "error": "Could not extract valid JSON from LLM response",
                        "raw_response": response_content[:500]
                    })

        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON in seed_ideas_json: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Error generating idea: {str(e)}"})

