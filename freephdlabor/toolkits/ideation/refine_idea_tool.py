from __future__ import annotations
from typing import Optional, Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import litellm

import json
from ..utils import extract_content_between_markers


class RefineIdeaToolInput(BaseModel):
    idea_to_refine_json: str = Field(description="JSON string containing the idea to refine")
    reflection_prompt: Optional[str] = Field(default=None, description="Specific prompt to direct the reflection/refinement (optional)")


class RefineIdeaTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "RefineIdeaTool"
    description: str = "Refines a research idea by calling LLM. Takes an idea JSON and returns the refined/improved version of the idea."
    args_schema: Type[BaseModel] = RefineIdeaToolInput
    model_id: str = ""

    def __init__(self, model=None, **kwargs: Any):
        """
        Initialize the tool with access to the LLM model.

        Args:
            model: The LLM model to use for refining ideas
        """
        model_id = model if isinstance(model, str) else getattr(model, 'model', str(model)) if model else ""
        super().__init__(model_id=model_id, **kwargs)

    def _run(self, idea_to_refine_json: str, reflection_prompt: str = "") -> str:
        """
        Refine a research idea by calling the LLM.

        Args:
            idea_to_refine_json: JSON string containing the idea to refine
            reflection_prompt: Specific prompt for reflection/refinement

        Returns:
            JSON string containing the refined idea
        """
        try:
            # Parse the idea to refine
            idea_to_refine = json.loads(idea_to_refine_json)

            # Create the reflection prompt based on ai_scientist/generate_ideas.py
            prompt = f"""In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created. Critically identify on the "Rationale" section of the idea json: is there any ambiguity or flaw in the logic or reasoning? Is there any unjustified claims about expected improvements? Think about how to resolve them step-by-step.
Ensure the idea is clear and well-justified, and the JSON is the correct format.

CRITICAL JSON FORMATTING RULES:
- USE PROPER JSON ESCAPING: Escape quotes (\"), backslashes (\\), and newlines (\n)
- AVOID UNICODE MATHEMATICAL SYMBOLS: Instead of τ, π, θ use tau, pi, theta
- AVOID RAW UNICODE ESCAPES: Don't use \\u03c4 - use "tau" instead
- USE ASCII CHARACTERS ONLY in field values to prevent parsing errors
- Example: Instead of "Let τ = (s_0, a_0)" write "Let tau = (s_0, a_0)"

In the next attempt, try and refine and improve your last idea.

{reflection_prompt}

Current idea to refine:
{json.dumps(idea_to_refine, indent=2)}

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

Carefully ensure that the NEW IDEA JSON has the same fields as the previous JSON with the following:
- "Name":
- "Title":
- "Experiment":
- "Technical_Details":
- "Rationale":
- "Implementation_Plan":

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

            # Call the LLM model to refine the idea
            if not self.model_id:
                return json.dumps({"error": "No model provided to RefineIdeaTool"})

            # Generate response using the model
            messages = [{"role": "user", "content": prompt}]
            response = litellm.completion(model=self.model_id, messages=messages)
            response_content = response.choices[0].message.content

            # Extract JSON content using the utility function
            json_content = extract_content_between_markers(response_content, "```json", "```")

            if json_content:
                try:
                    # Parse and validate the JSON
                    refined_idea_dict = json.loads(json_content)
                    return json.dumps(refined_idea_dict, indent=2)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"Invalid JSON extracted: {str(e)}", "raw_content": json_content[:500]})
            else:
                # No JSON block found, try to parse entire response as JSON
                try:
                    refined_idea_dict = json.loads(response_content)
                    return json.dumps(refined_idea_dict, indent=2)
                except json.JSONDecodeError:
                    return json.dumps({
                        "error": "Could not extract valid JSON from LLM response",
                        "raw_response": response_content[:500]
                    })

        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Error refining idea: {str(e)}"})

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError
