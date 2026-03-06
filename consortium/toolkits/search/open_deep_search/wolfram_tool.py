from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import wolframalpha
import os


class WolframAlphaInput(BaseModel):
    query: str = Field(description="The query to send to Wolfram Alpha")


class WolframAlphaTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "calculate"
    description: str = (
        "Performs computational, mathematical, and factual queries using "
        "Wolfram Alpha's computational knowledge engine."
    )
    args_schema: Type[BaseModel] = WolframAlphaInput

    app_id: str = ""

    def __init__(self, app_id: str, **kwargs):
        super().__init__(app_id=app_id, **kwargs)

    def _run(self, query: str) -> str:
        wolfram_client = wolframalpha.Client(self.app_id)
        try:
            res = wolfram_client.query(query)
            results = []
            for pod in res.pods:
                if pod.title:
                    for subpod in pod.subpods:
                        if subpod.plaintext:
                            results.append({"title": pod.title, "result": subpod.plaintext})

            final_result = "No result found."
            for r in results:
                if r.get("title") == "Result":
                    final_result = r.get("result", "").strip()
                    break
            if final_result == "No result found." and results:
                final_result = results[0]["result"]

            print(f"QUERY: {query}\n\nRESULT: {final_result}")
            return final_result
        except Exception as e:
            return f"Error querying Wolfram Alpha: {e}"
