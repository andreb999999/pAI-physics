from typing import Optional, Literal, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from .ods_agent import OpenDeepSearchAgent
import os
from dotenv import load_dotenv

load_dotenv()


class OpenDeepSearchInput(BaseModel):
    query: str = Field(description="The search query to perform")


class OpenDeepSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "web_search"
    description: str = (
        "Performs web search based on your query (think a Google search) "
        "then returns the final answer processed by an LLM."
    )
    args_schema: Type[BaseModel] = OpenDeepSearchInput

    model_name: Optional[str] = None
    reranker: str = "infinity"
    search_provider: str = "serper"
    search_tool: Optional[object] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        reranker: str = "infinity",
        search_provider: Literal["serper", "searxng"] = "serper",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            reranker=reranker,
            search_provider=search_provider,
            **kwargs,
        )
        self._setup()

    def _setup(self):
        serper_api_key = os.getenv("SERPER_API_KEY")
        searxng_instance_url = os.getenv("SEARXNG_INSTANCE_URL")
        searxng_api_key = os.getenv("SEARXNG_API_KEY")
        self.search_tool = OpenDeepSearchAgent(
            model_name=self.model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=serper_api_key,
            searxng_instance_url=searxng_instance_url,
            searxng_api_key=searxng_api_key,
        )

    def _run(self, query: str) -> str:
        answer = self.search_tool.ask_sync(query, max_sources=2, pro_mode=True)
        return answer

