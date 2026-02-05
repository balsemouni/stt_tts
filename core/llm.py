from langchain_nvidia_ai_endpoints import ChatNVIDIA

class LLMHandler:
    """Handles LLM inference using NVIDIA API"""

    def __init__(self, api_key, model="meta/llama-3.1-8b-instruct"):
        self.llm = ChatNVIDIA(
            model=model,
            nvidia_api_key=api_key
        )

    async def get_response(self, messages, loop):
        #This takes that blocking invoke call and runs it in a background thread.
        # .invoke() your whole program would stop moving until the AI finished typing
        response = await loop.run_in_executor(
            None,
            lambda: self.llm.invoke(messages)
        )
        return response.content