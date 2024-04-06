import logging
import os

import taskingai.assistant
from dotenv import load_dotenv, find_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


taskingai_api_key = os.getenv("TASKINGAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
assistant_id1 = os.getenv("ASSISTANT_ID1")
assistant_id2 = os.getenv("ASSISTANT_ID2")
use_fallback = os.getenv("USE_FALLBACK") == "true"


os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

taskingai.init(api_key=taskingai_api_key)

app = FastAPI()

chat_sessions = {}


def get_assistant():
    return taskingai.assistant


class ChatLogic:
    async def chat(self, assistant, user_id, user_message):
        raise NotImplementedError

    async def _chat_logic(
        self, assistant: taskingai.assistant, assistant_id, user_id, user_message
    ):
        try:
            chat_id = chat_sessions.get(user_id)
            if not chat_id:
                chat = assistant.create_chat(assistant_id=assistant_id)
                if not chat:
                    raise Exception("Failed to create chat session")
                chat_id = chat.chat_id
                chat_sessions[user_id] = chat_id

            assistant.create_message(
                assistant_id=assistant_id,
                chat_id=chat_id,
                text=user_message,
            )
            assistant_message = assistant.generate_message(
                assistant_id=assistant_id,
                chat_id=chat_id,
            )
            if not assistant_message:
                raise Exception("Failed to get assistant's response")

            return assistant_message.content.text
        except Exception as e:
            logger.error(f"Error in _chat_logic: {e}")
            return None


class ChatWithFallback(ChatLogic):
    async def chat(self, assistant, user_id, user_message):
        response = await self._chat_logic(
            assistant, assistant_id1, user_id, user_message
        )
        if response is None:
            logger.info("Using fallback assistant")
            response = await self._chat_logic(
                assistant, assistant_id2, user_id, user_message
            )
        return response


class ChatWithoutFallback(ChatLogic):
    async def chat(self, assistant, user_id, user_message):
        return await self._chat_logic(assistant, assistant_id1, user_id, user_message)


def chat_logic_factory():
    if use_fallback:
        return ChatWithFallback()
    else:
        return ChatWithoutFallback()


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat_endpoint(
    user_id: str,
    user_message: str = Body(..., embed=True),
    assistant: taskingai.assistant = Depends(get_assistant),
):
    chat_logic = chat_logic_factory()
    response = await chat_logic.chat(assistant, user_id, user_message)

    if response is None:
        raise HTTPException(
            status_code=500, detail="Failed to get a response from the assistant"
        )

    logger.info(f"Chat sessions: {chat_sessions}")

    return {"assistant_response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
