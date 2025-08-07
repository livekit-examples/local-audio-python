import asyncio
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "livekit-agents[deepgram,openai,cartesia,silero,elevenlabs,turn-detector,hume]~=1.0",
#   "livekit-plugins-noise-cancellation~=0.2",
#   "python-dotenv",
# ]
# ///

import json
import logging
from dotenv import load_dotenv

from livekit import agents
from livekit.rtc import RpcInvocationData, DataPacket
from livekit.agents import AgentSession, Agent, RoomInputOptions, get_job_context, RunContext
from livekit.agents.llm import function_tool, ChatMessage
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import (
    openai,
    elevenlabs,
    deepgram,
    noise_cancellation,
    silero,
)

load_dotenv()

logger = logging.getLogger("rover-agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""
                         You are a helpful assistant. 
                         You can start a task by calling the start_task function.
                         You can check the current task status by calling the get_current_task function.
                         """)

    @function_tool
    async def start_task(self, context: RunContext, task: str):
        """Start a task.
        
        Args:
            task: The task to start
        """
        
        logger.info("start task called with task: %s", task)
        
        try:
            ctx = get_job_context()
            room = ctx.room
            
            response = await room.local_participant.perform_rpc(
                destination_identity="robot",
                method="start_task",
                payload=task,
            )
            
            
            logger.info(f"Response from robot: {response}")
            return response
                
        except Exception as e:
            logger.error(f"Error starting task: {e}")
            return f"Error starting task: {str(e)}"

    @function_tool
    async def get_current_task(self, context: RunContext):
        """Get the current running ask status.
        
        Returns:
            The current task information from the robot
        """
        
        logger.info("get current task called")
        
        try:
            ctx = get_job_context()
            room = ctx.room
            
            response = await room.local_participant.perform_rpc(
                destination_identity="robot",
                method="get_current_task",
                payload="",
            )
            
            logger.info(f"Current task response from robot: {response}")
            return response
                
        except Exception as e:
            logger.error(f"Error getting current task: {e}")
            return f"Error getting current task: {str(e)}"

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=openai.LLM(model="gpt-4o"),
        tts=elevenlabs.TTS(
                model="eleven_multilingual_v2"
            ),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
    ) 
    
    agent = Assistant()
    
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            # participant_identity='rover',
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )
    
    # register rpc on agent side
    room = ctx.room
   
    async def _handle_data_received(data: DataPacket):
        update_ctx = data.data
        
        try:
            if isinstance(update_ctx, bytes):
                try:
                    update_ctx = update_ctx.decode("utf-8")
                except Exception:
                    update_ctx = str(update_ctx)

            updated_chat_context = agent.chat_ctx.copy()
            updated_chat_context.add_message(
                role="assistant",
                content=update_ctx,
            )

            logger.info(f"update_ctx: {update_ctx}")
            
            await agent.update_chat_ctx(updated_chat_context)
            
            await session.say(update_ctx, add_to_chat_ctx=True)
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")

    @room.on("data_received")
    def on_data_received(data: DataPacket):
        asyncio.create_task(_handle_data_received(data))
    


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="agent"))