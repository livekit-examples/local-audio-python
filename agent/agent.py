
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

from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, get_job_context, RunContext, RoomIO
from livekit.agents.llm import function_tool
from livekit.plugins import (
    openai,
    elevenlabs,
    deepgram,
    noise_cancellation,
    silero,
)

load_dotenv()

logger = logging.getLogger("robot_agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""
                         You are a useful assistant.  
                         Answer the users questions with concise answers.
                         Dont be too verbose.
                         Don't use any words that can't be pronounced. 
                         """)



async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=openai.LLM(model="gpt-4o"),
        # llm=openai.LLM.with_x_ai(model="grok-3"),
        tts=elevenlabs.TTS(
                model="eleven_multilingual_v2"
            ),
        vad=silero.VAD.load(),
        # turn_detection=EnglishModel(),
        min_interruption_words=2,
    ) 
    
    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()
    
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info("participant connected: %s %s", participant.sid, participant.identity)

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logging.info("participant disconnected: %s %s", participant.sid, participant.identity)
    

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            participant_identity='robot',
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
            close_on_disconnect=False,
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="robot_agent"))