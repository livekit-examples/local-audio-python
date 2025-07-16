
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "asyncio",
#   "livekit==1.0.11",
#   "livekit-agents==1.1.6",
#   "livekit-agents[deepgram,openai,cartesia,silero,elevenlabs,turn-detector,hume]~=1.1.6",
#   "livekit-plugins-noise-cancellation~=0.2",
#   "python-dotenv",
# ]
# ///

import asyncio
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
    
    

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            participant_identity='robot',
            noise_cancellation=noise_cancellation.BVC(), 
            close_on_disconnect=False,
        ),
    )
    
    # wait for either a participant to join or a shutdown signal
    shutdown_future: asyncio.Future[None] = asyncio.Future()
    
    # wait for either last participant to leave or a shutdown signal
    room_empty_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logging.info("participant connected: %s %s", participant.sid, participant.identity)
        if participant.identity == "phone":
            room_io.set_participant(participant.identity)
            logging.info("phone connected, using 'phone' audio stream")

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant, fut=room_empty_future):
        logging.info("participant disconnected: %s %s", participant.sid, participant.identity)
        if participant.identity == "phone":
            room_io.set_participant("robot")
            logging.info("phone disconnected, using 'robot' audio stream")
        if len(ctx.room.remote_participants) == 0 and not fut.done():
            fut.set_result(None)  
            
    # def _on_participant_disconnected(_: rtc.Participant, fut=room_empty_future) -> None:
        

    # ctx.room.on("participant_disconnected", _on_participant_disconnected)

    try:
        # blocking wait for either future to be set
        await asyncio.wait(
            [shutdown_future, room_empty_future], return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        # ctx.room.off("participant_disconnected", _on_participant_disconnected)
        await session.aclose()



if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="robot_agent"))