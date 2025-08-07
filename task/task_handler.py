#!/usr/bin/env python3
"""
Task Handler Library

A library for managing long-running tasks with progress tracking.
Only one task can run at a time, with 100-second execution and progress updates every second.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass


class TaskStatus(Enum):
    """Task execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TaskInfo:
    """Information about a running or completed task"""
    name: str
    status: TaskStatus
    progress: int  # 0-100
    start_time: float
    end_time: Optional[float] = None
    error_message: Optional[str] = None


class TaskHandler:
    """
    Manages single long-running tasks with progress tracking.
    
    Features:
    - Only one task can run at a time
    - 100-second task duration with 1-second progress updates
    - Progress tracking from 0-100%
    - Task state management
    - Error handling
    """
    
    def __init__(self, on_complete: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None):
        self.logger = logging.getLogger(__name__)
        self._current_task: Optional[TaskInfo] = None
        self._task_lock = asyncio.Lock()
        self._progress_task: Optional[asyncio.Task] = None
        self._on_complete_callback = on_complete
        
    @property
    def current_task(self) -> Optional[TaskInfo]:
        """Get current task information"""
        return self._current_task
    
    @property
    def is_running(self) -> bool:
        """Check if a task is currently running"""
        return (self._current_task is not None and 
                self._current_task.status == TaskStatus.RUNNING)
    
    async def start_task(self, task_name: str) -> Dict[str, Any]:
        """
        Start a new task.
        
        Args:
            task_name: Name of the task to start
            
        Returns:
            Dict with success status and message
            
        Raises:
            ValueError: If a task is already running
        """
        async with self._task_lock:
            # Check if a task is already running
            if self.is_running:
                error_msg = f"Cannot start task '{task_name}': Task '{self._current_task.name}' is already running"
                self.logger.warning(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "current_task": self._current_task.name,
                    "current_progress": self._current_task.progress
                }
            
            # Create new task
            self._current_task = TaskInfo(
                name=task_name,
                status=TaskStatus.RUNNING,
                progress=0,
                start_time=time.time()
            )
            
            self.logger.info(f"Starting task: {task_name}")
            
            # Start progress tracking task
            self._progress_task = asyncio.create_task(self._run_task_progress())
            
            return {
                "success": True,
                "message": f"Task '{task_name}' started successfully",
                "task_name": task_name,
                "start_time": self._current_task.start_time
            }
    
    async def get_current_task(self) -> Optional[Dict[str, Any]]:
        """
        Get current task status and progress.
        
        Returns:
            Dict with task info or None if no task is running
        """
        if self._current_task is None:
            return None
            
        return {
            "task_name": self._current_task.name,
            "status": self._current_task.status.value,
            "progress": self._current_task.progress,
            "start_time": self._current_task.start_time,
            "end_time": self._current_task.end_time,
            "error_message": self._current_task.error_message,
            "is_running": self.is_running
        }
    
    async def stop_current_task(self) -> Dict[str, Any]:
        """
        Stop the currently running task.
        
        Returns:
            Dict with operation result
        """
        async with self._task_lock:
            if not self.is_running:
                return {
                    "success": False,
                    "error": "No task is currently running"
                }
            
            # Cancel progress task
            if self._progress_task and not self._progress_task.done():
                self._progress_task.cancel()
                try:
                    await self._progress_task
                except asyncio.CancelledError:
                    pass
            
            # Mark task as completed (or cancelled)
            if self._current_task:
                self._current_task.status = TaskStatus.COMPLETED
                self._current_task.end_time = time.time()
                self.logger.info(f"Task '{self._current_task.name}' stopped manually")
            
            return {
                "success": True,
                "message": f"Task '{self._current_task.name}' stopped successfully"
            }
    
    async def reset_to_idle(self) -> Dict[str, Any]:
        """
        Reset the task handler to idle state, clearing any completed tasks.
        
        Returns:
            Dict with operation result
        """
        async with self._task_lock:
            if self.is_running:
                return {
                    "success": False,
                    "error": "Cannot reset to idle while a task is running"
                }
            
            # Clear completed task
            if self._current_task:
                task_name = self._current_task.name
                self._current_task = None
                self.logger.info(f"Reset to idle state, cleared task: {task_name}")
                return {
                    "success": True,
                    "message": f"Reset to idle state, cleared task: {task_name}"
                }
            else:
                return {
                    "success": True,
                    "message": "Already in idle state"
                }
    
    async def _run_task_progress(self):
        """
        Internal method to run task progress simulation.
        Updates progress every second for 100 seconds.
        """
        if not self._current_task:
            return
            
        task_name = self._current_task.name
        self.logger.info(f"Starting progress tracking for task: {task_name}")
        
        try:
            # Run for 100 seconds, updating progress every second
            for progress in range(0, 101):
                if not self.is_running:
                    break
                    
                # Update progress
                self._current_task.progress = progress
                
                # Log progress milestones
                if progress % 10 == 0:
                    self.logger.info(f"Task '{task_name}' progress: {progress}%")
                
                # Wait 1 second before next update (except for the last iteration)
                if progress < 100:
                    await asyncio.sleep(0.1)#1.0)
            
            # Task completed successfully
            if self._current_task and self._current_task.status == TaskStatus.RUNNING:
                self._current_task.status = TaskStatus.COMPLETED
                self._current_task.end_time = time.time()
                self._current_task.progress = 100
                
                duration = self._current_task.end_time - self._current_task.start_time
                self.logger.info(f"Task '{task_name}' completed successfully in {duration:.1f} seconds")
                
                # Call the on_complete callback if provided
                if self._on_complete_callback:
                    try:
                        task_info = {
                            "task_name": self._current_task.name,
                            "status": self._current_task.status.value,
                            "progress": self._current_task.progress,
                            "start_time": self._current_task.start_time,
                            "end_time": self._current_task.end_time,
                            "duration": duration,
                            "error_message": self._current_task.error_message
                        }
                        await self._on_complete_callback(task_info)
                        self.logger.info(f"Called on_complete callback for task '{task_name}'")
                    except Exception as e:
                        self.logger.error(f"Error calling on_complete callback for task '{task_name}': {e}")
                
                # Automatically reset to idle state when task reaches 100%
                self.logger.info(f"Task '{task_name}' reached 100%, resetting to idle state")
                self._current_task = None
                
        except asyncio.CancelledError:
            self.logger.info(f"Task '{task_name}' was cancelled")
            if self._current_task:
                self._current_task.status = TaskStatus.COMPLETED
                self._current_task.end_time = time.time()
            raise
            
        except Exception as e:
            # Task failed with error
            error_msg = f"Task '{task_name}' failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self._current_task:
                self._current_task.status = TaskStatus.ERROR
                self._current_task.end_time = time.time()
                self._current_task.error_message = error_msg
                
                # Call the on_complete callback for error cases too
                if self._on_complete_callback:
                    try:
                        duration = self._current_task.end_time - self._current_task.start_time
                        task_info = {
                            "task_name": self._current_task.name,
                            "status": self._current_task.status.value,
                            "progress": self._current_task.progress,
                            "start_time": self._current_task.start_time,
                            "end_time": self._current_task.end_time,
                            "duration": duration,
                            "error_message": self._current_task.error_message
                        }
                        await self._on_complete_callback(task_info)
                        self.logger.info(f"Called on_complete callback for failed task '{task_name}'")
                    except Exception as callback_error:
                        self.logger.error(f"Error calling on_complete callback for failed task '{task_name}': {callback_error}")
    
    async def cleanup(self):
        """
        Cleanup any running tasks and resources.
        """
        self.logger.info("Cleaning up task handler...")
        
        if self._progress_task and not self._progress_task.done():
            self._progress_task.cancel()
            try:
                await self._progress_task
            except asyncio.CancelledError:
                pass
        
        if self._current_task and self._current_task.status == TaskStatus.RUNNING:
            self._current_task.status = TaskStatus.COMPLETED
            self._current_task.end_time = time.time()
        
        self.logger.info("Task handler cleanup complete")


# Global task handler instance
_task_handler: Optional[TaskHandler] = None


def get_task_handler(on_complete: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None) -> TaskHandler:
    """
    Get the global task handler instance (singleton pattern).
    
    Args:
        on_complete: Optional callback to call when a task completes
    
    Returns:
        TaskHandler instance
    """
    global _task_handler
    if _task_handler is None:
        _task_handler = TaskHandler(on_complete=on_complete)
    return _task_handler


def reset_task_handler():
    """
    Reset the global task handler instance (useful for testing).
    """
    global _task_handler
    _task_handler = None


# Convenience functions for easy integration
async def start_task(task_name: str) -> Dict[str, Any]:
    """Convenience function to start a task"""
    handler = get_task_handler()
    return await handler.start_task(task_name)


async def get_current_task() -> Optional[Dict[str, Any]]:
    """Convenience function to get current task status"""
    handler = get_task_handler()
    return await handler.get_current_task()


async def stop_current_task() -> Dict[str, Any]:
    """Convenience function to stop current task"""
    handler = get_task_handler()
    return await handler.stop_current_task()


async def reset_to_idle() -> Dict[str, Any]:
    """Convenience function to reset task handler to idle state"""
    handler = get_task_handler()
    return await handler.reset_to_idle()
