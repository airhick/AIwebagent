import asyncio
import json
import logging
import traceback
from typing import Dict, Set
from aiohttp import web
from datetime import datetime

logger = logging.getLogger('task_queue')

class TaskQueue:
    """Manages asynchronous task execution and WebSocket communication"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_tasks = {}
        self.websockets = set()
        self._running = False
        self._task = None
        self.logger = logging.getLogger('TaskQueue')
        self.max_retries = 3

    async def start(self):
        """Start the task queue processor"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._process_queue())
            self.logger.info("Task queue started")

    async def stop(self):
        """Stop the task queue processor"""
        if self._running:
            self._running = False
            if self._task:
                await self._task
                self._task = None
            self.logger.info("Task queue stopped")

    async def add_task(self, task_data: Dict):
        """Add a new task to the queue"""
        task_id = task_data['id']
        self.logger.info(f"Adding task {task_id} to queue: {task_data['task']}")
        
        # Add metadata to task
        task_data.update({
            'added_time': datetime.now().isoformat(),
            'status': 'QUEUED',
            'retries': 0
        })
        
        await self.queue.put(task_data)
        await self._broadcast_task_update(
            task_id, 
            'QUEUED', 
            f"Task added to queue: {task_data['task']}"
        )

    async def _process_queue(self):
        """Process tasks from the queue"""
        self.logger.info("Starting task queue processor")
        
        while self._running:
            try:
                # Wait for a task with timeout
                try:
                    task_data = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                task_id = task_data['id']
                self.logger.info(f"Processing task {task_id}: {task_data['task']}")
                
                # Check if we have an assistant for this task
                if task_id not in self.active_tasks:
                    self.logger.error(f"No assistant found for task {task_id}")
                    await self._broadcast_task_update(
                        task_id, 
                        'ERROR', 
                        'Task processing failed: No assistant found'
                    )
                    continue

                # Update task status
                await self._broadcast_task_update(
                    task_id, 
                    'RUNNING', 
                    f"Executing task: {task_data['task']}"
                )
                
                try:
                    # Execute the task
                    assistant = self.active_tasks[task_id]['assistant']
                    self.logger.info(f"Executing task {task_id} with assistant")
                    
                    result = await assistant.execute_task(task_data['task'])
                    
                    if result:
                        status = 'COMPLETED'
                        message = 'Task completed successfully'
                    else:
                        if task_data['retries'] < self.max_retries:
                            # Retry the task
                            task_data['retries'] += 1
                            await self.queue.put(task_data)
                            status = 'RETRYING'
                            message = f"Retrying task (attempt {task_data['retries']}/{self.max_retries})"
                        else:
                            status = 'ERROR'
                            message = 'Task failed after max retries'
                    
                    self.logger.info(f"Task {task_id} {status.lower()}: {message}")
                    
                except Exception as e:
                    self.logger.error(f"Task {task_id} failed: {e}")
                    traceback.print_exc()
                    
                    if task_data['retries'] < self.max_retries:
                        # Retry the task
                        task_data['retries'] += 1
                        await self.queue.put(task_data)
                        status = 'RETRYING'
                        message = f"Error occurred, retrying (attempt {task_data['retries']}/{self.max_retries}): {str(e)}"
                    else:
                        status = 'ERROR'
                        message = f"Task failed after max retries: {str(e)}"
                
                # Broadcast final status
                await self._broadcast_task_update(task_id, status, message)
                
                # Clean up completed/failed task
                if status not in ['RETRYING']:
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                        self.logger.info(f"Cleaned up task {task_id}")
                
                # Mark task as done in queue
                self.queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in queue processing: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)  # Back off on error

    async def _broadcast_task_update(self, task_id: int, status: str, message: str):
        """Broadcast task status updates to all connected WebSocket clients"""
        update = {
            'type': 'task_update',
            'taskId': task_id,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        await self._broadcast_message(update)
        self.logger.info(f"Task {task_id} status update: {status} - {message}")

    async def _broadcast_message(self, message: Dict):
        """Send a message to all connected WebSocket clients"""
        if not self.websockets:
            return
            
        data = json.dumps(message)
        dead_sockets = set()
        
        for ws in self.websockets:
            try:
                await ws.send_str(data)
            except Exception as e:
                self.logger.error(f"Error broadcasting message: {e}")
                dead_sockets.add(ws)
        
        # Clean up dead connections
        self.websockets.difference_update(dead_sockets)

    def add_websocket(self, ws: web.WebSocketResponse):
        """Add a new WebSocket connection"""
        self.websockets.add(ws)
        self.logger.info("New WebSocket client connected")

    def remove_websocket(self, ws: web.WebSocketResponse):
        """Remove a WebSocket connection"""
        self.websockets.discard(ws)
        self.logger.info("WebSocket client disconnected") 