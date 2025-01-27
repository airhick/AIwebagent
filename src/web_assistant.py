from playwright.async_api import async_playwright
import cv2
import numpy as np
from PIL import Image
import io
import time
import re
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
import json
import asyncio
import sys
import os
import random
from enum import Enum
import traceback
import logging
import tempfile
from pathlib import Path
import websockets
import base64
from aiohttp import web
import aiohttp
from io import BytesIO
from datetime import datetime
from task_queue import TaskQueue
from task_planner import TaskPlanner
from task_executor import TaskExecutor

# Configure logging for cloud environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(tempfile.gettempdir(), 'web_assistant.log'))
    ]
)

logger = logging.getLogger('web_assistant')

class ErrorLevel(Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
    SUCCESS = 'SUCCESS'

class AIError:
    def __init__(self, code: str, message: str, level: ErrorLevel, details: Dict = None):
        self.code = code
        self.message = message
        self.level = level
        self.details = details or {}
        self.timestamp = time.time()
        self.traceback = traceback.format_exc() if level in [ErrorLevel.ERROR, ErrorLevel.CRITICAL] else None
        
        # Log the error
        logger = logging.getLogger('error_handler')
        log_message = f"{code}: {message}"
        if details:
            log_message += f" - Details: {json.dumps(details)}"
            
        if level == ErrorLevel.DEBUG:
            logger.debug(log_message)
        elif level == ErrorLevel.INFO:
            logger.info(log_message)
        elif level == ErrorLevel.WARNING:
            logger.warning(log_message)
        elif level == ErrorLevel.ERROR:
            logger.error(log_message)
        elif level == ErrorLevel.CRITICAL:
            logger.critical(log_message)

    def to_dict(self) -> Dict:
        """Convert error to dictionary format for cloud logging"""
        return {
            'code': self.code,
            'message': self.message,
            'level': self.level.name,
            'timestamp': self.timestamp,
            'details': self.details,
            'traceback': self.traceback
        }

    def __str__(self):
        return json.dumps(self.to_dict())

@dataclass
class WebElement:
    type: str  # button, link, input, text, etc.
    text: str  # visible text
    location: tuple  # (x, y, width, height)
    attributes: Dict[str, str]  # class, id, etc.
    is_visible: bool
    is_clickable: bool
    confidence: float

class WebAction:
    def __init__(self, action_type: str, selector: str = None, text: str = None, 
                 element: WebElement = None, sub_actions: List['WebAction'] = None):
        self.action_type = action_type
        self.selector = selector
        self.text = text
        self.element = element
        self.sub_actions = sub_actions or []
        self.completed = False
        self.success = False

class VisualTracker:
    def __init__(self, page):
        self.page = page
        self.tracking = False
        self.last_screenshot = None
        self.screenshot_buffer = []
        self.max_buffer_size = 10
        self.logger = logging.getLogger('visual_tracker')

    async def start_tracking(self):
        """Start visual tracking in headless mode"""
        self.tracking = True
        self.logger.info("Started visual tracking in headless mode")

    async def stop_tracking(self):
        """Stop visual tracking"""
        self.tracking = False
        self.screenshot_buffer.clear()
        self.logger.info("Stopped visual tracking")

    async def capture_screenshot(self) -> Optional[bytes]:
        """Capture screenshot in headless mode"""
        try:
            screenshot = await self.page.screenshot(
                type='jpeg',
                quality=80,
                full_page=True
            )
            self.last_screenshot = screenshot
            
            # Maintain screenshot buffer
            self.screenshot_buffer.append(screenshot)
            if len(self.screenshot_buffer) > self.max_buffer_size:
                self.screenshot_buffer.pop(0)
                
            return screenshot
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None

    async def capture_page_state(self) -> Dict:
        """Capture current page state including DOM and visual elements"""
        try:
            # Get page metrics
            metrics = await self.page.evaluate("""() => {
                return {
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    },
                    document: {
                        width: document.documentElement.scrollWidth,
                        height: document.documentElement.scrollHeight
                    }
                }
            }""")
            
            # Get all visible elements
            elements = await self.page.evaluate("""() => {
                const elements = [];
                const allElements = document.querySelectorAll('*');
                
                allElements.forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        elements.push({
                            tag: el.tagName.toLowerCase(),
                            id: el.id,
                            classes: Array.from(el.classList),
                            text: el.textContent.trim(),
                            location: {
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height
                            },
                            isVisible: window.getComputedStyle(el).display !== 'none'
                        });
                    }
                });
                
                return elements;
            }""")
            
            return {
                'timestamp': time.time(),
                'url': self.page.url,
                'metrics': metrics,
                'elements': elements
            }
            
        except Exception as e:
            self.logger.error(f"Page state capture failed: {e}")
            return {}

    async def move_cursor(self, x: float, y: float, clicking: bool = False):
        """Move cursor to specified coordinates with optional click"""
        try:
            if not self.tracking:
                await self.start_tracking()
                
            # Move cursor
            await self.page.mouse.move(x, y)
            
            if clicking:
                await self.page.mouse.down()
                await asyncio.sleep(0.1)
                await self.page.mouse.up()
                
        except Exception as e:
            print(f"Cursor movement error: {e}")
            
    async def highlight_element(self, element: WebElement):
        """Highlight an element on the page"""
        try:
            if not self.tracking or not element:
                return
                
            await self.page.evaluate("""element => {
                const oldOutline = element.style.outline;
                element.style.outline = '2px solid red';
                setTimeout(() => {
                    element.style.outline = oldOutline;
                }, 1000);
            }""", element)
            
        except Exception as e:
            print(f"Element highlight error: {e}")

class IntentAnalyzer:
    def __init__(self):
        self.action_patterns = {
            'navigation': [
                r'(?:go to|visit|open|navigate to) (?:the )?(?:website )?([a-zA-Z0-9\.-]+)',
                r'(?:take me to|bring me to) ([a-zA-Z0-9\.-]+)',
            ],
            'search': [
                r'(?:search for|look up|find|search) (.*)',
                r'(?:what is|who is|tell me about) (.*)',
            ],
            'click': [
                r'(?:click|press|select|choose) (?:the )?([^\.]+)',
                r'(?:select|choose) (?:the )?option ([^\.]+)',
            ],
            'type': [
                r'(?:type|enter|input|write) [\'"](.+?)[\'"]',
                r'(?:fill|put) [\'"](.+?)[\'"] in',
            ],
            'scroll': [
                r'(?:scroll|move) (?:down|up|to) (.*)',
            ],
            'form_fill': [
                r'(?:fill out|complete|submit) (?:the )?form',
            ]
        }
        
        self.context_memory = []
        
    def analyze_request(self, user_request: str) -> List[WebAction]:
        request_lower = user_request.lower()
        actions = []
        
        # Store request in context memory
        self.context_memory.append(request_lower)
        
        # Check each pattern category
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, request_lower)
                for match in matches:
                    action = self._create_action_from_match(action_type, match)
                    if action:
                        actions.append(action)
        
        # Handle complex multi-step actions
        if 'form' in request_lower:
            actions.extend(self._handle_form_filling())
        elif 'login' in request_lower:
            actions.extend(self._handle_login())
        
        return actions

    def _create_action_from_match(self, action_type: str, match) -> WebAction:
        if action_type == 'navigation':
            url = match.group(1)
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'
            return WebAction('navigate', text=url)
            
        elif action_type == 'search':
            search_text = match.group(1)
            return WebAction('search', text=search_text, sub_actions=[
                WebAction('type', selector='input[type="text"], input[type="search"]', text=search_text),
                WebAction('click', selector='button[type="submit"], input[type="submit"]')
            ])
            
        elif action_type == 'click':
            target = match.group(1)
            return WebAction('click', selector=f'text={target}')
            
        elif action_type == 'type':
            input_text = match.group(1)
            return WebAction('type', text=input_text)
            
        return None

    def _handle_form_filling(self) -> List[WebAction]:
        """Handle complex form filling tasks"""
        return [
            WebAction('form_fill', sub_actions=[
                WebAction('scan_form'),
                WebAction('identify_fields'),
                WebAction('fill_fields'),
                WebAction('submit_form')
            ])
        ]

class TaskParser:
    def __init__(self):
        self.common_tasks = {
            'navigation': {
                'patterns': [
                    r'(?:go to|visit|open|navigate to) (?:the )?(?:website )?([a-zA-Z0-9\.-]+)',
                    r'(?:take me to|bring me to) ([a-zA-Z0-9\.-]+)',
                    r'(?:open|launch) ([a-zA-Z0-9\.-]+)'
                ],
                'actions': ['navigate']
            },
            'search': {
                'patterns': [
                    r'(?:search for|look up|find|search) (.*)',
                    r'(?:what is|who is|tell me about) (.*)',
                    r'(?:show me|display) (.*)'
                ],
                'actions': ['type', 'submit']
            },
            'form_filling': {
                'patterns': [
                    r'(?:fill|complete|submit) (?:the )?form',
                    r'(?:sign up|register|create account)',
                    r'(?:login|log in|signin)'
                ],
                'actions': ['identify_fields', 'fill', 'submit']
            },
            'interaction': {
                'patterns': [
                    r'(?:click|press|select|choose) (?:the )?([^\.]+)',
                    r'(?:interact with|use) (?:the )?([^\.]+)'
                ],
                'actions': ['click']
            }
        }
        self.context_history = []
        
    def parse_task(self, user_request: str) -> Dict[str, Any]:
        task_info = {
            'original_request': user_request,
            'identified_tasks': [],
            'context': {},
            'required_actions': []
        }
        
        # Add to context history
        self.context_history.append(user_request)
        
        # Analyze request for each task type
        for task_type, task_data in self.common_tasks.items():
            for pattern in task_data['patterns']:
                matches = re.finditer(pattern, user_request, re.IGNORECASE)
                for match in matches:
                    task_info['identified_tasks'].append({
                        'type': task_type,
                        'matched_text': match.group(0),
                        'extracted_info': match.groups(),
                        'required_actions': task_data['actions']
                    })
        
        # Extract context from the request
        task_info['context'] = self._extract_context(user_request)
        
        return task_info
    
    def _extract_context(self, request: str) -> Dict[str, Any]:
        context = {
            'temporal': self._extract_temporal_context(request),
            'spatial': self._extract_spatial_context(request),
            'conditional': self._extract_conditions(request),
            'preferences': self._extract_preferences(request)
        }
        return context
    
    def _extract_temporal_context(self, request: str) -> Dict[str, Any]:
        temporal = {}
        # Add time-related pattern matching
        time_patterns = {
            'wait': r'(?:wait|pause) (?:for )?(\d+) ?(?:seconds|minutes|hours)',
            'until': r'(?:until|till) (.*?)(?:\.|\n|$)',
            'after': r'(?:after|following) (.*?)(?:\.|\n|$)'
        }
        for key, pattern in time_patterns.items():
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                temporal[key] = match.group(1)
        return temporal

    def _extract_spatial_context(self, request: str) -> Dict[str, Any]:
        spatial = {}
        # Add location-related pattern matching
        location_patterns = {
            'at': r'(?:at|in) (?:the )?(top|bottom|left|right|center)',
            'near': r'(?:near|close to|beside) (.*?)(?:\.|\n|$)',
            'between': r'between (.*?) and (.*?)(?:\.|\n|$)'
        }
        for key, pattern in location_patterns.items():
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                spatial[key] = match.groups()
        return spatial

    def _extract_conditions(self, request: str) -> List[str]:
        conditions = []
        # Add conditional pattern matching
        condition_patterns = [
            r'(?:if|when) (.*?) then',
            r'(?:only if|unless) (.*?)(?:\.|\n|$)',
            r'(?:after|before) (.*?)(?:\.|\n|$)'
        ]
        for pattern in condition_patterns:
            matches = re.finditer(pattern, request, re.IGNORECASE)
            conditions.extend(match.group(1) for match in matches)
        return conditions

    def _extract_preferences(self, request: str) -> Dict[str, Any]:
        preferences = {}
        # Add preference pattern matching
        preference_patterns = {
            'speed': r'(?:quickly|slowly|fast|slow)',
            'accuracy': r'(?:precisely|exactly|approximately)',
            'style': r'(?:style|manner|way) of (.*?)(?:\.|\n|$)'
        }
        for key, pattern in preference_patterns.items():
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                preferences[key] = match.group(0)
        return preferences

class VisualHighlighter:
    def __init__(self, page):
        self.page = page
        self.current_highlights = []
        
    async def highlight_element(self, element: WebElement, color: str = 'yellow', duration: int = 1000):
        """Highlight an element on the page"""
        highlight_js = """
        (element) => {
            const originalBackground = element.style.backgroundColor;
            const originalTransition = element.style.transition;
            element.style.transition = 'background-color 0.3s';
            element.style.backgroundColor = '%s';
            setTimeout(() => {
                element.style.backgroundColor = originalBackground;
                element.style.transition = originalTransition;
            }, %d);
        }
        """ % (color, duration)
        
        await self.page.evaluate(highlight_js, element)
        
    async def highlight_action(self, action: WebAction):
        """Highlight elements involved in an action"""
        if action.element:
            color = self._get_action_color(action.action_type)
            await self.highlight_element(action.element, color)
            
    def _get_action_color(self, action_type: str) -> str:
        colors = {
            'click': '#ffeb3b',  # Yellow
            'type': '#4caf50',   # Green
            'navigate': '#2196f3',# Blue
            'submit': '#ff9800',  # Orange
            'error': '#f44336'    # Red
        }
        return colors.get(action_type, '#e0e0e0')  # Default gray

class RecoveryStrategy:
    def __init__(self, page):
        self.page = page
        self.logger = logging.getLogger('recovery')
        self.recovery_history = []

    async def attempt_recovery(self, error: AIError) -> bool:
        """Attempt to recover from an error"""
        self.logger.info(f"Attempting recovery for error: {error.code}")
        
        recovery_actions = {
            'navigation_error': self._handle_navigation_error,
            'element_not_found': self._handle_element_not_found,
            'timeout_error': self._handle_timeout_error,
            'network_error': self._handle_network_error
        }
        
        handler = recovery_actions.get(error.code)
        if handler:
            try:
                success = await handler(error)
                self.recovery_history.append({
                    'timestamp': time.time(),
                    'error': error.to_dict(),
                    'success': success
                })
                return success
            except Exception as e:
                self.logger.error(f"Recovery failed: {e}")
                return False
        else:
            self.logger.warning(f"No recovery strategy for error code: {error.code}")
            return False

    async def _handle_navigation_error(self, error: AIError) -> bool:
        """Handle navigation-related errors"""
        try:
            await self.page.reload()
            await self.page.wait_for_load_state('networkidle')
            return True
        except Exception as e:
            self.logger.error(f"Navigation recovery failed: {e}")
            return False

    async def _handle_element_not_found(self, error: AIError) -> bool:
        """Handle element not found errors"""
        try:
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Element recovery failed: {e}")
            return False

    async def _handle_timeout_error(self, error: AIError) -> bool:
        """Handle timeout errors"""
        try:
            await self.page.reload(timeout=30000)
            return True
        except Exception as e:
            self.logger.error(f"Timeout recovery failed: {e}")
            return False

    async def _handle_network_error(self, error: AIError) -> bool:
        """Handle network-related errors"""
        try:
            await asyncio.sleep(2)
            await self.page.reload()
            return True
        except Exception as e:
            self.logger.error(f"Network recovery failed: {e}")
            return False

class ActionVerifier:
    def __init__(self, page):
        self.page = page
        self.verification_history = []
        
    async def verify_action(self, action: WebAction, pre_state: List[WebElement], 
                          post_state: List[WebElement]) -> bool:
        """Verify if an action was successful"""
        verification_result = {
            'action': action,
            'timestamp': time.time(),
            'success': False,
            'changes': [],
            'errors': []
        }
        
        try:
            # Verify based on action type
            if action.action_type == 'navigate':
                verification_result['success'] = await self._verify_navigation(action)
            elif action.action_type == 'click':
                verification_result['success'] = await self._verify_click(action, pre_state, post_state)
            elif action.action_type == 'type':
                verification_result['success'] = await self._verify_input(action)
                
            # Record state changes
            verification_result['changes'] = self._detect_state_changes(pre_state, post_state)
            
            # Store verification result
            self.verification_history.append(verification_result)
            
            return verification_result['success']
            
        except Exception as e:
            verification_result['errors'].append(str(e))
            self.verification_history.append(verification_result)
            return False
            
    async def _verify_navigation(self, action: WebAction) -> bool:
        """Verify navigation success"""
        try:
            current_url = self.page.url
            expected_url = action.text
            
            # Check if current URL matches expected URL pattern
            if expected_url in current_url:
                return True
                
            # Wait for any redirects to complete
            await self.page.wait_for_load_state('networkidle')
            
            # Check page title and content for relevance
            title = await self.page.title()
            return any(term in title.lower() for term in expected_url.lower().split('.'))
            
        except Exception as e:
            print(f"Navigation verification failed: {e}")
            return False
            
    async def _verify_click(self, action: WebAction, pre_state: List[WebElement],
                           post_state: List[WebElement]) -> bool:
        """Verify click action success"""
        try:
            # Check for visual changes
            changes = self._detect_state_changes(pre_state, post_state)
            if changes:
                return True
                
            # Check for new elements
            new_elements = await self.page.query_selector_all('[aria-expanded="true"], [data-open="true"]')
            if new_elements:
                return True
                
            # Check for URL changes
            if self.page.url != action.pre_url:
                return True
                
            return False
            
        except Exception as e:
            print(f"Click verification failed: {e}")
            return False
            
    async def _verify_input(self, action: WebAction) -> bool:
        """Verify input action success"""
        try:
            element = await self.page.query_selector(action.selector)
            if element:
                value = await element.get_attribute('value')
                return value == action.text
            return False
            
        except Exception as e:
            print(f"Input verification failed: {e}")
            return False
            
    def _detect_state_changes(self, pre_state: List[WebElement],
                            post_state: List[WebElement]) -> List[Dict]:
        """Detect and record state changes between pre and post action states"""
        changes = []
        
        # Compare element counts
        pre_count = len(pre_state)
        post_count = len(post_state)
        
        if pre_count != post_count:
            changes.append({
                'type': 'count_change',
                'pre': pre_count,
                'post': post_count
            })
            
        # Compare individual elements
        for pre_elem in pre_state:
            matching_post = next(
                (post_elem for post_elem in post_state 
                 if post_elem.location == pre_elem.location), None)
                 
            if not matching_post:
                changes.append({
                    'type': 'element_removed',
                    'element': pre_elem
                })
            elif matching_post.text != pre_elem.text:
                changes.append({
                    'type': 'content_change',
                    'element': pre_elem,
                    'pre_text': pre_elem.text,
                    'post_text': matching_post.text
                })
                
        # Find new elements
        pre_locations = {elem.location for elem in pre_state}
        new_elements = [elem for elem in post_state 
                       if elem.location not in pre_locations]
                       
        for new_elem in new_elements:
            changes.append({
                'type': 'element_added',
                'element': new_elem
            })
            
        return changes

class VisualMonitor:
    def __init__(self):
        self.is_monitoring = False
        self.last_screenshot = None
        self.screenshot_interval = 0.5  # Reduced interval for more responsive monitoring
        self.monitor_thread = None
        self.success_criteria = {}
        self.monitoring_window = None
        self.error_patterns = {
            'popup': ['cookie', 'accept', 'agree', 'consent', 'close', '×', 'x'],
            'error': ['error', 'warning', 'failed', 'invalid', 'incorrect', 'try again'],
            'captcha': ['captcha', 'verify', 'human', 'robot', 'prove'],
            'login': ['sign in', 'log in', 'login required', 'please login'],
            'loading': ['loading', 'please wait', 'processing']
        }
        self.last_error_time = 0
        self.error_cooldown = 2.0  # Seconds to wait before reporting same error type
        self.recovery_queue = []
        
    def start_monitoring(self, success_criteria: Dict[str, Any]):
        """Start visual monitoring with improved error handling"""
        try:
            self.success_criteria = success_criteria
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("👀 Visual monitoring started...")
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("👀 Visual monitoring stopped.")

    def _monitor_loop(self):
        """Monitor the screen for visual changes and errors"""
        try:
            while self.is_monitoring:
                # Capture screen
                screenshot = pyautogui.screenshot()
                screenshot_np = np.array(screenshot)
                screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

                # Process the screenshot
                if self.last_screenshot is not None:
                    # Detect changes
                    changes = self._detect_visual_changes(self.last_screenshot, screenshot_bgr)
                    
                    # Check for errors and obstacles
                    self._check_for_errors(screenshot_bgr)
                    
                    # Check success criteria
                    if self._check_success_criteria(screenshot_bgr, changes):
                        print("✅ Visual success criteria met")
                        self.is_monitoring = False
                        break

                self.last_screenshot = screenshot_bgr
                time.sleep(self.screenshot_interval)

        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            self.is_monitoring = False

    def _check_for_errors(self, frame):
        """Check for common UI obstacles and errors"""
        try:
            # Skip if we recently reported an error
            current_time = time.time()
            if current_time - self.last_error_time < self.error_cooldown:
                return

            # Convert frame to text using OCR (you'll need to implement this)
            text = self._extract_text_from_frame(frame)
            if not text:
                return

            # Check for each error pattern
            for error_type, patterns in self.error_patterns.items():
                if any(pattern.lower() in text.lower() for pattern in patterns):
                    self.last_error_time = current_time
                    recovery_action = self._create_recovery_action(error_type, text)
                    if recovery_action:
                        self.recovery_queue.append(recovery_action)
                        print(f"🔍 Detected {error_type}: {recovery_action['description']}")
                    break

        except Exception as e:
            print(f"❌ Error detection failed: {e}")

    def _create_recovery_action(self, error_type: str, text: str) -> Dict:
        """Create a recovery action based on the error type"""
        if error_type == 'popup':
            return {
                'type': 'click',
                'target': ['close', 'accept', 'agree', '×', 'x'],
                'description': 'Attempting to close popup'
            }
        elif error_type == 'captcha':
            return {
                'type': 'wait',
                'duration': 5,
                'description': 'Detected CAPTCHA, waiting for manual intervention'
            }
        elif error_type == 'login':
            return {
                'type': 'notify',
                'message': 'Login required',
                'description': 'Login wall detected'
            }
        elif error_type == 'loading':
            return {
                'type': 'wait',
                'duration': 2,
                'description': 'Waiting for content to load'
            }
        return None

    def _extract_text_from_frame(self, frame) -> str:
        """Extract text from frame using OCR"""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to get black text on white background
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours to identify text regions
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract text from each region
            extracted_text = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = gray[y:y+h, x:x+w]
                    
                    # Convert region to PIL Image for OCR
                    pil_image = Image.fromarray(roi)
                    
                    # Use pytesseract for OCR (you'll need to install pytesseract)
                    try:
                        import pytesseract
                        text = pytesseract.image_to_string(pil_image)
                        if text.strip():
                            extracted_text.append(text.strip())
                    except ImportError:
                        print("⚠️ Pytesseract not installed. OCR disabled.")
                        return ""
            
            return " ".join(extracted_text)
            
        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
            return ""

    def get_next_recovery_action(self) -> Dict:
        """Get the next recovery action from the queue"""
        return self.recovery_queue.pop(0) if self.recovery_queue else None

    def _detect_visual_changes(self, prev_frame, curr_frame) -> List[Dict]:
        """Detect visual changes between frames"""
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Threshold the difference
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            changes = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small changes
                    x, y, w, h = cv2.boundingRect(contour)
                    changes.append({
                        'region': (x, y, w, h),
                        'area': cv2.contourArea(contour)
                    })
            
            return changes
        except Exception as e:
            print(f"❌ Error detecting changes: {e}")
            return []

    def _check_success_criteria(self, current_frame, changes: List[Dict]) -> bool:
        """Check if success criteria are met"""
        try:
            if not self.success_criteria:
                return False

            if 'expected_text' in self.success_criteria:
                # Check for text in changed regions
                for change in changes:
                    if change['area'] > 1000:  # Significant change
                        return True

            if 'expected_element' in self.success_criteria:
                # Check for specific visual element
                template = self.success_criteria['expected_element']
                result = cv2.matchTemplate(current_frame, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                return max_val > 0.8

            return False
        except Exception as e:
            print(f"❌ Error checking success criteria: {e}")
            return False

class WebJourney:
    """Tracks a sequence of web navigation steps"""
    
    def __init__(self):
        self.steps = []
        self.start_time = datetime.now()
        self.context = {}
        self.errors = []

    def add_step(self, step_info: Dict):
        """Add a step to the journey with validation"""
        try:
            # Ensure required fields
            step = {
                'task': step_info.get('task', 'Unknown Task'),
                'timestamp': step_info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'status': step_info.get('status', 'pending'),
                'url': step_info.get('url', 'about:blank'),
                'title': step_info.get('title', 'No Title'),
                'error': step_info.get('error'),
                'duration': None,
                'retries': 0
            }
            
            # Add step number
            step['step_number'] = len(self.steps) + 1
            
            # Calculate duration if this is not the first step
            if self.steps:
                try:
                    last_time = datetime.strptime(self.steps[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
                    current_time = datetime.strptime(step['timestamp'], '%Y-%m-%d %H:%M:%S')
                    step['duration'] = (current_time - last_time).total_seconds()
                except:
                    step['duration'] = 0
            
            self.steps.append(step)
            
            # Update context with new information
            self._update_context(step)
            
        except Exception as e:
            logger.error(f"Error adding journey step: {e}")
            self.errors.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'step_info': step_info
            })

    def _update_context(self, step: Dict):
        """Update journey context with step information"""
        try:
            # Update current state
            self.context['current_url'] = step['url']
            self.context['current_title'] = step['title']
            self.context['last_action'] = step['task']
            self.context['last_status'] = step['status']
            
            # Track navigation history
            if 'navigation_history' not in self.context:
                self.context['navigation_history'] = []
            
            if step['url'] != 'about:blank':
                self.context['navigation_history'].append({
                    'url': step['url'],
                    'title': step['title'],
                    'timestamp': step['timestamp']
                })
            
            # Track errors
            if step.get('error'):
                if 'errors' not in self.context:
                    self.context['errors'] = []
                self.context['errors'].append({
                    'step': step['step_number'],
                    'error': step['error'],
                    'timestamp': step['timestamp']
                })
            
            # Update success/failure counts
            if 'stats' not in self.context:
                self.context['stats'] = {'success': 0, 'failure': 0, 'pending': 0}
            
            if step['status'] == 'completed':
                self.context['stats']['success'] += 1
            elif step['status'] == 'failed':
                self.context['stats']['failure'] += 1
            else:
                self.context['stats']['pending'] += 1
                
        except Exception as e:
            logger.error(f"Error updating journey context: {e}")
            self.errors.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'context': 'update_context'
            })

    def get_current_state(self) -> Dict:
        """Get current journey state"""
        try:
            if not self.steps:
                return {
                    'url': 'about:blank',
                    'title': 'No Title',
                    'status': 'initializing',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            latest_step = self.steps[-1]
            return {
                'url': latest_step.get('url', 'about:blank'),
                'title': latest_step.get('title', 'No Title'),
                'status': latest_step.get('status', 'unknown'),
                'timestamp': latest_step.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'error': latest_step.get('error')
            }
            
        except Exception as e:
            logger.error(f"Error getting journey state: {e}")
            return {
                'url': 'about:blank',
                'title': 'Error',
                'status': 'error',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }

    def get_journey_summary(self) -> Dict:
        """Get journey summary statistics"""
        try:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'total_steps': len(self.steps),
                'successful_steps': sum(1 for step in self.steps if step['status'] == 'completed'),
                'failed_steps': sum(1 for step in self.steps if step['status'] == 'failed'),
                'total_duration': duration,
                'average_step_duration': duration / len(self.steps) if self.steps else 0,
                'error_count': len(self.errors),
                'navigation_count': len(self.context.get('navigation_history', [])),
                'current_state': self.get_current_state()
            }
            
        except Exception as e:
            logger.error(f"Error generating journey summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

class KeyboardController:
    def __init__(self, page):
        self.page = page
        self.typing_speed = 50  # ms between keystrokes
        self.current_input = None
        self.is_typing = False
        self.keyboard_modifiers = {
            'shift': False,
            'control': False,
            'alt': False,
            'meta': False
        }
        
    async def type_text(self, text: str, element: WebElement = None, natural_typing: bool = True):
        """Type text with natural typing animation"""
        if self.is_typing:
            return False
            
        try:
            self.is_typing = True
            
            if element:
                # Move cursor to element first
                await self.page.evaluate("""(element) => {
                    element.focus();
                    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }""", element)
                
            if natural_typing:
                for char in text:
                    await self.page.keyboard.type(char)
                    # Random delay between keystrokes
                    delay = self.typing_speed + random.randint(-20, 20)
                    await asyncio.sleep(delay / 1000)
            else:
                await self.page.keyboard.type(text)
                
            return True
            
        except Exception as e:
            print(f"Typing error: {e}")
            return False
        finally:
            self.is_typing = False
            
    async def press_key(self, key: str, modifiers: List[str] = None):
        """Press a keyboard key with optional modifiers"""
        try:
            # Apply modifiers
            if modifiers:
                for mod in modifiers:
                    self.keyboard_modifiers[mod.lower()] = True
                    await self.page.keyboard.down(mod)
            
            # Press the key
            await self.page.keyboard.press(key)
            
            # Release modifiers
            if modifiers:
                for mod in modifiers:
                    self.keyboard_modifiers[mod.lower()] = False
                    await self.page.keyboard.up(mod)
                    
            return True
            
        except Exception as e:
            print(f"Key press error: {e}")
            return False
            
    async def send_shortcut(self, shortcut: str):
        """Send keyboard shortcut (e.g., 'Control+C')"""
        try:
            keys = shortcut.split('+')
            modifiers = keys[:-1]
            key = keys[-1]
            
            return await self.press_key(key, modifiers)
            
        except Exception as e:
            print(f"Shortcut error: {e}")
            return False
            
    async def clear_input(self, element: WebElement = None):
        """Clear input field content"""
        try:
            if element:
                await self.page.evaluate("""(element) => {
                    element.focus();
                    element.select();
                }""", element)
            
            await self.send_shortcut('Control+A')
            await self.page.keyboard.press('Backspace')
            return True
            
        except Exception as e:
            print(f"Clear input error: {e}")
            return False

class WebAssistant:
    def __init__(self):
        """Initialize WebAssistant attributes"""
        self.running = True
        self.user_data_dir = '/tmp/web_assistant/browser_data'
        self.playwright = None
        self.browser = None
        self.page = None
        self.visual_tracker = None
        self.recovery_strategy = None
        self.action_verifier = None
        self.visual_highlighter = None
        self.task_planner = TaskPlanner()
        self.task_executor = None
        self.user_interaction = UserInteraction()
        self.intent_analyzer = IntentAnalyzer()
        self.current_journey = None
        self.journey_history = []
        self.viewer_port = 3000
        self.websocket_clients = set()
        self.task_queue = TaskQueue()
        self.app = None
        self.runner = None
        self.site = None
        self._cleanup_required = False
        
        # Create necessary directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.screenshot_dir = os.path.join(self.base_dir, 'browser_screenshots')
        os.makedirs(self.screenshot_dir, exist_ok=True)

    @classmethod
    async def create(cls):
        """Factory method to create and initialize WebAssistant"""
        instance = cls()
        await instance._initialize()
        return instance

    async def _create_live_viewer(self):
        """Create live viewer using WebSocket"""
        from aiohttp import web
        import aiohttp
        
        async def websocket_handler(request):
            """Handle WebSocket connections"""
            ws = web.WebSocketResponse(heartbeat=30)
            await ws.prepare(request)
            
            print(f"New WebSocket connection from {request.remote}")
            self.websocket_clients.add(ws)
            
            try:
                # Send initial state
                if self.page:
                    screenshot = await self.page.screenshot(full_page=True)
                    await self._broadcast_state(
                        screenshot,
                        self.page.url,
                        log={'level': 'info', 'message': 'Connected to cloud browser'}
                    )
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        print(f'WebSocket connection closed with exception {ws.exception()}')
                        break
            finally:
                self.websocket_clients.remove(ws)
                print(f"WebSocket connection closed for {request.remote}")
            return ws

        async def index_handler(request):
            """Serve the viewer HTML page"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Live Browser View</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: #1e1e1e;
                        color: #fff;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    .header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                        padding: 10px;
                        background: #2d2d2d;
                        border-radius: 8px;
                    }
                    .status-bar {
                        display: flex;
                        gap: 20px;
                        font-size: 14px;
                    }
                    .status-item {
                        display: flex;
                        align-items: center;
                        gap: 5px;
                    }
                    .status-dot {
                        width: 8px;
                        height: 8px;
                        border-radius: 50%;
                        background: #4CAF50;
                        transition: background-color 0.3s ease;
                    }
                    .browser-container {
                        background: #2d2d2d;
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    .browser-header {
                        display: flex;
                        align-items: center;
                        padding: 10px;
                        background: #363636;
                        border-bottom: 1px solid #404040;
                    }
                    .url-bar {
                        flex-grow: 1;
                        margin: 0 10px;
                        padding: 5px 10px;
                        background: #1e1e1e;
                        border-radius: 4px;
                        color: #fff;
                        font-size: 14px;
                        font-family: monospace;
                    }
                    .browser-view {
                        position: relative;
                        background: #fff;
                        min-height: 600px;
                        overflow: hidden;
                    }
                    .browser-view img {
                        width: 100%;
                        height: auto;
                        display: block;
                    }
                    .overlay {
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        pointer-events: none;
                    }
                    .action-highlight {
                        position: absolute;
                        border: 2px solid #2196F3;
                        border-radius: 4px;
                        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.3);
                        transition: all 0.3s ease;
                    }
                    .console {
                        margin-top: 20px;
                        padding: 10px;
                        background: #2d2d2d;
                        border-radius: 8px;
                        font-family: monospace;
                        max-height: 200px;
                        overflow-y: auto;
                    }
                    .console-entry {
                        margin: 5px 0;
                        padding: 5px 10px;
                        border-left: 3px solid #404040;
                        font-size: 13px;
                        line-height: 1.4;
                        white-space: pre-wrap;
                        word-break: break-word;
                    }
                    .console-entry.info { border-color: #2196F3; }
                    .console-entry.success { border-color: #4CAF50; }
                    .console-entry.warning { border-color: #FFC107; }
                    .console-entry.error { border-color: #F44336; }
                    .loading {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        font-size: 16px;
                        color: #666;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Live Browser View</h1>
                        <div class="status-bar">
                            <div class="status-item">
                                <div class="status-dot"></div>
                                <span id="connection-status">Connecting...</span>
                            </div>
                            <div class="status-item" id="last-update"></div>
                        </div>
                    </div>
                    <div class="browser-container">
                        <div class="browser-header">
                            <div class="url-bar" id="current-url">about:blank</div>
                        </div>
                        <div class="browser-view">
                            <div class="loading">Connecting to cloud browser...</div>
                            <img id="browser-screen" src="" alt="Browser View" style="opacity: 0;">
                            <div class="overlay" id="action-overlay"></div>
                        </div>
                    </div>
                    <div class="console" id="console-log"></div>
                </div>
                <script>
                    const ws = new WebSocket(`ws://${location.hostname}:${location.port}/ws`);
                    const browserScreen = document.getElementById('browser-screen');
                    const currentUrl = document.getElementById('current-url');
                    const lastUpdate = document.getElementById('last-update');
                    const consoleLog = document.getElementById('console-log');
                    const connectionStatus = document.getElementById('connection-status');
                    const actionOverlay = document.getElementById('action-overlay');
                    let reconnectAttempts = 0;
                    const maxReconnectAttempts = 5;

                    function connect() {
                        ws.onopen = () => {
                            console.log('Connected to cloud browser');
                            connectionStatus.textContent = 'Connected';
                            document.querySelector('.status-dot').style.background = '#4CAF50';
                            document.querySelector('.loading').style.display = 'none';
                            browserScreen.style.opacity = '1';
                            reconnectAttempts = 0;
                            addConsoleEntry('success', 'Connected to cloud browser');
                        };

                        ws.onclose = () => {
                            console.log('Disconnected from cloud browser');
                            connectionStatus.textContent = 'Disconnected';
                            document.querySelector('.status-dot').style.background = '#F44336';
                            document.querySelector('.loading').style.display = 'block';
                            browserScreen.style.opacity = '0';
                            addConsoleEntry('error', 'Connection lost');
                            
                            if (reconnectAttempts < maxReconnectAttempts) {
                                reconnectAttempts++;
                                const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                                addConsoleEntry('info', `Reconnecting in ${delay/1000} seconds...`);
                                setTimeout(() => {
                                    addConsoleEntry('info', 'Attempting to reconnect...');
                                    location.reload();
                                }, delay);
                            } else {
                                addConsoleEntry('error', 'Failed to reconnect after multiple attempts. Please refresh the page manually.');
                            }
                        };

                        ws.onmessage = (event) => {
                            const data = JSON.parse(event.data);
                            
                            if (data.type === 'screenshot') {
                                browserScreen.src = 'data:image/png;base64,' + data.image;
                                currentUrl.textContent = data.url;
                                lastUpdate.textContent = new Date().toLocaleTimeString();
                                document.querySelector('.loading').style.display = 'none';
                                browserScreen.style.opacity = '1';
                            }
                            else if (data.type === 'action') {
                                showAction(data.action);
                            }
                            else if (data.type === 'log') {
                                addConsoleEntry(data.level, data.message);
                            }
                        };

                        ws.onerror = (error) => {
                            console.error('WebSocket error:', error);
                            addConsoleEntry('error', 'Connection error occurred');
                        };
                    }

                    function showAction(action) {
                        const highlight = document.createElement('div');
                        highlight.className = 'action-highlight';
                        highlight.style.left = action.x + 'px';
                        highlight.style.top = action.y + 'px';
                        highlight.style.width = action.width + 'px';
                        highlight.style.height = action.height + 'px';
                        
                        actionOverlay.appendChild(highlight);
                        setTimeout(() => highlight.remove(), 1000);
                    }

                    function addConsoleEntry(level, message) {
                        const entry = document.createElement('div');
                        entry.className = `console-entry ${level}`;
                        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                        consoleLog.appendChild(entry);
                        consoleLog.scrollTop = consoleLog.scrollHeight;
                        
                        // Keep only last 100 entries
                        while (consoleLog.children.length > 100) {
                            consoleLog.removeChild(consoleLog.firstChild);
                        }
                    }

                    // Start connection
                    connect();

                    // Handle visibility change
                    document.addEventListener('visibilitychange', () => {
                        if (document.visibilityState === 'visible' && ws.readyState !== WebSocket.OPEN) {
                            addConsoleEntry('info', 'Page visible, checking connection...');
                            location.reload();
                        }
                    });
                </script>
            </body>
            </html>
            """
            return web.Response(text=html_content, content_type='text/html')

        # Create web application
        app = web.Application()
        app.router.add_get('/', index_handler)
        app.router.add_get('/ws', websocket_handler)

        # Start the server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.viewer_port)
        await site.start()
        print(f"\n🌐 Live browser view available at: http://localhost:{self.viewer_port}")
        return True

    async def _broadcast_state(self, screenshot=None, url=None, log=None):
        """Broadcast current browser state to all connected clients"""
        try:
            if not hasattr(self, 'task_queue'):
                return
                
            # Prepare state with safe defaults
            state = {
                'status': 'Active',
                'url': url or 'about:blank',
                'title': await self._get_page_title(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'screenshot': None
            }
            
            # Handle screenshot if provided
            if screenshot:
                try:
                    screenshot_path = os.path.join(self.screenshot_dir, 'latest.png')
                    if isinstance(screenshot, bytes):
                        with open(screenshot_path, 'wb') as f:
                            f.write(screenshot)
                    elif isinstance(screenshot, str):
                        with open(screenshot_path, 'wb') as f:
                            f.write(base64.b64decode(screenshot))
                    state['screenshot'] = '/browser_screenshots/latest.png'
                except Exception as e:
                    logger.error(f"Screenshot save failed: {e}")
            
            # Broadcast state update
            await self.task_queue._broadcast_message({
                'type': 'state_update',
                'state': state
            })
            
            # Broadcast log message if provided
            if log:
                await self.task_queue._broadcast_message({
                    'type': 'console',
                    'level': log.get('level', 'info'),
                    'message': log.get('message', '')
                })
                
        except Exception as e:
            logger.error(f"State broadcast failed: {e}")
            traceback.print_exc()

    async def _index_handler(self, request):
        """Handle root path request by serving the index.html file"""
        try:
            index_path = os.path.join(self.base_dir, 'browser_viewer', 'index.html')
            with open(index_path, 'r') as f:
                content = f.read()
            return web.Response(text=content, content_type='text/html')
        except Exception as e:
            logger.error(f"Error serving index.html: {e}")
            return web.Response(text="Error loading interface", status=500)

    async def _websocket_handler(self, request):
        """Handle WebSocket connections and messages"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.task_queue.add_websocket(ws)
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data['type'] == 'task':
                            task_data = data['data']
                            self.task_queue.active_tasks[task_data['id']] = {'assistant': self}
                            await self.task_queue.add_task(task_data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {msg.data}")
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self.task_queue.remove_websocket(ws)
        
        return ws

    async def _initialize(self):
        """Initialize the web assistant and all its components"""
        try:
            print("\n📡 Initializing Cloud Environment")
            print("-" * 30)
            
            # Set up workspace and directories
            print("  ⚙️  Setting up workspace...")
            os.makedirs(self.user_data_dir, exist_ok=True)
            os.makedirs(self.screenshot_dir, exist_ok=True)
            
            # Initialize web server
            print("  🖥️  Starting live viewer interface...")
            self.app = web.Application()
            
            # Set up routes with proper static file handling
            self.app.router.add_get('/', self._index_handler)
            self.app.router.add_get('/ws', self._websocket_handler)
            self.app.router.add_static('/browser_screenshots', self.screenshot_dir)
            self.app.router.add_static('/static', os.path.join(self.base_dir, 'browser_viewer'))
            
            # Start the server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, 'localhost', self.viewer_port)
            await self.site.start()
            
            print(f"\n🌐 Live browser view available at: http://localhost:{self.viewer_port}")
            print("     ✓ Live viewer ready")
            
            # Initialize browser
            print("  🌐 Starting cloud browser...")
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox']
            )
            
            self.page = await self.browser.new_page(
                viewport={'width': 1280, 'height': 800}
            )
            
            # Initialize components
            print("  🔧 Initializing components...")
            self.visual_tracker = VisualTracker(self.page)
            self.recovery_strategy = RecoveryStrategy(self.page)
            self.action_verifier = ActionVerifier(self.page)
            self.visual_highlighter = VisualHighlighter(self.page)
            self.task_executor = TaskExecutor(self.page)
            
            # Start visual tracking
            print("  👁️  Starting visual tracking system...")
            await self.visual_tracker.start_tracking()
            
            # Load initial page
            print("  🌍 Loading initial page...")
            await self.page.goto('about:blank')
            
            # Start task queue
            await self.task_queue.start()
            
            # Capture and broadcast initial state
            print("  📸 Capturing initial state...")
            screenshot = await self.visual_tracker.capture_screenshot()
            await self._broadcast_state(
                screenshot=screenshot,
                url=self.page.url,
                log={'level': 'info', 'message': 'Cloud browser initialized and ready'}
            )
            
            print("\n✅ Cloud Browser System Ready")
            print("-" * 30)
            
            self._cleanup_required = True
            return True
            
        except Exception as e:
            print("\n❌ Cloud Initialization Failed:")
            print(f"   Error: {str(e)}")
            traceback.print_exc()
            return False

    def _setup_keyboard_listener(self):
        """Set up keyboard listener for interrupts"""
        def on_press(key):
            try:
                if key.char == 'q':
                    print("\n⚠️ Quit signal received...")
                    self.running = False
                    return False
            except AttributeError:
                pass
            
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        
    async def _handle_cookie_consent(self):
        """Handle cookie consent popups"""
        try:
            # Common cookie consent button selectors
            consent_selectors = [
                'text="Accept all"',
                'text="I agree"',
                'text="Accept cookies"',
                '[aria-label*="consent"]',
                '[aria-label*="cookie"]',
                'button:has-text("Accept")',
                '#consent button',
                '.cookie-consent button',
                '[data-cookieconsent] button'
            ]
            
            for selector in consent_selectors:
                try:
                    button = await self.page.wait_for_selector(selector, timeout=2000)
                    if button:
                        await button.click()
                        print("✅ Cookie consent handled")
                        return
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Cookie consent handling failed: {e}")
            
    async def cleanup(self):
        """Clean up resources and close browser"""
        try:
            self.running = False
            
            # Safely stop visual tracking
            if hasattr(self, 'visual_tracker') and self.visual_tracker:
                try:
                    await self.visual_tracker.stop_tracking()
                except Exception as e:
                    print(f"Visual tracker cleanup warning: {e}")
                
            # Safely close browser
            if hasattr(self, 'browser') and self.browser:
                try:
                    await self.browser.close()
                except Exception as e:
                    print(f"Browser cleanup warning: {e}")
                
            # Safely stop playwright
            if hasattr(self, 'playwright') and self.playwright:
                try:
                    await self.playwright.stop()
                except Exception as e:
                    print(f"Playwright cleanup warning: {e}")
            
            # Clear all component references
            self.visual_tracker = None
            self.recovery_strategy = None
            self.action_verifier = None
            self.visual_highlighter = None
            self.browser = None
            self.playwright = None
            self.page = None
                
            print("✨ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
            # Don't re-raise the exception to ensure cleanup continues

    async def execute_task(self, task_text: str) -> bool:
        """Execute a task and return success status"""
        try:
            logger.info(f"Executing task: {task_text}")
            
            # Parse and analyze the task
            task_info = self.task_planner.analyze_request(task_text)
            
            # Start a new journey if needed
            if not self.current_journey:
                self.current_journey = WebJourney()
            
            # Add task to journey with safe defaults
            step = {
                'task': task_text,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'started',
                'url': self.page.url if self.page else 'about:blank',
                'title': 'Initializing...'
            }
            self.current_journey.add_step(step)
            
            try:
                # Execute the task
                result = await self.task_executor.execute_task_step(task_info, {})
                
                # Update step with current state
                step.update({
                    'status': 'completed' if result else 'failed',
                    'url': self.page.url if self.page else 'about:blank',
                    'title': await self._get_page_title()
                })
                
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                step.update({
                    'status': 'failed',
                    'error': str(e)
                })
                result = False
            
            # Capture and broadcast new state
            try:
                screenshot = await self._capture_screenshot_safely()
                await self._broadcast_state(
                    screenshot=screenshot,
                    url=step.get('url'),
                    log={
                        'level': 'success' if result else 'error',
                        'message': f"Task {'completed successfully' if result else 'failed'}"
                    }
                )
            except Exception as e:
                logger.error(f"State broadcast failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            traceback.print_exc()
            
            try:
                if self.current_journey and self.current_journey.steps:
                    self.current_journey.steps[-1]['status'] = 'failed'
                    self.current_journey.steps[-1]['error'] = str(e)
                
                await self._broadcast_state(
                    url=self.page.url if self.page else 'about:blank',
                    log={
                        'level': 'error',
                        'message': f"Task failed: {str(e)}"
                    }
                )
            except:
                pass
            
            return False

    async def _get_page_title(self) -> str:
        """Safely get the page title"""
        try:
            if self.page:
                return await self.page.title() or 'No Title'
        except:
            pass
        return 'No Title'

    async def _capture_screenshot_safely(self) -> Optional[bytes]:
        """Safely capture a screenshot"""
        try:
            if self.page:
                return await self.page.screenshot(full_page=True)
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
        return None

    async def _verify_visual_change(self, before_screenshot, after_screenshot) -> bool:
        """Verify that visual changes occurred between screenshots"""
        try:
            # Convert screenshots to numpy arrays for comparison
            before_img = np.array(Image.open(io.BytesIO(before_screenshot)))
            after_img = np.array(Image.open(io.BytesIO(after_screenshot)))
            
            # Calculate difference between images
            diff = cv2.absdiff(before_img, after_img)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Calculate the percentage of pixels that changed
            change_percentage = (np.count_nonzero(gray_diff) * 100) / gray_diff.size
            
            # Consider the change significant if more than 5% of pixels changed
            return change_percentage > 5
            
        except Exception as e:
            print(f"⚠️ Visual verification error: {e}")
            return False

    async def _perform_element_action(self, step: Dict, element: WebElement):
        """Perform the specified action on the element"""
        bbox = element.location
        center_x = bbox[0] + bbox[2]/2
        center_y = bbox[1] + bbox[3]/2
        
        # Move cursor to element
        await self.visual_tracker.move_cursor(center_x, center_y)
        
        # Perform action based on type
        if step.get('action_type') == 'click':
            await self.visual_tracker.move_cursor(center_x, center_y, clicking=True)
            await self.page.click(f"text={element.text}")
            await asyncio.sleep(0.3)
            await self.visual_tracker.move_cursor(center_x, center_y, clicking=False)
            
        elif step.get('action_type') == 'type':
            await self.keyboard_controller.type_text(step['text'], element)
            
        elif step.get('action_type') == 'select':
            await self.page.select_option(f"select", step['value'])

    async def _find_relevant_elements(self, elements: List[WebElement], visual_cues: List[str]) -> List[WebElement]:
        """Find elements matching the visual cues"""
        relevant_elements = []
        for element in elements:
            if element.is_visible:
                # Check element text
                if any(cue.lower() in element.text.lower() for cue in visual_cues):
                    relevant_elements.append(element)
                    continue
                
                # Check element attributes
                for attr_value in element.attributes.values():
                    if any(cue.lower() in str(attr_value).lower() for cue in visual_cues):
                        relevant_elements.append(element)
                        break
        
        return relevant_elements

    async def _scroll_and_search(self, visual_cues: List[str]):
        """Scroll the page while looking for visual cues"""
        print("📜 Scrolling page to find relevant elements...")
        
        # Get page height
        page_height = await self.page.evaluate('document.documentElement.scrollHeight')
        current_position = 0
        scroll_step = 300
        
        while current_position < page_height:
            # Scroll down
            await self.page.evaluate(f'window.scrollBy(0, {scroll_step})')
            current_position += scroll_step
            
            # Wait for content to load
            await asyncio.sleep(0.5)
            
            # Check if we found any relevant elements
            elements = await self.visual_tracker.capture_page_state()
            if await self._find_relevant_elements(elements, visual_cues):
                print("✨ Found relevant elements!")
                return
        
        # Scroll back to top
        await self.page.evaluate('window.scrollTo(0, 0)')

    async def _execute_action(self, action: str, elements: List[WebElement], context: Dict) -> bool:
        """Execute a single action on the most relevant element"""
        try:
            if not elements:
                return False
                
            # Sort elements by relevance
            sorted_elements = sorted(elements, 
                                  key=lambda e: self._calculate_element_relevance(e, action, context),
                                  reverse=True)
            
            best_element = sorted_elements[0]
            
            if action == 'click':
                await self.page.click(self._build_selector(best_element))
                await self.page.wait_for_load_state('networkidle')
                return True
                
            elif action == 'type':
                text = self._get_text_for_element(best_element, context)
                if text:
                    await self.page.fill(self._build_selector(best_element), text)
                    return True
                    
            elif action == 'select':
                value = self._get_value_for_element(best_element, context)
                if value:
                    await self.page.select_option(self._build_selector(best_element), value)
                    return True
                    
            return False
            
        except Exception as e:
            print(f"❌ Action execution failed: {e}")
            return False

    async def _handle_obstacles(self):
        """Handle any obstacles that appear during task execution"""
        while True:
            recovery_action = self.visual_monitor.get_next_recovery_action()
            if not recovery_action:
                break
                
            print(f"🔄 Handling obstacle: {recovery_action['description']}")
            await self._handle_recovery_action(recovery_action)

    async def _verify_step_completion(self, step: Dict, initial_elements: List[WebElement], 
                                    final_elements: List[WebElement]) -> bool:
        """Verify if a step was completed successfully"""
        try:
            # Check for expected text
            if 'expected_text' in step['success_criteria']:
                expected_texts = step['success_criteria']['expected_text']
                page_text = await self.page.evaluate('document.body.innerText')
                if not any(text.lower() in page_text.lower() for text in expected_texts):
                    return False
            
            # Check for visual changes
            changes = self._detect_visual_changes(initial_elements, final_elements)
            if not changes:
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Step verification failed: {e}")
            return False

    def _calculate_element_relevance(self, element: WebElement, action: str, context: Dict) -> float:
        """Calculate how relevant an element is for the current action and context"""
        score = 0.0
        
        # Base score from element properties
        if element.is_clickable:
            score += 1.0
        if element.is_visible:
            score += 1.0
        
        # Action-specific scoring
        if action == 'click' and element.type in ['button', 'link', 'submit']:
            score += 2.0
        elif action == 'type' and element.type in ['input', 'textarea']:
            score += 2.0
        elif action == 'select' and element.type == 'select':
            score += 2.0
        
        # Context matching
        if context:
            text_lower = element.text.lower()
            for info in context.values():
                if isinstance(info, dict):
                    for value in info.values():
                        if str(value).lower() in text_lower:
                            score += 0.5
        
        return score

    def _get_text_for_element(self, element: WebElement, context: Dict) -> str:
        """Determine what text to enter into an element based on its attributes and context"""
        element_type = element.type.lower()
        element_name = element.attributes.get('name', '').lower()
        
        # Email field
        if 'email' in element_name or 'mail' in element_name:
            return "test@example.com"  # You should get this from user preferences
            
        # Name fields
        if 'name' in element_name:
            if 'first' in element_name:
                return "John"  # You should get this from user preferences
            if 'last' in element_name:
                return "Doe"  # You should get this from user preferences
            return "John Doe"  # You should get this from user preferences
            
        # Phone field
        if 'phone' in element_name or 'tel' in element_name:
            return "1234567890"  # You should get this from user preferences
            
        # Try to find matching context
        for info_type, info in context.items():
            if isinstance(info, dict):
                for key, value in info.items():
                    if key.lower() in element_name:
                        return str(value)
        
        return ""

    def _get_value_for_element(self, element: WebElement, context: Dict) -> str:
        """Determine what value to select for an element based on context"""
        element_name = element.attributes.get('name', '').lower()
        
        # Time selection
        if 'time' in element_name and context.get('time_info', {}).get('time'):
            return context['time_info']['time']
            
        # Date selection
        if 'date' in element_name and context.get('time_info', {}).get('date'):
            return context['time_info']['date']
            
        # Number of people/items
        if 'number' in element_name or 'quantity' in element_name:
            if context.get('quantity_info', {}).get('people'):
                return context['quantity_info']['people']
            if context.get('quantity_info', {}).get('items'):
                return context['quantity_info']['items']
        
        return ""

    async def _handle_recovery_action(self, action: Dict):
        """Handle a recovery action from the visual monitor"""
        try:
            if action['type'] == 'click':
                # Try each target text/pattern
                for target in action['target']:
                    try:
                        # Try multiple selector strategies
                        selectors = [
                            f'text="{target}"',
                            f'[aria-label*="{target}"]',
                            f'button:has-text("{target}")',
                            f'[role="button"]:has-text("{target}")',
                            f'[title*="{target}"]',
                            f'[class*="close"], [class*="dismiss"]'
                        ]
                        
                        for selector in selectors:
                            try:
                                element = await self.page.wait_for_selector(selector, timeout=2000)
                                if element:
                                    await element.click()
                                    print(f"✅ Successfully clicked {target}")
                                    return True
                            except:
                                continue
                    except:
                        continue
                        
            elif action['type'] == 'wait':
                print(f"⏳ Waiting for {action['duration']} seconds...")
                await asyncio.sleep(action['duration'])
                return True
                
            elif action['type'] == 'notify':
                print(f"ℹ️ {action['message']}")
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Recovery action failed: {e}")
            return False

    async def _infer_tasks_from_visual(self, user_request: str, current_elements: List[WebElement]) -> List[Dict]:
        """Infer tasks by analyzing the visual state of the page"""
        tasks = []
        request_lower = user_request.lower()
        
        try:
            # For restaurant booking
            if 'restaurant' in request_lower or 'book' in request_lower or 'table' in request_lower:
                # If we're not on a restaurant-related page, start with search
                if 'google.com' in self.page.url:
                    tasks.append({
                        'type': 'search',
                        'query': 'restaurants near me booking table',
                        'extracted_info': ['restaurants near me booking table'],
                        'required_actions': ['type', 'submit']
                    })
                
                # Look for booking-related elements
                booking_elements = [
                    elem for elem in current_elements
                    if any(term in elem.text.lower() for term in ['book', 'reserve', 'table', 'reservation'])
                    and elem.is_clickable
                ]
                
                if booking_elements:
                    tasks.append({
                        'type': 'click',
                        'matched_text': booking_elements[0].text,
                        'extracted_info': [booking_elements[0].text],
                        'required_actions': ['click']
                    })
            
            # If no specific tasks were inferred, try to find relevant clickable elements
            if not tasks:
                relevant_elements = [
                    elem for elem in current_elements
                    if elem.is_clickable and self._is_element_relevant_to_request(elem, request_lower)
                ]
                
                if relevant_elements:
                    best_element = max(relevant_elements, 
                                     key=lambda e: self._calculate_relevance_score(e, request_lower))
                    tasks.append({
                        'type': 'click',
                        'matched_text': best_element.text,
                        'extracted_info': [best_element.text],
                        'required_actions': ['click']
                    })
        
        except Exception as e:
            print(f"❌ Error inferring tasks: {e}")
        
        return tasks or [{
            'type': 'search',
            'query': user_request,
            'extracted_info': [user_request],
            'required_actions': ['type', 'submit']
        }]

    async def _attempt_visual_recovery(self, failed_task: Dict = None, user_request: str = None) -> bool:
        """Attempt recovery by analyzing the visual state of the page"""
        try:
            # Capture current visual state
            current_elements = await self.visual_tracker.capture_page_state()
            
            # Look for error messages or popups that might need to be dismissed
            error_elements = [
                elem for elem in current_elements
                if any(term in elem.text.lower() for term in ['error', 'warning', 'close', 'dismiss', 'ok'])
                and elem.is_clickable
            ]
            
            for error_elem in error_elements:
                try:
                    await self.page.click(self._build_selector(error_elem))
                    await asyncio.sleep(1)
                except:
                    continue
            
            # Try to find elements relevant to the original request
            if user_request:
                relevant_elements = [
                    elem for elem in current_elements
                    if elem.is_clickable and self._is_element_relevant_to_request(elem, user_request.lower())
                ]
                
                if relevant_elements:
                    best_element = max(relevant_elements, 
                                     key=lambda e: self._calculate_relevance_score(e, user_request.lower()))
                    try:
                        await self.page.click(self._build_selector(best_element))
                        return True
                    except:
                        pass
            
            # If we're stuck, try common recovery actions
            recovery_actions = [
                lambda: self.page.go_back(),
                lambda: self.page.reload(),
                lambda: self.page.wait_for_load_state('networkidle'),
                lambda: asyncio.sleep(2)
            ]
            
            for action in recovery_actions:
                try:
                    await action()
                    await asyncio.sleep(1)
                    return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            print(f"❌ Visual recovery failed: {e}")
            return False

    def _is_element_relevant_to_request(self, element: WebElement, request: str) -> bool:
        """Determine if an element is relevant to the user's request"""
        if not element.is_visible:
            return False
            
        element_text = element.text.lower()
        
        # Check for direct text matches
        if any(word in element_text for word in request.split()):
            return True
            
        # Check for semantic relevance
        relevant_terms = {
            'restaurant': ['book', 'reserve', 'table', 'dining', 'reservation', 'eat'],
            'search': ['search', 'find', 'look', 'discover'],
            'navigation': ['menu', 'home', 'about', 'go', 'visit'],
            'booking': ['book', 'reserve', 'schedule', 'appointment', 'time']
        }
        
        request_terms = set(request.split())
        for category, terms in relevant_terms.items():
            if any(term in request_terms for term in terms):
                if any(term in element_text for term in terms):
                    return True
        
        return False

    async def _execute_task_step(self, task_info: Dict, context: Dict) -> bool:
        """Execute a single task step"""
        try:
            action_type = task_info.get('action_type', 'search')
            logger.info(f"Executing task step: {action_type}")
            
            if action_type == 'navigate':
                url = task_info.get('url', '')
                if not url.startswith(('http://', 'https://')):
                    url = f'https://{url}'
                logger.info(f"Navigating to: {url}")
                await self.page.goto(url, wait_until='networkidle')
                return True
                
            elif action_type == 'search':
                query = task_info.get('query', '')
                if not self.page.url.startswith('https://www.google.com'):
                    logger.info("Navigating to Google")
                    await self.page.goto('https://www.google.com', wait_until='networkidle')
                
                logger.info(f"Searching for: {query}")
                # Handle cookie consent if present
                await self._handle_cookie_consent()
                
                # Type search query
                await self.page.fill('input[name="q"]', query)
                await self.page.press('input[name="q"]', 'Enter')
                await self.page.wait_for_load_state('networkidle')
                return True
                
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Task step execution failed: {e}")
            traceback.print_exc()
            return False

    async def _perform_action(self, action: WebAction) -> bool:
        """Perform a web action with improved error handling"""
        try:
            if action.action_type == 'navigate':
                await self.page.goto(action.text, wait_until='networkidle')
                return True
            
            elif action.action_type == 'click':
                element = await self.page.wait_for_selector(action.selector, timeout=5000)
                if element:
                    await element.click()
                    await self.page.wait_for_load_state('networkidle')
                    return True
            
            elif action.action_type == 'type':
                element = await self.page.wait_for_selector(action.selector, timeout=5000)
                if element:
                    await element.fill(action.text)
                    return True
            
            elif action.action_type == 'select':
                element = await self.page.wait_for_selector(action.selector, timeout=5000)
                if element:
                    await element.select_option(value=action.text)
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Action execution failed: {e}")
            return False

    async def _extract_context_from_action(self, action: WebAction, 
                                       elements_before: List[WebElement],
                                       elements_after: List[WebElement]) -> Dict:
        """Extract context from action and state changes"""
        context = {}
        
        # Store action-specific context
        if action.action_type == 'navigate':
            context['last_url'] = action.text
            context['navigation_timestamp'] = time.time()
            
        elif action.action_type == 'search':
            context['last_search'] = action.text
            context['search_timestamp'] = time.time()
            
        elif action.action_type == 'form_fill':
            context['form_data'] = action.text
            context['form_timestamp'] = time.time()
        
        # Analyze state changes
        new_elements = [elem for elem in elements_after 
                       if elem not in elements_before]
        for elem in new_elements:
            if elem.type == 'text' and elem.is_visible:
                context[f'extracted_text_{len(context)}'] = elem.text
        
        return context

    async def _extract_information(self, elements: List[WebElement], task: Dict) -> Dict:
        """Extract relevant information from page elements based on task"""
        info = {}
        
        # Extract based on task type
        if task['type'] == 'search':
            info['search_results'] = [
                elem.text for elem in elements 
                if elem.type in ['link', 'heading'] and elem.is_visible
            ]
            
        elif task['type'] == 'navigation':
            info['page_title'] = self.page.title()
            info['current_url'] = self.page.url
            
        elif task['type'] == 'form_fill':
            info['form_fields'] = [
                elem.attributes for elem in elements
                if elem.type in ['input', 'select', 'textarea']
            ]
        
        info['timestamp'] = time.time()
        info['type'] = task['type']
        
        return info

    def _infer_actions_from_elements(self, user_request: str, elements: List[WebElement]) -> List[WebAction]:
        """Infer actions from visual elements when direct parsing fails"""
        actions = []
        request_lower = user_request.lower()
        
        # Handle restaurant booking specific case
        if 'restaurant' in request_lower and ('book' in request_lower or 'reservation' in request_lower):
            # First, search for restaurants
            search_text = f"restaurants in {self._extract_location(request_lower)}"
            actions.append(WebAction('type', 
                                  selector='input[type="text"], input[type="search"]', 
                                  text=search_text))
            actions.append(WebAction('click', 
                                  selector='button[type="submit"], input[type="submit"]'))
        
        # Handle other common cases
        for element in elements:
            if self._is_relevant_element_for_request(element, request_lower):
                actions.append(WebAction(
                    action_type=self._determine_action_type(element, request_lower),
                    selector=self._build_selector(element),
                    element=element
                ))
        
        return actions

    def _extract_location(self, request: str) -> str:
        """Extract location from request"""
        location_patterns = [
            r'in ([a-zA-Z\s]+)',
            r'at ([a-zA-Z\s]+)',
            r'near ([a-zA-Z\s]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, request)
            if match:
                return match.group(1).strip()
        return ""

    def _is_relevant_element_for_request(self, element: WebElement, request: str) -> bool:
        """Determine if an element is relevant for the current request"""
        if not element.is_visible:
            return False
            
        element_text = element.text.lower()
        element_type = element.type.lower()
        
        relevant_terms = {
            'restaurant': ['book', 'reserve', 'reservation', 'table', 'dining'],
            'search': ['search', 'find', 'look'],
            'navigation': ['menu', 'home', 'about'],
            'booking': ['book now', 'reserve', 'schedule', 'appointment']
        }
        
        for category, terms in relevant_terms.items():
            if any(term in request for term in terms):
                if any(term in element_text for term in terms):
                    return True
                
        return False

    def _determine_action_type(self, element: WebElement, request: str) -> str:
        """Determine what type of action to take on an element"""
        element_type = element.type.lower()
        
        if element_type in ['input', 'textarea']:
            return 'type'
        elif element_type in ['button', 'a'] or element.is_clickable:
            return 'click'
        elif element_type == 'select':
            return 'select'
        
        return 'click'  # Default to click

    def _build_selector(self, element: WebElement) -> str:
        """Build a reliable selector for an element"""
        selectors = []
        
        # Try ID first
        if element.attributes.get('id'):
            selectors.append(f"#{element.attributes['id']}")
        
        # Try other attributes
        if element.attributes.get('name'):
            selectors.append(f"[name='{element.attributes['name']}']")
        
        if element.attributes.get('class'):
            selectors.append(f".{element.attributes['class'].replace(' ', '.')}")
        
        # Use text content
        if element.text:
            selectors.append(f"text={element.text}")
        
        return selectors[0] if selectors else f"{element.type}"

    def _determine_success_criteria(self, user_request: str) -> Dict[str, Any]:
        """Determine visual success criteria based on user request"""
        criteria = {}
        
        # Extract key terms from request
        terms = user_request.lower().split()
        
        if 'go to' in user_request.lower():
            # For navigation, look for the website name
            site_name = user_request.lower().split('go to')[-1].strip()
            criteria['expected_text'] = site_name
        
        return criteria

    async def _capture_visual_state(self) -> Dict:
        """Capture the current visual state of the page"""
        visual_state = {
            'timestamp': time.time(),
            'url': self.page.url,
            'viewport': await self.page.viewport_size(),
            'elements': []
        }
        
        try:
            # Capture screenshot
            screenshot = await self.page.screenshot(type='jpeg', quality=50)
            visual_state['screenshot'] = screenshot
            
            # Get visible elements
            elements = await self.visual_tracker.capture_page_state()
            
            # Store element information
            for element in elements:
                if element.is_visible:
                    element_info = {
                        'type': element.type,
                        'text': element.text,
                        'location': element.location,
                        'is_clickable': element.is_clickable,
                        'confidence': element.confidence
                    }
                    visual_state['elements'].append(element_info)
            
            # Capture page metrics
            metrics = await self.page.evaluate("""() => ({
                scrollHeight: document.documentElement.scrollHeight,
                scrollWidth: document.documentElement.scrollWidth,
                clientHeight: document.documentElement.clientHeight,
                clientWidth: document.documentElement.clientWidth
            })""")
            visual_state['metrics'] = metrics
            
            # Capture visible text content
            text_content = await self.page.evaluate("""() => {
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                let text = [];
                let node;
                while(node = walker.nextNode()) {
                    const element = node.parentElement;
                    if (element.offsetWidth > 0 && element.offsetHeight > 0) {
                        text.push(node.textContent.trim());
                    }
                }
                return text.filter(t => t.length > 0);
            }""")
            visual_state['visible_text'] = text_content
            
        except Exception as e:
            print(f"⚠️ Error capturing visual state: {e}")
            # Return partial state if capture failed
            visual_state['capture_error'] = str(e)
        
        return visual_state

    def _format_url(self, url: str) -> str:
        """Format URL properly with protocol and domain"""
        if not url:
            return 'https://www.google.com'  # Default to Google if no URL provided
            
        url = str(url).strip().lower()
        
        # Common domain mappings
        domain_mappings = {
            'reddit': 'www.reddit.com',
            'google': 'www.google.com',
            'facebook': 'www.facebook.com',
            'twitter': 'twitter.com',
            'youtube': 'www.youtube.com',
            'amazon': 'www.amazon.com',
            'github': 'github.com',
            'linkedin': 'www.linkedin.com'
        }
        
        # Remove common prefixes if present
        for prefix in ['go to ', 'visit ', 'open ', 'navigate to ']:
            if url.startswith(prefix):
                url = url[len(prefix):]
        
        # Check if it's a known domain
        for key, domain in domain_mappings.items():
            if key in url:
                url = domain
                break
        
        # Add www. if needed
        if not any(p in url for p in ['www.', 'http://', 'https://']):
            if '.' not in url:
                url = f'www.{url}.com'
            elif not url.startswith('www.'):
                url = f'www.{url}'
        
        # Add https:// if needed
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        return url

    async def _handle_navigation(self, task: Dict, context: Dict) -> WebAction:
        """Handle navigation tasks with context awareness"""
        try:
            url = task.get('extracted_info', [None])[0] if task.get('extracted_info') else None
            
            # Format URL properly
            formatted_url = self._format_url(url)
            
            # Check if we're already on the page
            current_url = self.page.url
            if formatted_url == current_url:
                return None
                
            return WebAction('navigate', text=formatted_url)
            
        except Exception as e:
            print(f"❌ Navigation handling error: {e}")
            return WebAction('navigate', text='https://www.google.com')  # Fallback to Google

    async def _handle_search(self, task: Dict, context: Dict) -> WebAction:
        """Handle search tasks with context awareness"""
        search_text = task['extracted_info'][0] if task['extracted_info'] else ''
        
        # Incorporate context if relevant
        if context.get('last_search'):
            # Refine search based on previous searches
            if search_text.startswith('more about'):
                search_text = f"{context['last_search']} {search_text[11:]}"
        
        return WebAction('search', text=search_text, sub_actions=[
            WebAction('type', selector='input[type="text"], input[type="search"]', text=search_text),
            WebAction('click', selector='button[type="submit"], input[type="submit"]')
        ])

    async def _handle_click(self, task: Dict, context: Dict) -> WebAction:
        """Handle click tasks with context awareness"""
        target = task['extracted_info'][0] if task['extracted_info'] else ''
        
        # Try to find the best selector
        selectors = [
            f'text="{target}"',
            f'[aria-label*="{target}"]',
            f'[title*="{target}"]',
            f'#{target}',
            f'.{target}'
        ]
        
        # Check each selector
        for selector in selectors:
            element = self.page.query_selector(selector)
            if element:
                return WebAction('click', selector=selector)
        
        # If no exact match, try fuzzy matching with visible elements
        visible_elements = self.visual_tracker.last_elements
        best_match = None
        best_score = 0
        
        for element in visible_elements:
            if element.is_clickable and element.text:
                score = self._calculate_text_similarity(target, element.text)
                if score > best_score and score > 0.7:  # Threshold for similarity
                    best_score = score
                    best_match = element
        
        if best_match:
            return WebAction('click', selector=self._build_selector(best_match))
            
        return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple case-insensitive containment check
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.8
        
        # Calculate word overlap
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        return overlap / max(len(words1), len(words2))

    def _handle_form(self, task: Dict, context: Dict) -> WebAction:
        """Handle form filling tasks with context awareness"""
        form_data = task.get('form_data', {})
        
        # Incorporate saved form data from context
        if context.get('form_data'):
            form_data.update(context['form_data'])
        
        sub_actions = []
        
        # Find form fields
        form_fields = self.page.query_selector_all('input, select, textarea')
        for field in form_fields:
            field_type = field.get_attribute('type')
            field_name = field.get_attribute('name')
            field_id = field.get_attribute('id')
            
            # Match field with form data
            value = None
            for key, val in form_data.items():
                if key.lower() in [field_name, field_id, field_type]:
                    value = val
                    break
            
            if value:
                selector = f'#{field_id}' if field_id else f'[name="{field_name}"]'
                sub_actions.append(WebAction('type', selector=selector, text=value))
        
        # Add submit action
        sub_actions.append(WebAction('click', selector='button[type="submit"], input[type="submit"]'))
        
        return WebAction('form_fill', sub_actions=sub_actions)

    async def _handle_generic_action(self, task: Dict, context: Dict) -> WebAction:
        """Handle generic tasks by inferring action from context"""
        # Try to infer action from visible elements
        elements = self.visual_tracker.last_elements
        actions = self._infer_actions_from_elements(task['matched_text'], elements)
        
        if actions:
            return actions[0]  # Return first inferred action
            
        # If no action could be inferred, try recovery strategies
        return await self._fallback_action(task, context)

    async def _fallback_action(self, task: Dict, context: Dict) -> WebAction:
        """Provide fallback actions when primary action fails"""
        # Try to find any interactive element that might be relevant
        relevant_elements = [
            elem for elem in self.visual_tracker.last_elements
            if elem.is_clickable and self._is_element_relevant(elem, task['matched_text'])
        ]
        
        if relevant_elements:
            best_element = max(relevant_elements, 
                             key=lambda e: self._calculate_relevance_score(e, task['matched_text']))
            return WebAction('click', selector=self._build_selector(best_element))
        
        return None

    def _calculate_relevance_score(self, element: WebElement, task_text: str) -> float:
        """Calculate how relevant an element is to the task"""
        score = 0.0
        
        # Text content similarity
        if element.text:
            score += self._calculate_text_similarity(task_text, element.text) * 2.0
        
        # Attribute similarity
        for attr in ['title', 'aria-label', 'placeholder', 'name']:
            if attr in element.attributes:
                score += self._calculate_text_similarity(task_text, element.attributes[attr])
        
        # Boost score for certain element types
        if element.type in ['button', 'link']:
            score *= 1.5
        
        # Boost score for high confidence elements
        score *= element.confidence
        
        return score

    async def start_journey(self, user_request: str):
        """Start a new web journey based on user request"""
        print(f"\n🎯 Starting new journey: {user_request}")
        self.current_journey = WebJourney()
        
        # Parse the complex task into subtasks
        subtasks = await self._parse_complex_task(user_request)
        
        for subtask in subtasks:
            try:
                # Execute subtask with visual monitoring
                success = await self._execute_subtask(subtask)
                
                if not success:
                    # Try alternative approach
                    success = await self._try_alternative_approach(subtask)
                
                # Record journey step
                self.current_journey.add_step({
                    'url': self.page.url,
                    'action': subtask,
                    'visual_state': await self._capture_visual_state(),
                    'success': success
                })
                
                if not success:
                    print(f"❌ Failed to complete subtask: {subtask}")
                    break
                    
            except Exception as e:
                print(f"❌ Error during journey: {e}")
                await self._attempt_recovery()
        
        # Store completed journey
        self.journey_history.append(self.current_journey)

    async def _parse_complex_task(self, user_request: str) -> List[Dict]:
        """Parse complex task into sequence of subtasks"""
        subtasks = []
        request_lower = user_request.lower()
        
        # Example: "Book a table at an Italian restaurant in Geneva for tomorrow"
        if 'restaurant' in request_lower and 'book' in request_lower:
            location = self._extract_location(request_lower)
            cuisine = self._extract_cuisine(request_lower)
            time_info = self._extract_time_info(request_lower)
            
            subtasks = [
                {
                    'type': 'search',
                    'query': f'best {cuisine} restaurants in {location}',
                    'success_criteria': {'expected_text': ['restaurant', 'booking', 'reserve']}
                },
                {
                    'type': 'analyze_results',
                    'criteria': ['rating', 'reviews', 'availability'],
                    'success_criteria': {'expected_element': 'restaurant_card'}
                },
                {
                    'type': 'select_restaurant',
                    'criteria': ['high_rating', 'has_booking'],
                    'success_criteria': {'expected_text': ['book', 'reserve', 'reservation']}
                },
                {
                    'type': 'booking_flow',
                    'details': {'time': time_info, 'cuisine': cuisine},
                    'success_criteria': {'expected_text': ['confirmation', 'success']}
                }
            ]
        
        return subtasks

    async def _execute_subtask(self, subtask: Dict) -> bool:
        """Execute a single subtask with visual verification"""
        print(f"\n📝 Executing subtask: {subtask['type']}")
        
        # Start visual monitoring for this subtask
        self.visual_monitor.start_monitoring(subtask['success_criteria'])
        
        try:
            if subtask['type'] == 'search':
                success = await self._handle_search_subtask(subtask)
            elif subtask['type'] == 'analyze_results':
                success = await self._handle_analysis_subtask(subtask)
            elif subtask['type'] == 'select_restaurant':
                success = await self._handle_selection_subtask(subtask)
            elif subtask['type'] == 'booking_flow':
                success = await self._handle_booking_subtask(subtask)
            else:
                print(f"❌ Unknown subtask type: {subtask['type']}")
                return False
            
            # Wait for visual verification
            await asyncio.sleep(2)
            return success
            
        except Exception as e:
            print(f"❌ Subtask execution failed: {e}")
            return False
        finally:
            self.visual_monitor.stop_monitoring()

    async def _handle_search_subtask(self, subtask: Dict) -> bool:
        """Handle search subtask with improved error handling and retry logic"""
        try:
            # Ensure we're on Google
            if 'google.com' not in self.page.url:
                await self.page.goto('https://www.google.com', wait_until='networkidle')
                await self._handle_cookie_consent()  # Handle consent after navigation
            
            # Get the query
            query = subtask.get('query', 'restaurants near me booking table')
            
            # Wait for the page to be fully interactive
            await self.page.wait_for_load_state('domcontentloaded')
            await self.page.wait_for_load_state('networkidle')
            
            # Try different search input selectors with better error handling
            search_selectors = [
                'textarea[name="q"]',  # Google's new textarea search input
                'input[name="q"]',
                'input[type="search"]',
                'input[title*="Search"]',
                'input[title*="Rechercher"]'  # French version
            ]
            
            for selector in search_selectors:
                try:
                    # Wait for the element to be visible and enabled
                    search_input = await self.page.wait_for_selector(selector, 
                                                                   state='visible',
                                                                   timeout=5000)
                    if search_input:
                        # Clear existing text if any
                        await search_input.fill('')
                        await self.page.wait_for_timeout(100)  # Small delay for stability
                        
                        # Type the search query
                        await search_input.fill(query)
                        await self.page.wait_for_timeout(100)  # Small delay for stability
                        
                        # Try different submit methods
                        try:
                            await search_input.press('Enter')
                        except:
                            submit_button = await self.page.query_selector('button[type="submit"]')
                            if submit_button:
                                await submit_button.click()
                        
                        # Wait for search results
                        await self.page.wait_for_load_state('networkidle')
                        
                        # Verify search was successful
                        if await self.page.query_selector('#search'):
                            print(f"✅ Successfully searched for: {query}")
                            return True
                except Exception as e:
                    print(f"⚠️ Search attempt failed with selector {selector}: {e}")
                    continue
            
            raise Exception("All search attempts failed")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False

    async def _handle_analysis_subtask(self, subtask: Dict) -> bool:
        """Handle results analysis subtask"""
        try:
            # Wait for results to load
            await self.page.wait_for_selector('.g', timeout=5000)
            
            # Get all results
            results = await self.page.query_selector_all('.g')
            
            # Analyze based on criteria
            for result in results:
                if await self._meets_criteria(result, subtask['criteria']):
                    return True
            
            return False
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False

    async def _handle_selection_subtask(self, subtask: Dict) -> bool:
        """Handle restaurant selection subtask"""
        try:
            # Find the best matching restaurant based on criteria
            restaurants = await self.page.query_selector_all('.restaurant-card')
            best_match = None
            best_score = 0
            
            for restaurant in restaurants:
                score = await self._calculate_restaurant_score(restaurant, subtask['criteria'])
                if score > best_score:
                    best_score = score
                    best_match = restaurant
            
            if best_match:
                await best_match.click()
                await self.page.wait_for_load_state('networkidle')
                return True
                
            return False
        except Exception as e:
            print(f"❌ Selection failed: {e}")
            return False

    async def _handle_booking_subtask(self, subtask: Dict) -> bool:
        """Handle booking flow subtask"""
        try:
            # Find and fill booking form
            await self._fill_booking_form(subtask['details'])
            
            # Submit booking
            submit_button = await self.page.query_selector('button[type="submit"]')
            if submit_button:
                await submit_button.click()
                await self.page.wait_for_load_state('networkidle')
                
                # Check for confirmation
                confirmation = await self.page.query_selector('.confirmation-message')
                return bool(confirmation)
            
            return False
        except Exception as e:
            print(f"❌ Booking failed: {e}")
            return False

    async def _meets_criteria(self, element, criteria: List[str]) -> bool:
        """Check if an element meets the specified criteria"""
        try:
            text = await element.inner_text()
            return all(criterion.lower() in text.lower() for criterion in criteria)
        except:
            return False

    async def _calculate_restaurant_score(self, restaurant, criteria: List[str]) -> float:
        """Calculate a score for a restaurant based on criteria"""
        try:
            score = 0.0
            text = await restaurant.inner_text()
            
            # Check ratings
            if 'high_rating' in criteria:
                rating_match = re.search(r'(\d\.\d)/5', text)
                if rating_match:
                    score += float(rating_match.group(1))
            
            # Check booking availability
            if 'has_booking' in criteria and 'book' in text.lower():
                score += 2.0
                
            return score
        except:
            return 0.0

    async def _fill_booking_form(self, details: Dict):
        """Fill out a booking form with provided details"""
        try:
            # Fill time if provided
            if details.get('time'):
                time_input = await self.page.query_selector('input[type="time"]')
                if time_input:
                    await time_input.fill(details['time'])
            
            # Fill other details as needed
            if details.get('cuisine'):
                cuisine_input = await self.page.query_selector('input[name="cuisine"]')
                if cuisine_input:
                    await cuisine_input.fill(details['cuisine'])
                    
        except Exception as e:
            print(f"❌ Form filling failed: {e}")
            raise

    async def _try_alternative_approach(self, failed_subtask: Dict) -> bool:
        """Try alternative approaches when primary approach fails"""
        print("🔄 Trying alternative approach...")
        
        alternatives = {
            'search': [
                self._try_direct_url,
                self._try_different_search_engine
            ],
            'analyze_results': [
                self._try_different_filters,
                self._try_sorting_options
            ],
            'select_restaurant': [
                self._try_different_selection_criteria,
                self._try_alternative_booking_site
            ],
            'booking_flow': [
                self._try_phone_number_extraction,
                self._try_alternative_booking_platform
            ]
        }
        
        task_type = failed_subtask['type']
        if task_type in alternatives:
            for alternative in alternatives[task_type]:
                try:
                    if await alternative(failed_subtask):
                        return True
                except Exception as e:
                    print(f"❌ Alternative approach failed: {e}")
                    continue
        
        return False

    def _extract_page_text(self) -> str:
        """Extract visible text from the current page"""
        try:
            return self.page.evaluate('() => document.body.innerText')
        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
            return ""

    async def _attempt_recovery(self):
        """Attempt to recover from a failed task"""
        print("🔄 Attempting recovery...")
        try:
            # Check if browser is still alive
            if not self.browser or not self.page:
                await self._initialize()
                return
            
            # Try to reload the page
            try:
                await self.page.reload(wait_until='networkidle')
                print("✅ Page reloaded successfully")
                return
            except Exception as e:
                print(f"⚠️ Page reload failed: {e}")
            
            # If reload fails, try to create a new page
            try:
                self.page = await self.browser.new_page()
                print("✅ Created new page")
                return
            except Exception as e:
                print(f"⚠️ New page creation failed: {e}")
            
            # If all else fails, reinitialize
            await self._initialize()
            
        except Exception as e:
            print(f"⚠️ Recovery failed: {e}")
            # Don't raise the exception to allow the task to continue

    async def _try_direct_url(self, failed_subtask: Dict) -> bool:
        """Try direct URL approach when primary approach fails"""
        print("🔄 Trying direct URL approach...")
        
        # Implement direct URL approach logic here
        # This is a placeholder and should be replaced with actual logic
        return False

    async def _try_different_search_engine(self, failed_subtask: Dict) -> bool:
        """Try different search engine when primary approach fails"""
        print("🔄 Trying different search engine...")
        
        # Implement different search engine logic here
        # This is a placeholder and should be replaced with actual logic
        return False

    async def _try_different_filters(self, failed_subtask: Dict) -> bool:
        """Try different filters when primary approach fails"""
        print("🔄 Trying different filters...")
        
        # Implement different filters logic here
        # This is a placeholder and should be replaced with actual logic
        return False

    async def _try_sorting_options(self, failed_subtask: Dict) -> bool:
        """Try different sorting options when primary approach fails"""
        print("🔄 Trying different sorting options...")
        
        # Implement different sorting options logic here
        # This is a placeholder and should be replaced with actual logic
        return False

    async def _try_different_selection_criteria(self, failed_subtask: Dict) -> bool:
        """Try different selection criteria when primary approach fails"""
        print("🔄 Trying different selection criteria...")
        
        # Implement different selection criteria logic here
        # This is a placeholder and should be replaced with actual logic
        return False

    async def _try_alternative_booking_site(self, failed_subtask: Dict) -> bool:
        """Try alternative booking site when primary approach fails"""
        print("🔄 Trying alternative booking site...")
        
        # Implement alternative booking site logic here
        # This is a placeholder and should be replaced with actual logic
        return False

    async def _try_phone_number_extraction(self, failed_subtask: Dict) -> bool:
        """Try phone number extraction when primary approach fails"""
        print("🔄 Trying phone number extraction...")
        
        # Implement phone number extraction logic here
        # This is a placeholder and should be replaced with actual logic
        return False

    async def _try_alternative_booking_platform(self, failed_subtask: Dict) -> bool:
        """Try alternative booking platform when primary approach fails"""
        print("🔄 Trying alternative booking platform...")
        
        # Implement alternative booking platform logic here
        # This is a placeholder and should be replaced with actual logic
        return False 

class TaskPlanner:
    """Plans and analyzes web tasks"""
    
    def __init__(self):
        self.common_actions = {
            'navigate': ['go to', 'visit', 'open', 'browse', 'navigate'],
            'search': ['search for', 'find', 'look up', 'search'],
            'click': ['click', 'press', 'select', 'choose'],
            'type': ['type', 'enter', 'input', 'write'],
            'scroll': ['scroll', 'move down', 'move up'],
            'wait': ['wait', 'pause', 'delay']
        }

    def analyze_request(self, request: str) -> Dict:
        """Analyze user request and create task plan"""
        request_lower = request.lower()
        
        # Check for navigation actions
        for action in self.common_actions['navigate']:
            if action in request_lower:
                url = self._extract_url(request)
                if url:
                    return {
                        'action_type': 'navigate',
                        'url': url,
                        'original_request': request
                    }
        
        # Check for search actions
        for action in self.common_actions['search']:
            if action in request_lower:
                search_term = self._extract_search_term(request)
                if search_term:
                    return {
                        'action_type': 'search',
                        'query': search_term,
                        'original_request': request
                    }
        
        # Default to navigation if it looks like a URL
        if self._looks_like_url(request):
            return {
                'action_type': 'navigate',
                'url': request,
                'original_request': request
            }
            
        # Default to search
        return {
            'action_type': 'search',
            'query': request,
            'original_request': request
        }

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from text"""
        words = text.split()
        for word in words:
            if self._looks_like_url(word):
                return word
        return None

    def _looks_like_url(self, text: str) -> bool:
        """Check if text looks like a URL"""
        url_patterns = [
            r'https?://',
            r'www\.',
            r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}',
        ]
        return any(re.search(pattern, text) for pattern in url_patterns)

    def _extract_search_term(self, text: str) -> str:
        """Extract search term from text"""
        for action in self.common_actions['search']:
            if action in text.lower():
                # Get everything after the action word
                parts = text.lower().split(action, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return text.strip()

class UserInteraction:
    def __init__(self):
        self.interaction_history = []
        self.user_preferences = {}
        self.required_fields = {
            'restaurant_booking': {
                'personal': ['name', 'email', 'phone'],
                'booking': ['date', 'time', 'guests'],
                'preferences': ['cuisine', 'price_range', 'location']
            },
            'online_shopping': {
                'personal': ['name', 'email', 'phone'],
                'shipping': ['address', 'city', 'postal_code'],
                'payment': ['card_type', 'card_number', 'expiry']
            }
        }
        
    async def gather_missing_info(self, flow_type: str, context: Dict) -> Dict:
        """Gather missing information from user while maintaining context"""
        if not flow_type or flow_type not in self.required_fields:
            return context
            
        print("\n📝 I need some additional information to proceed:")
        
        # Track what we're asking for and why
        missing_info = {}
        required_fields = self.required_fields[flow_type]
        
        for category, fields in required_fields.items():
            for field in fields:
                # Check if we already have this info in context
                if not self._has_field_info(field, context):
                    # Get field value from user
                    value = await self._ask_user_for_field(field, category, context)
                    if value:
                        missing_info[field] = value
                        # Update interaction history
                        self._log_interaction('user_input', field, value)
                        
        # Merge new info with existing context
        return {**context, **missing_info}
        
    def _has_field_info(self, field: str, context: Dict) -> bool:
        """Check if field information exists in context"""
        # Check direct field presence
        if field in context:
            return True
            
        # Check in nested dictionaries
        for value in context.values():
            if isinstance(value, dict) and field in value:
                return True
                
        return False
        
    async def _ask_user_for_field(self, field: str, category: str, context: Dict) -> str:
        """Ask user for specific information with context-aware prompts"""
        # Format field name for display
        display_name = field.replace('_', ' ').title()
        
        # Create context-aware prompt
        prompt = self._create_context_aware_prompt(field, category, context)
        
        # Get user input with validation
        while True:
            print(f"\n💬 {prompt}")
            value = input("> ").strip()
            
            if not value:
                print("⚠️ This information is required to proceed. Please provide a value.")
                continue
                
            if self._validate_input(field, value):
                return value
            else:
                print(f"⚠️ Invalid {display_name}. Please try again.")
                
    def _create_context_aware_prompt(self, field: str, category: str, context: Dict) -> str:
        """Create a context-aware prompt for user input"""
        prompts = {
            'name': "What's your name?",
            'email': "What's your email address?",
            'phone': "What's your phone number?",
            'date': self._get_date_prompt(context),
            'time': self._get_time_prompt(context),
            'guests': "How many people are in your party?",
            'cuisine': "What type of cuisine do you prefer?",
            'price_range': "What's your preferred price range (budget/moderate/expensive)?",
            'location': "In which area are you looking for?",
            'address': "What's your shipping address?",
            'city': "What city are you in?",
            'postal_code': "What's your postal code?",
            'card_type': "What type of payment card will you use?",
            'card_number': "What's your card number?",
            'expiry': "What's your card expiry date (MM/YY)?"
        }
        
        return prompts.get(field, f"Please enter your {field.replace('_', ' ')}:")
        
    def _get_date_prompt(self, context: Dict) -> str:
        """Create a context-aware date prompt"""
        if 'time_info' in context and context['time_info'].get('time'):
            return f"For what date would you like to book at {context['time_info']['time']}?"
        return "What date would you like to book for?"
        
    def _get_time_prompt(self, context: Dict) -> str:
        """Create a context-aware time prompt"""
        if 'time_info' in context and context['time_info'].get('date'):
            return f"What time would you like to book on {context['time_info']['date']}?"
        return "What time would you like to book for?"
        
    def _validate_input(self, field: str, value: str) -> bool:
        """Validate user input based on field type"""
        validators = {
            'email': lambda x: '@' in x and '.' in x,
            'phone': lambda x: len(x) >= 10 and x.replace('+', '').replace('-', '').isdigit(),
            'date': lambda x: bool(re.match(r'^\d{1,2}(/|-)\d{1,2}(/|-)\d{2,4}$', x)),
            'time': lambda x: bool(re.match(r'^\d{1,2}(:\d{2})?\s*(am|pm)?$', x.lower())),
            'guests': lambda x: x.isdigit() and 1 <= int(x) <= 100,
            'postal_code': lambda x: len(x) >= 4,
            'card_number': lambda x: len(x.replace(' ', '')) >= 15,
            'expiry': lambda x: bool(re.match(r'^\d{2}/\d{2}$', x))
        }
        
        if field in validators:
            return validators[field](value)
        return True  # No validation for other fields
        
    def _log_interaction(self, interaction_type: str, field: str, value: str):
        """Log user interaction for context maintenance"""
        self.interaction_history.append({
            'timestamp': time.time(),
            'type': interaction_type,
            'field': field,
            'value': value if field not in ['card_number', 'expiry'] else '****'  # Mask sensitive data
        })
        
    def get_interaction_summary(self) -> str:
        """Get a summary of user interactions"""
        if not self.interaction_history:
            return "No interactions recorded"
            
        summary = "\n📋 Interaction Summary:"
        for interaction in self.interaction_history:
            timestamp = time.strftime('%H:%M:%S', time.localtime(interaction['timestamp']))
            summary += f"\n{timestamp} - Asked for {interaction['field']}: {interaction['value']}"
            
        return summary

class RequestAnalyzer:
    def __init__(self):
        # Initialize transformers for NLP tasks
        self.nlp_model = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.context_memory = []
        
    async def analyze_request(self, request: str, context: Dict = None) -> Dict:
        """Analyze user request using AI to extract intents and entities"""
        analysis = {
            'original_request': request,
            'timestamp': time.time(),
            'intents': [],
            'entities': {},
            'complexity': 0,
            'required_capabilities': set(),
            'context_references': [],
            'temporal_info': {},
            'spatial_info': {}
        }
        
        try:
            # Load models on first use
            if not self.nlp_model:
                from transformers import pipeline
                self.nlp_model = pipeline('text-classification', model='distilbert-base-uncased')
                self.intent_classifier = pipeline('zero-shot-classification',
                                               candidate_labels=['navigation', 'search', 'booking', 
                                                              'form_fill', 'purchase', 'interaction'])
                self.entity_extractor = pipeline('ner')
            
            # Classify primary intent
            intent_result = await asyncio.to_thread(self.intent_classifier, 
                                                  request)
            analysis['intents'] = [
                {'label': label, 'score': score} 
                for label, score in zip(intent_result['labels'], 
                                      intent_result['scores'])
                if score > 0.3  # Only keep confident predictions
            ]
            
            # Extract entities
            entities = await asyncio.to_thread(self.entity_extractor, request)
            for entity in entities:
                entity_type = entity['entity'].lower()
                if entity_type not in analysis['entities']:
                    analysis['entities'][entity_type] = []
                analysis['entities'][entity_type].append({
                    'text': entity['word'],
                    'score': entity['score']
                })
            
            # Analyze complexity
            analysis['complexity'] = self._analyze_complexity(request, analysis)
            
            # Extract temporal and spatial information
            analysis['temporal_info'] = self._extract_temporal_info(request)
            analysis['spatial_info'] = self._extract_spatial_info(request)
            
            # Identify required capabilities
            analysis['required_capabilities'] = self._identify_capabilities(analysis)
            
            # Find context references
            if context:
                analysis['context_references'] = self._find_context_references(request, context)
            
            # Update context memory
            self.context_memory.append({
                'timestamp': time.time(),
                'request': request,
                'analysis': analysis
            })
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ Request analysis error: {e}")
            # Return basic analysis if AI processing fails
            return self._fallback_analysis(request)
    
    def _analyze_complexity(self, request: str, analysis: Dict) -> int:
        """Analyze task complexity based on various factors"""
        complexity = 0
        
        # Add complexity for each detected intent
        complexity += len(analysis['intents'])
        
        # Add complexity for each entity
        complexity += sum(len(entities) for entities in analysis['entities'].values())
        
        # Add complexity for conjunctions
        complexity += request.lower().count(' and ') + request.lower().count(' then ')
        
        # Add complexity for conditional statements
        complexity += request.lower().count(' if ') + request.lower().count(' unless ')
        
        return complexity
    
    def _extract_temporal_info(self, request: str) -> Dict:
        """Extract temporal information from request"""
        temporal_info = {}
        
        # Extract dates
        date_patterns = [
            r'(?:on|for|at) (\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* ?(?:\d{4})?)',
            r'(?:on|for|at) (tomorrow|today|next week|next month)',
            r'in (\d+) (?:days?|weeks?|months?)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                temporal_info['date'] = match.group(1)
                break
        
        # Extract times
        time_patterns = [
            r'(?:at|by) (\d{1,2}(?::\d{2})? ?(?:am|pm)?)',
            r'(\d{1,2} o\'clock)',
            r'(\d{1,2}(?::\d{2})? hours?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                temporal_info['time'] = match.group(1)
                break
        
        return temporal_info
    
    def _extract_spatial_info(self, request: str) -> Dict:
        """Extract spatial/location information from request"""
        spatial_info = {}
        
        # Extract locations
        location_patterns = [
            r'(?:in|at|near|around) ([\w\s]+(?:city|town|area|region|country))',
            r'(?:in|at|near|around) ([\w\s]+)(?=\W|$)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                spatial_info['location'] = match.group(1)
                break
        
        return spatial_info
    
    def _identify_capabilities(self, analysis: Dict) -> Set[str]:
        """Identify required capabilities based on analysis"""
        capabilities = set()
        
        # Add capabilities based on intents
        intent_capability_map = {
            'navigation': {'web_navigation', 'url_handling'},
            'search': {'text_search', 'result_analysis'},
            'booking': {'form_filling', 'date_handling', 'payment_processing'},
            'form_fill': {'form_filling', 'data_validation'},
            'purchase': {'payment_processing', 'cart_management'},
            'interaction': {'click_handling', 'keyboard_input'}
        }
        
        for intent in analysis['intents']:
            if intent['label'] in intent_capability_map:
                capabilities.update(intent_capability_map[intent['label']])
        
        # Add capabilities based on entities
        if analysis['entities'].get('date') or analysis['entities'].get('time'):
            capabilities.add('temporal_processing')
        if analysis['entities'].get('location'):
            capabilities.add('location_processing')
        
        return capabilities
    
    def _find_context_references(self, request: str, context: Dict) -> List[Dict]:
        """Find references to previous context in the request"""
        references = []
        
        # Look for pronouns and context references
        reference_patterns = [
            r'(?:it|that|this|there|them|those|these)',
            r'(?:the same|previous|last|earlier)',
            r'(?:again|repeat|another)'
        ]
        
        for pattern in reference_patterns:
            matches = re.finditer(pattern, request, re.IGNORECASE)
            for match in matches:
                references.append({
                    'type': 'pronoun_reference',
                    'text': match.group(),
                    'span': match.span(),
                    'possible_referents': self._find_possible_referents(match.group(), context)
                })
        
        return references
    
    def _find_possible_referents(self, reference: str, context: Dict) -> List[str]:
        """Find possible referents for a context reference"""
        referents = []
        
        # Check recent context memory
        for past_context in reversed(self.context_memory[-5:]):  # Look at last 5 interactions
            if 'entities' in past_context['analysis']:
                for entity_type, entities in past_context['analysis']['entities'].items():
                    for entity in entities:
                        referents.append({
                            'text': entity['text'],
                            'type': entity_type,
                            'confidence': entity['score'],
                            'recency': time.time() - past_context['timestamp']
                        })
        
        return sorted(referents, key=lambda x: (x['recency'], -x['confidence']))
    
    def _fallback_analysis(self, request: str) -> Dict:
        """Provide basic analysis when AI processing fails"""
        return {
            'original_request': request,
            'timestamp': time.time(),
            'intents': [{'label': 'generic', 'score': 1.0}],
            'entities': {},
            'complexity': 1,
            'required_capabilities': {'basic_web_interaction'},
            'context_references': [],
            'temporal_info': {},
            'spatial_info': {}
        }

class AIDebugger:
    def __init__(self):
        self.debug_history = []
        self.max_history_size = 100
        self.error_analyzer = None
    
    async def analyze_failure(self, failed_action: Dict, error: Exception, 
                            visual_state: Dict, context: Dict) -> Dict:
        try:
            if not self.error_analyzer:
                from transformers import pipeline
                self.error_analyzer = pipeline('zero-shot-classification',
                                            candidate_labels=[
                                                'element_not_found',
                                                'element_not_clickable',
                                                'network_error',
                                                'timeout',
                                                'invalid_state',
                                                'permission_denied',
                                                'navigation_blocked',
                                                'form_validation',
                                                'resource_missing'
                                            ])
            
            error_text = f"{str(error)} {self._get_error_context(failed_action)}"
            error_classification = await asyncio.to_thread(self.error_analyzer, error_text)
            
            error_type = error_classification['labels'][0]
            confidence = error_classification['scores'][0]
            
            if confidence < 0.5:
                error = AIError(
                    "LOW_CONFIDENCE_CLASSIFICATION",
                    "Low confidence in error classification",
                    ErrorLevel.WARNING,
                    {
                        "confidence": f"{confidence:.2%}",
                        "suggested_type": error_type,
                        "action": "Manual verification recommended"
                    }
                )
                ErrorHandler.log_error(error)
            
            diagnosis = self._build_diagnosis(error_type, failed_action, visual_state)
            recovery_plan = self._generate_recovery_plan(diagnosis, context)
            
            return {
                'timestamp': time.time(),
                'action': failed_action,
                'error': str(error),
                'error_type': error_type,
                'confidence': confidence,
                'diagnosis': diagnosis,
                'recovery_plan': recovery_plan,
                'visual_state': visual_state,
                'context': context
            }
            
        except Exception as e:
            error = AIError(
                "DEBUG_ANALYSIS_ERROR",
                "Failed to analyze debug information",
                ErrorLevel.ERROR,
                {
                    "cause": str(e),
                    "action": failed_action.get('action_type', 'unknown'),
                    "traceback": traceback.format_exc()
                }
            )
            ErrorHandler.log_error(error)
            return {
                'timestamp': time.time(),
                'action': failed_action,
                'error': str(error),
                'error_type': 'analysis_failed',
                'confidence': 0.0,
                'diagnosis': None,
                'recovery_plan': None,
                'visual_state': visual_state,
                'context': context
            }
    
    def _get_error_context(self, failed_action: Dict) -> str:
        context_parts = []
        if 'selector' in failed_action:
            context_parts.append(f"Selector: {failed_action['selector']}")
        if 'element' in failed_action:
            context_parts.append(f"Element type: {failed_action['element'].get('type')}")
        if 'visual_state' in failed_action:
            context_parts.append("Visual state available")
        return ' '.join(context_parts)
    
    def _build_diagnosis(self, error_type: str, failed_action: Dict, visual_state: Dict) -> Dict:
        diagnosis_map = {
            'element_not_found': self._diagnose_missing_element,
            'element_not_clickable': self._diagnose_not_clickable,
            'network_error': self._diagnose_network_error,
            'timeout': self._diagnose_timeout
        }
        
        diagnosis = {
            'error_type': error_type,
            'description': '',
            'likely_causes': [],
            'visual_factors': [],
            'state_issues': []
        }
        
        diagnose_func = diagnosis_map.get(error_type, self._diagnose_generic_error)
        diagnosis.update(diagnose_func(failed_action, visual_state))
        return diagnosis
    
    def _diagnose_missing_element(self, action: Dict, visual_state: Dict) -> Dict:
        diagnosis = {
            'description': 'Target element could not be found on the page',
            'likely_causes': [
                'Element not yet loaded',
                'Element in different frame/shadow DOM',
                'Dynamic content not rendered',
                'Incorrect selector'
            ],
            'visual_factors': [],
            'state_issues': []
        }
        
        if visual_state:
            elements = visual_state.get('elements', [])
            if any(elem.get('type') == 'loading' for elem in elements):
                diagnosis['likely_causes'].insert(0, 'Page still loading')
                diagnosis['visual_factors'].append('Loading indicator visible')
            
            if any(elem.get('type') == 'iframe' for elem in elements):
                diagnosis['likely_causes'].insert(0, 'Element might be in iframe')
                diagnosis['visual_factors'].append('Iframes present on page')
        
        return diagnosis
    
    def _diagnose_not_clickable(self, action: Dict, visual_state: Dict) -> Dict:
        diagnosis = {
            'description': 'Element found but not clickable',
            'likely_causes': [
                'Element covered by overlay/modal',
                'Element disabled',
                'Element outside viewport',
                'Element too small/invisible'
            ],
            'visual_factors': [],
            'state_issues': []
        }
        
        if visual_state and action.get('element'):
            target_bbox = action['element'].get('bbox', (0, 0, 0, 0))
            
            for elem in visual_state.get('elements', []):
                if elem.get('bbox') and self._check_overlap(target_bbox, elem['bbox']):
                    diagnosis['visual_factors'].append(
                        f"Element overlapped by {elem.get('type', 'unknown')}"
                    )
            
            viewport = visual_state.get('viewport', {'width': 0, 'height': 0})
            if not self._is_in_viewport(target_bbox, viewport):
                diagnosis['visual_factors'].append('Element outside viewport')
        
        return diagnosis
    
    def _diagnose_network_error(self, action: Dict) -> Dict:
        return {
            'description': 'Network-related error occurred',
            'likely_causes': [
                'Connection interrupted',
                'Resource not available',
                'CORS/Security policy',
                'Server error'
            ],
            'visual_factors': [],
            'state_issues': ['Network request failed']
        }
    
    def _diagnose_timeout(self, action: Dict, visual_state: Dict) -> Dict:
        diagnosis = {
            'description': 'Operation timed out',
            'likely_causes': [
                'Slow network connection',
                'Heavy page load',
                'Resource not responding',
                'Infinite loading state'
            ],
            'visual_factors': [],
            'state_issues': ['Operation exceeded timeout']
        }
        
        if visual_state:
            loading_elements = [
                elem for elem in visual_state.get('elements', [])
                if elem.get('type') == 'loading'
            ]
            if loading_elements:
                diagnosis['visual_factors'].append('Perpetual loading indicator')
                diagnosis['likely_causes'].insert(0, 'Page stuck in loading state')
        
        return diagnosis
    
    def _diagnose_generic_error(self, action: Dict) -> Dict:
        return {
            'description': 'Unknown error occurred',
            'likely_causes': [
                'Unexpected page state',
                'JavaScript error',
                'Browser compatibility issue',
                'Resource conflict'
            ],
            'visual_factors': [],
            'state_issues': ['Unknown error condition']
        }
    
    def _generate_recovery_plan(self, diagnosis: Dict, context: Dict) -> List[Dict]:
        recovery_plan = [{
            'step': 'verify_state',
            'description': 'Verify current page state',
            'action': 'check_page_state'
        }]
        
        recovery_map = {
            'element_not_found': self._plan_missing_element_recovery,
            'element_not_clickable': self._plan_not_clickable_recovery,
            'network_error': self._plan_network_error_recovery,
            'timeout': self._plan_timeout_recovery
        }
        
        if diagnosis['error_type'] in recovery_map:
            recovery_plan.extend(recovery_map[diagnosis['error_type']](diagnosis))
        
        recovery_plan.append({
            'step': 'verify_recovery',
            'description': 'Verify recovery success',
            'action': 'verify_state'
        })
        
        return recovery_plan
    
    def _plan_missing_element_recovery(self, diagnosis: Dict) -> List[Dict]:
        steps = []
        
        if 'Page still loading' in diagnosis['likely_causes']:
            steps.append({
                'step': 'wait_for_load',
                'description': 'Wait for page to finish loading',
                'action': 'wait_for_load_state',
                'timeout': 10000
            })
        
        if 'Element might be in iframe' in diagnosis['likely_causes']:
            steps.append({
                'step': 'check_iframes',
                'description': 'Check for element in iframes',
                'action': 'search_iframes'
            })
        
        steps.append({
            'step': 'try_alternatives',
            'description': 'Try alternative element selectors',
            'action': 'try_alternative_selectors'
        })
        
        return steps
    
    def _plan_not_clickable_recovery(self, diagnosis: Dict) -> List[Dict]:
        steps = []
        
        if any('overlapped' in factor for factor in diagnosis['visual_factors']):
            steps.append({
                'step': 'handle_overlay',
                'description': 'Remove overlapping elements',
                'action': 'clear_overlays'
            })
        
        if 'Element outside viewport' in diagnosis['visual_factors']:
            steps.append({
                'step': 'scroll_to_element',
                'description': 'Scroll element into view',
                'action': 'scroll_into_view'
            })
        
        return steps
    
    def _plan_network_error_recovery(self, diagnosis: Dict) -> List[Dict]:
        return [
            {
                'step': 'retry_connection',
                'description': 'Retry network connection',
                'action': 'retry_request',
                'max_retries': 3
            },
            {
                'step': 'check_connectivity',
                'description': 'Verify network connectivity',
                'action': 'check_network'
            }
        ]
    
    def _plan_timeout_recovery(self, diagnosis: Dict) -> List[Dict]:
        steps = [{
            'step': 'extend_timeout',
            'description': 'Increase timeout duration',
            'action': 'set_timeout',
            'timeout': 30000
        }]
        
        if 'Perpetual loading indicator' in diagnosis['visual_factors']:
            steps.append({
                'step': 'force_refresh',
                'description': 'Force page refresh',
                'action': 'refresh_page'
            })
        
        return steps
    
    def _check_overlap(self, bbox1: tuple, bbox2: tuple) -> bool:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def _is_in_viewport(self, bbox: tuple, viewport: Dict) -> bool:
        x, y, w, h = bbox
        return (0 <= x <= viewport['width'] and
                0 <= y <= viewport['height'] and
                0 <= x + w <= viewport['width'] and
                0 <= y + h <= viewport['height'])
    
    def _update_history(self, analysis: Dict):
        self.debug_history.append(analysis)
        if len(self.debug_history) > self.max_history_size:
            self.debug_history = self.debug_history[-self.max_history_size:]

class ErrorLevel(Enum):
    DEBUG = '\033[96m'    # Cyan
    INFO = '\033[94m'     # Blue
    WARNING = '\033[93m'  # Yellow
    ERROR = '\033[91m'    # Red
    CRITICAL = '\033[41m' # Red background
    SUCCESS = '\033[92m'  # Green
    RESET = '\033[0m'     # Reset color

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger('error_handler')
        self.error_history = []
        self.max_history = 100

    def log_error(self, error: AIError):
        """Log an error and maintain history"""
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

    def get_error_stats(self) -> Dict:
        """Get error statistics for monitoring"""
        stats = {
            'total_errors': len(self.error_history),
            'error_types': {},
            'error_levels': {},
            'recent_errors': []
        }
        
        for error in self.error_history[-10:]:  # Last 10 errors
            stats['error_types'][error.code] = stats['error_types'].get(error.code, 0) + 1
            stats['error_levels'][error.level.name] = stats['error_levels'].get(error.level.name, 0) + 1
            stats['recent_errors'].append(error.to_dict())
            
        return stats

class RecoveryStrategy:
    def __init__(self, page):
        self.page = page
        self.logger = logging.getLogger('recovery')
        self.recovery_history = []

    async def attempt_recovery(self, error: AIError) -> bool:
        """Attempt to recover from an error"""
        self.logger.info(f"Attempting recovery for error: {error.code}")
        
        recovery_actions = {
            'navigation_error': self._handle_navigation_error,
            'element_not_found': self._handle_element_not_found,
            'timeout_error': self._handle_timeout_error,
            'network_error': self._handle_network_error
        }
        
        handler = recovery_actions.get(error.code)
        if handler:
            try:
                success = await handler(error)
                self.recovery_history.append({
                    'timestamp': time.time(),
                    'error': error.to_dict(),
                    'success': success
                })
                return success
            except Exception as e:
                self.logger.error(f"Recovery failed: {e}")
                return False
        else:
            self.logger.warning(f"No recovery strategy for error code: {error.code}")
            return False

    async def _handle_navigation_error(self, error: AIError) -> bool:
        """Handle navigation-related errors"""
        try:
            await self.page.reload()
            await self.page.wait_for_load_state('networkidle')
            return True
        except Exception as e:
            self.logger.error(f"Navigation recovery failed: {e}")
            return False

    async def _handle_element_not_found(self, error: AIError) -> bool:
        """Handle element not found errors"""
        try:
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Element recovery failed: {e}")
            return False

    async def _handle_timeout_error(self, error: AIError) -> bool:
        """Handle timeout errors"""
        try:
            await self.page.reload(timeout=30000)
            return True
        except Exception as e:
            self.logger.error(f"Timeout recovery failed: {e}")
            return False

    async def _handle_network_error(self, error: AIError) -> bool:
        """Handle network-related errors"""
        try:
            await asyncio.sleep(2)
            await self.page.reload()
            return True
        except Exception as e:
            self.logger.error(f"Network recovery failed: {e}")
            return False

class ElementDetector:
    def detect_elements(self, image):
        try:
            # Convert image to proper format
            if isinstance(image, str):
                image = cv2.imread(image)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            
            # Prepare image and run detection
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            
            # Process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]
            
            elements = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                element_type = self.element_types[int(label) % len(self.element_types)]
                box = [int(i) for i in box.tolist()]
                
                element = {
                    'type': element_type,
                    'bbox': tuple(box),
                    'confidence': float(score),
                    'attributes': self._extract_element_attributes(image, box)
                }
                elements.append(element)
            
            self._update_visual_memory(elements, image)
            return elements
            
        except cv2.error as e:
            error = ErrorHandler.format_cv_error(e)
            ErrorHandler.log_error(error)
            return []
        except Exception as e:
            if "CUDA" in str(e) or "model" in str(e).lower():
                error = ErrorHandler.format_ai_error(e)
            else:
                error = AIError(
                    "DETECTION_ERROR",
                    "Failed to detect elements in image",
                    ErrorLevel.ERROR,
                    {
                        "cause": str(e),
                        "traceback": traceback.format_exc()
                    }
                )
            ErrorHandler.log_error(error)
            return []

    def _extract_element_attributes(self, image, bbox):
        try:
            x1, y1, x2, y2 = bbox
            element_region = image.crop((x1, y1, x2, y2))
            element_np = np.array(element_region)
            
            return {
                'size': (x2 - x1, y2 - y1),
                'aspect_ratio': (x2 - x1) / (y2 - y1) if y2 > y1 else 0,
                'position': {'x': x1, 'y': y1},
                'is_clickable': self._is_likely_clickable(element_np),
                'has_text': self._has_text(element_np),
                'dominant_color': self._get_dominant_color(element_np),
                'visual_features': self._extract_visual_features(element_np)
            }
            
        except Exception as e:
            error = AIError(
                "ATTRIBUTE_EXTRACTION_ERROR",
                "Failed to extract element attributes",
                ErrorLevel.WARNING,
                {
                    "element_bbox": bbox,
                    "cause": str(e)
                }
            )
            ErrorHandler.log_error(error)
            return {}

class AIDebugger:
    async def analyze_failure(self, failed_action: Dict, error: Exception, 
                            visual_state: Dict, context: Dict) -> Dict:
        try:
            if not self.error_analyzer:
                from transformers import pipeline
                self.error_analyzer = pipeline('zero-shot-classification',
                                            candidate_labels=[
                                                'element_not_found',
                                                'element_not_clickable',
                                                'network_error',
                                                'timeout',
                                                'invalid_state',
                                                'permission_denied',
                                                'navigation_blocked',
                                                'form_validation',
                                                'resource_missing'
                                            ])
            
            error_text = f"{str(error)} {self._get_error_context(failed_action)}"
            error_classification = await asyncio.to_thread(self.error_analyzer, error_text)
            
            error_type = error_classification['labels'][0]
            confidence = error_classification['scores'][0]
            
            if confidence < 0.5:
                error = AIError(
                    "LOW_CONFIDENCE_CLASSIFICATION",
                    "Low confidence in error classification",
                    ErrorLevel.WARNING,
                    {
                        "confidence": f"{confidence:.2%}",
                        "suggested_type": error_type,
                        "action": "Manual verification recommended"
                    }
                )
                ErrorHandler.log_error(error)
            
            diagnosis = self._build_diagnosis(error_type, failed_action, visual_state)
            recovery_plan = self._generate_recovery_plan(diagnosis, context)
            
            return {
                'timestamp': time.time(),
                'action': failed_action,
                'error': str(error),
                'error_type': error_type,
                'confidence': confidence,
                'diagnosis': diagnosis,
                'recovery_plan': recovery_plan,
                'visual_state': visual_state,
                'context': context
            }
            
        except Exception as e:
            error = AIError(
                "DEBUG_ANALYSIS_ERROR",
                "Failed to analyze debug information",
                ErrorLevel.ERROR,
                {
                    "cause": str(e),
                    "action": failed_action.get('action_type', 'unknown'),
                    "traceback": traceback.format_exc()
                }
            )
            ErrorHandler.log_error(error)
            return {
                'timestamp': time.time(),
                'action': failed_action,
                'error': str(error),
                'error_type': 'analysis_failed',
                'confidence': 0.0,
                'diagnosis': None,
                'recovery_plan': None,
                'visual_state': visual_state,
                'context': context
            }