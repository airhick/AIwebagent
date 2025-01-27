import logging
import traceback
from typing import Dict, Optional
import asyncio
import re

logger = logging.getLogger('task_executor')

class TaskExecutor:
    def __init__(self, page):
        self.page = page
        self.timeout = 30000  # 30 seconds default timeout
        self.navigation_timeout = 60000  # 60 seconds for navigation
        self.max_retries = 3

    async def execute_task_step(self, task_info: Dict, context: Dict) -> bool:
        """Execute a single task step"""
        try:
            action_type = task_info.get('action_type', 'search')
            logger.info(f"Executing task step: {action_type}")
            
            # Extract URL for navigation tasks
            if action_type == 'navigate':
                url = task_info.get('url', '')
                if not url.startswith(('http://', 'https://')):
                    url = f'https://{url}'
                logger.info(f"Direct navigation to: {url}")
                return await self._handle_navigation({'url': url})
            
            # For all other tasks, ensure we're on a valid page first
            current_url = self.page.url
            if not current_url or current_url == 'about:blank':
                logger.info("No active page, navigating to Google first")
                await self._handle_navigation({'url': 'https://www.google.com'})
            
            if action_type == 'search':
                return await self._handle_search(task_info)
            elif action_type == 'click':
                return await self._handle_click(task_info)
            elif action_type == 'type':
                return await self._handle_type(task_info)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Task step execution failed: {e}")
            traceback.print_exc()
            return False

    async def _handle_navigation(self, task_info: Dict) -> bool:
        """Handle navigation tasks with retries"""
        url = task_info.get('url', '')
        if not url:
            return False
            
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Navigation attempt {attempt + 1}/{self.max_retries} to: {url}")
                
                # Set navigation timeout
                self.page.set_default_navigation_timeout(self.navigation_timeout)
                
                # Try navigation with different wait_until strategies
                try:
                    await self.page.goto(url, wait_until='networkidle')
                except:
                    try:
                        await self.page.goto(url, wait_until='domcontentloaded')
                    except:
                        await self.page.goto(url, wait_until='load')
                
                # Verify navigation success
                current_url = self.page.url
                if current_url and current_url != 'about:blank':
                    logger.info(f"Successfully navigated to: {current_url}")
                    
                    # Handle cookie consent if present
                    await self._handle_cookie_consent()
                    
                    # Wait for page to be stable
                    await self._wait_for_page_stable()
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Navigation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
                    continue
                return False
        
        return False

    async def _wait_for_page_stable(self):
        """Wait for page to be stable"""
        try:
            # Wait for network to be idle
            await self.page.wait_for_load_state('networkidle', timeout=5000)
        except:
            pass
            
        try:
            # Wait for no visible network requests
            await self.page.wait_for_function('''
                () => !document.querySelector('.loading, .spinner, .progress')
            ''', timeout=5000)
        except:
            pass
            
        # Small delay to ensure stability
        await asyncio.sleep(0.5)

    async def _handle_search(self, task_info: Dict) -> bool:
        """Handle search tasks with retries"""
        query = task_info.get('query', '')
        if not query:
            return False
            
        for attempt in range(self.max_retries):
            try:
                # Ensure we're on Google
                if not self.page.url.startswith('https://www.google.com'):
                    logger.info("Navigating to Google")
                    if not await self._handle_navigation({'url': 'https://www.google.com'}):
                        continue
                
                logger.info(f"Searching for: {query}")
                
                # Handle cookie consent if present
                await self._handle_cookie_consent()
                
                # Find and fill search input
                search_input = await self.page.wait_for_selector(
                    'input[name="q"]',
                    timeout=self.timeout
                )
                
                if not search_input:
                    logger.error("Search input not found")
                    continue
                
                # Clear existing text and type new query
                await search_input.click()
                await search_input.fill('')
                await search_input.fill(query)
                await search_input.press('Enter')
                
                # Wait for search results
                await self._wait_for_page_stable()
                
                logger.info("Search completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return False
        
        return False

    async def _handle_click(self, task_info: Dict) -> bool:
        """Handle click tasks"""
        try:
            selector = task_info.get('selector', '')
            text = task_info.get('text', '')
            
            if text:
                # Try to find element by text
                element = await self.page.wait_for_selector(f'text="{text}"', timeout=self.timeout)
                if not element:
                    # Try partial text match
                    element = await self.page.wait_for_selector(f'text="{text}"', timeout=self.timeout)
            elif selector:
                element = await self.page.wait_for_selector(selector, timeout=self.timeout)
            else:
                logger.error("No selector or text provided for click action")
                return False
            
            if element:
                # Scroll element into view
                await element.scroll_into_view_if_needed()
                # Click the element
                await element.click()
                # Wait for any navigation or network activity
                await self.page.wait_for_load_state('networkidle')
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Click action failed: {e}")
            return False

    async def _handle_type(self, task_info: Dict) -> bool:
        """Handle type tasks"""
        try:
            selector = task_info.get('selector', '')
            text = task_info.get('text', '')
            
            if not text:
                logger.error("No text provided for type action")
                return False
            
            if selector:
                element = await self.page.wait_for_selector(selector, timeout=self.timeout)
                if element:
                    await element.fill(text)
                    return True
            else:
                # Try to find a focused element or common input types
                for input_selector in ['input:focus', 'textarea:focus', 'input[type="text"]', 'textarea']:
                    try:
                        element = await self.page.wait_for_selector(input_selector, timeout=5000)
                        if element:
                            await element.fill(text)
                            return True
                    except:
                        continue
            
            return False
            
        except Exception as e:
            logger.error(f"Type action failed: {e}")
            return False

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
                        logger.info("Cookie consent handled")
                        await self.page.wait_for_load_state('networkidle')
                        return
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Cookie consent handling failed: {e}")

    async def _wait_for_navigation(self, timeout: int = None):
        """Wait for page navigation to complete"""
        try:
            await self.page.wait_for_load_state('networkidle', timeout=timeout or self.navigation_timeout)
            await self.page.wait_for_load_state('domcontentloaded', timeout=timeout or self.navigation_timeout)
        except Exception as e:
            logger.warning(f"Navigation wait failed: {e}")

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """Extract URL from text if present"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        match = re.search(url_pattern, text)
        if match:
            return match.group(0)
        return None 