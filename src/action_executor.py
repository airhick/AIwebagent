class ActionExecutor:
    def __init__(self, page):
        self.page = page
    
    def execute_action(self, element, action_type):
        # Convert element coordinates to page coordinates
        x1, y1, x2, y2 = element['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        if action_type == 'click':
            self.page.mouse.click(center_x, center_y)
        elif action_type == 'type':
            self.page.mouse.click(center_x, center_y)
            self.page.keyboard.type(element['text'])
        elif action_type == 'scroll':
            self.page.mouse.wheel(0, 100)
    
    def wait_for_element(self, element_type, timeout=5000):
        # Wait for element to be visible
        try:
            self.page.wait_for_selector(f"[data-type='{element_type}']", 
                                      timeout=timeout)
            return True
        except:
            return False 