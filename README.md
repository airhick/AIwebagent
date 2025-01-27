# AI Web Assistant

A powerful web automation assistant that provides real-time browser control and task execution through an interactive web interface.

## Features

- **Real-Time Browser Control**: Live view of browser actions through WebSocket connection
- **Interactive Console**: Execute tasks directly from the web interface
- **Task Queue Management**: Handle multiple sequential tasks with proper queuing and execution
- **Robust Error Handling**: Automatic retries and recovery strategies
- **Visual Feedback**: Real-time status updates and browser screenshots
- **Cookie Consent Handling**: Automatic handling of cookie consent popups
- **Extensible Architecture**: Easy to add new task types and capabilities

## Requirements

- Python 3.8+
- Playwright
- aiohttp
- OpenCV (cv2)
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/airhick/AIwebagent.git
cd AIwebagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright browsers:
```bash
playwright install
```

## Usage

1. Start the assistant:
```bash
python src/main.py
```

2. Open your browser and navigate to `http://localhost:3000`

3. Enter tasks in the interactive console, for example:
- "go to google"
- "search for python programming"
- "go to reddit"

## Project Structure

```
.
├── src/
│   ├── main.py                 # Entry point
│   ├── web_assistant.py        # Core assistant class
│   ├── task_executor.py        # Task execution logic
│   ├── task_planner.py         # Task analysis and planning
│   ├── task_queue.py           # Async task queue management
│   └── element_detector.py     # Visual element detection
├── browser_viewer/
│   ├── index.html             # Web interface
│   └── browser_state.json     # Current browser state
└── browser_screenshots/       # Screenshot storage
```

## Features in Detail

### Real-Time Browser Control
- WebSocket-based communication
- Live screenshot updates
- Task status monitoring
- Interactive console input

### Task Management
- Asynchronous task queue
- Automatic retries on failure
- Progress tracking
- Error recovery strategies

### Visual Processing
- Screenshot capture and analysis
- Element detection
- Visual state verification
- Change detection

### Error Handling
- Automatic recovery attempts
- Detailed error logging
- Status updates
- Retry mechanisms

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Playwright for browser automation
- aiohttp for async web server
- OpenCV for visual processing 