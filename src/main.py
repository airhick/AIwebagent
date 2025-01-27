from web_assistant import WebAssistant
import asyncio
import sys
import json

async def main():
    assistant = None
    try:
        print("\n" + "="*50)
        print("🚀 Starting Cloud Web Assistant")
        print("="*50 + "\n")
        
        print("📡 Initializing cloud environment...")
        assistant = await WebAssistant.create()
        
        print("\n💻 Cloud Browser Interface:")
        print("  • Access the live view at: http://localhost:3000")
        print("  • Real-time updates and interactions")
        print("  • Console output and status monitoring")
        print("\n" + "-"*50)
        
        while True:
            try:
                user_request = input("\n🤖 Enter task (or 'exit' to quit): ")
                
                if user_request.lower() == 'exit':
                    print("\n🔄 Shutting down cloud session...")
                    break
                
                if not user_request.strip():
                    continue
                
                print("\n▶️ Executing cloud task...")
                result = await assistant.execute_task(user_request)
                
                # Output result in a structured format
                response = {
                    "status": "success" if result else "error",
                    "task": user_request,
                    "timestamp": assistant.current_journey.steps[-1]['timestamp'] if assistant.current_journey and assistant.current_journey.steps else None
                }
                print("\n📊 Task Result:")
                print(json.dumps(response, indent=2))
                
            except Exception as e:
                error_response = {
                    "status": "error",
                    "task": user_request if 'user_request' in locals() else None,
                    "error": str(e)
                }
                print("\n❌ Task Error:")
                print(json.dumps(error_response, indent=2))
                continue
            
    except Exception as e:
        print("\n💥 Critical Error:")
        print(json.dumps({"status": "critical_error", "error": str(e)}, indent=2))
    finally:
        if assistant:
            print("\n" + "="*50)
            print("🔄 Cleaning up cloud resources...")
            await assistant.cleanup()
            print("✅ Cloud session ended successfully")
            print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main()) 