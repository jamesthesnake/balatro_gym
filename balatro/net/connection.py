# balatro/net/connection.py
import asyncio
import json
from typing import Optional, Dict
import logging

logger = logging.getLogger('balatro.connection')
logger.setLevel(logging.DEBUG)  # Enable debug logging

class Connection:
    """Network connection handler for Balatro protocol"""
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.buffer = b""
        logger.debug("Connection initialized")
    
    async def send_message(self, kind: str, body: Optional[Dict] = None):
        """Send a message using the Balatro protocol (kind!body)"""
        if body is None or body == {}:
            message = f"{kind}!\n"  # Add newline
        else:
            body_json = json.dumps(body)
            message = f"{kind}!{body_json}\n"  # Add newline
        
        logger.debug(f"Sending: {repr(message)}")
        self.writer.write(message.encode())
        await self.writer.drain()
    
    async def read_message(self) -> Optional[Dict]:
        """Read a message from the connection"""
        while True:
            # Read more data if we don't have a complete message
            if b"!" not in self.buffer:
                try:
                    chunk = await self.reader.read(1024)
                    if not chunk:
                        logger.debug("No data received, connection closed")
                        return None
                    self.buffer += chunk
                    logger.debug(f"Buffer now: {self.buffer}")
                except Exception as e:
                    logger.error(f"Read error: {e}")
                    return None
            
            # Process messages in buffer
            while b"!" in self.buffer:
                exclamation_index = self.buffer.index(b"!")
                
                # Extract message up to the !
                message_bytes = self.buffer[:exclamation_index]
                # Keep the rest in buffer (skip the !)
                self.buffer = self.buffer[exclamation_index + 1:]
                
                # Decode and clean the message
                try:
                    message = message_bytes.decode('utf-8').strip()
                except:
                    logger.error(f"Failed to decode message bytes: {message_bytes}")
                    continue
                
                if not message:
                    continue
                
                logger.debug(f"Received message: '{message}'")
                
                # Handle ping/pong immediately
                if message == "ping":
                    logger.debug("Got ping, sending pong")
                    await self.send_message("pong")
                    continue
                
                if message == "pong":
                    logger.debug("Got pong")
                    continue
                
                # For result messages, parse the body
                body = {}
                if message.startswith("result/"):
                    # Look for JSON body after the message type
                    # Try to parse JSON from the remaining buffer
                    if self.buffer:
                        # Find where the JSON ends (look for balanced braces)
                        json_str = ""
                        brace_count = 0
                        i = 0
                        
                        while i < len(self.buffer):
                            char = chr(self.buffer[i])
                            json_str += char
                            
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found complete JSON
                                    try:
                                        body = json.loads(json_str)
                                        self.buffer = self.buffer[i+1:]  # Remove parsed JSON
                                        logger.debug(f"Parsed body: {body}")
                                    except:
                                        logger.error(f"Failed to parse JSON: {json_str}")
                                    break
                            i += 1
                
                return {"kind": message, "body": body}
    
    async def send_request(self, kind: str, body: Optional[Dict] = None) -> Dict:
        """Send a request and wait for the response"""
        logger.info(f"Sending request: {kind}")
        await self.send_message(kind, body)
        
        # Map request types to expected response types
        response_map = {
            "screen/get": "result/screen/current",
            "main_menu/start_run": "result/blind_select/info",
            "blind_select/select": "result/play/hand",
            "blind_select/skip": "result/blind_select/info",
            "play/click": "result/play/hand",
            "play/play": "result/play/play/result",
            "play/discard": "result/play/discard/result",
            "shop/continue": "result/shop/continue/result",  # Fixed this
            "shop/buymain": "result/shop/buymain/result",
            "shop/buyuse": "result/shop/buyuse/result",
            "shop/buyvoucher": "result/shop/buyvoucher/result",
            "shop/buybooster": "result/shop/buybooster/result",
            "shop/reroll": "result/shop/reroll",
            "overview/cash_out": "result/shop/info"
        }
        
        expected_result = response_map.get(kind, f"result/{kind}")
        logger.debug(f"Waiting for: {expected_result}")
        
        while True:
            response = await self.read_message()
            if not response:
                raise ConnectionError("Connection closed")
            
            logger.debug(f"Got response: {response['kind']}")
            
            if response["kind"] == expected_result:
                logger.info(f"Got expected response for {kind}")
                return response
            
            # Log unexpected messages but keep waiting
            logger.warning(f"Unexpected message while waiting for {expected_result}: {response}")
