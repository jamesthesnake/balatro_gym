from dataclasses import dataclass
from typing import Optional

@dataclass
class CurrentBlind:
    """Current blind information"""
    chips: float
    kind: Optional[str] = None

class SelectBlind:
    """Blind selection screen handler"""
    def __init__(self, blind_info: dict, connection):
        self.info = blind_info
        self.connection = connection
    
    def small(self) -> dict:
        return self.info["small"]
    
    def big(self) -> dict:
        return self.info["big"]
    
    def boss(self) -> dict:
        return self.info["boss"]
    
    async def select(self):
        """Select the current blind"""
        response = await self.connection.send_request("blind_select/select")
        
        if "Ok" in response["body"]:
            from .play import Play
            return Play(response["body"]["Ok"], self.connection)
        else:
            raise Exception(f"Select blind failed: {response['body']['Err']}")
    
    async def skip(self):
        """Skip the current blind"""
        response = await self.connection.send_request("blind_select/skip")
        
        if "Ok" in response["body"]:
            return SelectBlind(response["body"]["Ok"], self.connection)
        else:
            raise Exception(f"Skip blind failed: {response['body']['Err']}")

