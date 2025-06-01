class RoundOverview:
    """Round overview screen handler"""
    def __init__(self, overview_info: dict, connection):
        self.info = overview_info
        self.connection = connection
    
    def total_earned(self) -> int:
        return self.info["total_earned"]
    
    def earnings(self) -> list:
        return self.info["earnings"]
    
    async def cash_out(self):
        """Cash out and continue to shop"""
        response = await self.connection.send_request("overview/cash_out")
        
        if "Ok" in response["body"]:
            from .shop import Shop
            return Shop(response["body"]["Ok"], self.connection)
        else:
            raise Exception(f"Cash out failed: {response['body']['Err']}")

class GameOverview:
    """Game over screen handler"""
    def __init__(self, connection):
        self.connection = connection

