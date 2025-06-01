class Shop:
    """Shop screen handler"""
    def __init__(self, shop_info: dict, connection):
        self.info = shop_info
        self.connection = connection
    
    def main_cards(self) -> list:
        return self.info.get("main", [])
    
    def vouchers(self) -> list:
        return self.info.get("vouchers", [])
    
    def boosters(self) -> list:
        return self.info.get("boosters", [])
    
    async def buy_main(self, index: int):
        """Buy a main shop item"""
        response = await self.connection.send_request("shop/buymain", {"index": index})
        
        if "Ok" in response["body"]:
            self.info = response["body"]["Ok"]
            return self
        else:
            raise Exception(f"Buy failed: {response['body']['Err']}")
    
    async def leave(self):
        """Leave the shop"""
        response = await self.connection.send_request("shop/continue")
        
        if "Ok" in response["body"]:
            from .blinds import SelectBlind
            return SelectBlind(response["body"]["Ok"], self.connection)
        else:
            raise Exception(f"Leave shop failed: {response['body']['Err']}")

