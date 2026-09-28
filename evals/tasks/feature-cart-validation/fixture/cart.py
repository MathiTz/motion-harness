class Cart:
    def __init__(self):
        self.items = []

    def add_item(self, name, price, qty):
        self.items.append({"name": name, "price": price, "qty": qty})

    def total(self):
        return sum(i["price"] * i["qty"] for i in self.items)
