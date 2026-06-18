class PaymentService:
    def __init__(self, payment_repo):
        self._repo = payment_repo

    def get_payment(self, payment_id):
        return self._repo.find_by_id(payment_id)

    def create_payment(self, amount_cents: int):
        return self._repo.create(amount_cents=amount_cents)
