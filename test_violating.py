# Violates: "Never call the ORM directly from a service class."
class PaymentService:
    def get_payment(self, payment_id):
        return Payment.objects.get(id=payment_id)

    def create_payment(self, amount):
        payment = Payment.objects.create(amount=amount)
        return payment
