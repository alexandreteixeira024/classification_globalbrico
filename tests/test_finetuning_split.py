"""Testes unitários para a divisão estratificada 80/20 e helpers de fine-tuning."""

import unittest

from src.finetuning import stratified_split


class TestFinetuningSplit(unittest.TestCase):
    def test_stratified_split_proportions(self):
        # Cria amostra com 20 Pedidos de Informação, 20 SPAM e 10 Pedidos de Encomenda
        examples = []
        for i in range(20):
            examples.append((f"info_{i}", f"Texto info {i}", "Pedido de Informação"))
        for i in range(20):
            examples.append((f"spam_{i}", f"Texto spam {i}", "SPAM"))
        for i in range(10):
            examples.append((f"enc_{i}", f"Texto enc {i}", "Pedido de Encomenda"))

        total = len(examples)  # 50
        train, test = stratified_split(examples, test_size=0.20, seed=42)

        self.assertEqual(len(train) + len(test), total)
        # 20% de 50 é 10
        self.assertEqual(len(test), 10)
        self.assertEqual(len(train), 40)

        # Proporções exatas por classe no teste:
        # 20 * 0.2 = 4 info
        # 20 * 0.2 = 4 spam
        # 10 * 0.2 = 2 enc
        info_test = sum(1 for _, _, y in test if y == "Pedido de Informação")
        spam_test = sum(1 for _, _, y in test if y == "SPAM")
        enc_test = sum(1 for _, _, y in test if y == "Pedido de Encomenda")

        self.assertEqual(info_test, 4)
        self.assertEqual(spam_test, 4)
        self.assertEqual(enc_test, 2)

    def test_stratified_split_single_example_class(self):
        # Se uma classe tiver apenas 1 exemplo, ela deve ir obrigatoriamente para o treino
        examples = [
            ("info_1", "Texto 1", "Pedido de Informação"),
            ("info_2", "Texto 2", "Pedido de Informação"),
            ("info_3", "Texto 3", "Pedido de Informação"),
            ("info_4", "Texto 4", "Pedido de Informação"),
            ("info_5", "Texto 5", "Pedido de Informação"),
            ("spam_1", "Texto 6", "SPAM"),
            ("spam_2", "Texto 7", "SPAM"),
            ("spam_3", "Texto 8", "SPAM"),
            ("spam_4", "Texto 9", "SPAM"),
            ("spam_5", "Texto 10", "SPAM"),
            ("enc_1", "Texto 11", "Pedido de Encomenda"),  # Apenas 1!
        ]

        train, test = stratified_split(examples, test_size=0.20, seed=42)

        # Verifica que o exemplo único de encomenda ficou no treino
        enc_train = [uid for uid, _, y in train if y == "Pedido de Encomenda"]
        enc_test = [uid for uid, _, y in test if y == "Pedido de Encomenda"]

        self.assertEqual(len(enc_train), 1)
        self.assertEqual(len(enc_test), 0)
        self.assertEqual(enc_train[0], "enc_1")

    def test_determinism_with_seed(self):
        examples = [(f"id_{i}", f"Texto {i}", "SPAM" if i % 2 == 0 else "Pedido de Informação") for i in range(30)]

        train1, test1 = stratified_split(examples, test_size=0.20, seed=42)
        train2, test2 = stratified_split(examples, test_size=0.20, seed=42)

        self.assertEqual([x[0] for x in train1], [x[0] for x in train2])
        self.assertEqual([x[0] for x in test1], [x[0] for x in test2])


if __name__ == "__main__":
    unittest.main()
