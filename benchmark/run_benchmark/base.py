"""Inference budgets, answer parsing and shared explicit-silence chat protocol."""
class Adapter:
    native = False
    description_budget = diagnosis_budget = 256
    choice_budget = 32

    @staticmethod
    def parse_choice(text):
        answer = text.strip()
        return answer if answer in ("A", "B", "C", "D") else None

    def end_case(self):
        pass

    def close(self):
        self.end_case()
