from src.framework.base_model import BaseModel
from src.models.model_utils import ModelInfo
from src.utils import model_info_from_name


class Primary(BaseModel):
    """The primary model.  Takes user input and generates a response as normal"""

    def __init__(self, model_info: ModelInfo):
        self.model, self.tokenizer = self.from_pretrained(model_info)

    @classmethod
    def from_model_name(cls, primary_model_name: str):
        """Creates a primary model from its name

        Args:
            primary_model_name: The name of the primary model
        Returns:
            An instance of a primary model"""

        p_model_info = ModelInfo(*model_info_from_name(primary_model_name), model_task="conversational")
        return cls(p_model_info)

    @property
    def name(self):
        return f"P({self.model.name_or_path})"

    def generate(self, prompt: str, **kwargs):
        """Generates a response to the given prompt using the primary model"""

        return self.generate_using(prompt, self.model, self.tokenizer, **kwargs)
