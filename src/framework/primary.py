from src.framework.framework_model import FrameworkModel
from src.framework.framework_utils import ModelInfo, from_pretrained
from src.utils import model_info_from_name


class Primary(FrameworkModel):
    """The primary model.  Takes user input and generates a response as normal"""

    def __init__(self, model_info: ModelInfo):
        """
        Args:
            model_info: The model information for the primary model"""

        self.model, self.tokenizer = from_pretrained(model_info)

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
        return self.generate_using(prompt, self.model, self.tokenizer, **kwargs)
