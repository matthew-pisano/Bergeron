from universalmodels import ModelInfo, pretrained_from_info, model_info_from_name, ModelSrc

from src.framework.framework_model import FrameworkModel


class Primary(FrameworkModel):
    """The primary model.  Takes user input and generates a response as normal"""

    def __init__(self, model_info: ModelInfo):
        """
        Args:
            model_info: The model information for the primary model"""

        self.model, self.tokenizer = pretrained_from_info(model_info)

    @classmethod
    def from_model_name(cls, primary_model_name: str, model_src: ModelSrc = ModelSrc.AUTO):
        """Creates a primary model from its name

        Args:
            primary_model_name: The name of the primary model
            model_src: The suggested source of the model to load. Defaults to AUTO
        Returns:
            An instance of a primary model"""

        p_model_info = model_info_from_name(primary_model_name, model_src=model_src, model_task="conversational")
        return cls(p_model_info)

    @property
    def name(self):
        return f"P({self.model.name_or_path})"

    def generate(self, prompt: str, **kwargs):
        return self.generate_using(prompt, self.model, self.tokenizer, **kwargs)
