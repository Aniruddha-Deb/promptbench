import jinja2

class PromptGenerator:

    def __init__(self, prompt_dir):

        self.env = jinja2.Environment(loader=jinja2.PackageLoader('promptbench', package_path=prompt_dir))

    def create_prompt(self, prompt_name, **kwargs):
        template = self.env.get_template(f"{prompt_name}.txt")

        return template.render(**kwargs)
