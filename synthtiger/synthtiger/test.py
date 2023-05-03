import synthtiger
from pprint import pprint
import numpy as np
import pandas as pd


config = synthtiger.read_config(
    "/Users/jongbeomkim/Desktop/workspace/scene_text_image_generator/synthtiger/examples/config.yaml"
)
pprint(config)

seed = 33
synthtiger.set_global_random_seed(seed)

template_path = "/Users/jongbeomkim/Desktop/workspace/scene_text_image_generator/synthtiger/examples/synthtiger/template.py"
template_name = "Singleline"
template = synthtiger.read_template(path=template_path, name=template_name, config=config)
template.init_save("/Users/jongbeomkim/Documents/synthtiger/")

generator = synthtiger.generator(
    path=template_path,
    name=template_name,
    config=config,
    count=50,
    worker=4,
    seed=seed,
    retry=True,
    verbose=True,
)
list(generator)


from synthtiger.components.color import GrayMap

colormap = GrayMap(**config["colormap2"])
colormap.colorize = 0
fg_color, bg_color = colormap.sample()
fg_color, bg_color



from synthtiger import components, layers, templates, utils


style = components.Switch(
    component=components.Selector(
        [
            components.TextBorder(),
            components.TextShadow(),
            components.TextExtrusion(),
        ]
    ),
    **config.get("style", {}),
)
style.component
fg_style = style.sample()
fg_style


angle = 45
radian = np.radians(angle)
offsets = np.array([np.cos(radian), -np.sin(radian)])
offsets
np.linalg.norm(offsets)


df = pd.read_csv("/Users/jongbeomkim/Documents/synthtiger/style.csv")
df.sort_values(by="image_path", inplace=True)
df.head()

percents = [[-0.5, 0.5], [-0.5, 0.5]]
percents = tuple(
    np.random.uniform(percent[0], percent[1]) for percent in percents
)
percents
np.tile(percents, 2)[: 2]