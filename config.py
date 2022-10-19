from pathlib import Path


ORI_PATH = Path(__file__).parent.resolve() / "data/ori/"#Path("./data/ori/")
TRAIN_PATH = Path(__file__).parent.resolve() / "data/train/" #Path("./data/train/")
VAL_PATH = Path(__file__).parent.resolve() / "data/val/" # Path("./data/val/")
TEST_PATH = Path(__file__).parent.resolve() / "data/test/" # Path("./data/test/")
FAKE_PATH = Path(__file__).parent.resolve() / "data/fake/" # Path("./data/fake/")

ORI_SAVE_ALL = Path(__file__).parent.resolve() / "data/all/" # Path("./data/all/")
RENDER_BG_SAVE_ALL = Path(__file__).parent.resolve() / "data/all_render_bg/" # Path("./data/all_render_bg/")
NOBG_SAVE_ALL = Path(__file__).parent.resolve() / "data/all_no_bg/"  # Path("./data/all_no_bg/")


MODEL_SAVE_ONLY_AE = Path(__file__).parent.resolve() / "results/only_ae/" # Path("./results/only_ae/")

MODEL_SAVE_CLASS_INIT = Path(__file__).parent.resolve() / "results/class_init/" # Path("./results/only_ae/")