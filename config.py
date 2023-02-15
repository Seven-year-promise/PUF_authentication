from pathlib import Path

FILE_PATH = Path(__file__).parent.resolve()

PAPER_SAVE_PATH = FILE_PATH / "for_paper/"

# annotations
TRAIN_PATH = FILE_PATH / "data/annotations/train/" #Path("./data/train/")
VAL_PATH = FILE_PATH / "data/annotations/val/" # Path("./data/val/")
CANDIDATE_PATH = FILE_PATH / "data/annotations/candidate/" # Path("./data/test/")
FAKE_PATH = FILE_PATH / "data/annotations/fake/" # Path("./data/fake/")
ADD_PATH = FILE_PATH / "data/annotations/add/"
TEST_PATH = FILE_PATH / "data/annotations/test/"


# the images after rendering
ORI_SAVE_ALL = FILE_PATH / "data/all/" # Path("./data/all/")
RENDER_BG_SAVE_ALL = FILE_PATH / "data/all_render_bg/" # Path("./data/all_render_bg/")
NOBG_SAVE_ALL = FILE_PATH / "data/all_no_bg/"  # Path("./data/all_no_bg/")
CROP_SAVE_ALL = FILE_PATH / "data/all_crop/"  # Path("./data/all_no_bg/")
RGB_SAVE_ALL = FILE_PATH / "data/all_rgb/"  # Path("./data/all_no_bg/")
TEST_SAVEL_ALL_SMALLER = FILE_PATH / "data/all_test_smaller/"  # Path("./data/all_no_bg/")

MODEL_SAVE_ONLY_AE = FILE_PATH / "results/only_ae/" # Path("./results/only_ae/")

MODEL_SAVE_CLASS_INIT = FILE_PATH / "results/class_init/" # Path("./results/only_ae/")
MODEL_SAVE_CLASS_ADD = FILE_PATH / "results/class_add/" # Path("./results/only_ae/")

INIT_FEATURE_PATH = FILE_PATH / "data/features/init/"

RESULT_PATH = FILE_PATH / "results/"