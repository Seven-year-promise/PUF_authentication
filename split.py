from pathlib import Path
from  config import *
import json

def split(percentages=[]):
    im_path = list(ORI_SAVE_ALL.rglob("*.jpg"))

    data_num = len(im_path)
    print(data_num)
    train_num = int(data_num * percentages[0])
    val_num = int(data_num * percentages[1])
    test_num = int(data_num * percentages[2])
    fake_num =  data_num - train_num - val_num - test_num

    train_dict = {}
    for i in range(train_num):
        train_dict[i] = {}
        train_dict[i]["PUF"] = str(ORI_SAVE_ALL / (str(i)+".jpg"))
        train_dict[i]["height"] = 768
        train_dict[i]["width"] = 768
        train_dict[i]["PUF_no_bg"] = str(NOBG_SAVE_ALL / (str(i) + ".jpg"))
        train_dict[i]["PUF_render_bg"] = str(RENDER_BG_SAVE_ALL / (str(i) + ".jpg"))
        train_dict[i]["label"] = i

    val_dict = {}
    for i in range(train_num, train_num+val_num):
        val_dict[i-train_num] = {}
        val_dict[i-train_num]["PUF"] = str(ORI_SAVE_ALL / (str(i) + ".jpg"))
        val_dict[i-train_num]["height"] = 768
        val_dict[i-train_num]["width"] = 768
        val_dict[i-train_num]["PUF_no_bg"] = str(NOBG_SAVE_ALL / (str(i) + ".jpg"))
        val_dict[i-train_num]["PUF_render_bg"] = str(RENDER_BG_SAVE_ALL / (str(i) + ".jpg"))
        val_dict[i-train_num]["label"] = i

    test_dict = {}
    for i in range(train_num + val_num, train_num + val_num + test_num):
        test_dict[i-train_num-val_num] = {}
        test_dict[i-train_num-val_num]["PUF"] = str(ORI_SAVE_ALL / (str(i) + ".jpg"))
        test_dict[i-train_num-val_num]["height"] = 768
        test_dict[i-train_num-val_num]["width"] = 768
        test_dict[i-train_num-val_num]["PUF_no_bg"] = str(NOBG_SAVE_ALL / (str(i) + ".jpg"))
        test_dict[i-train_num-val_num]["PUF_render_bg"] = str(RENDER_BG_SAVE_ALL / (str(i) + ".jpg"))
        test_dict[i-train_num-val_num]["label"] = i

    fake_dict = {}
    for i in range(train_num + val_num + test_num, data_num):
        fake_dict[i-train_num-val_num-test_num] = {}
        fake_dict[i-train_num-val_num-test_num]["PUF"] = str(ORI_SAVE_ALL / (str(i) + ".jpg"))
        fake_dict[i-train_num-val_num-test_num]["height"] = 768
        fake_dict[i-train_num-val_num-test_num]["width"] = 768
        fake_dict[i-train_num-val_num-test_num]["PUF_no_bg"] = str(NOBG_SAVE_ALL / (str(i) + ".jpg"))
        fake_dict[i-train_num-val_num-test_num]["PUF_render_bg"] = str(RENDER_BG_SAVE_ALL / (str(i) + ".jpg"))
        fake_dict[i-train_num-val_num-test_num]["label"] = i

    with open(TRAIN_PATH/"train.json", "w", encoding="utf-8") as f:
        print(train_dict)
        json.dump(train_dict, f, ensure_ascii=False, indent=4)

    with open(VAL_PATH/"val.json", "w", encoding="utf-8") as f:
        json.dump(val_dict, f, ensure_ascii=False, indent=4)

    with open(TEST_PATH/"test.json", "w", encoding="utf-8") as f:
        json.dump(test_dict, f, ensure_ascii=False, indent=4)

    with open(FAKE_PATH/"fake.json", "w", encoding="utf-8") as f:
        json.dump(fake_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    split([0.5, 0.0, 0.25, 0.25])