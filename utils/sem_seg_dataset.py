import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
# from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, SINGLE_ANSWER_LIST, MULTI_ANSWER_LIST, EXPAND_QUESTION_LIST
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, SINGLE_ANSWER_LIST, MULTI_ANSWER_LIST, EXPAND_QUESTION_LIST


def init_mapillary(base_image_dir, local_rank):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir, local_rank):
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir, local_rank):
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir, local_rank):
    
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir, local_rank):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
        
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        num_classes_per_question=1,
        use_expand_question_list=False,
        local_rank=0,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size) 
        self.local_rank = local_rank

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.single_answer_list = SINGLE_ANSWER_LIST
        self.multi_answer_list = MULTI_ANSWER_LIST   
        self.num_classes_per_question = num_classes_per_question

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        if use_expand_question_list:
            self.short_question_list.extend(EXPAND_QUESTION_LIST)

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir, self.local_rank)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }
        self.local_rank = local_rank

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
        
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            max_num_classes_per_sample = self.num_classes_per_question * self.num_classes_per_sample
            if len(anns) >= max_num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=max_num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part  + "{seg}"
                    else:
                        name = "the {} of the {}".format(part, obj) + "{seg}"
                else:
                    name = sampled_cls + "{seg}"
                sampled_classes.append(name)
            sampled_anns, sampled_classes = allocate_class(sampled_anns, sampled_classes, max_question_num=self.num_classes_per_sample, max_class_per_question=self.num_classes_per_question)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            max_num_classes_per_sample = self.num_classes_per_question * self.num_classes_per_sample
            if len(classes) >= max_num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=max_num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes
            _, sampled_classes = allocate_class(None, sampled_classes, max_question_num=self.num_classes_per_sample, max_class_per_question=self.num_classes_per_question)
            
        questions = []
        answers = []
        class_ids = []

        if ds == 'paco_lvis':
            for sampled_classes_per_question in sampled_classes:
                target = ''
                _seg = []
                for i, sampled_cls in enumerate(sampled_classes_per_question):
                    text = sampled_cls
                    assert len(text.split("||")) == 1
                    if i == len(sampled_classes_per_question) - 1:
                        if text.count('{seg}') > 0:
                            _seg.append('[SEG]')
                            target = target + (' and '  + text) if i != 0 else target + text
                        elif text.count('{seg}') > 0:
                            _seg.append('[SEG]')
                            target = target + (' and '  + text) if i != 0 else target + text
                    elif i == 0:
                        if text.count('{seg}') > 0:
                            target += text
                            _seg.append('[SEG]')
                        elif text.count('{seg}') > 0:
                            target += text
                            _seg.append('[SEG]')
                    else:
                        if text.count('{seg}') > 0:
                            target += (', '  + text)  
                            _seg.append('[SEG]')
                        elif text.count('{seg}') > 0:
                            target += (', '  + text)
                            _seg.append('[SEG]')
                    if ds in ["paco_lvis", "pascal_part"]:
                        continue
                    class_id = self.data2classes[ds].tolist().index(sampled_cls)
                    class_ids.append(class_id) 
                if len(_seg) > 1:
                    part1 = ', '.join(_seg[:-1])
                    part2 = ' and ' + _seg[-1]
                    _seg = part1 + part2 
                else:
                    _seg = _seg[0]
            
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=target.lower().format(seg="")))
                separate_answer = random.randint(0, 1)
                
                if len(sampled_classes_per_question) == 1:
                    _target = target.format(seg="")
                    choice_list = self.answer_list
                    answer_temp = random.choice(choice_list)
                    if target.count('{seg}') > 0: 
                        answer_temp = answer_temp.format(class_name=_target.lower(), seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                    elif target.count('{seg}') > 0: 
                        answer_temp = answer_temp.format(class_name=_target.lower(), seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                    answers.append(answer_temp)
                    
                elif separate_answer:
                    target_answer = []
                    answer_temp = random.choice(self.single_answer_list)
                    for i, sampled_cls in enumerate(sampled_classes_per_question):
                        if sampled_cls.count('{seg}') > 0: 
                            _answer_temp = answer_temp.format(class_name=sampled_cls.format(seg=""), seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                        if sampled_cls.count('{seg}') > 0: 
                            _answer_temp = answer_temp.format(class_name=sampled_cls.format(seg=""), seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                        target_answer.append(_answer_temp[:-1])
                        
                    if len(target_answer) > 1:
                        part1 = ', '.join(target_answer[:-1])
                        part2 = ' and ' + target_answer[-1]
                        target_answer = part1 + part2 + '.'
                    else:
                        target_answer = target_answer[0] + '.'
                    answers.append(target_answer)
                else:
                    _target = target.format(seg="")
                    answer_temp = random.choice(self.multi_answer_list)
                    _answer_temp = answer_temp.format(class_name=_target.lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                    answers.append(_answer_temp)
        elif ds == 'pascal_part':
            for sampled_classes_per_question in sampled_classes:
                target = ''
                _seg = []
                for i, sampled_cls in enumerate(sampled_classes_per_question):
                    text = sampled_cls
                    assert len(text.split("||")) == 1
                    if i == len(sampled_classes_per_question) - 1:
                        if text.count('{seg}') > 0:
                            _seg.append('[SEG]')
                            target = target + (' and '  + text) if i != 0 else target + text
                    elif i == 0:
                        if text.count('{seg}') > 0:
                            target += text
                            _seg.append('[SEG]')
                    else:
                        if text.count('{seg}') > 0:
                            target += (', '  + text)  
                            _seg.append('[SEG]')
                            
                    if ds in ["paco_lvis", "pascal_part"]:
                        continue
                    class_id = self.data2classes[ds].tolist().index(sampled_cls)
                    class_ids.append(class_id) 
                if len(_seg) > 1:
                    part1 = ', '.join(_seg[:-1])
                    part2 = ' and ' + _seg[-1]
                    _seg = part1 + part2 
                else:
                    _seg = _seg[0]
            
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=target.lower().format(seg="")))
                separate_answer = random.randint(0, 1)
            
                if len(sampled_classes_per_question) == 1:
                    _target = target.format(seg="")
                    choice_list = self.answer_list
                    answer_temp = random.choice(choice_list)
                    answer_temp = answer_temp.format(class_name=_target.lower(), seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                    answers.append(answer_temp)
                elif separate_answer:
                    target_answer = []
                    answer_temp = random.choice(self.single_answer_list)
                    for i, sampled_cls in enumerate(sampled_classes_per_question):
                        _answer_temp = answer_temp.format(class_name=sampled_cls.format(seg=""), seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                        target_answer.append(_answer_temp[:-1])
                    if len(target_answer) > 1:
                        part1 = ', '.join(target_answer[:-1])
                        part2 = ' and ' + target_answer[-1]
                        target_answer = part1 + part2 + '.'
                    else:
                        target_answer = target_answer[0] + '.'
                    answers.append(target_answer)
                else:
                    _target = target.format(seg="")
                    answer_temp = random.choice(self.multi_answer_list)
                    _answer_temp = answer_temp.format(class_name=_target.lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                    answers.append(_answer_temp)
                
        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            for sampled_classes_per_question in sampled_classes:
                target = ''
                _seg = []
                for i, sampled_cls in enumerate(sampled_classes_per_question):
                    text = sampled_cls
                    assert len(text.split("||")) == 1
                    if i == len(sampled_classes_per_question) - 1:
                        _seg.append('[SEG]')
                        target = target + (' and '  + text) if i != 0 else target + text
                    elif i == 0:
                        target += text
                        _seg.append('[SEG]')
                    else:
                        _seg.append('[SEG]')
                        target += (', '  + text)

                    if ds in ["paco_lvis", "pascal_part"]:
                        continue
                    class_id = self.data2classes[ds].tolist().index(sampled_cls)
                    class_ids.append(class_id) 
                if len(_seg) > 1:
                    part1 = ', '.join(_seg[:-1])
                    part2 = ' and ' + _seg[-1]
                    _seg = part1 + part2 
                else:
                    _seg = _seg[0]
                    
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=target.lower()))
                separate_answer = random.randint(0, 1)

                if len(sampled_classes_per_question) == 1:
                    choice_list = self.answer_list
                    answer_temp = random.choice(choice_list)
                    answer_temp = answer_temp.format(class_name=target.lower(), seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                    answers.append(answer_temp)
                elif separate_answer:
                    target_answer = []
                    answer_temp = random.choice(self.single_answer_list)
                    for i, sampled_cls in enumerate(sampled_classes_per_question):
                        _answer_temp = answer_temp.format(class_name=sampled_cls, seg="[SEG]") if "{class_name}" in answer_temp else answer_temp.format(seg="[SEG]")
                        target_answer.append(_answer_temp[:-1])
                    if len(target_answer) > 1:
                        part1 = ', '.join(target_answer[:-1])
                        part2 = ' and ' + target_answer[-1]
                        target_answer = part1 + part2 + '.'
                    else:
                        target_answer = target_answer[0] + '.'
                    answers.append(target_answer)
                else:
                    answer_temp = random.choice(self.multi_answer_list)
                    _answer_temp = answer_temp.format(class_name=target.lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                    answers.append(_answer_temp)

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for sampled_anns_per_question in sampled_anns:
                for ann in sampled_anns_per_question:
                    try:
                        masks.append(coco_api.annToMask(ann))
                    except Exception as e:
                        print(e)
                        return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)
        

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )


def allocate_class(sampled_anns, sampled_ann_classes, max_question_num=3, max_class_per_question=3):
    if len(sampled_ann_classes) < max_question_num:
        max_question_num = len(sampled_ann_classes)
    sample_num = len(sampled_ann_classes)
    question_id = np.arange(max_question_num)
    class_counts = np.arange(max_question_num) * 0
    new_sampled_ann_ids = [[] for _ in range(max_question_num)] 
    new_sampled_ann_classes = [[] for _ in range(max_question_num)] 
    sample_ids = np.arange(sample_num)
    np.random.shuffle(sample_ids)
    for i in range(sample_num):
        if 0 in class_counts:
            choose_id = np.random.choice(np.where(class_counts == 0)[0], size=1)[0]
        else:
            choose_id = np.random.choice(np.where(class_counts < max_class_per_question)[0], size=1)[0]
        
        class_counts[choose_id] += 1
        sample_id = sample_ids[i]
        if sampled_anns is not None:
            new_sampled_ann_ids[choose_id].append(sampled_anns[sample_id])
        new_sampled_ann_classes[choose_id].append(sampled_ann_classes[sample_id])

    return new_sampled_ann_ids, new_sampled_ann_classes
