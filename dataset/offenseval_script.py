# coding=utf-8
# Lint as: python3

from __future__ import absolute_import, division, print_function

import csv
import logging
import os

import datasets


_CITATION = """\
@inproceedings{zampieri-etal-2019-semeval,
    title = "{S}em{E}val-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media ({O}ffens{E}val)",
    author = "Zampieri, Marcos  and
      Malmasi, Shervin  and
      Nakov, Preslav  and
      Rosenthal, Sara  and
      Farra, Noura  and
      Kumar, Ritesh",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-2010",
}
"""


_DESCRIPTION = """\
Offensive Language Identification Dataset (OLID), which contains over 14,000 English tweets, and it featured three sub-tasks.
"""

_FOLDER_NAME = "offenseval-{split}"


class OffensEvalConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(OffensEvalConfig, self).__init__(**kwargs)


class Offenseval(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        OffensEvalConfig(
            name="offenseval",
            version=datasets.Version("1.0.0"),
            description="OffensEval: Identifying and Categorizing Offensive Language in Social Media",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "tweet": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["NOT", "OFF"]),
                    # "subtask_b": datasets.features.ClassLabel(names=["NULL", "TIN", "UNT"]),
                    # "subtask_c": datasets.features.ClassLabel(names=["NULL", "IND", "GRP"])
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        "dataset", _FOLDER_NAME.format(split="train"), "olid-training-v1.0.tsv"
                    ),
                    "labelpath": None,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        "dataset", _FOLDER_NAME.format(split="test"), "testset-levela.tsv"
                    ),
                    "labelpath": os.path.join(
                        "dataset", _FOLDER_NAME.format(split="test"), "labels-levela.csv"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath, labelpath):
        logging.info("⏳ Generating examples from = %s", filepath)

        if labelpath:
            with open(filepath, encoding="utf-8") as f:
                with open(labelpath, encoding="utf-8") as f2:
                    reader_testset = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    reader_label = csv.DictReader(
                        f2, delimiter=",", quoting=csv.QUOTE_NONE, fieldnames=["id", "subtask_a"]
                    )
                    list_label = list(reader_label)
                    for idx, row in enumerate(reader_testset):
                        row_label = list_label[idx]
                        yield idx, {"idx": row["id"], "tweet": row["tweet"], "label": row_label["subtask_a"]}
        else:
            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for idx, row in enumerate(reader):
                    yield idx, {"idx": row["id"], "tweet": row["tweet"], "label": row["subtask_a"]}
