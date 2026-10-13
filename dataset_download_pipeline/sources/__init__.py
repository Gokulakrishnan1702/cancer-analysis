"""
Sources module exposing a registry of download functions for all 15 cancer types.
"""

from sources.lung import download_lung
from sources.breast import download_breast
from sources.colorectal import download_colorectal
from sources.melanoma import download_melanoma
from sources.ovarian import download_ovarian
from sources.prostate import download_prostate
from sources.pancreatic import download_pancreatic
from sources.renal import download_renal
from sources.gastric import download_gastric
from sources.endometrial import download_endometrial
from sources.bladder import download_bladder
from sources.esophageal import download_esophageal
from sources.thyroid import download_thyroid
from sources.lymphoma import download_lymphoma
from sources.liver import download_liver

SOURCE_DOWNLOADERS = {
    "lung": download_lung,
    "breast": download_breast,
    "colorectal": download_colorectal,
    "melanoma": download_melanoma,
    "ovarian": download_ovarian,
    "prostate": download_prostate,
    "pancreatic": download_pancreatic,
    "renal": download_renal,
    "gastric": download_gastric,
    "endometrial": download_endometrial,
    "bladder": download_bladder,
    "esophageal": download_esophageal,
    "thyroid": download_thyroid,
    "lymphoma": download_lymphoma,
    "liver": download_liver,
}
