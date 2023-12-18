from cog import BaseModel, Input, Path, File
import os
import torch
import torchaudio
import traceback
import logging
import io
import argparse
import sys
import tempfile
import librosa
import numpy as np

from trainers.formatter import format_audio_list
#from tts.TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
#from tts.TTS.tts.configs.xtts_config import XttsConfig
#from tts.TTS.tts.models.xtts import Xtts

class TrainingOutput(BaseModel):
    weights: Path

def train(
    prefix: str = Input(description="data you wish to save", default="Training was successful."),
) -> TrainingOutput:
    weights = Path("output.txt")
    with open(weights, "w") as f:
        f.write(prefix)

    return TrainingOutput(weights=weights)
