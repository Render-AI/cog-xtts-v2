from cog import BaseModel, Input, Path, File
import logging
import io
import argparse
import os
import sys
import tempfile
import librosa.display
import numpy as np
import os
import torch
import torchaudio
import traceback
from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class TrainingOutput(BaseModel):
    weights: Path

def train(
    prefix: str = Input(description="data you wish to save", default="Test"),
) -> TrainingOutput:
    weights = Path("output.txt")
    with open(weights, "w") as f:
        f.write(prefix)

    return TrainingOutput(weights=weights)
