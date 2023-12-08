# Prediction interface for Cog
from cog import BasePredictor, Input, Path, BaseModel
import os
from TTS.api import TTS


class ModelOutput(BaseModel):
    audio_out: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize",
            default="Hi there, I'm your new voice clone. Try your best to upload quality audio"
        ),
        speaker: Path = Input(
            description="Original speaker audio (wav, mp3, m4a, ogg, or flv)"),
        language: str = Input(
            description="Output language for the synthesised speech",
            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr",
                     "ru", "nl", "cs", "ar", "zh", "hu", "ko", "hi"],
            default="en"
        ),
        cleanup_voice: bool = Input(
            description="Whether to apply denoising to the input audio (good for laptop/phone recordings)",
            default=False),
        cleanup_output: bool = Input(
            description="Whether to apply additional processing to the output audio (microphone recordings)",
            default=False
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        speaker_wav = speaker
        filter = "highpass=75,lowpass=8000,"
        trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02"
        # ffmpeg convert to wav and apply afftn denoise filter. y to overwrite and avoid caching
        if cleanup_voice:
            os.system(
                f"ffmpeg -i {speaker} -af {filter}{trim_silence} -y {speaker_wav}")
        else:
            os.system(f"ffmpeg -i {speaker} -y {speaker_wav}")

        path = self.model.tts_to_file(
            text=text,
            file_path="/tmp/output.wav",
            speaker_wav=speaker_wav,
            language=language
        )

        if cleanup_output is not False:
            # see: https://github.com/gemelo-ai/vocos            
            import torchaudio
            from vocos import Vocos

            vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

            y, sr = torchaudio.load('/tmp/output.wav')
            if y.size(0) > 1:  # mix to mono
                y = y.mean(dim=0, keepdim=True)
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=44100)
            y_hat = vocos(y)
            torchaudio.save("output.wav", y_hat, 44100)
            
            # get the current working directory
            current_working_directory = os.getcwd()

            from voicefixer import VoiceFixer
            voicefixer = VoiceFixer()
            # or voicefixer = VoiceFixer(model='voicefixer/voicefixer')
            # Mode 0: Original Model (suggested by default)
            # Mode 1: Add preprocessing module (remove higher frequency)
            # Mode 2: Train mode (might work sometimes on seriously degraded real speech)
            for mode in [0,1,2]:
                print("Testing mode",mode)
                voicefixer.restore(
                    input=os.path.join(current_working_directory,"output.wav"), # low quality .wav/.flac file
                    output=os.path.join(current_working_directory,"output-cleaned.wav"), # save file path
                    cuda=True, # GPU acceleration
                    mode=mode
                )
                if (mode != 2):
                    check("output_mode_" + str(mode) + ".flac")
                print("Pass")
            
            # return ModelOutput(audio_out=Path('output.wav'))
            return Path('output-cleaned.wav')

        else:
            # return Path(path)
            # return ModelOutput(audio_out=Path('/tmp/output.wav'))
            return Path('/tmp/output.wav')
