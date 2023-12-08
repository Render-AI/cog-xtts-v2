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
        normalize_voice: bool = Input(
            description=" Whether to normalize the conditioning audio.",
            default=False),            
        cleanup_output: bool = Input(
            description="Whether to apply additional processing to the output audio (microphone recordings)",
            default=False),
        use_vocoder: bool = Input(
            description="Whether to resample with vocoder.",
            default=False),            
        cleanup_output_mode: int = Input(
            description="Output processing mode",
            choices=[0, 1, 2],
            default=0),                  
        output_speaking_rate: float = Input(
            description="Output speaking rate.",
            ge=0.1, # GE = min value (Greater than, or Equal to)
            le=2, # LE = max value (Less than, or Equal to)
            default=1,),
        output_temperature: float = Input(
            description="Higher temperature produces more diversity / more unexpected outputs.",
            ge=0.1, # GE = min value (Greater than, or Equal to)
            le=10, # LE = max value (Less than, or Equal to)
            default=0.65,            
        ),    
        output_top_k: int = Input(
            description="Higher value produces more unexpected outputs.",
            ge=1, # GE = min value (Greater than, or Equal to)            
            default=35,            
        ),    
        output_top_p: float = Input(
            description="Higher value produces more unexpected outputs.",
            ge=1, # GE = min value (Greater than, or Equal to)            
            default=0.5,            
        ),   
        gpt_cond_len:  float = Input(
            description="Secs audio to be used as conditioning for the autoregressive model.",
            ge=6, # GE = min value (Greater than, or Equal to)            
            default=12,            
        ),   
        max_ref_len:  int = Input(
            description=" Maximum number of seconds of audio to be used as conditioning for the decoder.",
            ge=6, # GE = min value (Greater than, or Equal to)            
            default=30,            
        ),   
        output_length_penalty: float = Input(
            description="Higher value causes the model to produce more short/terse outputs.",
            ge=0.1, # GE = min value (Greater than, or Equal to)
            le=10, # LE = max value (Less than, or Equal to)
            default=1,            
        ),    
        output_repetition_penalty: float = Input(
            description="Repetition penalty (prevents long pauses and repetition of sounds like 'umm' and 'ahh').",
            ge=0.1, # GE = min value (Greater than, or Equal to)
            le=10, # LE = max value (Less than, or Equal to)
            default=2,            
        ),     
        enable_text_splitting: bool = Input(
            description="Whether to split the text into sentences and generate audio for each sentence. It allows you to have infinite input length but might loose important context between sentences.",
            default=False),       
        output_sample_rate: int = Input(
            description="Output sample rate.",
            choices=[22050, 24000, 44100, 48000],
            default=24000),           
        output_format: str = Input(
            description="Output format",
            choices=["wav", "mp3"],
            default="mp3"),
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
            language=language,
            length_penalty=output_length_penalty,
            temperature=output_temperature,
            top_p=output_top_p,
            top_k=output_top_k,
            repetition_penalty=output_repetition_penalty,
            speed=output_speaking_rate,
            enable_text_splitting=enable_text_splitting,
            gpt_cond_len=gpt_cond_len,
            # max_ref_len=max_ref_len
        )

        if cleanup_output is not False:
            # see: https://github.com/gemelo-ai/vocos
            import torchaudio
            filePath = '/tmp/output.wav'
            
            # Vocos
            if use_vocoder == True:
                from vocos import Vocos
                vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
                y, sr = torchaudio.load(filePath)
                if y.size(0) > 1:  # mix to mono
                    y = y.mean(dim=0, keepdim=True)
                if(sr != output_sample_rate):
                    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=output_sample_rate)
                y_hat = vocos(y)
                torchaudio.save("output.wav", y_hat, output_sample_rate)
                filePath = "output.wav"
                    
            # VoiceFixer
            from voicefixer import VoiceFixer
            voicefixer = VoiceFixer()
            current_working_directory = os.getcwd()
            # Mode 0: Original Model (suggested by default)
            # Mode 1: Add preprocessing module (remove higher frequency)
            # Mode 2: Train mode (might work sometimes on seriously degraded real speech)
            voicefixer.restore(
                # low quality .wav/.flac file
                input=os.path.join(current_working_directory, filePath),
                output=os.path.join(current_working_directory,
                                    "output-cleaned.wav"),  # save file path
                cuda=True,  # GPU acceleration
                mode=cleanup_output_mode                
            )
            filePath = "output-cleaned.wav"
            
            if output_format == 'mp3':
                from pydub import AudioSegment
                compressed = AudioSegment.from_wav(filePath)
                compressed.export("output-cleaned.mp3")
                filePath = "output-cleaned.mp3"

            return Path(filePath)

        else:
            # other Cog output formats:
            # return Path(path)
            # return ModelOutput(audio_out=Path('/tmp/output.wav'))
            filePath = "/tmp/output.wav"
            if output_format == 'mp3':
                from pydub import AudioSegment
                compressed = AudioSegment.from_wav(filePath)
                compressed.export("output-cleaned.mp3")
                filePath = "output-cleaned.mp3"
            return Path(filePath)
