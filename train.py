from cog import BasePredictor, File, Input, Path
import io

class TrainingOutput(BaseModel):
    # weights: Path
    out_path: str

def train(out_path: Input('out_path', default="/temp")) -> File:
    # return io.StringIO("hello " + out_path)
    return TrainingOutput(out_path=Path(out_path))
