from cog import BaseModel, Input, Path

class TrainingOutput(BaseModel):
    weights: Path

def train(
    prefix: str = Input(description="data you wish to save", default="Test"),
) -> TrainingOutput:
    weights = Path("output.txt")
    with open(weights, "w") as f:
        f.write(prefix)

    return TrainingOutput(weights=weights)
